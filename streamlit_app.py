import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from datetime import date
from difflib import get_close_matches
import calendar
import re

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Finance Analytics System", layout="wide")

# -------------------------------------------------
# DB / ENGINE
# -------------------------------------------------
@st.cache_resource
def get_engine():
    """Create a SQLAlchemy Engine with sensible defaults for Streamlit Cloud."""
    url = st.secrets["DATABASE_URL"]
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=3,
        max_overflow=2,
    )

def test_connection(engine):
    with engine.connect() as conn:
        return conn.execute(text("select 1")).scalar()

engine = get_engine()

# -------------------------------------------------
# UI HEADER + DB STATUS
# -------------------------------------------------
st.title("📊 Finance Analytics System")

try:
    _ = test_connection(engine)
    st.success("Database connected ✅")
except OperationalError as e:
    st.error("Database connection failed")
    try:
        real_err = str(e.orig)
    except Exception:
        real_err = str(e)
    st.code(real_err)
    st.stop()

# -------------------------------------------------
# HELPERS (DB)
# -------------------------------------------------
@st.cache_data(ttl=3600)
def get_source_relation() -> str:
    """
    Return the best available relation for analytics queries.
    Prefer semantic view if it exists; otherwise fall back to raw table.
    """
    with engine.connect() as conn:
        try:
            conn.execute(text("SELECT 1 FROM public.v_finance_semantic LIMIT 1"))
            return "public.v_finance_semantic"
        except Exception:
            return "public.gl_register"


@st.cache_data(ttl=3600)
def relation_exists(rel: str) -> bool:
    """Check whether a table/view exists and is selectable."""
    with engine.connect() as conn:
        try:
            conn.execute(text(f"SELECT 1 FROM {rel} LIMIT 1"))
            return True
        except Exception:
            return False

def pick_relation(preferred: str, fallback: str = "public.v_finance_semantic") -> str:
    """
    Prefer a specific semantic view for a module. If it doesn't exist, fall back to
    v_finance_semantic, then finally raw gl_register.
    """
    if relation_exists(preferred):
        return preferred
    if relation_exists(fallback):
        return fallback
    return "public.gl_register"

REL = {
    "semantic": "public.v_finance_semantic",
    "revenue": "public.v_revenue",
    "expense": "public.v_expense",
    "receivable": "public.v_receivable",
    "recoup_pending": "public.v_recoup_pending",
    "recoup_completed": "public.v_recoup_completed",
    "cashflow": "public.v_cashflow",
    "fiscal_calendar": "public.v_fiscal_calendar",
}
@st.cache_data(ttl=3600)
def get_distinct(col: str):
    # col is controlled by code (not user). Restrict to known identifiers.
    allowed = {
        "pay_to", "func_code", "bank", "account", "attribute", "head_name",
        "bill_no", "status"
    }
    if col not in allowed:
        raise ValueError(f"Unsupported distinct column: {col}")

    rel = get_source_relation()
    q = text(f'SELECT DISTINCT {col} FROM {rel} WHERE {col} IS NOT NULL ORDER BY {col}')
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(q).fetchall()]

@st.cache_data(ttl=3600)
def get_known_payees():
    return get_distinct("pay_to")

KNOWN_PAYEES = get_known_payees()
KNOWN_FUNC_CODES = get_distinct("func_code")

@st.cache_data(ttl=3600)
def get_distinct_years() -> list[int]:
    """Return a list of distinct calendar years from the transaction dates."""
    rel = get_source_relation()
    q = text(f'SELECT DISTINCT EXTRACT(YEAR FROM "date")::int AS year FROM {rel} ORDER BY year')
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(q).fetchall()]

def best_payee_match(name: str | None):
    """
    Return the best matching payee name from KNOWN_PAYEES.
    If no close match is found, return None instead of the raw name to avoid
    accidentally treating generic words like 'head' or 'bank' as payees.
    """
    if not name:
        return None
    name = name.strip()
    if not name:
        return None
    matches = get_close_matches(name.title(), KNOWN_PAYEES, n=1, cutoff=0.75)
    return matches[0] if matches else None

# -------------------------------------------------
# NLP-ish ROUTING (DETERMINISTIC)
# -------------------------------------------------
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}

def detect_intent(q: str) -> str:
    ql = q.lower()
    if "trial balance" in ql or "tb" in ql or "balance as of" in ql:
        return "trial_balance"
    if "cashflow" in ql or "cash flow" in ql:
        return "cashflow"
    if "recoup" in ql:
        return "recoup"
    if any(w in ql for w in ["revenue", "income", "grant"]):
        return "revenue"
    if any(w in ql for w in ["expense", "cost", "paid", "payment", "wages", "salary"]):
        return "expense"
    return "search"

def parse_month_range(q: str, default_year: int | None = None):
    """Parses phrases like 'July to January' and returns (start_date, end_exclusive)."""
    ql = q.lower()
    default_year = default_year or date.today().year
    month_pattern = r"\b(" + "|".join(sorted(MONTHS.keys(), key=len, reverse=True)) + r")\b"
    found = re.findall(month_pattern, ql)
    found = [m.lower() for m in found if m.lower() in MONTHS]

    if len(found) >= 2:
        m1, m2 = MONTHS[found[0]], MONTHS[found[1]]
        y1 = default_year
        y2 = default_year + (1 if m2 < m1 else 0)
        start = date(y1, m1, 1)
        end_excl = date(y2 + 1, 1, 1) if m2 == 12 else date(y2, m2 + 1, 1)
        return start, end_excl

    return None, None

def infer_date_sql(q: str):
    """Returns (sql_fragment, params_dict) or (None, {})."""
    ql = q.lower()

    if "last month" in ql:
        return (
            "\"date\" >= date_trunc('month', current_date) - interval '1 month' "
            "and \"date\" < date_trunc('month', current_date)",
            {}
        )

    if "this month" in ql:
        return (
            "\"date\" >= date_trunc('month', current_date) "
            "and \"date\" < date_trunc('month', current_date) + interval '1 month'",
            {}
        )

    return None, {}



def extract_payee(q: str):
    """
    Extract a payee name from the question. Only matches patterns like
    'to <name>' to avoid capturing structures such as 'by head' or 'by bank'.
    Returns None if the extracted name is not a known payee.
    """
    ql = q.lower()
    # Only match patterns like 'to <payee>' and avoid 'by head', etc.
    m = re.search(r"(?:to)\s+([a-z\s]+?)(?:\s+with|\s+month|\s+for|$)", ql)
    if m:
        return best_payee_match(m.group(1))
    return None

def parse_explicit_func_code(q: str) -> str | None:
    """If user explicitly writes 'function X' / 'func_code X', return best match."""
    ql = q.lower()
    m = re.search(r"\b(?:function|func_code|func)\s*[:=]?\s*([a-z0-9_\-\s]+)", ql)
    if not m:
        return None
    raw = m.group(1).strip()
    if not raw:
        return None
    matches = get_close_matches(raw.title(), KNOWN_FUNC_CODES, n=1, cutoff=0.6)
    return matches[0] if matches else raw.title()

def detect_structure(q: str):
    ql = q.lower()
    return {
        "by_head": ("by head" in ql) or ("head wise" in ql) or ("head-wise" in ql) or ("head" in ql and "by" in ql),
        "by_bank": ("by bank" in ql) or ("bank wise" in ql) or ("bank-wise" in ql) or ("bank" in ql and "by" in ql),
        "monthly": ("monthly" in ql) or ("per month" in ql) or ("month wise" in ql) or ("month-wise" in ql) or ("trend" in ql),
        "top": ("top" in ql) or ("highest" in ql) or ("largest" in ql),
    }

# -------------------------------------------------
# FILTER BUILDER (keeps UI filters; override ONLY affects SQL)
# -------------------------------------------------
USE_UI = object()  # sentinel

def build_where_from_ui(
    df,
    dt,
    bank,
    head,
    account,
    attribute,
    func_code,
    *,
    fy_label: str | None = None,
    func_override=USE_UI,
) -> tuple[list[str], dict, str]:
    """
    Build a list of SQL conditions and a parameters dict based on UI selections.

    Args:
        df: start date (inclusive)
        dt: end date (inclusive)
        bank, head, account, attribute, func_code: selected filter values or "ALL".
        fy_label: optional fiscal year label like "FY2025-26". If provided and not
            "ALL", overrides the date range to the fiscal year window (July 1–June 30).
        func_override: if not USE_UI, overrides any selected func_code. Use None to
            ignore func_code entirely.

    Returns:
        where: list of SQL condition strings
        params: dict of parameters for bound variables
        effective_func: the func_code actually applied ("ALL" if none)
    """
    where = []
    params: dict[str, any] = {}

    # Apply date range or fiscal year
    if fy_label and fy_label != "ALL":
        # Expect format FYYYYY-YY (e.g., FY2025-26). Extract start year.
        try:
            start_year = int(fy_label.replace("FY", "").split("-")[0])
            # Fiscal year runs from July 1 to June 30 of next year.
            fy_start = date(start_year, 7, 1)
            fy_end = date(start_year + 1, 6, 30)
            where.append('"date" >= :fy_start')
            where.append('"date" <= :fy_end')
            params["fy_start"] = fy_start
            params["fy_end"] = fy_end
        except Exception:
            # fall back to explicit df/dt if parsing fails
            where.append('"date" between :df and :dt')
            params["df"] = df
            params["dt"] = dt
    else:
        # explicit date range
        where.append('"date" between :df and :dt')
        params["df"] = df
        params["dt"] = dt

    # Bank filter
    if bank != "ALL":
        where.append("bank = :bank"); params["bank"] = bank
    # Head filter
    if head != "ALL":
        where.append("head_name = :head_name"); params["head_name"] = head
    # Account filter
    if account != "ALL":
        where.append("account = :account"); params["account"] = account
    # Attribute filter
    if attribute != "ALL":
        where.append("attribute = :attribute"); params["attribute"] = attribute

    # Determine which func_code to use
    if func_override is USE_UI:
        effective_func = func_code if func_code != "ALL" else None
    elif func_override in (None, "ALL"):
        effective_func = None
    else:
        effective_func = func_override

    if effective_func is not None:
        where.append("func_code = :func_code"); params["func_code"] = effective_func

    return where, params, (effective_func if effective_func is not None else "ALL")

def apply_intent_func_override(intent: str, question: str, ui_func_code: str):
    """Return USE_UI, None, or a string func_code."""
    explicit = parse_explicit_func_code(question)
    if explicit:
        return explicit  # respect explicit request

    if intent == "revenue":
        return "Revenue"  # force Revenue for revenue intent
    if intent in ("expense", "recoup", "cashflow", "trial_balance", "search"):
        return None  # ignore UI func_code filter
    return USE_UI

# -------------------------------------------------
# POWERPIVOT / DAX EQUIVALENT METRICS (SQL)
# -------------------------------------------------
# These mirror the original Excel/PowerPivot measures you shared, but run directly in Postgres.

RECoup_START_DATE = date(2025, 7, 1)
BANK_REVENUE_DEFAULT = "Revenue:4069284635"
BANK_ASSIGNMENT_DEFAULT = "Assignment Account 1169255177"

def _is_blank_sql(col: str) -> str:
    # status can be blank or spaces; treat both as blank
    return f"NULLIF(BTRIM({col}), '') IS NULL"

def _not_blank_sql(col: str) -> str:
    return f"NULLIF(BTRIM({col}), '') IS NOT NULL"


def run_scalar(sql: str, params: dict, *, rel: str | None = None) -> float:
    """
    Execute a scalar SQL query safely with bound params.

    IMPORTANT:
    - `rel` is a server-side identifier chosen by code (NOT user input).
    - Use `{rel}` placeholder in SQL if you want the relation injected.
    """
    if rel:
        sql = sql.format(rel=rel)
    with engine.connect() as conn:
        v = conn.execute(text(sql), params).scalar()
    return float(v or 0)

def run_df(sql: str, params: dict, columns: list[str] | None = None, *, rel: str | None = None) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame."""
    if rel:
        sql = sql.format(rel=rel)
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    df_out = pd.DataFrame(rows)
    if columns and not df_out.empty:
        df_out.columns = columns
    return df_out
@st.cache_data(ttl=600)
def compute_powerpivot_metrics(where_sql: str, params: dict, bank_revenue: str = BANK_REVENUE_DEFAULT, bank_assignment: str = BANK_ASSIGNMENT_DEFAULT):
    """
    Return KPIs equivalent to your DAX measures, under current UI filter context.

    This function is cached so that repeated calls with the same SQL filter and parameters
    do not recompute the same aggregations on every interaction.  A TTL of 600
    seconds (10 minutes) is used to ensure that the cache stays reasonably fresh.
    """
    rel = pick_relation(REL['semantic'])

    total_deposit = run_scalar(
        f"""
        select coalesce(sum(coalesce(credit_deposit,0)),0)
        from {rel}
        where {where_sql}
        """,
        params,
    )

    pending_recoup_debit = run_scalar(
        f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from {rel}
        where {where_sql}
          and bill_no ilike '%recoup%'
          and {_is_blank_sql('status')}
          and coalesce(account,'') <> coalesce(bank,'')
          and "date" >= :recoup_start
          and coalesce(bank,'') <> :bank_assignment
        """,
        {**params, "recoup_start": RECoup_START_DATE, "bank_assignment": bank_assignment}, rel=rel)

    completed_recoup = run_scalar(
        f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from {rel}
        where {where_sql}
          and bill_no ilike '%recoup%'
          and {_not_blank_sql('status')}
          and coalesce(account,'') <> coalesce(bank,'')
        """,
        params,
    )

    pending_recoup_minus_deposit = run_scalar(
        f"""
        with p as (
          select
            coalesce(sum(coalesce(debit_payment,0)),0) as p_debit,
            coalesce(sum(coalesce(credit_deposit,0)),0) as p_credit
          from {rel}
          where {where_sql}
            and bill_no ilike '%recoup%'
            and {_is_blank_sql('status')}
            and coalesce(account,'') <> coalesce(bank,'')
            and "date" >= :recoup_start
        )
        select (p_debit - p_credit) from p
        """,
        {**params, "recoup_start": RECoup_START_DATE}, rel=rel)

    recoup_amount_revenue_bank = run_scalar(
        f"""
        select coalesce(sum(coalesce(credit_deposit,0)),0)
        from {rel}
        where {where_sql}
          and bill_no ilike '%recoup%'
          and bank = :bank_revenue
        """,
        {**params, "bank_revenue": bank_revenue}, rel=rel)

    revenue_exp_not_recoup = run_scalar(
        f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from {rel}
        where {where_sql}
          and bill_no ilike '%recoup%'
          and bank = :bank_revenue
        """,
        {**params, "bank_revenue": bank_revenue}, rel=rel)

    exp_recoup_from_assignment = run_scalar(
        f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from {rel}
        where {where_sql}
          and bill_no ilike '%recoup%'
          and bank = :bank_assignment
        """,
        {**params, "bank_assignment": bank_assignment}, rel=rel)

    total_expenses_revenue_dr = run_scalar(
        f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from {rel}
        where {where_sql}
          and head_name = 'Expense'
          and bank = :bank_revenue
        """,
        {**params, "bank_revenue": bank_revenue}, rel=rel)

    total_expenses_revenue_cr = run_scalar(
        f"""
        select coalesce(sum(coalesce(credit_deposit,0)),0)
        from {rel}
        where {where_sql}
          and head_name = 'Expense'
          and bank = :bank_revenue
        """,
        {**params, "bank_revenue": bank_revenue}, rel=rel)

    return {
        "Total Deposit": total_deposit,
        "Payments to be Recoup (Pending Debit)": pending_recoup_debit,
        "Completed Recoup (Debit)": completed_recoup,
        "Pending Recoup - Deposit": pending_recoup_minus_deposit,
        "Recoup Amount (Revenue Bank Credit)": recoup_amount_revenue_bank,
        "Revenue Bank Recoup Debit": revenue_exp_not_recoup,
        "Recoup Debit (Assignment Bank)": exp_recoup_from_assignment,
        "Total Expenses Revenue Dr": total_expenses_revenue_dr,
        "Total Expenses Revenue Cr": total_expenses_revenue_cr,
    }

# -------------------------------------------------
# UI FILTERS
# -------------------------------------------------
#
# To reduce unnecessary reruns, wrap all filter controls in a form.  The form will
# only trigger a rerun when the user clicks the "Apply Filters" button.  Selected
# filter values are stored in st.session_state so they persist across reruns.
if "filters_applied" not in st.session_state:
    # Initialize defaults on first run
    st.session_state.filters_applied = False
    st.session_state.df = date(2025, 1, 1)
    st.session_state.dt = date.today()
    st.session_state.bank = "ALL"
    st.session_state.head = "ALL"
    st.session_state.account = "ALL"
    st.session_state.attribute = "ALL"
    st.session_state.func_code = "ALL"
    st.session_state.fy_label = "ALL"

with st.form(key="filter_form"):
    c1, c2 = st.columns(2)
    with c1:
        new_df = st.date_input("From Date", value=st.session_state.df)
    with c2:
        new_dt = st.date_input("To Date", value=st.session_state.dt)

    # Pre-fetch distinct lists once for index lookups. These functions are cached via @st.cache_data.
    banks = ["ALL"] + get_distinct("bank")
    heads = ["ALL"] + get_distinct("head_name")
    accounts = ["ALL"] + get_distinct("account")
    try:
        attributes = get_distinct("attribute")
    except Exception:
        attributes = []
    attrs_list = ["ALL"] + sorted(attributes)
    funcs = ["ALL"] + get_distinct("func_code")

    # Compute indices for current selections to preserve state on rerun
    b_idx = banks.index(st.session_state.bank) if st.session_state.bank in banks else 0
    h_idx = heads.index(st.session_state.head) if st.session_state.head in heads else 0
    a_idx = accounts.index(st.session_state.account) if st.session_state.account in accounts else 0
    attr_idx = attrs_list.index(st.session_state.attribute) if st.session_state.attribute in attrs_list else 0
    f_idx = funcs.index(st.session_state.func_code) if st.session_state.func_code in funcs else 0

    new_bank = st.selectbox("Bank", banks, index=b_idx)
    new_head = st.selectbox("Head", heads, index=h_idx)
    new_account = st.selectbox("Account", accounts, index=a_idx)
    new_attribute = st.selectbox("Attribute", attrs_list, index=attr_idx)
    new_func_code = st.selectbox("Function Code", funcs, index=f_idx)
    # Fiscal Year filter: compute options from distinct years
    years = get_distinct_years()
    fy_options = ["ALL"] + [f"FY{y}-{(y+1)%100:02d}" for y in years]
    fy_idx = fy_options.index(st.session_state.fy_label) if st.session_state.fy_label in fy_options else 0
    new_fy_label = st.selectbox("Fiscal Year", fy_options, index=fy_idx)

    apply_filters = st.form_submit_button("Apply Filters")

if apply_filters or not st.session_state.filters_applied:
    # Update session state with new values
    st.session_state.filters_applied = True
    st.session_state.df = new_df
    st.session_state.dt = new_dt
    st.session_state.bank = new_bank
    st.session_state.head = new_head
    st.session_state.account = new_account
    st.session_state.attribute = new_attribute
    st.session_state.func_code = new_func_code
    st.session_state.fy_label = new_fy_label

# Read filter values from session state
df = st.session_state.df
dt = st.session_state.dt
bank = st.session_state.bank
head = st.session_state.head
account = st.session_state.account
attribute = st.session_state.attribute
func_code = st.session_state.func_code
fy_label = st.session_state.fy_label

# -------------------------------------------------
# TABS

# -------------------------------------------------
# TABS
# -------------------------------------------------

tab_rev, tab_exp, tab_cf, tab_tb, tab_rec_kpi, tab_receivables, tab_qa, tab_search = st.tabs(
    [
        "Revenue",
        "Expense",
        "Cashflow",
        "Trial Balance",
        "Recoup KPIs",
        "Receivables",
        "AI Q&A",
        "Search Description",
    ]
)

# --------------- Column-aware metric expressions ---------------
@st.cache_data(ttl=600)
def get_relation_columns(_engine, relation: str) -> set[str]:
    """Return a set of column names for a given table/view (schema-qualified)."""
    if '.' in relation:
        schema, name = relation.split('.', 1)
    else:
        schema, name = 'public', relation
    sql = """
      select lower(column_name) as column_name
      from information_schema.columns
      where table_schema = :schema and table_name = :name
    """
    with _engine.connect() as conn:
        rows = conn.execute(text(sql), {"schema": schema, "name": name}).fetchall()
    return {r[0] for r in rows}

def revenue_expr(cols: set[str]) -> str:
    # Prefer net_flow sign convention: revenue is inflow, often net_flow < 0
    if 'net_flow' in cols:
        return "sum(case when net_flow < 0 then -net_flow else 0 end)"
    if 'credit_deposit' in cols:
        return "sum(coalesce(credit_deposit,0))"
    if 'gl_amount' in cols:
        return "sum(abs(gl_amount))/2"
    return "sum(0)"

def expense_expr(cols: set[str]) -> str:
    # Expense is cash outflow, often net_flow > 0
    if 'net_flow' in cols:
        return "sum(case when net_flow > 0 then net_flow else 0 end)"
    if 'debit_payment' in cols:
        return "sum(coalesce(debit_payment,0))"
    if 'gl_amount' in cols:
        return "sum(case when gl_amount > 0 then gl_amount else 0 end)"
    return "sum(0)"

def cash_amount_expr(cols: set[str]) -> str:
    # Use net_flow if present (signed), otherwise gl_amount.
    if 'net_flow' in cols:
        return "sum(net_flow)"
    if 'gl_amount' in cols:
        return "sum(gl_amount)"
    return "sum(0)"

# ---------------- Revenue tab ----------------
with tab_rev:
    st.subheader("Revenue (Monthly)")

    rel = pick_relation(engine, "public.v_revenue")
    cols = get_relation_columns(engine, rel)

    where, params, _ = build_where_from_ui(
        df, dt, bank, head, account, attribute, func_code,
        fy_label=fy_label,
        func_override="Revenue",
    )

    sql = f"""
      select date_trunc('month', "date") as month,
             {revenue_expr(cols)} as revenue
      from {rel}
      where {' and '.join(where)}
      group by 1
      order by 1
    """

    df_rev = run_df(engine, sql, params)
    if not df_rev.empty:
        df_rev.columns = ["Month", "Revenue"]
        st.dataframe(df_rev, use_container_width=True)
        st.line_chart(df_rev.set_index("Month"))
        st.success(f"Total Revenue: {df_rev['Revenue'].sum():,.0f} PKR")
    else:
        st.info("No revenue rows found for selected filters/date range.")

# ---------------- Expense tab ----------------
with tab_exp:
    st.subheader("Expense (Net Cash Outflow, Monthly)")

    rel = pick_relation(engine, "public.v_expense")
    cols = get_relation_columns(engine, rel)

    # IMPORTANT: do not exclude recoup; v_expense should already implement semantic truth.
    where, params, _ = build_where_from_ui(
        df, dt, bank, head, account, attribute, func_code,
        fy_label=fy_label,
        func_override=None,
    )

    sql = f"""
      select date_trunc('month', "date") as month,
             {expense_expr(cols)} as expense
      from {rel}
      where {' and '.join(where)}
      group by 1
      order by 1
    """

    df_exp = run_df(engine, sql, params)
    if not df_exp.empty:
        df_exp.columns = ["Month", "Expense"]
        st.dataframe(df_exp, use_container_width=True)
        st.line_chart(df_exp.set_index("Month"))
        st.success(f"Total Expense (Outflow): {df_exp['Expense'].sum():,.0f} PKR")
    else:
        st.info("No expense rows found for selected filters/date range.")

# ---------------- Cashflow tab ----------------
with tab_cf:
    st.subheader("Cashflow Summary (By Bank & Direction)")

    rel = pick_relation(engine, "public.v_cashflow")
    cols = get_relation_columns(engine, rel)

    where, params, _ = build_where_from_ui(
        df, dt, bank, head, account, attribute, func_code,
        fy_label=fy_label,
        func_override=None,
    )

    # If direction column doesn't exist, derive it from net_flow/gl_amount.
    direction_sql = "direction" if "direction" in cols else (
        "case when coalesce(net_flow, gl_amount, 0) < 0 then 'in' else 'out' end"
    )

    sql = f"""
      select coalesce(bank, 'UNKNOWN') as bank,
             {direction_sql} as direction,
             {cash_amount_expr(cols)} as amount
      from {rel}
      where {' and '.join(where)}
      group by 1,2
      order by 1,2
    """

    df_cf = run_df(engine, sql, params)
    if not df_cf.empty:
        df_cf.columns = ["Bank", "Direction", "Amount"]
        st.dataframe(df_cf, use_container_width=True)

        inflow = df_cf[df_cf["Direction"] == "in"]["Amount"].sum()
        outflow = df_cf[df_cf["Direction"] == "out"]["Amount"].sum()
        st.success(f"Inflow: {inflow:,.0f} PKR | Outflow: {outflow:,.0f} PKR | Net: {(inflow+outflow):,.0f} PKR")
    else:
        st.info("No cashflow rows found for selected filters/date range.")

# ---------------- Trial Balance tab ----------------
with tab_tb:
    st.subheader("Trial Balance")

    rel = pick_relation(engine, "public.v_finance_semantic")
    where, params, _ = build_where_from_ui(df, dt, bank, head, account, attribute, func_code, fy_label=fy_label)

    # Column-aware balance calc: prefer net_flow/gl_amount.
    cols = get_relation_columns(engine, rel)
    amt = "sum(net_flow)" if "net_flow" in cols else "sum(gl_amount)" if "gl_amount" in cols else "sum(0)"

    sql = f"""
      select coalesce(account, head_name, 'UNKNOWN') as ledger,
             {amt} as balance
      from {rel}
      where {' and '.join(where)}
      group by 1
      order by 2 desc
    """

    df_tb = run_df(engine, sql, params)
    if not df_tb.empty:
        df_tb.columns = ["Ledger", "Balance"]
        st.dataframe(df_tb, use_container_width=True)
    else:
        st.info("No rows for trial balance under selected filters.")

# ---------------- Recoup KPIs tab ----------------
with tab_rec_kpi:
    st.subheader("Recoup KPIs")

    rel_pending = pick_relation(engine, "public.v_recoup_pending")
    rel_done = pick_relation(engine, "public.v_recoup_completed")

    where, params, _ = build_where_from_ui(df, dt, bank, head, account, attribute, func_code, fy_label=fy_label)

    # KPI blocks
    k1 = run_df(engine, f"select count(*) as n, sum(coalesce(net_flow, gl_amount, 0)) as amt from {rel_pending} where {' and '.join(where)}", params)
    k2 = run_df(engine, f"select count(*) as n, sum(coalesce(net_flow, gl_amount, 0)) as amt from {rel_done} where {' and '.join(where)}", params)

    c1, c2 = st.columns(2)
    if not k1.empty:
        c1.metric("Recoup Pending (Count)", int(k1.iloc[0,0] or 0))
        c1.metric("Recoup Pending (Amount)", f"{(k1.iloc[0,1] or 0):,.0f}")
    if not k2.empty:
        c2.metric("Recoup Completed (Count)", int(k2.iloc[0,0] or 0))
        c2.metric("Recoup Completed (Amount)", f"{(k2.iloc[0,1] or 0):,.0f}")

# ---------------- Receivables tab ----------------
with tab_receivables:
    st.subheader("Receivables Ledger")

    rel = pick_relation(engine, "public.v_receivable")
    where, params, _ = build_where_from_ui(df, dt, bank, head, account, attribute, func_code, fy_label=fy_label)

    sql = f"""
      select coalesce(head_name, account, 'UNKNOWN') as party,
             sum(coalesce(net_flow, gl_amount, 0)) as balance
      from {rel}
      where {' and '.join(where)}
      group by 1
      order by 2 desc
    """

    df_ar = run_df(engine, sql, params)
    if not df_ar.empty:
        df_ar.columns = ["Party", "Balance"]
        st.dataframe(df_ar, use_container_width=True)
    else:
        st.info("No receivables rows for selected filters.")

# ---------------- AI Q&A tab ----------------
with tab_qa:
    st.subheader("AI Q&A (Semantic Views)")
    st.caption("Uses semantic views to avoid crashing on raw table details. If OpenAI key not configured, this tab will not run.")

    # (Keep your existing AI code block here if you have one; ensure it queries v_finance_semantic + specific views.)
    st.info("AI Q&A module placeholder — keep your current implementation, but point it to v_finance_semantic / v_revenue / v_expense / v_cashflow.")

# ---------------- Search Description tab ----------------
with tab_search:
    st.subheader("Search Description")

    rel = pick_relation(engine, "public.v_finance_semantic")
    cols = get_relation_columns(engine, rel)

    q = st.text_input("Search text (description / pay_to / account / head_name)")

    where, params, _ = build_where_from_ui(df, dt, bank, head, account, attribute, func_code, fy_label=fy_label)

    if q:
        like_cols = [c for c in ["description", "pay_to", "account", "head_name"] if c in cols]
        if like_cols:
            where.append("(" + " or ".join([f"lower({c}) like lower(:q)" for c in like_cols]) + ")")
            params["q"] = f"%{q}%"

        sql = f"""
          select "date", bank, account, head_name, pay_to, description,
                 coalesce(net_flow, gl_amount, 0) as net_flow
          from {rel}
          where {' and '.join(where)}
          order by "date" desc
          limit 500
        """

        df_s = run_df(engine, sql, params)
        if not df_s.empty:
            st.dataframe(df_s, use_container_width=True)
            st.success(f"Net impact (sum net_flow): {df_s['net_flow'].sum():,.0f}")
        else:
            st.info("No matches.")
    else:
        st.info("Type a keyword above to search.")
