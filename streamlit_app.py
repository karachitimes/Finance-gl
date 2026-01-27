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
def get_distinct(col: str):
    # col is controlled by code (not user), safe to interpolate.
    q = text(f'SELECT DISTINCT {col} FROM public.v_finance_logic WHERE {col} IS NOT NULL ORDER BY {col}')
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
    q = text('SELECT DISTINCT EXTRACT(YEAR FROM "date")::int AS year FROM public.v_finance_logic ORDER BY year')
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

def run_scalar(sql: str, params: dict) -> float:
    with engine.connect() as conn:
        v = conn.execute(text(sql), params).scalar()
    return float(v or 0)

def run_df(sql: str, params: dict, columns: list[str] | None = None) -> pd.DataFrame:
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

    total_deposit = run_scalar(
        f"""
        select coalesce(sum(coalesce(credit_deposit,0)),0)
        from public.v_finance_logic
        where {where_sql}
        """,
        params,
    )

    pending_recoup_debit = run_scalar(
        f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from public.v_finance_logic
        where {where_sql}
          and bill_no ilike '%recoup%'
          and {_is_blank_sql('status')}
          and coalesce(account,'') <> coalesce(bank,'')
          and "date" >= :recoup_start
          and coalesce(bank,'') <> :bank_assignment
        """,
        {**params, "recoup_start": RECoup_START_DATE, "bank_assignment": bank_assignment},
    )

    completed_recoup = run_scalar(
        f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from public.v_finance_logic
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
          from public.v_finance_logic
          where {where_sql}
            and bill_no ilike '%recoup%'
            and {_is_blank_sql('status')}
            and coalesce(account,'') <> coalesce(bank,'')
            and "date" >= :recoup_start
        )
        select (p_debit - p_credit) from p
        """,
        {**params, "recoup_start": RECoup_START_DATE},
    )

    recoup_amount_revenue_bank = run_scalar(
        f"""
        select coalesce(sum(coalesce(credit_deposit,0)),0)
        from public.v_finance_logic
        where {where_sql}
          and bill_no ilike '%recoup%'
          and bank = :bank_revenue
        """,
        {**params, "bank_revenue": bank_revenue},
    )

    revenue_exp_not_recoup = run_scalar(
        f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from public.v_finance_logic
        where {where_sql}
          and bill_no ilike '%recoup%'
          and bank = :bank_revenue
        """,
        {**params, "bank_revenue": bank_revenue},
    )

    exp_recoup_from_assignment = run_scalar(
        f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from public.v_finance_logic
        where {where_sql}
          and bill_no ilike '%recoup%'
          and bank = :bank_assignment
        """,
        {**params, "bank_assignment": bank_assignment},
    )

    total_expenses_revenue_dr = run_scalar(
        f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from public.v_finance_logic
        where {where_sql}
          and head_name = 'Expense'
          and bank = :bank_revenue
        """,
        {**params, "bank_revenue": bank_revenue},
    )

    total_expenses_revenue_cr = run_scalar(
        f"""
        select coalesce(sum(coalesce(credit_deposit,0)),0)
        from public.v_finance_logic
        where {where_sql}
          and head_name = 'Expense'
          and bank = :bank_revenue
        """,
        {**params, "bank_revenue": bank_revenue},
    )

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
# ---------------- Revenue tab ----------------
with tab_rev:
    st.subheader("Revenue (Monthly)")
    where, params, _ = build_where_from_ui(df, dt, bank, head, account, attribute, func_code, fy_label=fy_label, func_override="Revenue")

    # Use abs(signed_amount)/2 to avoid double‐counting mirrored debit/credit rows.
    sql = f"""
    select date_trunc('month', "date") as month,
           sum(abs(signed_amount))/2 as revenue
    from public.v_finance_logic
    where {' and '.join(where)}
      and entry_type = 'revenue'
    group by 1
    order by 1
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    if rows:
        df_rev = pd.DataFrame(rows, columns=["Month", "Revenue"])
        st.dataframe(df_rev, use_container_width=True)
        st.line_chart(df_rev.set_index("Month"))
        st.success(f"Total Revenue: {df_rev['Revenue'].sum():,.0f} PKR")
    else:
        st.info("No revenue rows found for selected filters/date range.")

# ---------------- Expense tab ----------------
with tab_exp:
    st.subheader("Expenses (Monthly)")
    where, params, _ = build_where_from_ui(df, dt, bank, head, account, attribute, func_code, fy_label=fy_label, func_override=None)

    # Sum expense_amount; expense transactions typically aren't duplicated the same way as revenue, so no division by 2.
    sql = f"""
    select date_trunc('month', "date") as month,
           sum(coalesce(expense_amount,0)) as expense
    from public.v_finance_logic
    where {' and '.join(where)}
      and entry_type = 'expense'
    group by 1
    order by 1
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    if rows:
        df_exp = pd.DataFrame(rows, columns=["Month", "Expense"])
        st.dataframe(df_exp, use_container_width=True)
        st.line_chart(df_exp.set_index("Month"))
        st.success(f"Total Expense: {df_exp['Expense'].sum():,.0f} PKR")
    else:
        st.info("No expense rows found for selected filters/date range.")

# ---------------- Cashflow tab ----------------
with tab_cf:
    st.subheader("Cashflow Summary (By Bank & Direction)")
    where, params, _ = build_where_from_ui(df, dt, bank, head, account, attribute, func_code, fy_label=fy_label, func_override=None)

    sql = f"""
    select
      coalesce(bank, 'UNKNOWN') as bank,
      direction,
      sum(signed_amount) as amount
    from public.v_finance_logic
    where {' and '.join(where)}
    group by 1,2
    order by 1,2
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    if rows:
        df_cf = pd.DataFrame(rows, columns=["Bank", "Direction", "Amount"])
        st.dataframe(df_cf, use_container_width=True)

        inflow = df_cf[df_cf["Direction"] == "in"]["Amount"].sum()
        outflow = df_cf[df_cf["Direction"] == "out"]["Amount"].sum()  # likely negative
        st.success(
            f"Inflow: {inflow:,.0f} PKR  |  Outflow: {abs(outflow):,.0f} PKR  |  Net: {(inflow+outflow):,.0f} PKR"
        )
    else:
        st.info("No rows found for selected filters/date range.")

# ---------------- Trial balance tab ----------------
with tab_tb:
    st.subheader("Trial Balance (As of To Date)")

    where = ['"date" <= :dt']
    params = {"dt": dt}

    if bank != "ALL":
        where.append("bank = :bank"); params["bank"] = bank
    if head != "ALL":
        where.append("head_name = :head_name"); params["head_name"] = head
    if account != "ALL":
        where.append("account = :account"); params["account"] = account
    if func_code != "ALL":
        where.append("func_code = :func_code"); params["func_code"] = func_code

    sql = f"""
    select
      account,
      sum(signed_amount) as balance
    from public.v_finance_logic
    where {' and '.join(where)}
    group by 1
    order by 1
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    if rows:
        df_tb = pd.DataFrame(rows, columns=["Account", "Balance"])
        st.dataframe(df_tb, use_container_width=True)
        st.success(f"Net (sum of balances): {df_tb['Balance'].sum():,.0f} PKR")
    else:
        st.info("No rows found for trial balance with current filters.")


# ---------------- Recoup KPIs tab (PowerPivot logic) ----------------
with tab_rec_kpi:
    st.subheader("Recoup KPIs (PowerPivot/DAX equivalent)")

    c0, c1 = st.columns(2)
    with c0:
        bank_revenue = st.text_input("Revenue Bank (for specific KPIs)", value=BANK_REVENUE_DEFAULT)
    with c1:
        bank_assignment = st.text_input("Assignment Bank (exclude / specific KPIs)", value=BANK_ASSIGNMENT_DEFAULT)

    # Use UI filters but ignore func_code for recoup KPIs by default
    where, params, _ = build_where_from_ui(df, dt, bank, head, account, attribute, func_code, fy_label=fy_label, func_override=None)
    where_sql = " and ".join(where)

    kpis = compute_powerpivot_metrics(where_sql, params, bank_revenue=bank_revenue, bank_assignment=bank_assignment)

    cols = st.columns(3)
    items = list(kpis.items())
    for i, (k, v) in enumerate(items):
        with cols[i % 3]:
            st.metric(k, f"{v:,.0f}")

    st.divider()
    st.caption("Pending Recoup (Net) by Head")

    pending_by_head_sql = f"""
        select head_name,
               coalesce(sum(coalesce(debit_payment,0) - coalesce(credit_deposit,0)),0) as pending_net
        from public.v_finance_logic
        where {where_sql}
          and bill_no ilike '%recoup%'
          and {_is_blank_sql('status')}
          and coalesce(account,'') <> coalesce(bank,'')
          and "date" >= :recoup_start
        group by 1
        order by 2 desc
        limit 100
    """
    df_pending = run_df(pending_by_head_sql, {**params, "recoup_start": RECoup_START_DATE}, ["Head", "Pending Net"])
    if df_pending.empty:
        st.info("No pending recoup rows under current filters.")
    else:
        st.dataframe(df_pending, use_container_width=True)

# ---------------- Receivables tab ----------------
with tab_receivables:
    st.subheader("Receivables (Billing & Collection)")
    # Build base filter ignoring func_code for receivables
    where_base, params_base, _ = build_where_from_ui(
        df,
        dt,
        bank,
        head,
        account,
        attribute,
        func_code,
        fy_label=fy_label,
        func_override=None,
    )
    # Build a safe WHERE clause string; if no conditions, default to 1=1
    where_clause = ' and '.join(where_base) if where_base else '1=1'
    # Only include receivable func_codes
    # Compute billing (AR raised)
    bill_sql = f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from public.v_finance_logic
        where {where_clause}
          and func_code in ('AGR','AMC','PAR','WAR')
          and coalesce(debit_payment,0) > 0
    """
    # Compute collection (AR collected)
    collect_sql = f"""
        select coalesce(sum(coalesce(credit_deposit,0)),0)
        from public.v_finance_logic
        where {where_clause}
          and func_code in ('AGR','AMC','PAR','WAR')
          and coalesce(credit_deposit,0) > 0
    """
    try:
        billed = run_scalar(bill_sql, params_base)
        collected = run_scalar(collect_sql, params_base)
    except Exception:
        billed = 0
        collected = 0
    outstanding = billed - collected
    c0, c1, c2 = st.columns(3)
    with c0:
        st.metric("AR Raised (Debit)", f"{billed:,.0f}")
    with c1:
        st.metric("AR Collected (Credit)", f"{collected:,.0f}")
    with c2:
        st.metric("Outstanding AR", f"{outstanding:,.0f}")

    st.divider()
    st.caption("Receivable Ledger (Last 1000 rows)")
    ledger_sql = f"""
        select
          "date",
          account,
          head_name,
          pay_to,
          description,
          debit_payment,
          credit_deposit,
          signed_amount as gl_amount,
          bill_no,
          voucher_no,
          reference_no
        from public.v_finance_logic
        where {where_clause}
          and func_code in ('AGR','AMC','PAR','WAR')
        order by "date" desc
        limit 1000
    """
    


# ---------------- AI Q&A tab ----------------
with tab_qa:
    st.subheader("Ask a Finance Question (Deterministic + Search)")
    st.caption(
        "Examples: revenue by head | revenue by head monthly | expense by head | monthly revenue trend | pending recoup amount | trial balance"
    )

    q = st.text_input("Ask anything…", placeholder="revenue by head monthly")

    if q:
        intent = detect_intent(q)
        payee = extract_payee(q)

        # Build WHERE from UI + intent override (SQL-only)
        func_override = apply_intent_func_override(intent, q, func_code)
        where, params, effective_func = build_where_from_ui(df, dt, bank, head, account, attribute, func_code, fy_label=fy_label, func_override=func_override)

        # Override date filter if question specifies relative dates
        date_sql, date_params = infer_date_sql(q)
        if date_sql:
            
            where = [
                w for w in where
                if "between :df and :dt" not in w
                and '"date" between' not in w
            ]
            where.insert(0, date_sql)
            params.update(date_params)

        # Month-range overrides date filter
        m_start, m_end_excl = parse_month_range(q)
        if m_start and m_end_excl:
            where = [
                w for w in where
                if "between :df and :dt" not in w
                and '"date" between' not in w
                and '"date" >=' not in w
                and '"date" <' not in w
            ]

            where.insert(0, '"date" >= :m_start and "date" < :m_end')
            params["m_start"] = m_start
            params["m_end"] = m_end_excl

        if payee:
            where.append("pay_to ilike :payee")
            params["payee"] = f"%{payee}%"

        where_sql = " and ".join(where)
        struct = detect_structure(q)
        ql = q.lower()

        label = ""
        sql = ""

        # ---------- Revenue ----------
        if intent == "revenue":
            if struct["by_head"] and struct["monthly"]:
                label = "Revenue by Head (Monthly)"
                # Sum credit_deposit only (treat AGR/AMC as revenue) and exclude PAR/WAR and recoup transactions
                sql = f"""
                select date_trunc('month',"date") as month,
                       head_name,
                       sum(coalesce(credit_deposit,0)) as revenue
                from public.v_finance_logic
                where {where_sql}
                  and func_code in ('Revenue')
                  and credit_deposit > 0
                  and func_code not in ('Power','Water')
                  and coalesce(bill_no,'') not ilike '%recoup%'
                group by 1,2
                order by 1,3 desc
                """
            elif struct["by_head"]:
                label = "Revenue by Head"
                sql = f"""
                select head_name,
                       sum(coalesce(credit_deposit,0)) as revenue
                from public.v_finance_logic
                where {where_sql}
                  and func_code in ('Revenue')
                  and credit_deposit > 0
                  and func_code not in ('Power','Water')
                  and coalesce(bill_no,'') not ilike '%recoup%'
                group by 1
                order by 2 desc
                limit 50
                """
            elif struct["by_bank"]:
                label = "Revenue by Bank"
                sql = f"""
                select coalesce(bank,'UNKNOWN') as bank,
                       sum(coalesce(credit_deposit,0)) as revenue
                from public.v_finance_logic
                where {where_sql}
                  and func_code in ('Revenue')
                  and credit_deposit > 0
                  and func_code not in ('Power','Water')
                  and coalesce(bill_no,'') not ilike '%recoup%'
                group by 1
                order by 2 desc
                """
            elif struct["monthly"]:
                label = "Monthly Revenue"
                sql = f"""
                select date_trunc('month',"date") as month,
                       sum(coalesce(credit_deposit,0)) as revenue
                from public.v_finance_logic
                where {where_sql}
                  and func_code in ('Revenue')
                  and credit_deposit > 0
                  and func_code not in ('Power','Water')
                  and coalesce(bill_no,'') not ilike '%recoup%'
                group by 1
                order by 1
                """
            else:
                label = "Total Revenue"
                sql = f"""
                select coalesce(sum(coalesce(credit_deposit,0)),0) as revenue
                from public.v_finance_logic
                where {where_sql}
                  and func_code in ('Revenue')
                  and credit_deposit > 0
                  and func_code not in ('Power','Water')
                  and coalesce(bill_no,'') not ilike '%recoup%'
                """

        # ---------- Expense ----------
        elif intent == "expense":
            if struct["by_head"] and struct["monthly"]:
                label = "Expense by Head (Monthly)"
                # Expense is net outflow (signed_amount positive) for non-revenue/non-grant func codes
                sql = f"""
                select date_trunc('month',"date") as month,
                       head_name,
                       sum(signed_amount) as expense
                from public.v_finance_logic
                where {where_sql}
                  and func_code not in ('Revenue','Loan/Advance','Power','Water')
                  and signed_amount > 0
                  and coalesce(bill_no,'') not ilike '%recoup%'
                group by 1,2
                order by 1,3 desc
                """
            elif struct["by_head"]:
                label = "Expense by Head"
                sql = f"""
                select head_name,
                       sum(signed_amount) as expense
                from public.v_finance_logic
                where {where_sql}
                  and func_code not in ('Revenue','Loan/Advance','Power','Water')
                  and signed_amount > 0
                  and coalesce(bill_no,'') not ilike '%recoup%'
                group by 1
                order by 2 desc
                limit 50
                """
            elif struct["monthly"]:
                label = "Monthly Expense"
                sql = f"""
                select date_trunc('month',"date") as month,
                       sum(signed_amount) as expense
                from public.v_finance_logic
                where {where_sql}
                  and func_code not in ('Revenue','Loan/Advance','Power','Water')
                  and signed_amount > 0
                  and coalesce(bill_no,'') not ilike '%recoup%'
                group by 1
                order by 1
                """
            else:
                label = "Total Expense"
                sql = f"""
                select coalesce(sum(signed_amount),0) as expense
                from public.v_finance_logic
                where {where_sql}
                  and func_code not in ('Revenue','Loan/Advance','Power','Water')
                  and signed_amount > 0
                  and coalesce(bill_no,'') not ilike '%recoup%'
                """

        # ---------- Recoup ----------
        elif intent == "recoup":
            pending = ("pending" in ql) or ("outstanding" in ql) or ("not recouped" in ql)
            recouped = ("recouped" in ql) or ("settled" in ql)
            pending_minus_deposit = (
                ("pending recoup - deposit" in ql)
                or ("pending recoup minus deposit" in ql)
                or ("recoup - deposit" in ql)
            )
            if pending_minus_deposit:
                label = "Pending Recoup - Deposit"
                sql = f"""
                with p as (
                  select
                    coalesce(sum(coalesce(debit_payment,0)),0) as p_debit,
                    coalesce(sum(coalesce(credit_deposit,0)),0) as p_credit
                  from public.v_finance_logic
                  where {where_sql}
                    and bill_no ilike '%recoup%'
                    and {_is_blank_sql('status')}
                    and coalesce(account,'') <> coalesce(bank,'')
                    and "date" >= :recoup_start
                )
                select (p_debit - p_credit) as pending_minus_deposit from p
                """
                params["recoup_start"] = RECoup_START_DATE
            elif pending:
                label = "Pending Recoup Amount"
                sql = f"""
                select coalesce(sum(coalesce(recoup_pending_amount,0)),0) as pending_recoup
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='recoup'
                  and recoup_state='pending'
                  and bill_no ilike '%recoup%'
                  and {_is_blank_sql('status')}
                """
            elif recouped:
                label = "Recouped Total"
                sql = f"""
                select coalesce(sum(abs(signed_amount)),0) as recouped_total
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='recoup'
                  and recoup_state='recouped'
                  and bill_no ilike '%recoup%'
                  and {_not_blank_sql('status')}
                """
            else:
                label = "Recoup Total"
                sql = f"""
                select coalesce(sum(abs(signed_amount)),0) as recoup_total
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='recoup'
                  and bill_no ilike '%recoup%'
                """

        # ---------- Cashflow ----------
        elif intent == "cashflow":
            label = "Cashflow"
            sql = f"""
            select coalesce(bank,'UNKNOWN') as bank,
                   direction,
                   sum(signed_amount) as amount
            from public.v_finance_logic
            where {where_sql}
            group by 1,2
            order by 1,2
            """

        # ---------- Trial balance ----------
        elif intent == "trial_balance":
            label = "Trial Balance"
            sql = f"""
            select account,
                   sum(signed_amount) as balance
            from public.v_finance_logic
            where {where_sql}
            group by 1
            order by 1
            """

        # ---------- Search fallback ----------
        else:
            label = "Search matched total"
            params["q"] = q
            sql = f"""
            select coalesce(sum(signed_amount),0) as total
            from public.v_finance_logic
            where {where_sql}
              and (
                search_text % :q
                or search_tsv @@ plainto_tsquery('simple', :q)
              )
            """

        # ---------- Execute + Render ----------
        with engine.connect() as conn:
            if "group by" in sql.lower() or intent in ("cashflow", "trial_balance"):
                rows = conn.execute(text(sql), params).fetchall()
                if not rows:
                    st.warning("No rows found for this question with current filters.")
                else:
                    df_out = pd.DataFrame(rows)

                    # Set friendly column names when possible
                    if intent == "cashflow":
                        df_out.columns = ["Bank", "Direction", "Amount"]
                        st.dataframe(df_out, use_container_width=True)
                        inflow = df_out[df_out["Direction"] == "in"]["Amount"].sum()
                        outflow = df_out[df_out["Direction"] == "out"]["Amount"].sum()
                        st.success(f"Inflow: {inflow:,.0f} PKR | Outflow: {abs(outflow):,.0f} PKR | Net: {(inflow+outflow):,.0f} PKR")
                    elif intent == "trial_balance":
                        df_out.columns = ["Account", "Balance"]
                        st.dataframe(df_out, use_container_width=True)
                        st.success(f"Net (sum of balances): {df_out['Balance'].sum():,.0f} PKR")
                    else:
                        # special handling for head-by-month reports: pivot months to columns
                        if "Revenue by Head (Monthly)" in label or "Expense by Head (Monthly)" in label:
                            # Standardize column names
                            if "Revenue" in label:
                                df_out.columns = ["Month", "Head", "Revenue"]
                                value_col = "Revenue"
                            else:
                                df_out.columns = ["Month", "Head", "Expense"]
                                value_col = "Expense"
                            # Convert Month to datetime for proper sorting
                            df_out["Month"] = pd.to_datetime(df_out["Month"])
                            # Pivot to get months as columns
                            df_pivot = df_out.pivot(index="Head", columns="Month", values=value_col).fillna(0)
                            # Sort month columns chronologically
                            df_pivot = df_pivot.reindex(sorted(df_pivot.columns), axis=1)
                            # Rename columns to abbreviated month-year format
                            df_pivot.columns = [dt.strftime('%b-%y') for dt in df_pivot.columns]
                            # Reset index to turn Head into a column
                            df_pivot = df_pivot.reset_index().rename(columns={"Head": "Head Name"})
                            st.subheader(label)
                            st.dataframe(df_pivot, use_container_width=True)
                        else:
                            # heuristic naming for other reports
                            if label == "Monthly Revenue":
                                df_out.columns = ["Month", "Revenue"]
                                st.line_chart(df_out.set_index("Month"))
                            elif label == "Monthly Expense":
                                df_out.columns = ["Month", "Expense"]
                                st.line_chart(df_out.set_index("Month"))
                            elif label in ("Revenue by Head", "Expense by Head"):
                                df_out.columns = ["Head", "Amount"]
                            elif label == "Revenue by Bank":
                                df_out.columns = ["Bank", "Revenue"]
                            elif "Revenue by Head (Monthly)" in label:
                                df_out.columns = ["Month", "Head", "Revenue"]
                            elif "Expense by Head (Monthly)" in label:
                                df_out.columns = ["Month", "Head", "Expense"]
                            # Display table
                            st.subheader(label)
                            st.dataframe(df_out, use_container_width=True)
            else:
                val = conn.execute(text(sql), params).scalar() or 0
                st.success(f"{label}: {val:,.0f} PKR")

        with st.expander("🔍 Why this result?"):
            st.write(f"Intent detected: `{intent}`")
            st.write(f"Effective function filter: `{effective_func}`")
            if payee:
                st.write(f"Payee: `{payee}`")
            if m_start and m_end_excl:
                st.write(f"Month range: `{m_start}` to `{m_end_excl}` (end exclusive)")
            st.write("Filters applied:")
            st.write(f"- Bank: `{bank}`  |  Head: `{head}`  |  Account: `{account}`  |  Attribute: `{attribute}`  |  Function: `{func_code}`")
            st.write(f"- From: `{df}`  |  To: `{dt}`")
            st.write("SQL (debug):")
            st.code(sql.strip())
            st.write("Params (debug):")
            st.json({k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in params.items()})
