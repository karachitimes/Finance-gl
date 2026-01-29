import streamlit as st
import pandas as pd
import numpy as np
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
def relation_exists(fq: str) -> bool:
    with engine.connect() as conn:
        try:
            conn.execute(text(f"select 1 from {fq} limit 1"))
            return True
        except Exception:
            return False


@st.cache_data(ttl=3600)
def has_column(fq_relation: str, col: str) -> bool:
    """Check if a relation has a given column using information_schema."""
    if "." in fq_relation:
        schema, rel = fq_relation.split(".", 1)
    else:
        schema, rel = "public", fq_relation

    sql = text(
        """
        select 1
        from information_schema.columns
        where table_schema = :schema
          and table_name = :rel
          and column_name = :col
        limit 1
        """
    )
    with engine.connect() as conn:
        return conn.execute(sql, {"schema": schema, "rel": rel, "col": col}).scalar() is not None


def cashflow_net_expr(rel: str) -> str:
    """Return best available expression representing cash movement for a relation."""
    if has_column(rel, "net_flow"):
        return "coalesce(net_flow,0)"
    if has_column(rel, "gl_amount"):
        return "coalesce(gl_amount,0)"
    return "coalesce(debit_payment,0) - coalesce(credit_deposit,0)"


def cashflow_dir_expr(rel: str, net_expr: str) -> str:
    """Return direction column if present; otherwise compute from sign."""
    if has_column(rel, "direction"):
        return "direction"
    return f"case when {net_expr} >= 0 then 'out' else 'in' end"

@st.cache_data(ttl=3600)
def get_source_relation() -> str:
    """
    Return the best available relation for analytics queries.
    Prefer semantic view if it exists; otherwise fall back to raw table.
    """
    if relation_exists("public.v_finance_semantic"):
        return "public.v_finance_semantic"
    return "public.gl_register"

def expense_amount_expr(rel: str) -> str:
    """Best available expense outflow expression for a relation."""
    if has_column(rel, "net_flow"):
        return "coalesce(net_flow,0)"
    if has_column(rel, "gl_amount"):
        return "coalesce(gl_amount,0)"
    return "coalesce(debit_payment,0) - coalesce(credit_deposit,0)"


def tb_amount_expr(rel: str) -> str:
    """Best available amount expression for trial balance."""
    if has_column(rel, "gl_amount"):
        return "coalesce(gl_amount,0)"
    if has_column(rel, "net_flow"):
        return "coalesce(net_flow,0)"
    return "coalesce(debit_payment,0) - coalesce(credit_deposit,0)"


def tb_account_expr(rel: str) -> str:
    """Best available grouping key for trial balance."""
    if has_column(rel, "account"):
        return "account"
    if has_column(rel, "head_name"):
        return "head_name"
    if has_column(rel, "bank"):
        return "bank"
    return "'UNKNOWN'"


def pick_relation(candidates: list[str]) -> str:
    for c in candidates:
        if relation_exists(c):
            return c
    return candidates[-1]

# Prefer semantic views when available
REL_SEM = pick_relation(["public.v_finance_semantic", "public.gl_register"])
REL_REV = pick_relation(["public.v_revenue", REL_SEM])
REL_EXP = pick_relation(["public.v_expense", REL_SEM])
REL_CF  = pick_relation(["public.v_cashflow", REL_SEM])
REL_AR  = pick_relation(["public.v_receivable", REL_SEM])
REL_RP  = pick_relation(["public.v_recoup_pending", REL_SEM])
REL_RC  = pick_relation(["public.v_recoup_completed", REL_SEM])

@st.cache_data(ttl=3600)
def get_distinct(col: str):
    # col is controlled by code (not user). Restrict to known identifiers.
    allowed = {
        "pay_to", "func_code", "bank", "account", "attribute", "head_name",
        "bill_no", "status"
    }
    if col not in allowed:
        raise ValueError(f"Unsupported distinct column: {col}")

    q = text(f'SELECT DISTINCT {col} FROM {REL_SEM} WHERE {col} IS NOT NULL ORDER BY {col}')
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
    q = text(f'SELECT DISTINCT EXTRACT(YEAR FROM "date")::int AS year FROM {REL_SEM} ORDER BY year')
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
        "by_head": ("by head" in ql) or ("head wise" in ql) or ("head-wise" in ql),
        "by_bank": ("by bank" in ql) or ("bank wise" in ql) or ("bank-wise" in ql),
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
    Returns: (where_conditions, params, effective_func_label)
    """
    where = []
    params: dict[str, any] = {}

    # Apply date range or fiscal year
    if fy_label and fy_label != "ALL":
        try:
            start_year = int(fy_label.replace("FY", "").split("-")[0])
            fy_start = date(start_year, 7, 1)
            fy_end = date(start_year + 1, 6, 30)
            where.append('"date" >= :fy_start')
            where.append('"date" <= :fy_end')
            params["fy_start"] = fy_start
            params["fy_end"] = fy_end
        except Exception:
            where.append('"date" between :df and :dt')
            params["df"] = df
            params["dt"] = dt
    else:
        where.append('"date" between :df and :dt')
        params["df"] = df
        params["dt"] = dt

    if bank != "ALL":
        where.append("bank = :bank"); params["bank"] = bank
    if head != "ALL":
        where.append("head_name = :head_name"); params["head_name"] = head
    if account != "ALL":
        where.append("account = :account"); params["account"] = account
    if attribute != "ALL":
        where.append("attribute = :attribute"); params["attribute"] = attribute

    if func_override is USE_UI:
        effective_func = func_code if func_code != "ALL" else None
    elif func_override in (None, "ALL"):
        effective_func = None
    else:
        effective_func = func_override

    if effective_func is not None:
        where.append("func_code = :func_code"); params["func_code"] = effective_func

    return where, params, (effective_func if effective_func is not None else "ALL")

def apply_intent_func_override(intent: str, question: str):
    explicit = parse_explicit_func_code(question)
    if explicit:
        return explicit
    if intent == "revenue":
        return "Revenue"
    if intent in ("expense", "recoup", "cashflow", "trial_balance", "search"):
        return None
    return USE_UI

# -------------------------------------------------
# QUERY RUNNERS
# -------------------------------------------------
def run_scalar(sql: str, params: dict) -> float:
    with engine.connect() as conn:
        v = conn.execute(text(sql), params).scalar()
    return float(v or 0)

def run_df(sql: str, params: dict, columns: list[str] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    out = pd.DataFrame(rows)
    if columns and not out.empty:
        out.columns = columns
    return out


# -------------------------------------------------
# DATAFRAME DISPLAY HELPERS (Totals everywhere)
# -------------------------------------------------
def add_totals_table(df: pd.DataFrame, label_col: str | None = None, total_label: str = "TOTAL") -> pd.DataFrame:
    """Append a totals row for all numeric columns."""
    if df is None or df.empty:
        return df
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return out

    total_row = {c: "" for c in out.columns}
    for c in num_cols:
        total_row[c] = float(pd.to_numeric(out[c], errors="coerce").fillna(0).sum())

    if label_col and label_col in out.columns:
        total_row[label_col] = total_label
    else:
        for c in out.columns:
            if c not in num_cols:
                total_row[c] = total_label
                break

    return pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)


def add_totals_pivot(pivot: pd.DataFrame, total_label: str = "TOTAL") -> pd.DataFrame:
    """Add row totals + column totals + grand total to a pivot/cube."""
    if pivot is None or pivot.empty:
        return pivot
    out = pivot.copy()
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0)
    out[total_label] = out.sum(axis=1)
    total_row = out.sum(axis=0).to_frame().T
    total_row.index = [total_label]
    return pd.concat([out, total_row], axis=0)


def show_df(df: pd.DataFrame, *, label_col: str | None = None):
    if df is None or df.empty:
        st.info("No rows for selected filters.")
        return
    st.dataframe(add_totals_table(df, label_col=label_col), use_container_width=True)


def show_pivot(pivot: pd.DataFrame):
    if pivot is None or pivot.empty:
        st.info("No rows for selected filters.")
        return
    st.dataframe(add_totals_pivot(pivot), use_container_width=True)


# -------------------------------------------------
# POWERPIVOT / DAX EQUIVALENT METRICS (SQL)
# -------------------------------------------------
RECoup_START_DATE = date(2025, 7, 1)
BANK_REVENUE_DEFAULT = "Revenue:4069284635"
BANK_ASSIGNMENT_DEFAULT = "Assignment Account 1169255177"

def _is_blank_sql(col: str) -> str:
    return f"NULLIF(BTRIM({col}), '') IS NULL"

def _not_blank_sql(col: str) -> str:
    return f"NULLIF(BTRIM({col}), '') IS NOT NULL"

@st.cache_data(ttl=600)
def compute_powerpivot_metrics(where_sql: str, params: dict, bank_revenue: str = BANK_REVENUE_DEFAULT, bank_assignment: str = BANK_ASSIGNMENT_DEFAULT):
    total_deposit = run_scalar(
        f"""
        select coalesce(sum(coalesce(credit_deposit,0)),0)
        from {REL_SEM}
        where {where_sql}
        """,
        params,
    )

    pending_recoup_debit = run_scalar(
        f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from {REL_SEM}
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
        from {REL_SEM}
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
          from {REL_SEM}
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
        from {REL_SEM}
        where {where_sql}
          and bill_no ilike '%recoup%'
          and bank = :bank_revenue
        """,
        {**params, "bank_revenue": bank_revenue},
    )

    exp_recoup_from_assignment = run_scalar(
        f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from {REL_SEM}
        where {where_sql}
          and bill_no ilike '%recoup%'
          and bank = :bank_assignment
        """,
        {**params, "bank_assignment": bank_assignment},
    )

    return {
        "Total Deposit": total_deposit,
        "Payments to be Recoup (Pending Debit)": pending_recoup_debit,
        "Completed Recoup (Debit)": completed_recoup,
        "Pending Recoup - Deposit": pending_recoup_minus_deposit,
        "Recoup Amount (Revenue Bank Credit)": recoup_amount_revenue_bank,
        "Recoup Debit (Assignment Bank)": exp_recoup_from_assignment,
    }

# -------------------------------------------------
# UI FILTERS
# -------------------------------------------------
if "filters_applied" not in st.session_state:
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

    banks = ["ALL"] + get_distinct("bank")
    heads = ["ALL"] + get_distinct("head_name")
    accounts = ["ALL"] + get_distinct("account")
    try:
        attributes = get_distinct("attribute")
    except Exception:
        attributes = []
    attrs_list = ["ALL"] + sorted(attributes)
    funcs = ["ALL"] + get_distinct("func_code")

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

    years = get_distinct_years()
    fy_options = ["ALL"] + [f"FY{y}-{(y+1)%100:02d}" for y in years]
    fy_idx = fy_options.index(st.session_state.fy_label) if st.session_state.fy_label in fy_options else 0
    new_fy_label = st.selectbox("Fiscal Year", fy_options, index=fy_idx)

    apply_filters = st.form_submit_button("Apply Filters")

if apply_filters or not st.session_state.filters_applied:
    st.session_state.filters_applied = True
    st.session_state.df = new_df
    st.session_state.dt = new_dt
    st.session_state.bank = new_bank
    st.session_state.head = new_head
    st.session_state.account = new_account
    st.session_state.attribute = new_attribute
    st.session_state.func_code = new_func_code
    st.session_state.fy_label = new_fy_label

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
    st.subheader("Revenue Dashboard (Monthly)")

    where, params, _ = build_where_from_ui(
        df, dt, bank, head, account, attribute, func_code,
        fy_label=fy_label,
        func_override="Revenue",
    )
    where_sql = " and ".join(where) if where else "1=1"

    sql_month = f"""
    select
      date_trunc('month', "date")::date as month_start,
      to_char(date_trunc('month', "date"), 'Mon-YY') as month_label,
      sum(coalesce(credit_deposit,0)) as revenue_amount
    from {REL_REV}
    where {where_sql}
    group by 1,2
    order by 1
    """
    df_rev = run_df(sql_month, params, ["month_start", "month_label", "revenue_amount"])
    show_df(df_rev, label_col=\"month_label\")

    st.divider()
    st.caption("Monthly Revenue by Head (Pivot) — Head rows × Month columns")

    sql_pivot = f"""
    select
      to_char(date_trunc('month', "date"), 'Mon-YY') as month_label,
      head_name,
      sum(coalesce(credit_deposit,0)) as revenue_amount
    from {REL_REV}
    where {where_sql}
    group by 1,2
    order by 2,1
    """
    df_pivot = run_df(sql_pivot, params, ["month_label", "head_name", "revenue_amount"])

    if not df_pivot.empty:
        cube = df_pivot.pivot_table(
            index="head_name",
            columns="month_label",
            values="revenue_amount",
            aggfunc="sum",
            fill_value=0,
        )
        show_pivot(cube)
    else:
        st.info("No revenue data for selected filters.")

    st.divider()
    st.subheader("Finance Semantic Layer (How to read these numbers)")
    st.markdown("""
**Revenue cube**
- Rows: **Head Name**
- Columns: **Month**
- Measure: **credit_deposit** (revenue inflow)
- Source: `v_revenue` (fallback: `v_finance_semantic` / `gl_register`)

**Expense cube**
- Rows: **Head Name**
- Columns: **Month**
- Measure: **net_flow** (cash outflow)
- Rule: Expense is based on **column1 = 'Expense'**; it is **not removed** just because bill_no = 'Recoup'.

**Cashflow cube**
- Rows: **Bank × Direction**
- Columns: **Month**
- Measure: **abs(net_flow)** for in/out; net is **sum(net_flow)**

**Workflow separation (recoup ≠ expense)**
- **Recoup** is a settlement/workflow label (`bill_no='Recoup'`), tracked separately.
- **Expense** remains expense even if later recouped; recoup status is reported in recoup views.
""")

# ---------------- Expense tab ----------------
with tab_exp:
    st.subheader("Expenses (Monthly Net Cash Outflow)")

    where, params, _ = build_where_from_ui(
        df, dt, bank, head, account, attribute, func_code,
        fy_label=fy_label,
        func_override=None,
    )
    where_sql = " and ".join(where) if where else "1=1"
    exp_expr = expense_amount_expr(REL_EXP)


    sql = f"""
    select
      date_trunc('month', "date")::date as month,
      to_char(date_trunc('month', "date"), 'Mon-YY') as month_label,
      sum({exp_expr}) as expense_outflow
    from {REL_EXP}
    where {where_sql}
    group by 1,2
    order by 1
    """
    df_exp = run_df(sql, params, ["month", "month_label", "expense_outflow"])
    if not df_exp.empty:
        show_df(df_exp, label_col=\"month_label\")
        st.line_chart(df_exp.set_index("month_label")["expense_outflow"])
        st.success(f"Total Expense Outflow: {df_exp['expense_outflow'].sum():,.0f} PKR")
    else:
        st.info("No expense rows found for selected filters/date range.")

    st.divider()
    st.caption("Expense Cube — Head × Month (Pivot)")
    sql_exp_pivot = f"""
    select
      to_char(date_trunc('month', "date"), 'Mon-YY') as month_label,
      head_name,
      sum({exp_expr}) as outflow
    from {REL_EXP}
    where {where_sql}
    group by 1,2
    order by 2,1
    """
    df_exp_p = run_df(sql_exp_pivot, params, ["month_label", "head_name", "outflow"])
    if not df_exp_p.empty:
        cube = df_exp_p.pivot_table(index="head_name", columns="month_label", values="outflow", aggfunc="sum", fill_value=0)
        show_pivot(cube)

# ---------------- Cashflow tab ----------------
with tab_cf:
    st.subheader("Cashflow Summary (By Bank & Direction)")

    where, params, _ = build_where_from_ui(
        df, dt, bank, head, account, attribute, func_code,
        fy_label=fy_label,
        func_override=None,
    )
    where_sql = " and ".join(where) if where else "1=1"

    # Robust cashflow query
    net_expr = cashflow_net_expr(REL_CF)
    dir_expr = cashflow_dir_expr(REL_CF, net_expr)

    sql = f"""
    select
      coalesce(bank, 'UNKNOWN') as bank,
      {dir_expr} as direction,
      sum({net_expr}) as amount
    from {REL_CF}
    where {where_sql}
    group by 1,2
    order by 1,2
    """
    df_cf = run_df(sql, params, ["Bank", "Direction", "Amount"])

    if not df_cf.empty:
        show_df(df_cf, label_col=\"Bank\")

        inflow = df_cf[df_cf["Direction"] == "in"]["Amount"].abs().sum()
        outflow = df_cf[df_cf["Direction"] == "out"]["Amount"].abs().sum()
        net = inflow - outflow

        st.success(
            f"Inflow: {inflow:,.0f} PKR  |  "
            f"Outflow: {outflow:,.0f} PKR  |  "
            f"Net: {net:,.0f} PKR"
        )
    else:
        st.info("No rows found for selected filters/date range.")

    st.divider()
    st.caption("Cashflow Cube — Direction × Month (Pivot)")

    net_expr = cashflow_net_expr(REL_CF)
    dir_expr = cashflow_dir_expr(REL_CF, net_expr)

    sql_cf_pivot = f"""
    select
      to_char(date_trunc('month', "date"), 'Mon-YY') as month_label,
      {dir_expr} as direction,
      sum(abs({net_expr})) as amount
    from {REL_CF}
    where {where_sql}
    group by 1,2
    order by 1,2
    """
    df_cf_p = run_df(sql_cf_pivot, params, ["month_label", "direction", "amount"])

    if not df_cf_p.empty:
        cube = df_cf_p.pivot_table(
            index="direction",
            columns="month_label",
            values="amount",
            aggfunc="sum",
            fill_value=0,
        )
        show_pivot(cube)
    else:
        st.info("No cashflow rows for pivot under current filters.")

# ---------------- Trial balance tab ----------------
with tab_tb:
    st.subheader("Trial Balance (As of To Date)")

    where = ['"date" <= :dt']
    params = {"dt": dt}

    if bank != "ALL" and has_column(REL_SEM, "bank"):
        where.append("bank = :bank"); params["bank"] = bank
    if head != "ALL" and has_column(REL_SEM, "head_name"):
        where.append("head_name = :head_name"); params["head_name"] = head
    if account != "ALL" and has_column(REL_SEM, "account"):
        where.append("account = :account"); params["account"] = account
    if func_code != "ALL" and has_column(REL_SEM, "func_code"):
        where.append("func_code = :func_code"); params["func_code"] = func_code
    if attribute != "ALL" and has_column(REL_SEM, "attribute"):
        where.append("attribute = :attribute"); params["attribute"] = attribute

    acct_expr = tb_account_expr(REL_SEM)
    amt_expr = tb_amount_expr(REL_SEM)

    sql = f"""
    select
      {acct_expr} as "Account",
      sum({amt_expr}) as "Balance"
    from {REL_SEM}
    where {' and '.join(where)}
    group by 1
    order by 1
    """
    df_tb = run_df(sql, params)
    if not df_tb.empty:
        # Prefer totals-aware renderer if present
        if "show_df" in globals():
            show_df(df_tb, label_col="Account")
        else:
            st.dataframe(df_tb, use_container_width=True)
            st.success(f"Net (sum of balances): {df_tb['Balance'].sum():,.0f} PKR")
    else:
        st.info("No rows found for trial balance with current filters.")


# ---------------- Recoup KPIs tab ----------------
with tab_rec_kpi:
    st.subheader("Recoup KPIs (PowerPivot/DAX equivalent)")

    c0, c1 = st.columns(2)
    with c0:
        bank_revenue = st.text_input("Revenue Bank (for specific KPIs)", value=BANK_REVENUE_DEFAULT)
    with c1:
        bank_assignment = st.text_input("Assignment Bank (exclude / specific KPIs)", value=BANK_ASSIGNMENT_DEFAULT)

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
    select
      head_name,
      coalesce(sum(coalesce(debit_payment,0) - coalesce(credit_deposit,0)),0) as pending_net
    from {REL_SEM}
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
        show_df(df_pending, label_col=\"Head\")

# ---------------- Receivables tab ----------------
with tab_receivables:
    st.subheader("Receivables (Billing & Collection)")

    where_base, params_base, _ = build_where_from_ui(
        df, dt, bank, head, account, attribute, func_code,
        fy_label=fy_label,
        func_override=None,
    )
    where_clause = " and ".join(where_base) if where_base else "1=1"

    bill_sql = f"""
    select coalesce(sum(coalesce(debit_payment,0)),0)
    from {REL_AR}
    where {where_clause}
      and func_code in ('AGR','AMC','PAR','WAR')
      and coalesce(debit_payment,0) > 0
    """
    collect_sql = f"""
    select coalesce(sum(coalesce(credit_deposit,0)),0)
    from {REL_AR}
    where {where_clause}
      and func_code in ('AGR','AMC','PAR','WAR')
      and coalesce(credit_deposit,0) > 0
    """

    billed = run_scalar(bill_sql, params_base)
    collected = run_scalar(collect_sql, params_base)
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

    # Schema-safe ledger SELECT list
    base_cols = ['"date"', "account", "head_name", "pay_to", "description",
                 "debit_payment", "credit_deposit", "gl_amount", "net_flow", "bill_no"]
    optional_cols = ["voucher_no", "reference_no", "status", "bank", "func_code", "attribute"]

    select_cols = []
    for c in base_cols:
        if c == '"date"' or has_column(REL_AR, c):
            select_cols.append(c)
    for c in optional_cols:
        if has_column(REL_AR, c):
            select_cols.append(c)

    ar_filter = ""
    if has_column(REL_AR, "func_code"):
        ar_filter = "and func_code in ('AGR','AMC','PAR','WAR')"

    ledger_sql = f"""
    select {', '.join(select_cols)}
    from {REL_AR}
    where {where_clause}
      {ar_filter}
    order by "date" desc
    limit 1000
    """
    df_ledger = run_df(ledger_sql, params_base)
    show_df(df_ledger)

# ---------------- AI Q&A tab ----------------
with tab_qa:
    st.subheader("Ask a Finance Question (Deterministic + Safe)")
    st.caption("Examples: revenue by head monthly | expense by head | cashflow by bank | pending recoup | trial balance | search vendor name")

    q = st.text_input("Ask anything…", placeholder="revenue by head monthly")

    if q:
        intent = detect_intent(q)
        struct = detect_structure(q)
        payee = extract_payee(q)

        func_override = apply_intent_func_override(intent, q)
        where, params, effective_func = build_where_from_ui(
            df, dt, bank, head, account, attribute, func_code,
            fy_label=fy_label,
            func_override=func_override,
        )

        # Relative date overrides
        date_sql, date_params = infer_date_sql(q)
        if date_sql:
            where = [w for w in where if "between :df and :dt" not in w and '"date" between' not in w]
            where.insert(0, date_sql)
            params.update(date_params)

        # Month-range overrides
        m_start, m_end_excl = parse_month_range(q)
        if m_start and m_end_excl:
            where = [w for w in where if "between :df and :dt" not in w and '"date" between' not in w]
            where.insert(0, '"date" >= :m_start and "date" < :m_end')
            params["m_start"] = m_start
            params["m_end"] = m_end_excl

        if payee:
            where.append("pay_to ilike :payee")
            params["payee"] = f"%{payee}%"

        where_sql = " and ".join(where) if where else "1=1"

        if intent == "revenue":
            rel = REL_REV

            if struct["by_head"] and struct["monthly"]:
                st.subheader("Revenue by Head (Monthly) — Pivot")
                sql = f"""
                select date_trunc('month',"date")::date as month,
                       head_name,
                       sum(coalesce(credit_deposit,0)) as revenue
                from {rel}
                where {where_sql}
                group by 1,2
                order by 1,3 desc
                """
                df_out = run_df(sql, params, ["Month", "Head", "Revenue"])
                if not df_out.empty:
                    df_out["Month"] = pd.to_datetime(df_out["Month"])
                    pv = df_out.pivot_table(index="Head", columns="Month", values="Revenue", aggfunc="sum", fill_value=0)
                    pv = pv.reindex(sorted(pv.columns), axis=1)
                    pv.columns = [d.strftime("%b-%y") for d in pv.columns]
                    show_pivot(pv)
                else:
                    st.warning("No rows found.")

            elif struct["by_head"]:
                st.subheader("Revenue by Head")
                sql = f"""
                select head_name,
                       sum(coalesce(credit_deposit,0)) as revenue
                from {rel}
                where {where_sql}
                group by 1
                order by 2 desc
                limit 50
                """
                df_out = run_df(sql, params, ["Head", "Revenue"])
                show_df(df_out, label_col="Head")

            elif struct["monthly"]:
                st.subheader("Monthly Revenue")
                sql = f"""
                select date_trunc('month',"date")::date as month,
                       to_char(date_trunc('month',"date"), 'Mon-YY') as month_label,
                       sum(coalesce(credit_deposit,0)) as revenue
                from {rel}
                where {where_sql}
                group by 1,2
                order by 1
                """
                df_out = run_df(sql, params, ["Month", "Month Label", "Revenue"])
                show_df(df_out, label_col="Month Label")

            else:
                sql = f"""
                select coalesce(sum(coalesce(credit_deposit,0)),0) as revenue
                from {rel}
                where {where_sql}
                """
                st.success(f"Total Revenue: {run_scalar(sql, params):,.0f} PKR")

        elif intent == "expense":
            rel = REL_EXP
            exp_expr = expense_amount_expr(rel)

            if struct["by_head"] and struct["monthly"]:
                st.subheader("Expense by Head (Monthly) — Pivot")
                sql = f"""
                select date_trunc('month',"date")::date as month,
                       head_name,
                       sum({exp_expr}) as outflow
                from {rel}
                where {where_sql}
                group by 1,2
                order by 1,3 desc
                """
                df_out = run_df(sql, params, ["Month", "Head", "Outflow"])
                if not df_out.empty:
                    df_out["Month"] = pd.to_datetime(df_out["Month"])
                    pv = df_out.pivot_table(index="Head", columns="Month", values="Outflow", aggfunc="sum", fill_value=0)
                    pv = pv.reindex(sorted(pv.columns), axis=1)
                    pv.columns = [d.strftime("%b-%y") for d in pv.columns]
                    show_pivot(pv)
                else:
                    st.warning("No rows found.")

            elif struct["by_head"]:
                st.subheader("Expense by Head")
                sql = f"""
                select head_name,
                       sum({exp_expr}) as outflow
                from {rel}
                where {where_sql}
                group by 1
                order by 2 desc
                limit 50
                """
                df_out = run_df(sql, params, ["Head", "Outflow"])
                show_df(df_out, label_col="Head")

            elif struct["monthly"]:
                st.subheader("Monthly Expense")
                sql = f"""
                select date_trunc('month',"date")::date as month,
                       to_char(date_trunc('month',"date"), 'Mon-YY') as month_label,
                       sum({exp_expr}) as outflow
                from {rel}
                where {where_sql}
                group by 1,2
                order by 1
                """
                df_out = run_df(sql, params, ["Month", "Month Label", "Outflow"])
                show_df(df_out, label_col="Month Label")

            else:
                sql = f"""
                select coalesce(sum({exp_expr}),0) as outflow
                from {rel}
                where {where_sql}
                """
                st.success(f"Total Expense Outflow: {run_scalar(sql, params):,.0f} PKR")

        elif intent == "cashflow":
            rel = REL_CF
            st.subheader("Cashflow by Bank & Direction")

            net_expr = cashflow_net_expr(rel)
            dir_expr = cashflow_dir_expr(rel, net_expr)

            sql = f"""
            select coalesce(bank,'UNKNOWN') as bank,
                   {dir_expr} as direction,
                   sum({net_expr}) as amount
            from {rel}
            where {where_sql}
            group by 1,2
            order by 1,2
            """
            df_out = run_df(sql, params, ["Bank", "Direction", "Amount"])
            show_df(df_out, label_col="Bank")

        elif intent == "trial_balance":
            rel = REL_SEM
            st.subheader("Trial Balance")

            acct_expr = tb_account_expr(rel)
            amt_expr = tb_amount_expr(rel)

            sql = f"""
            select {acct_expr} as "Account",
                   sum({amt_expr}) as "Balance"
            from {rel}
            where {where_sql}
            group by 1
            order by 1
            """
            df_out = run_df(sql, params)
            show_df(df_out, label_col="Account")

        elif intent == "recoup":
            rel = REL_SEM
            st.subheader("Recoup (Pending vs Completed)")

            sql = f"""
            select
              case when {_is_blank_sql('status')} then 'pending' else 'completed' end as recoup_state,
              coalesce(sum(coalesce(debit_payment,0) - coalesce(credit_deposit,0)),0) as amount
            from {rel}
            where {where_sql}
              and bill_no ilike '%recoup%'
              and coalesce(account,'') <> coalesce(bank,'')
            group by 1
            order by 1
            """
            df_out = run_df(sql, params, ["State", "Amount"])
            show_df(df_out, label_col="State")

        else:
            rel = REL_SEM
            st.subheader("Search results (latest rows)")

            term = q.strip()
            params2 = dict(params)
            params2["q"] = f"%{term}%"

            base_cols = ['"date"', "bank", "account", "head_name", "pay_to", "description"]
            optional_cols = ["debit_payment", "credit_deposit", "gl_amount", "net_flow", "bill_no", "status",
                             "voucher_no", "reference_no", "func_code", "attribute"]

            select_cols = []
            for c in base_cols:
                if c == '"date"' or has_column(rel, c):
                    select_cols.append(c)
            for c in optional_cols:
                if has_column(rel, c):
                    select_cols.append(c)
            if not select_cols:
                select_cols = ['"date"', "description"]

            sql = f"""
            select {', '.join(select_cols)}
            from {rel}
            where {where_sql}
              and (
                coalesce(description,'') ilike :q
                or coalesce(pay_to,'') ilike :q
                or coalesce(account,'') ilike :q
                or coalesce(head_name,'') ilike :q
              )
            order by "date" desc
            limit 500
            """
            df_out = run_df(sql, params2)
            show_df(df_out)

        with st.expander("🔍 Debug (SQL + Params)"):
            st.write(f"Intent: `{intent}` | Effective func filter: `{effective_func}`")
            st.code(where_sql)
            st.json({k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in params.items()})


# ---------------- Search Description tab ----------------
with tab_search:
    st.subheader("Search Description / Payee / Head / Account")
    term = st.text_input("Search text", value="", placeholder="e.g., recoup, salary, AGR, vendor name...")
    limit = st.slider("Rows", 50, 2000, 300, step=50)

    where, params, _ = build_where_from_ui(
        df, dt, bank, head, account, attribute, func_code,
        fy_label=fy_label,
        func_override=None,
    )
    where_sql = " and ".join(where) if where else "1=1"

    if term.strip():
        params["q"] = f"%{term.strip()}%"

        # Schema-safe select list (REL_SEM may be a view without all columns)
        base_cols = ['"date"', "bank", "account", "head_name", "pay_to", "description"]
        optional_cols = ["attribute", "func_code", "debit_payment", "credit_deposit", "gl_amount", "net_flow", "bill_no", "status"]

        select_cols = []
        for c in base_cols:
            if c == '"date"' or has_column(REL_SEM, c):
                select_cols.append(c)
        for c in optional_cols:
            if has_column(REL_SEM, c):
                select_cols.append(c)
        if not select_cols:
            select_cols = ['"date"', "description"]

        sql = f"""
        select {', '.join(select_cols)}
        from {REL_SEM}
        where {where_sql}
          and (
            coalesce(description,'') ilike :q
            or coalesce(pay_to,'') ilike :q
            or coalesce(account,'') ilike :q
            or coalesce(head_name,'') ilike :q
          )
        order by "date" desc
        limit {int(limit)}
        """
        df_s = run_df(sql, params)

        # Net effect caption (if available)
        if not df_s.empty:
            if "net_flow" in df_s.columns:
                st.caption(f"Net effect (sum net_flow): {df_s['net_flow'].sum():,.0f}")
            elif "gl_amount" in df_s.columns:
                st.caption(f"Net effect (sum gl_amount): {df_s['gl_amount'].sum():,.0f}")

        show_df(df_s)
    else:
        st.info("Enter a search term to show results.")
