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
# SEMANTIC VIEWS (Supabase)
# -------------------------------------------------
# Expect these views to exist in Supabase (run the SQL script once):
# public.v_fiscal_calendar
# public.v_finance_semantic
# public.v_revenue
# public.v_expense
# public.v_receivable
# public.v_recoup_pending
# public.v_recoup_completed
# public.v_cashflow
BASE_SEMANTIC_VIEW = "public.v_finance_semantic"
V_REVENUE = "public.v_revenue"
V_EXPENSE = "public.v_expense"
V_RECEIVABLE = "public.v_receivable"
V_RECOUP_PENDING = "public.v_recoup_pending"
V_RECOUP_COMPLETED = "public.v_recoup_completed"
V_CASHFLOW = "public.v_cashflow"

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
# HELPERS (SQL PARAM SAFETY)
# -------------------------------------------------
_PARAM_RE = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")

def prune_params(sql: str, params: dict) -> dict:
    """Keep only parameters that actually appear in the SQL string (:param)."""
    needed = set(_PARAM_RE.findall(sql))
    return {k: v for k, v in params.items() if k in needed}

# -------------------------------------------------
# HELPERS (DB)
# -------------------------------------------------
@st.cache_data(ttl=3600)
def get_distinct(col: str):
    # col is controlled by code (not user), safe to interpolate.
    q = text(f'SELECT DISTINCT {col} FROM {BASE_SEMANTIC_VIEW} WHERE {col} IS NOT NULL ORDER BY {col}')
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(q).fetchall()]

@st.cache_data(ttl=3600)
def get_known_payees():
    return get_distinct("pay_to")

KNOWN_PAYEES = get_known_payees()
KNOWN_FUNC_CODES = get_distinct("func_code")

@st.cache_data(ttl=3600)
def get_distinct_fiscal_years() -> list[str]:
    q = text(f"SELECT DISTINCT fiscal_year FROM {BASE_SEMANTIC_VIEW} WHERE fiscal_year IS NOT NULL ORDER BY fiscal_year")
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
        "by_head": ("by head" in ql) or ("head wise" in ql) or ("head-wise" in ql) or ("head" in ql and "by" in ql),
        "by_bank": ("by bank" in ql) or ("bank wise" in ql) or ("bank-wise" in ql) or ("bank" in ql and "by" in ql),
        "monthly": ("monthly" in ql) or ("per month" in ql) or ("month wise" in ql) or ("month-wise" in ql) or ("trend" in ql),
        "top": ("top" in ql) or ("highest" in ql) or ("largest" in ql),
        }
    
# -------------------------------------------------
# FILTER BUILDER
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

    fy_label expects "ALL" or something like "FY2025-26" (FY starts July).
    If fy_label is provided (and not ALL), it overrides date range.
    """
    where = []
    params: dict[str, any] = {}

    # Apply date range or fiscal year window
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

    # func_code behavior
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
    explicit = parse_explicit_func_code(question)
    if explicit:
        return explicit  # respect explicit request

    # With semantic views, intent decides the source view; func_code filter usually unnecessary.
    return None

# -------------------------------------------------
# DB EXEC HELPERS
# -------------------------------------------------
def run_scalar(sql: str, params: dict) -> float:
    params2 = prune_params(sql, params)
    with engine.connect() as conn:
        v = conn.execute(text(sql), params2).scalar()
    return float(v or 0)

def run_df(sql: str, params: dict, columns: list[str] | None = None) -> pd.DataFrame:
    params2 = prune_params(sql, params)
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params2).fetchall()
    df_out = pd.DataFrame(rows)
    if columns and not df_out.empty:
        df_out.columns = columns
    return df_out

# -------------------------------------------------
# Recoup KPI metrics (semantic-view based)
# -------------------------------------------------
@st.cache_data(ttl=600)
def compute_recoup_kpis(where_sql: str, params: dict) -> dict:
    pending_amount = run_scalar(
        f"""
        select coalesce(sum(net_flow),0)
        from {V_RECOUP_PENDING}
        where {where_sql}
        """,
        params,
    )
    completed_amount = run_scalar(
        f"""
        select coalesce(sum(net_flow),0)
        from {V_RECOUP_COMPLETED}
        where {where_sql}
        """,
        params,
    )
    pending_count = run_scalar(
        f"""
        select coalesce(count(*),0)
        from {V_RECOUP_PENDING}
        where {where_sql}
        """,
        params,
    )
    completed_count = run_scalar(
        f"""
        select coalesce(count(*),0)
        from {V_RECOUP_COMPLETED}
        where {where_sql}
        """,
        params,
    )
    return {
        "Pending amount (net)": pending_amount,
        "Completed amount (net)": completed_amount,
        "Pending count": pending_count,
        "Completed count": completed_count,
    }
    
# -------------------------------------------------
# UI FILTERS (Form)
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
    attrs_list = ["ALL"] + sorted([a for a in attributes if a is not None])
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
    new_func_code = st.selectbox("Function Code (optional)", funcs, index=f_idx)

    # Fiscal Year filter: take from semantic view (already FY starts July)
    fiscal_years = get_distinct_fiscal_years()
    fy_options = ["ALL"] + [f"FY{fy}" if not str(fy).startswith("FY") else str(fy) for fy in fiscal_years]
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
    st.subheader("Revenue (Monthly)")
    where, params, _ = build_where_from_ui(df, dt, bank, head, account, attribute, func_code, fy_label=fy_label, func_override=None)
    where_sql = " and ".join(where)

    sql = f"""
    select date_trunc('month', "date") as month,
           sum(coalesce(credit_deposit,0)) as revenue
    from {V_REVENUE}
    where {where_sql}
    group by 1
    order by 1
    """
    df_rev = run_df(sql, params, ["Month", "Revenue"])
    if not df_rev.empty:
        st.dataframe(df_rev, use_container_width=True)
        st.line_chart(df_rev.set_index("Month"))
        st.success(f"Total Revenue: {df_rev['Revenue'].sum():,.0f} PKR")
    else:
        st.info("No revenue rows found for selected filters/date range.")

# ---------------- Expense tab ----------------
with tab_exp:
    st.subheader("Expenses (Monthly) — Net Cash Outflow (+/- included)")
    where, params, _ = build_where_from_ui(df, dt, bank, head, account, attribute, func_code, fy_label=fy_label, func_override=None)
    where_sql = " and ".join(where)

    sql = f"""
    select date_trunc('month', "date") as month,
           sum(coalesce(net_flow,0)) as expense_net
    from {V_EXPENSE}
    where {where_sql}
    group by 1
    order by 1
    """
    df_exp = run_df(sql, params, ["Month", "Expense Net"])
    if not df_exp.empty:
        st.dataframe(df_exp, use_container_width=True)
        st.line_chart(df_exp.set_index("Month"))
        st.success(f"Total Expense (net): {df_exp['Expense Net'].sum():,.0f} PKR")
    else:
        st.info("No expense rows found for selected filters/date range.")

# ---------------- Cashflow tab ----------------
with tab_cf:
    st.subheader("Cashflow Summary (By Bank & Direction)")
    where, params, _ = build_where_from_ui(df, dt, bank, head, account, attribute, func_code, fy_label=fy_label, func_override=None)
    where_sql = " and ".join(where)

    sql = f"""
    select
      coalesce(bank, 'UNKNOWN') as bank,
      direction,
      sum(coalesce(net_flow,0)) as amount
    from {V_CASHFLOW}
    where {where_sql}
    group by 1,2
    order by 1,2
    """
    df_cf = run_df(sql, params, ["Bank", "Direction", "Amount"])
    if not df_cf.empty:
        st.dataframe(df_cf, use_container_width=True)
        inflow = df_cf[df_cf["Direction"] == "in"]["Amount"].sum()
        outflow = df_cf[df_cf["Direction"] == "out"]["Amount"].sum()
        st.success(
            f"Inflow: {inflow:,.0f} PKR  |  Outflow: {abs(outflow):,.0f} PKR  |  Net: {(inflow+outflow):,.0f} PKR"
        )
    else:
        st.info("No cashflow rows found for selected filters/date range.")

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
    if attribute != "ALL":
        where.append("attribute = :attribute"); params["attribute"] = attribute
    if fy_label and fy_label != "ALL":
        # override to FY window
        try:
            start_year = int(fy_label.replace("FY", "").split("-")[0])
            params["fy_start"] = date(start_year, 7, 1)
            params["fy_end"] = date(start_year + 1, 6, 30)
            where = [w for w in where if w != '"date" <= :dt']
            where.append('"date" >= :fy_start')
            where.append('"date" <= :fy_end')
        except Exception:
            pass

    where_sql = " and ".join(where)

    sql = f"""
    select
      account,
      sum(coalesce(net_flow,0)) as balance
    from {BASE_SEMANTIC_VIEW}
    where {where_sql}
    group by 1
    order by 1
    """
    df_tb = run_df(sql, params, ["Account", "Balance"])
    if not df_tb.empty:
        st.dataframe(df_tb, use_container_width=True)
        st.success(f"Net (sum of balances): {df_tb['Balance'].sum():,.0f} PKR")
    else:
        st.info("No rows found for trial balance with current filters.")
        
# ---------------- Recoup KPIs tab ----------------
with tab_rec_kpi:
    st.subheader("Recoup KPIs (Pending vs Completed)")

    # Use UI filters; ignore func_code
    where, params, _ = build_where_from_ui(df, dt, bank, head, account, attribute, func_code, fy_label=fy_label, func_override=None)
    where_sql = " and ".join(where)

    kpis = compute_recoup_kpis(where_sql, params)

    cols = st.columns(4)
    items = list(kpis.items())
    for i, (k, v) in enumerate(items):
        with cols[i % 4]:
            st.metric(k, f"{v:,.0f}")

    st.divider()
    st.caption("Pending Recoup (Net) by Head")

    pending_by_head_sql = f"""
        select head_name,
               coalesce(sum(coalesce(net_flow,0)),0) as pending_net
        from {V_RECOUP_PENDING}
        where {where_sql}
        group by 1
        order by 2 desc
        limit 100
    """
    df_pending = run_df(pending_by_head_sql, params, ["Head", "Pending Net"])
    if df_pending.empty:
        st.info("No pending recoup rows under current filters.")
    else:
        st.dataframe(df_pending, use_container_width=True)

# ---------------- Receivables tab ----------------
with tab_receivables:
    st.subheader("Receivables (Billing & Collection)")
    where_base, params_base, _ = build_where_from_ui(
        df, dt, bank, head, account, attribute, func_code, fy_label=fy_label, func_override=None
    )
    where_clause = " and ".join(where_base) if where_base else "1=1"

    bill_sql = f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from {V_RECEIVABLE}
        where {where_clause}
          and coalesce(debit_payment,0) > 0
    """
    collect_sql = f"""
        select coalesce(sum(coalesce(credit_deposit,0)),0)
        from {V_RECEIVABLE}
        where {where_clause}
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

    ledger_sql = f"""
        select
          "date",
          account,
          head_name,
          pay_to,
          description,
          debit_payment,
          credit_deposit,
          net_flow,
          bill_no,
          status,
          bank,
          attribute,
          fiscal_year
        from {V_RECEIVABLE}
        where {where_clause}
        order by "date" desc
        limit 1000
    """
    df_ledger = run_df(
        ledger_sql,
        params_base,
        ["Date","Account","Head","Pay To","Description","Debit","Credit","Net Flow","Bill No","Status","Bank","Attribute","Fiscal Year"],
    )
    if df_ledger.empty:
        st.info("No receivable rows under current filters.")
    else:
        st.dataframe(df_ledger, use_container_width=True)

# ---------------- AI Q&A tab ----------------
with tab_qa:
    st.subheader("Ask a Finance Question (Deterministic)")
    st.caption("Examples: revenue by head | revenue by head monthly | expense by head | pending recoup amount | trial balance | cashflow")

    q = st.text_input("Ask anything…", placeholder="revenue by head monthly")

    if q:
        intent = detect_intent(q)
        payee = extract_payee(q)

        func_override = apply_intent_func_override(intent, q, func_code)
        where, params, effective_func = build_where_from_ui(df, dt, bank, head, account, attribute, func_code, fy_label=fy_label, func_override=func_override)

        # Relative date overrides
        date_sql, date_params = infer_date_sql(q)
        if date_sql:
            where = [w for w in where if "between :df and :dt" not in w and '"date" between' not in w and '"date" >=' not in w and '"date" <=' not in w]
            where.insert(0, date_sql)
            params.update(date_params)

        # Month range overrides
        m_start, m_end_excl = parse_month_range(q)
        if m_start and m_end_excl:
            where = [w for w in where if '"date"' not in w or (":df" not in w and ":dt" not in w and ":fy_start" not in w and ":fy_end" not in w)]
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
            src = V_REVENUE
            if struct["by_head"] and struct["monthly"]:
                label = "Revenue by Head (Monthly)"
                sql = f"""
                select date_trunc('month',"date") as month,
                       head_name,
                       sum(coalesce(credit_deposit,0)) as revenue
                from {src}
                where {where_sql}
                group by 1,2
                order by 1,3 desc
                """
            elif struct["by_head"]:
                label = "Revenue by Head"
                sql = f"""
                select head_name,
                       sum(coalesce(credit_deposit,0)) as revenue
                from {src}
                where {where_sql}
                group by 1
                order by 2 desc
                limit 50
                """
            elif struct["by_bank"]:
                label = "Revenue by Bank"
                sql = f"""
                select coalesce(bank,'UNKNOWN') as bank,
                       sum(coalesce(credit_deposit,0)) as revenue
                from {src}
                where {where_sql}
                group by 1
                order by 2 desc
                """
            elif struct["monthly"]:
                label = "Monthly Revenue"
                sql = f"""
                select date_trunc('month',"date") as month,
                       sum(coalesce(credit_deposit,0)) as revenue
                from {src}
                where {where_sql}
                group by 1
                order by 1
                """
            else:
                label = "Total Revenue"
                sql = f"""
                select coalesce(sum(coalesce(credit_deposit,0)),0) as revenue
                from {src}
                where {where_sql}
                """

        # ---------- Expense ----------
        elif intent == "expense":
            src = V_EXPENSE
            if struct["by_head"] and struct["monthly"]:
                label = "Expense by Head (Monthly)"
                sql = f"""
                select date_trunc('month',"date") as month,
                       head_name,
                       sum(coalesce(net_flow,0)) as expense
                from {src}
                where {where_sql}
                group by 1,2
                order by 1,3 desc
                """
            elif struct["by_head"]:
                label = "Expense by Head"
                sql = f"""
                select head_name,
                       sum(coalesce(net_flow,0)) as expense
                from {src}
                where {where_sql}
                group by 1
                order by 2 desc
                limit 50
                """
            elif struct["monthly"]:
                label = "Monthly Expense"
                sql = f"""
                select date_trunc('month',"date") as month,
                       sum(coalesce(net_flow,0)) as expense
                from {src}
                where {where_sql}
                group by 1
                order by 1
                """
            else:
                label = "Total Expense"
                sql = f"""
                select coalesce(sum(coalesce(net_flow,0)),0) as expense
                from {src}
                where {where_sql}
                """

        # ---------- Recoup ----------
        elif intent == "recoup":
            pending = ("pending" in ql) or ("outstanding" in ql) or ("not recouped" in ql)
            recouped = ("recouped" in ql) or ("settled" in ql) or ("completed" in ql)
            if pending:
                label = "Pending Recoup Amount (net)"
                sql = f"""
                select coalesce(sum(coalesce(net_flow,0)),0) as pending_recoup
                from {V_RECOUP_PENDING}
                where {where_sql}
                """
            elif recouped:
                label = "Completed Recoup Amount (net)"
                sql = f"""
                select coalesce(sum(coalesce(net_flow,0)),0) as completed_recoup
                from {V_RECOUP_COMPLETED}
                where {where_sql}
                """
            else:
                label = "Recoup Total (net)"
                sql = f"""
                select coalesce(sum(coalesce(net_flow,0)),0) as recoup_total
                from {BASE_SEMANTIC_VIEW}
                where {where_sql} and bill_no = 'Recoup'
                """

        # ---------- Cashflow ----------
        elif intent == "cashflow":
            label = "Cashflow"
            sql = f"""
            select coalesce(bank,'UNKNOWN') as bank,
                   direction,
                   sum(coalesce(net_flow,0)) as amount
            from {V_CASHFLOW}
            where {where_sql}
            group by 1,2
            order by 1,2
            """

        # ---------- Trial balance ----------
        elif intent == "trial_balance":
            label = "Trial Balance"
            sql = f"""
            select account,
                   sum(coalesce(net_flow,0)) as balance
            from {BASE_SEMANTIC_VIEW}
            where {where_sql}
            group by 1
            order by 1
            """
        
        # ---------- Search fallback ----------
        else:
            label = "Search results (description/pay_to)"
            params["q"] = q.strip()
            sql = f"""
            select
              "date",
              account,
              head_name,
              pay_to,
              description,
              debit_payment,
              credit_deposit,
              net_flow,
              bank,
              attribute,
              bill_no,
              status
            from {BASE_SEMANTIC_VIEW}
            where {where_sql}
              and (
                description ilike :q_like
                or coalesce(pay_to,'') ilike :q_like
                or coalesce(account,'') ilike :q_like
                or coalesce(head_name,'') ilike :q_like
              )
            order by "date" desc
            limit 200
            """
            params["q_like"] = f"%{params['q']}%"

        # ---------- Execute + Render ----------
        if "group by" in sql.lower() or intent in ("cashflow", "trial_balance", "search", "revenue", "expense"):
            df_out = run_df(sql, params)
            if df_out.empty:
                st.warning("No rows found for this question with current filters.")
            else:
                # Pivot for head-by-month monthly reports
                if label in ("Revenue by Head (Monthly)", "Expense by Head (Monthly)"):
                    if "Revenue" in label:
                        df_out.columns = ["Month", "Head", "Revenue"]
                        value_col = "Revenue"
                    else:
                        df_out.columns = ["Month", "Head", "Expense"]
                        value_col = "Expense"
                    df_out["Month"] = pd.to_datetime(df_out["Month"])
                    df_pivot = df_out.pivot(index="Head", columns="Month", values=value_col).fillna(0)
                    df_pivot = df_pivot.reindex(sorted(df_pivot.columns), axis=1)
                    df_pivot.columns = [d.strftime("%b-%y") for d in df_pivot.columns]
                    df_pivot = df_pivot.reset_index().rename(columns={"Head": "Head Name"})
                    st.subheader(label)
                    st.dataframe(df_pivot, use_container_width=True)
                else:
                    st.subheader(label)
                    st.dataframe(df_out, use_container_width=True)
                    if label in ("Monthly Revenue", "Monthly Expense") and df_out.shape[1] == 2:
                        # chart on month
                        try:
                            df_out.columns = ["Month", "Amount"]
                            st.line_chart(df_out.set_index("Month"))
                        except Exception:
                            pass
        else:
            val = run_scalar(sql, params)
            st.success(f"{label}: {val:,.0f} PKR")

        with st.expander("🔍 Why this result?"):
            st.write(f"Intent detected: `{intent}`")
            if payee:
                st.write(f"Payee: `{payee}`")
            if m_start and m_end_excl:
                st.write(f"Month range: `{m_start}` to `{m_end_excl}` (end exclusive)")
            st.write("Filters applied:")
            st.write(f"- Bank: `{bank}`  |  Head: `{head}`  |  Account: `{account}`  |  Attribute: `{attribute}`  |  Function: `{func_code}`  |  Fiscal Year: `{fy_label}`")
            st.write(f"- From: `{df}`  |  To: `{dt}`")
            st.write("SQL (debug):")
            st.code(sql.strip())
            st.write("Params (debug):")
            st.json({k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in prune_params(sql, params).items()})

# ---------------- Search Description tab ----------------
with tab_search:
    st.subheader("Search Description (Ledger rows)")
    st.caption("Search inside description / pay_to / account / head_name, and get the net effect (net_flow). Uses global slicers.")

    s = st.text_input("Search text", placeholder="e.g., POL chg May-25")
    limit = st.slider("Max rows", min_value=50, max_value=5000, value=500, step=50)

    if s and s.strip():
        where, params, _ = build_where_from_ui(df, dt, bank, head, account, attribute, func_code, fy_label=fy_label, func_override=None)
        where_sql = " and ".join(where)
        params["q_like"] = f"%{s.strip()}%"

        fts_sql = f"""
        select
          "date" as "Date",
          fiscal_year as "Fiscal Year",
          bank as "Bank",
          account as "Account",
          head_name as "Head Name",
          pay_to as "Pay To",
          description as "Description",
          debit_payment as "Debit",
          credit_deposit as "Credit",
          net_flow as "Net Flow",
          bill_no as "Bill No",
          status as "Status",
          attribute as "Attribute"
        from {BASE_SEMANTIC_VIEW}
        where {where_sql}
          and (
            description ilike :q_like
            or coalesce(pay_to,'') ilike :q_like
            or coalesce(account,'') ilike :q_like
            or coalesce(head_name,'') ilike :q_like
          )
        order by "date" desc
        limit :lim
        """
        params["lim"] = int(limit)

        df_search = run_df(fts_sql, params)
        if df_search.empty:
            st.info("No matches under current filters.")
        else:
            st.dataframe(df_search, use_container_width=True)
            st.success(f"Rows: {len(df_search):,} | Net Flow sum: {df_search['Net Flow'].sum():,.0f} PKR")
    else:
        st.info("Type search text above to find matching ledger rows.")
