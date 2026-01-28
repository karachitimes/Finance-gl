import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from datetime import date

# =================================================
# CONFIG
# =================================================
st.set_page_config(page_title="Finance Analytics System", layout="wide")

# Prefer Streamlit secrets, fallback to env var
DATABASE_URL = None
if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
    DATABASE_URL = st.secrets["DATABASE_URL"]
else:
    DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    st.error("DATABASE_URL is not set. Add it to Streamlit secrets or environment variables.")
    st.stop()


# =================================================
# DB HELPERS
# =================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=3,
        max_overflow=2,
    )


@st.cache_data(ttl=300, show_spinner=False)
def run_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run a SELECT query safely with bound params and return a dataframe."""
    params = params or {}
    eng = get_engine()
    with eng.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


@st.cache_data(ttl=600, show_spinner=False)
def table_exists(schema: str, rel: str) -> bool:
    q = """
    select 1
    from information_schema.tables
    where table_schema = :schema and table_name = :rel
    union all
    select 1
    from information_schema.views
    where table_schema = :schema and table_name = :rel
    limit 1
    """
    df = run_df(q, {"schema": schema, "rel": rel})
    return not df.empty


def pick_relation(candidates: list[str]) -> str:
    """Pick first existing relation from list; returns last candidate even if missing."""
    for fq in candidates:
        if "." in fq:
            schema, rel = fq.split(".", 1)
        else:
            schema, rel = "public", fq
        if table_exists(schema, rel):
            return fq
    return candidates[-1]


# =================================================
# FILTER BUILDER
# =================================================
def where_builder(
    *,
    date_from: date | None,
    date_to: date | None,
    fiscal_year: str | None,
    bank: str | None,
    account: str | None,
    attribute: str | None,
    func_code: str | None,
    head_name: str | None,
    column1: str | None,
    extra_clauses: list[str] | None = None,
) -> tuple[str, dict]:
    """
    Builds a safe WHERE clause and params dict.
    Only includes keys actually used.
    Assumes the underlying relation has:
      "date", fiscal_year, bank, account, attribute, func_code, head_name, column1
    """
    clauses: list[str] = []
    params: dict = {}

    if date_from:
        clauses.append('"date" >= :date_from')
        params["date_from"] = date_from
    if date_to:
        clauses.append('"date" <= :date_to')
        params["date_to"] = date_to

    def add_eq(col: str, key: str, val: str | None):
        if val and val != "All":
            clauses.append(f'{col} = :{key}')
            params[key] = val

    add_eq("fiscal_year", "fiscal_year", fiscal_year)
    add_eq("bank", "bank", bank)
    add_eq("account", "account", account)
    add_eq("attribute", "attribute", attribute)
    add_eq("func_code", "func_code", func_code)
    add_eq("head_name", "head_name", head_name)
    add_eq("column1", "column1", column1)

    if extra_clauses:
        for c in extra_clauses:
            if c:
                clauses.append(c)

    where_sql = ""
    if clauses:
        where_sql = "WHERE " + " AND ".join(clauses)

    return where_sql, params


def options_for(rel: str, col: str, limit: int = 500) -> list[str]:
    """Distinct options for a slicer column from a relation (view/table)."""
    try:
        sql = f"""
        select distinct {col} as v
        from {rel}
        where {col} is not null and nullif(trim({col}::text),'') is not null
        order by 1
        limit {limit}
        """
        df = run_df(sql)
        vals = df["v"].astype(str).tolist()
        return ["All"] + vals
    except Exception:
        # If relation/column missing, return minimal options.
        return ["All"]


# =================================================
# UI
# =================================================
st.title("📊 Finance Analytics System")

# Choose base relations (views first, fallbacks last)
REL_SEM = pick_relation(["public.v_finance_semantic", "public.gl_register"])
REL_REV = pick_relation(["public.v_revenue", "public.v_finance_semantic", "public.gl_register"])
REL_EXP = pick_relation(["public.v_expense", "public.v_finance_semantic", "public.gl_register"])
REL_AR  = pick_relation(["public.v_receivable", "public.v_finance_semantic", "public.gl_register"])
REL_RP  = pick_relation(["public.v_recoup_pending", "public.v_finance_semantic", "public.gl_register"])
REL_RC  = pick_relation(["public.v_recoup_completed", "public.v_finance_semantic", "public.gl_register"])
REL_CF  = pick_relation(["public.v_cashflow", "public.v_finance_semantic", "public.gl_register"])

with st.sidebar:
    st.subheader("Global Filters")

    # Date range (always available)
    # default: FY start Jul 1 this year to today
    today = date.today()
    default_from = date(today.year, 7, 1) if today.month >= 7 else date(today.year - 1, 7, 1)

    date_from = st.date_input("From", value=default_from)
    date_to = st.date_input("To", value=today)

    fiscal_year = st.selectbox("Fiscal Year", options=options_for(REL_SEM, "fiscal_year"))
    bank = st.selectbox("Bank", options=options_for(REL_SEM, "bank"))
    account = st.selectbox("Account", options=options_for(REL_SEM, "account"))
    attribute = st.selectbox("Attribute", options=options_for(REL_SEM, "attribute"))
    func_code = st.selectbox("Func Code", options=options_for(REL_SEM, "func_code"))
    head_name = st.selectbox("Head Name", options=options_for(REL_SEM, "head_name"))
    column1 = st.selectbox("Column1", options=options_for(REL_SEM, "column1"))

    st.caption("These slicers apply everywhere. No DDL runs in Streamlit.")

# Tabs
tabs = st.tabs(["Revenue", "Expense (Net Cash Outflow)", "Cashflow", "Receivables", "Recoup", "Trial Balance", "Search Description", "AI Q&A"])

# =================================================
# Revenue
# =================================================
with tabs[0]:
    st.subheader("Revenue Dashboard")
    where_sql, params = where_builder(
        date_from=date_from, date_to=date_to,
        fiscal_year=fiscal_year, bank=bank, account=account, attribute=attribute,
        func_code=func_code, head_name=head_name, column1=column1,
        extra_clauses=None
    )

    # Revenue view includes credit_deposit; use it as amount basis
    sql = f"""
    select
      date_trunc('month', "date")::date as month_start,
      to_char(date_trunc('month', "date"), 'Mon-YY') as month_label,
      sum(coalesce(credit_deposit,0)) as revenue_amount
    from {REL_REV}
    {where_sql}
    group by 1,2
    order by 1
    """
    try:
        df = run_df(sql, params)
        total = float(df["revenue_amount"].sum()) if not df.empty else 0.0
        st.metric("Total Revenue (credit deposits)", f"{total:,.0f}")
        st.dataframe(df, use_container_width=True)
    except SQLAlchemyError:
        st.error("Revenue query failed. Check that v_revenue (or v_finance_semantic) exists and includes credit_deposit.")

# =================================================
# Expense
# =================================================
with tabs[1]:
    st.subheader("Expense Dashboard (Net Cash Outflow)")
    where_sql, params = where_builder(
        date_from=date_from, date_to=date_to,
        fiscal_year=fiscal_year, bank=bank, account=account, attribute=attribute,
        func_code=func_code, head_name=head_name, column1=column1,
        extra_clauses=None
    )

    sql = f"""
    select
      date_trunc('month', "date")::date as month_start,
      to_char(date_trunc('month', "date"), 'Mon-YY') as month_label,
      sum(coalesce(net_flow,0)) as expense_outflow
    from {REL_EXP}
    {where_sql}
    group by 1,2
    order by 1
    """
    try:
        df = run_df(sql, params)
        total = float(df["expense_outflow"].sum()) if not df.empty else 0.0
        st.metric("Total Expense Outflow (net_flow)", f"{total:,.0f}")
        st.dataframe(df, use_container_width=True)

        st.divider()
        st.caption("Top expense heads (by net outflow)")
        sql2 = f"""
        select head_name, sum(coalesce(net_flow,0)) as outflow
        from {REL_EXP}
        {where_sql}
        group by 1
        order by 2 desc
        limit 50
        """
        df2 = run_df(sql2, params)
        st.dataframe(df2, use_container_width=True)
    except SQLAlchemyError:
        st.error("Expense query failed. Check that v_expense (or v_finance_semantic) exists and includes net_flow.")

# =================================================
# Cashflow
# =================================================
with tabs[2]:
    st.subheader("Cashflow Dashboard")
    where_sql, params = where_builder(
        date_from=date_from, date_to=date_to,
        fiscal_year=fiscal_year, bank=bank, account=account, attribute=attribute,
        func_code=func_code, head_name=head_name, column1=column1,
        extra_clauses=None
    )

    sql = f"""
    select
      date_trunc('month', "date")::date as month_start,
      to_char(date_trunc('month', "date"), 'Mon-YY') as month_label,
      sum(case when direction='in' then abs(coalesce(net_flow,0)) else 0 end) as inflow,
      sum(case when direction='out' then abs(coalesce(net_flow,0)) else 0 end) as outflow,
      sum(coalesce(net_flow,0)) as net
    from {REL_CF}
    {where_sql}
    group by 1,2
    order by 1
    """
    try:
        df = run_df(sql, params)
        inflow = float(df["inflow"].sum()) if not df.empty else 0.0
        outflow = float(df["outflow"].sum()) if not df.empty else 0.0
        st.metric("Total Inflow", f"{inflow:,.0f}")
        st.metric("Total Outflow", f"{outflow:,.0f}")
        st.dataframe(df, use_container_width=True)
    except SQLAlchemyError:
        st.error("Cashflow query failed. Check that v_cashflow exists (or use v_finance_semantic).")

# =================================================
# Receivables
# =================================================
with tabs[3]:
    st.subheader("Receivables (PAR/WAR/AGR/AMC)")
    where_sql, params = where_builder(
        date_from=date_from, date_to=date_to,
        fiscal_year=fiscal_year, bank=bank, account=account, attribute=attribute,
        func_code=func_code, head_name=head_name, column1=column1,
        extra_clauses=None
    )

    sql = f"""
    select
      func_code,
      head_name,
      sum(coalesce(debit_payment,0)) as debit,
      sum(coalesce(credit_deposit,0)) as credit,
      sum(coalesce(net_flow,0)) as net_flow
    from {REL_AR}
    {where_sql}
    group by 1,2
    order by 1,2
    """
    try:
        df = run_df(sql, params)
        st.dataframe(df, use_container_width=True)
    except SQLAlchemyError:
        st.error("Receivables query failed. Check that v_receivable exists and includes debit_payment/credit_deposit/net_flow.")

# =================================================
# Recoup
# =================================================
with tabs[4]:
    st.subheader("Recoup Module")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Pending Recoup**")
        where_sql, params = where_builder(
            date_from=date_from, date_to=date_to,
            fiscal_year=fiscal_year, bank=bank, account=account, attribute=attribute,
            func_code=func_code, head_name=head_name, column1=column1,
            extra_clauses=None
        )
        sqlp = f"""
        select
          count(*) as rows,
          sum(coalesce(net_flow,0)) as amount
        from {REL_RP}
        {where_sql}
        """
        try:
            d = run_df(sqlp, params)
            st.metric("Count", int(d.loc[0,"rows"]) if not d.empty else 0)
            st.metric("Amount (net_flow)", f"{float(d.loc[0,'amount']) if not d.empty else 0:,.0f}")
            st.caption("Details")
            sqld = f"""
            select "date", bank, account, head_name, description, bill_no, status, net_flow
            from {REL_RP}
            {where_sql}
            order by "date" desc
            limit 200
            """
            st.dataframe(run_df(sqld, params), use_container_width=True)
        except SQLAlchemyError:
            st.error("Pending recoup query failed. Check v_recoup_pending.")

    with colB:
        st.markdown("**Completed Recoup**")
        where_sql, params = where_builder(
            date_from=date_from, date_to=date_to,
            fiscal_year=fiscal_year, bank=bank, account=account, attribute=attribute,
            func_code=func_code, head_name=head_name, column1=column1,
            extra_clauses=None
        )
        sqlc = f"""
        select
          count(*) as rows,
          sum(coalesce(net_flow,0)) as amount
        from {REL_RC}
        {where_sql}
        """
        try:
            d = run_df(sqlc, params)
            st.metric("Count", int(d.loc[0,"rows"]) if not d.empty else 0)
            st.metric("Amount (net_flow)", f"{float(d.loc[0,'amount']) if not d.empty else 0:,.0f}")
            st.caption("Details")
            sqld = f"""
            select "date", bank, account, head_name, description, bill_no, status, net_flow
            from {REL_RC}
            {where_sql}
            order by "date" desc
            limit 200
            """
            st.dataframe(run_df(sqld, params), use_container_width=True)
        except SQLAlchemyError:
            st.error("Completed recoup query failed. Check v_recoup_completed.")

# =================================================
# Trial Balance
# =================================================
with tabs[5]:
    st.subheader("Trial Balance (from semantic view)")
    where_sql, params = where_builder(
        date_from=date_from, date_to=date_to,
        fiscal_year=fiscal_year, bank=bank, account=account, attribute=attribute,
        func_code=func_code, head_name=head_name, column1=column1,
        extra_clauses=None
    )
    sql = f"""
    select
      account,
      head_name,
      sum(coalesce(debit_payment,0)) as debit,
      sum(coalesce(credit_deposit,0)) as credit,
      sum(coalesce(net_flow,0)) as net_flow
    from {REL_SEM}
    {where_sql}
    group by 1,2
    order by 1,2
    """
    try:
        df = run_df(sql, params)
        st.dataframe(df, use_container_width=True)
    except SQLAlchemyError:
        st.error("Trial balance query failed. Check v_finance_semantic exists.")

# =================================================
# Search Description
# =================================================
with tabs[6]:
    st.subheader("Search Description / Payee / Head / Account")
    q = st.text_input("Search text", value="", placeholder="e.g., recoup, salary, AGR, vendor name...")
    limit = st.slider("Rows", 50, 2000, 200, step=50)

    extra = []
    params_extra = {}
    if q.strip():
        extra.append("(description ilike :q OR pay_to ilike :q OR account ilike :q OR head_name ilike :q)")
        params_extra["q"] = f"%{q.strip()}%"

    where_sql, params = where_builder(
        date_from=date_from, date_to=date_to,
        fiscal_year=fiscal_year, bank=bank, account=account, attribute=attribute,
        func_code=func_code, head_name=head_name, column1=column1,
        extra_clauses=extra
    )
    params = {**params, **params_extra}

    sql = f"""
    select
      "date", bank, account, attribute, func_code, column1,
      head_name, pay_to, description,
      debit_payment, credit_deposit, net_flow,
      bill_no, status
    from {REL_SEM}
    {where_sql}
    order by "date" desc
    limit {int(limit)}
    """
    try:
        df = run_df(sql, params)
        st.caption(f"Net effect (sum net_flow): {df['net_flow'].sum():,.0f}" if not df.empty else "No rows.")
        st.dataframe(df, use_container_width=True)
    except SQLAlchemyError:
        st.error("Search failed. Check v_finance_semantic exists and columns are present.")

# =================================================
# AI Q&A (Safe)
# =================================================
with tabs[7]:
    st.subheader("AI Q&A (safe semantic querying)")
    st.caption("This module does NOT run arbitrary SQL. It maps questions to safe templates over semantic views.")

    question = st.text_area("Ask a question", placeholder="e.g., total revenue this fiscal year by month, top expense heads, pending recoup amount...")
    go = st.button("Run", type="primary")

    if go and question.strip():
        ql = question.lower()

        # Simple intent routing (extend safely later)
        if "revenue" in ql and ("month" in ql or "monthly" in ql):
            where_sql, params = where_builder(
                date_from=date_from, date_to=date_to,
                fiscal_year=fiscal_year, bank=bank, account=account, attribute=attribute,
                func_code=func_code, head_name=head_name, column1=column1,
            )
            sql = f"""
            select to_char(date_trunc('month',"date"), 'Mon-YY') as month,
                   sum(coalesce(credit_deposit,0)) as revenue
            from {REL_REV}
            {where_sql}
            group by 1
            order by min(date_trunc('month',"date"))
            """
            st.dataframe(run_df(sql, params), use_container_width=True)

        elif "expense" in ql and ("top" in ql or "head" in ql):
            where_sql, params = where_builder(
                date_from=date_from, date_to=date_to,
                fiscal_year=fiscal_year, bank=bank, account=account, attribute=attribute,
                func_code=func_code, head_name=head_name, column1=column1,
            )
            sql = f"""
            select head_name, sum(coalesce(net_flow,0)) as outflow
            from {REL_EXP}
            {where_sql}
            group by 1
            order by 2 desc
            limit 25
            """
            st.dataframe(run_df(sql, params), use_container_width=True)

        elif "recoup" in ql and ("pending" in ql or "obligation" in ql):
            where_sql, params = where_builder(
                date_from=date_from, date_to=date_to,
                fiscal_year=fiscal_year, bank=bank, account=account, attribute=attribute,
                func_code=func_code, head_name=head_name, column1=column1,
            )
            sql = f"""
            select count(*) as rows, sum(coalesce(net_flow,0)) as amount
            from {REL_RP}
            {where_sql}
            """
            st.dataframe(run_df(sql, params), use_container_width=True)

        elif "cashflow" in ql and ("inflow" in ql or "outflow" in ql):
            where_sql, params = where_builder(
                date_from=date_from, date_to=date_to,
                fiscal_year=fiscal_year, bank=bank, account=account, attribute=attribute,
                func_code=func_code, head_name=head_name, column1=column1,
            )
            sql = f"""
            select direction,
                   sum(abs(coalesce(net_flow,0))) as amount
            from {REL_CF}
            {where_sql}
            group by 1
            order by 1
            """
            st.dataframe(run_df(sql, params), use_container_width=True)

        else:
            st.info("Supported: monthly revenue, top expense heads, pending recoup summary, cashflow inflow/outflow. Add more templates safely as needed.")
