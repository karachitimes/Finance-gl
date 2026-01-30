import streamlit as st
from sqlalchemy import text
from datetime import date

# -------------------------------------------------
# RELATIONS (prefer views when available)
# -------------------------------------------------
REL_SEM = "public.v_finance_semantic"
REL_REV = "public.v_revenue"
REL_EXP = "public.v_expense"
REL_CF  = "public.v_cashflow"
REL_AR  = "public.v_receivable"

RECoup_START_DATE = date(2025, 7, 1)
BANK_REVENUE_DEFAULT = "Revenue:4069284635"
BANK_ASSIGNMENT_DEFAULT = "Assignment Account 1169255177"

@st.cache_data(ttl=3600)
def relation_exists(engine, rel: str) -> bool:
    with engine.connect() as conn:
        try:
            conn.execute(text(f"SELECT 1 FROM {rel} LIMIT 1"))
            return True
        except Exception:
            return False

@st.cache_data(ttl=3600)
def pick_view(engine, preferred: str, fallback: str) -> str:
    return preferred if relation_exists(engine, preferred) else fallback

@st.cache_data(ttl=3600)
def get_source_relation(engine) -> str:
    """Prefer semantic view if it exists; otherwise fall back to raw table."""
    with engine.connect() as conn:
        try:
            conn.execute(text(f"SELECT 1 FROM {REL_SEM} LIMIT 1"))
            return REL_SEM
        except Exception:
            return "public.gl_register"

@st.cache_data(ttl=3600)
def has_column(engine, rel: str, col: str) -> bool:
    if "." in rel:
        schema, name = rel.split(".", 1)
    else:
        schema, name = "public", rel
    with engine.connect() as conn:
        q = text("""
            select 1
            from information_schema.columns
            where table_schema = :schema
              and table_name = :name
              and column_name = :col
            limit 1
        """)
        return conn.execute(q, {"schema": schema, "name": name, "col": col}).scalar() is not None

def expense_amount_expr(engine, rel: str) -> str:
    if has_column(engine, rel, "net_flow"):
        return "greatest(coalesce(net_flow,0),0)"
    if has_column(engine, rel, "gl_amount"):
        return "greatest(coalesce(gl_amount,0),0)"
    return "greatest(coalesce(debit_payment,0) - coalesce(credit_deposit,0),0)"

def cashflow_net_expr(engine, rel: str) -> str:
    if has_column(engine, rel, "net_flow"):
        return "coalesce(net_flow,0)"
    if has_column(engine, rel, "gl_amount"):
        return "coalesce(gl_amount,0)"
    return "(coalesce(debit_payment,0) - coalesce(credit_deposit,0))"

def cashflow_dir_expr(engine, rel: str, net_expr: str) -> str:
    if has_column(engine, rel, "direction"):
        return "direction"
    return f"(case when {net_expr} >= 0 then 'out' else 'in' end)"

def tb_amount_expr(engine, rel: str) -> str:
    # prefer gl_amount if present, else net_flow, else debit-credit
    if has_column(engine, rel, "gl_amount"):
        return "coalesce(gl_amount,0)"
    if has_column(engine, rel, "net_flow"):
        return "coalesce(net_flow,0)"
    return "(coalesce(debit_payment,0) - coalesce(credit_deposit,0))"
