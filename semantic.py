import streamlit as st
from datetime import date
from sqlalchemy import text

from db import get_engine

# -----------------------------
# Relations (views)
# -----------------------------
REL_SEM = "public.v_finance_semantic"
REL_REV = "public.v_revenue"
REL_EXP = "public.v_expense"
REL_AR  = "public.v_receivable"
REL_CF  = "public.v_cashflow"

# Recoup constants used by recoup.py (keep defaults consistent with your earlier app)
RECoup_START_DATE = date(2010, 1, 1)
BANK_REVENUE_DEFAULT = "Revenue:4069284635"
BANK_ASSIGNMENT_DEFAULT = "RM:4069284626"  # keep existing default; user can edit in UI


# -------------------------------------------------
# Source relation detection (schema-safe)
# IMPORTANT: cache_data must not receive SQLAlchemy engine
# -------------------------------------------------
@st.cache_data(ttl=300)
def get_source_relation_cached() -> str:
    engine = get_engine()
    with engine.connect() as conn:
        try:
            conn.execute(text(f"select 1 from {REL_SEM} limit 1"))
            return REL_SEM
        except Exception:
            return "public.gl_register"

def get_source_relation(engine) -> str:
    # Keep signature so callers don't change; engine intentionally unused here.
    return get_source_relation_cached()


# -------------------------------------------------
# View picker: prefer a specialized view if it exists, else fallback to base
# -------------------------------------------------
def pick_view(engine, preferred_rel: str, fallback_rel: str) -> str:
    try:
        with engine.connect() as conn:
            conn.execute(text(f"select 1 from {preferred_rel} limit 1"))
        return preferred_rel
    except Exception:
        return fallback_rel


# -------------------------------------------------
# Schema helpers
# -------------------------------------------------
@st.cache_data(ttl=3600)
def _columns_cached(rel: str) -> set[str]:
    engine = get_engine()
    schema, table = rel.split(".", 1) if "." in rel else ("public", rel)
    sql = text("""
        select column_name
        from information_schema.columns
        where table_schema = :schema and table_name = :table
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"schema": schema, "table": table}).fetchall()
    return {r[0] for r in rows}

def has_column(engine, rel: str, col: str) -> bool:
    # engine unused; kept for compatibility
    return col in _columns_cached(rel)


# -------------------------------------------------
# Amount expressions (avoid missing-column crashes)
# -------------------------------------------------
def expense_amount_expr(engine, rel: str) -> str:
    # Prefer net_flow when available, else debit-credit
    if has_column(engine, rel, "net_flow"):
        return "coalesce(net_flow,0)"
    # fallback
    return "(coalesce(debit_payment,0) - coalesce(credit_deposit,0))"

def cashflow_net_expr(engine, rel: str) -> str:
    if has_column(engine, rel, "net_flow"):
        return "coalesce(net_flow,0)"
    return "(coalesce(debit_payment,0) - coalesce(credit_deposit,0))"

def cashflow_dir_expr(engine, rel: str, net_expr: str) -> str:
    # Use direction column if present (v_cashflow), else derive from net
    if has_column(engine, rel, "direction"):
        return "direction"
    return f"(case when ({net_expr}) >= 0 then 'out' else 'in' end)"

def tb_amount_expr(engine, rel: str) -> str:
    # Trial balance uses net flow logic
    if has_column(engine, rel, "net_flow"):
        return "coalesce(net_flow,0)"
    return "(coalesce(debit_payment,0) - coalesce(credit_deposit,0))"
