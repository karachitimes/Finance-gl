import streamlit as st
from sqlalchemy import text

from db import run_df
from filters import build_where_from_ui
from utils import show_df
from semantic import has_column

def _safe_select_cols(engine, rel: str, preferred: list[str]) -> list[str]:
    cols = []
    for c in preferred:
        try:
            if has_column(engine, rel, c):
                cols.append(c)
        except Exception:
            # If schema check fails, keep minimal safe columns
            pass
    return cols

def render_search_tab(engine, f, *, rel: str):
    st.subheader("Search Transactions")
    st.caption("Search in description, pay_to, head_name, account (and bill/voucher/reference if available).")

    term = st.text_input("Search term", placeholder="e.g. vendor name, cheque no, bill no, head…")
    if not term:
        st.info("Type a search term to see results.")
        return

    # Global filters apply (same behavior as other tabs)
    where, params, _ = build_where_from_ui(
        f["df"], f["dt"], f["bank"], f["head"], f["account"], f["attribute"], f["func_code"],
        fy_label=f.get("fy_label", "ALL"),
        func_override=None,
    )

    # Search condition
    params = dict(params)
    params["q"] = f"%{term.strip()}%"

    # Build search OR clauses only for columns that exist
    searchable = ["description", "pay_to", "head_name", "account", "bill_no", "voucher_no", "reference_no", "status"]
    or_clauses = []
    for c in searchable:
        if has_column(engine, rel, c):
            or_clauses.append(f"coalesce({c},'') ilike :q")

    if not or_clauses:
        st.error("Search columns not found in this relation. Please use v_finance_semantic or gl_register.")
        return

    where.append("(" + " or ".join(or_clauses) + ")")
    where_sql = " and ".join(where) if where else "1=1"

    # Select columns (only if they exist)
    preferred_cols = [
        '"date"', "bank", "account", "head_name", "pay_to", "description",
        "debit_payment", "credit_deposit", "gl_amount", "bill_no", "status", "voucher_no", "reference_no"
    ]
    cols = _safe_select_cols(engine, rel, [c.replace('"','').replace('"','') for c in preferred_cols])
    # Re-add quoting for date if present
    select_cols = []
    for c in cols:
        if c == "date":
            select_cols.append('"date"')
        else:
            select_cols.append(c)

    if not select_cols:
        select_cols = ['"date"', "bank", "account", "head_name", "pay_to", "description"]

    sql = f"""
        select {", ".join(select_cols)}
        from {rel}
        where {where_sql}
        order by "date" desc
        limit 500
    """

    df = run_df(engine, sql, params, rel=rel)
    if df.empty:
        st.warning("No rows for selected filters / term.")
        return

    show_df(df)
