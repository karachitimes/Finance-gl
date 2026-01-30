import streamlit as st
import pandas as pd

from db import run_df
from filters import build_where_from_ui
from semantic import get_source_relation, pick_view, has_column, REL_EXP
from utils import show_df


def render_classification_tab(engine, f, *, rel: str):
    st.subheader("Auto-Classification (Suggestions)")
    st.caption("Generates suggested head mappings based on historical patterns. This module does not write back to the database.")

    rel_sem = get_source_relation(engine)
    rel0 = rel or rel_sem
    rele = pick_view(engine, REL_EXP, rel0)

    # Filters
    where, params, _ = build_where_from_ui(
        f["df"], f["dt"], f["bank"], f["head"], f["account"], f["attribute"], f["func_code"],
        fy_label=f["fy_label"], func_override=None
    )
    where_sql = " and ".join(where) if where else "1=1"

    st.markdown("### 1) Find unmapped expenses")
    max_rows = st.slider("Max rows", min_value=50, max_value=2000, value=500, step=50)

    # Identify "unmapped": blank head_name (or placeholder)
    unmapped_sql = f"""
        select "date", bank, account, pay_to, head_name, bill_no, folio_chq_no, description,
               coalesce(debit_payment,0) as debit_payment,
               coalesce(credit_deposit,0) as credit_deposit
        from {rele}
        where {where_sql}
          and nullif(trim(coalesce(head_name,'')),'') is null
        order by "date" desc
        limit {int(max_rows)}
    """
    df_unmapped = run_df(engine, unmapped_sql, params, rel=rele)
    if df_unmapped.empty:
        st.success("No unmapped rows in the current filter/date range.")
        return
    show_df(df_unmapped)

    st.markdown("### 2) Generate suggestions")
    basis = st.selectbox("Suggest using", ["pay_to + account", "pay_to only", "account only"], index=0)
    min_support = st.slider("Minimum historical examples", min_value=3, max_value=50, value=8, step=1)
    confidence = st.slider("Minimum confidence", min_value=0.50, max_value=0.99, value=0.80, step=0.01)

    # Build a training summary from the *same filtered window* but only mapped rows
    if basis == "pay_to + account":
        group_keys = ["pay_to", "account"]
    elif basis == "pay_to only":
        group_keys = ["pay_to"]
    else:
        group_keys = ["account"]

    keys_sql = ", ".join(group_keys)
    hist_sql = f"""
        select {keys_sql}, head_name, count(*) as n
        from {rele}
        where {where_sql}
          and nullif(trim(coalesce(head_name,'')),'') is not null
        group by {keys_sql}, head_name
    """
    df_hist = run_df(engine, hist_sql, params, rel=rele)
    if df_hist.empty:
        st.warning("No historical mapped rows found in the current filter/date range. Expand date range and try again.")
        return

    # Compute top head per key and confidence
    df_hist["n"] = pd.to_numeric(df_hist["n"], errors="coerce").fillna(0)
    df_tot = df_hist.groupby(group_keys)["n"].sum().reset_index().rename(columns={"n": "total_n"})
    df_best = df_hist.sort_values("n", ascending=False).groupby(group_keys, as_index=False).first()
    df_best = df_best.merge(df_tot, on=group_keys, how="left")
    df_best["confidence"] = (df_best["n"] / df_best["total_n"]).replace([pd.NA, float("inf")], 0)

    df_best = df_best[(df_best["total_n"] >= int(min_support)) & (df_best["confidence"] >= float(confidence))]
    if df_best.empty:
        st.info("No suggestions meet the support/confidence thresholds. Lower thresholds or widen the date range.")
        return

    # Join suggestions onto unmapped rows
    df_s = df_unmapped.copy()
    for k in group_keys:
        if k not in df_s.columns:
            df_s[k] = None
    df_s = df_s.merge(df_best[group_keys + ["head_name", "confidence", "total_n"]], on=group_keys, how="left", suffixes=("", "_suggested"))
    df_s = df_s.rename(columns={"head_name_suggested": "Suggested Head", "confidence": "Confidence", "total_n": "Support"})

    out_cols = ["date", "bank", "account", "pay_to", "description", "Suggested Head", "Confidence", "Support", "bill_no", "folio_chq_no"]
    out_cols = [c for c in out_cols if c in df_s.columns]
    df_out = df_s[out_cols].copy()

    st.markdown("### Suggested mappings")
    show_df(df_out, label_col="Suggested Head")

    csv = df_out.to_csv(index=False).encode("utf-8")
    st.download_button("Download suggestions (CSV)", data=csv, file_name="head_mapping_suggestions.csv", mime="text/csv")
