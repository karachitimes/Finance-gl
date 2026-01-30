import streamlit as st
import pandas as pd
import numpy as np

from db import run_df
from filters import build_where_from_ui
from semantic import get_source_relation, has_column
from utils import show_df


def _recoup_amount_expr(rel: str) -> str:
    return "(coalesce(debit_payment,0) - coalesce(credit_deposit,0))"


def render_recoup_kpi_tab(engine, f, *, rel: str):
    render_recoup_intelligence_tab(engine, f, rel=rel)


def render_recoup_intelligence_tab(engine, f, *, rel: str):
    st.subheader("Recoup Intelligence Panel")
    st.caption("Recoup = bill_no='Recoup'. status blank = pending; status filled = completed.")

    rel_sem = get_source_relation(engine)
    rel0 = rel or rel_sem

    where, params, _ = build_where_from_ui(
        f["df"], f["dt"], f["bank"], f["head"], f["account"], f["attribute"], f["func_code"],
        fy_label=f["fy_label"], func_override=None
    )
    where_sql = " and ".join(where) if where else "1=1"
    amt_expr = _recoup_amount_expr(rel0)

    def sel(cols: list[str]) -> list[str]:
        out = []
        for c in cols:
            if c == '"date"':
                out.append(c)
            else:
                if has_column(engine, rel0, c):
                    out.append(c)
        return out

    # ---- Summary
    sql_summary = f"""
        select
          case when nullif(trim(coalesce(status,'')),'') is null then 'pending' else 'completed' end as recoup_state,
          coalesce(sum({amt_expr}),0) as amount
        from {rel0}
        where {where_sql}
          and bill_no = 'Recoup'
        group by 1
        order by 1
    """
    df_sum = run_df(engine, sql_summary, params, ["State","Amount"], rel=rel0)
    if df_sum.empty:
        st.info("No recoup rows for selected filters/date range.")
        return
    show_df(df_sum, label_col="State")

    st.divider()

    # ---- Pending rows (safe select)
    st.subheader("Aging buckets (Pending)")
    pending_cols = sel([
        '"date"',
        "bank", "account", "head_name", "pay_to",
        "bill_no", "status",
        "ref", "folio_chq_no",
        "description",
    ])

    sql_pending = f"""
        select
          "date"::date as date,
          (current_date - ("date"::date))::int as age_days,
          {", ".join(pending_cols[1:])},
          {amt_expr} as amount
        from {rel0}
        where {where_sql}
          and bill_no = 'Recoup'
          and nullif(trim(coalesce(status,'')),'') is null
        order by "date" asc
        limit 5000
    """
    df_pending = run_df(engine, sql_pending, params, rel=rel0)

    if df_pending.empty:
        st.success("No pending recoup rows in selected range.")
        return

    df_pending["amount"] = pd.to_numeric(df_pending["amount"], errors="coerce").fillna(0.0)
    df_pending["age_days"] = pd.to_numeric(df_pending["age_days"], errors="coerce").fillna(0).astype(int)

    buckets = [(0,30,"0–30"), (31,60,"31–60"), (61,90,"61–90"), (91,180,"91–180"), (181,99999,"180+")]
    def _bucket(age: int) -> str:
        for lo, hi, label in buckets:
            if lo <= age <= hi:
                return label
        return "180+"

    df_pending["bucket"] = df_pending["age_days"].apply(_bucket)
    df_b = df_pending.groupby("bucket", as_index=False)["amount"].sum().rename(columns={"amount":"Amount"})
    order = [b[2] for b in buckets]
    df_b["bucket"] = pd.Categorical(df_b["bucket"], categories=order, ordered=True)
    df_b = df_b.sort_values("bucket").rename(columns={"bucket":"Aging Bucket"})
    show_df(df_b, label_col="Aging Bucket")
