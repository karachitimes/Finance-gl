import streamlit as st
import pandas as pd
import numpy as np

from db import run_df
from filters import build_where_from_ui
from semantic import get_source_relation, has_column
from utils import show_df


# -----------------------------
# Helpers
# -----------------------------
def _recoup_amount_expr() -> str:
    # unified recoup amount logic
    return "(coalesce(debit_payment,0) - coalesce(credit_deposit,0))"


def _safe_cols(engine, rel, cols):
    """Return only columns that actually exist in the relation"""
    out = []
    for c in cols:
        if c == '"date"':
            out.append(c)
        elif has_column(engine, rel, c):
            out.append(c)
    return out


# -----------------------------
# Public entry (keeps app.py compatibility)
# -----------------------------
def render_recoup_kpi_tab(engine, f, *, rel: str):
    render_recoup_intelligence_tab(engine, f, rel=rel)


# -----------------------------
# Main Intelligence Panel
# -----------------------------
def render_recoup_intelligence_tab(engine, f, *, rel: str):

    st.subheader("Recoup Intelligence Panel")
    st.caption("Logic: bill_no='Recoup' | status blank = pending | status filled = completed")

    # Resolve semantic relation
    rel_sem = get_source_relation(engine)
    rel0 = rel or rel_sem

    # Filters
    where, params, _ = build_where_from_ui(
        f["df"], f["dt"], f["bank"], f["head"], f["account"],
        f["attribute"], f["func_code"],
        fy_label=f["fy_label"], func_override=None
    )
    where_sql = " and ".join(where) if where else "1=1"

    amt_expr = _recoup_amount_expr()

    # =========================================================
    # 1) Summary (Pending vs Completed)
    # =========================================================
    sql_summary = f"""
        select
          case
            when nullif(trim(coalesce(status,'')),'') is null then 'pending'
            else 'completed'
          end as recoup_state,
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

    st.markdown("### Recoup Status Summary")
    show_df(df_sum, label_col="State")

    st.divider()

    # =========================================================
    # 2) Pending Recoup (Row-level Intelligence)
    # =========================================================
    st.markdown("### Pending Recoup (Row Level)")

    pending_cols = _safe_cols(engine, rel0, [
        '"date"',
        "bank", "account", "head_name", "pay_to",
        "bill_no", "status",
        "ref", "folio_chq_no",
        "description"
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
        st.success("No pending recoup in selected range.")
        return

    # Clean numeric
    df_pending["amount"] = pd.to_numeric(df_pending["amount"], errors="coerce").fillna(0.0)
    df_pending["age_days"] = pd.to_numeric(df_pending["age_days"], errors="coerce").fillna(0).astype(int)

    show_df(df_pending)

    st.divider()

    # =========================================================
    # 3) Aging Buckets
    # =========================================================
    st.markdown("### Aging Buckets")

    buckets = [
        (0, 30, "0–30"),
        (31, 60, "31–60"),
        (61, 90, "61–90"),
        (91, 180, "91–180"),
        (181, 10_000, "180+"),
    ]

    def _bucket(age):
        for lo, hi, label in buckets:
            if lo <= age <= hi:
                return label
        return "180+"

    df_pending["aging_bucket"] = df_pending["age_days"].apply(_bucket)

    df_aging = (
        df_pending
        .groupby("aging_bucket", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "Amount"})
    )

    order = [b[2] for b in buckets]
    df_aging["aging_bucket"] = pd.Categorical(df_aging["aging_bucket"], categories=order, ordered=True)
    df_aging = df_aging.sort_values("aging_bucket")

    show_df(df_aging, label_col="aging_bucket")

    st.divider()

    # =========================================================
    # 4) Recovery Efficiency
    # =========================================================
    st.markdown("### Recovery Efficiency")

    pending_amount = df_sum.loc[df_sum["State"] == "pending", "Amount"].sum()
    completed_amount = df_sum.loc[df_sum["State"] == "completed", "Amount"].sum()
    total = pending_amount + completed_amount

    recovery_rate = (completed_amount / total) * 100 if total != 0 else 0
    pending_ratio = (pending_amount / total) * 100 if total != 0 else 0

    df_eff = pd.DataFrame([
        {"Metric": "Completed Recovery %", "Value": round(recovery_rate, 2)},
        {"Metric": "Pending Exposure %", "Value": round(pending_ratio, 2)},
        {"Metric": "Completed Amount", "Value": round(completed_amount, 2)},
        {"Metric": "Pending Amount", "Value": round(pending_amount, 2)},
    ])

    show_df(df_eff, label_col="Metric")

    st.divider()

    # =========================================================
    # 5) Loss Risk Model
    # =========================================================
    st.markdown("### Loss Risk (Operational Model)")

    # Define high-risk threshold (configurable)
    RISK_DAYS = 180

    df_risk = df_pending[df_pending["age_days"] >= RISK_DAYS]

    loss_risk_amount = df_risk["amount"].sum()

    st.metric("High Risk Amount (>{} days)".format(RISK_DAYS), f"{loss_risk_amount:,.0f}")

    if not df_risk.empty and "pay_to" in df_risk.columns:
        st.markdown("#### Top Risk Payees")
        df_risk_payees = (
            df_risk
            .groupby("pay_to", as_index=False)["amount"]
            .sum()
            .sort_values("amount", ascending=False)
            .head(10)
        )
        show_df(df_risk_payees, label_col="pay_to")

    st.divider()

    # =========================================================
    # 6) Cycle Time (Approximation Model)
    # =========================================================
    st.markdown("### Cycle Time (Approximation)")

    # Cycle model: group by reference identity if available
    group_key = None
    if has_column(engine, rel0, "ref"):
        group_key = "ref"
    elif has_column(engine, rel0, "folio_chq_no"):
        group_key = "folio_chq_no"
    else:
        group_key = "bill_no"

    sql_cycle = f"""
        select
            {group_key} as recoup_id,
            min("date"::date) as start_date,
            max("date"::date) as last_seen_date,
            max(case when nullif(trim(coalesce(status,'')),'') is not null then "date"::date else null end) as completed_date
        from {rel0}
        where {where_sql}
          and bill_no = 'Recoup'
        group by {group_key}
    """

    df_cycle = run_df(engine, sql_cycle, params, rel=rel0)

    if not df_cycle.empty:
        df_cycle["cycle_days"] = (
            pd.to_datetime(df_cycle["completed_date"]) - pd.to_datetime(df_cycle["start_date"])
        ).dt.days

        df_cycle["cycle_days"] = df_cycle["cycle_days"].fillna(
            (pd.Timestamp.today().normalize() - pd.to_datetime(df_cycle["start_date"])).dt.days
        )

        st.markdown("#### Cycle Time Distribution")
        show_df(df_cycle[[group_key, "start_date", "completed_date", "cycle_days"]])

        st.metric("Average Cycle Time (days)", round(df_cycle["cycle_days"].mean(), 2))
        st.metric("Max Cycle Time (days)", int(df_cycle["cycle_days"].max()))
