# intelligence_cockpit.py
import streamlit as st
import pandas as pd

from db import run_df, run_scalar
from semantic import get_source_relation
from utils import show_df
from ai.anomaly_engine import detect_anomalies


def _money(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return "0"


def _get_monthly_series(engine, rel, f, *, kind="expense"):
    """
    Returns a monthly series:
      - expense: sum of net_flow
      - revenue: sum of credit_deposit where func_code='Revenue'
    """
    where_sql = f.get("where_sql", "1=1")
    params = f.get("params", {})

    if kind == "expense":
        sql = f"""
            select date_trunc('month',"date")::date as month,
                   coalesce(sum(coalesce(net_flow,0)),0) as amount
            from {rel}
            where {where_sql}
            group by 1
            order by 1
        """
    else:
        sql = f"""
            select date_trunc('month',"date")::date as month,
                   coalesce(sum(coalesce(credit_deposit,0)),0) as amount
            from {rel}
            where {where_sql}
              and func_code = 'Revenue'
            group by 1
            order by 1
        """

    df = run_df(engine, sql, params, rel=rel)
    if df is None or df.empty:
        return pd.DataFrame(columns=["month", "amount"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df


def _top_contributors(engine, rel, f, *, kind="expense", dim="head_name", limit=10):
    """
    Practical explanation driver: top contributors by dimension (head/vendor/bank/payee).
    """
    where_sql = f.get("where_sql", "1=1")
    params = f.get("params", {})

    if kind == "expense":
        # vendor/payee explanation uses pay_to; heads use head_name; banks use bank
        sql = f"""
            select coalesce(nullif(trim({dim}),''), '(blank)') as key,
                   coalesce(sum(coalesce(net_flow,0)),0) as amount
            from {rel}
            where {where_sql}
            group by 1
            order by 2 desc
            limit {int(limit)}
        """
    else:
        sql = f"""
            select coalesce(nullif(trim({dim}),''), '(blank)') as key,
                   coalesce(sum(coalesce(credit_deposit,0)),0) as amount
            from {rel}
            where {where_sql}
              and func_code='Revenue'
            group by 1
            order by 2 desc
            limit {int(limit)}
        """

    df = run_df(engine, sql, params, rel=rel)
    if df is None or df.empty:
        return pd.DataFrame(columns=["key", "amount"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df


def _recoup_snapshot(engine, rel, f):
    """
    Uses your recoup logic:
      bill_no='Recoup' and status blank => pending
      bill_no='Recoup' and status not blank => completed
    """
    where_sql = f.get("where_sql", "1=1")
    params = f.get("params", {})

    sql = f"""
        select
          sum(case when bill_no='Recoup' and nullif(trim(coalesce(status,'')),'') is null
              then (coalesce(debit_payment,0) - coalesce(credit_deposit,0)) else 0 end) as pending_amt,
          sum(case when bill_no='Recoup' and nullif(trim(coalesce(status,'')),'') is not null
              then (coalesce(debit_payment,0) - coalesce(credit_deposit,0)) else 0 end) as completed_amt
        from {rel}
        where {where_sql}
    """
    df = run_df(engine, sql, params, rel=rel)
    if df is None or df.empty:
        return 0.0, 0.0
    pending = float(df.iloc[0].get("pending_amt") or 0)
    completed = float(df.iloc[0].get("completed_amt") or 0)
    return pending, completed


def render_intelligence_cockpit(engine, f, *, rel=None):
    st.subheader("🧠 Intelligence Cockpit")
    st.caption("Revenue + Expense intelligence, anomaly detection, explanations, and executive narrative in one place.")

    rel0 = rel or get_source_relation(engine)

    # ----------------------------
    # Section A — KPI Strip (fast)
    # ----------------------------
    df_rev_m = _get_monthly_series(engine, rel0, f, kind="revenue")
    df_exp_m = _get_monthly_series(engine, rel0, f, kind="expense")

    rev_total = float(df_rev_m["amount"].sum()) if not df_rev_m.empty else 0.0
    exp_total = float(df_exp_m["amount"].sum()) if not df_exp_m.empty else 0.0

    # stability proxies (lower std = more stable)
    rev_std = float(df_rev_m["amount"].std(ddof=0)) if len(df_rev_m) >= 2 else 0.0
    exp_std = float(df_exp_m["amount"].std(ddof=0)) if len(df_exp_m) >= 2 else 0.0

    pending_recoup, completed_recoup = _recoup_snapshot(engine, rel0, f)
    recoup_total = pending_recoup + completed_recoup
    recoup_eff = (completed_recoup / recoup_total * 100) if recoup_total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", _money(rev_total))
    c2.metric("Total Expense", _money(exp_total))
    c3.metric("Revenue Volatility (σ)", _money(rev_std))
    c4.metric("Expense Volatility (σ)", _money(exp_std))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Recoup Pending", _money(pending_recoup))
    c6.metric("Recoup Completed", _money(completed_recoup))
    c7.metric("Recoup Recovery %", f"{recoup_eff:.1f}%")
    c8.metric("Net Result", _money(rev_total - exp_total))

    st.divider()

    # ----------------------------
    # Section B — AI Anomaly Panel
    # ----------------------------
    st.markdown("## 🔍 AI Anomaly Panel (Expenses)")

    if df_exp_m.empty:
        st.warning("Not enough expense history for anomaly detection under current filters.")
        return

    sensitivity = st.slider("Anomaly sensitivity", 2.5, 6.0, 3.5, 0.1)
    df_anom = df_exp_m.rename(columns={"amount": "expense"})
    df_anom = detect_anomalies(df_anom, "expense", z=sensitivity)  # your pure-Python MAD/zscore engine

    anom_count = int(df_anom["is_anomaly"].sum())
    st.metric("Anomalous months detected", anom_count)

    show_df(df_anom[df_anom["is_anomaly"]].sort_values("month", ascending=False))

    st.divider()

    # ----------------------------
    # Section C — ML Explanation Panel (Practical drivers)
    # ----------------------------
    st.markdown("## 🧠 ML Explanation Panel (Top Drivers)")
    st.caption("Explainability done the reliable way: show which heads/vendors/banks drive the totals under current filters.")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Expense drivers by Head")
        df_head = _top_contributors(engine, rel0, f, kind="expense", dim="head_name", limit=10)
        show_df(df_head, label_col="key")

        st.markdown("### Expense drivers by Payee/Vendor")
        df_pay = _top_contributors(engine, rel0, f, kind="expense", dim="pay_to", limit=10)
        show_df(df_pay, label_col="key")

    with colB:
        st.markdown("### Revenue drivers by Head")
        df_rhead = _top_contributors(engine, rel0, f, kind="revenue", dim="head_name", limit=10)
        show_df(df_rhead, label_col="key")

        st.markdown("### Exposure by Bank")
        df_bank = _top_contributors(engine, rel0, f, kind="expense", dim="bank", limit=10)
        show_df(df_bank, label_col="key")

    st.divider()

    # ----------------------------
    # Section D — Executive Summary Generator
    # ----------------------------
    st.markdown("## 📝 Executive Summary Generator")

    # pick a few notable facts
    top_exp_head = df_head.iloc[0]["key"] if not df_head.empty else "(n/a)"
    top_exp_amt = float(df_head.iloc[0]["amount"]) if not df_head.empty else 0.0

    top_rev_head = df_rhead.iloc[0]["key"] if not df_rhead.empty else "(n/a)"
    top_rev_amt = float(df_rhead.iloc[0]["amount"]) if not df_rhead.empty else 0.0

    # simple narrative
    summary = (
        f"During the selected period, total revenue was {_money(rev_total)} and total expense was {_money(exp_total)}, "
        f"resulting in a net position of {_money(rev_total - exp_total)}. "
        f"Revenue volatility (σ) is {_money(rev_std)} and expense volatility (σ) is {_money(exp_std)}, "
        f"with {anom_count} anomalous expense month(s) detected. "
        f"The largest revenue head is '{top_rev_head}' at approximately {_money(top_rev_amt)}, "
        f"and the largest expense head is '{top_exp_head}' at approximately {_money(top_exp_amt)}. "
        f"Recoup recovery efficiency is {recoup_eff:.1f}%, with pending recoup {_money(pending_recoup)} and completed recoup {_money(completed_recoup)}."
    )

    st.text_area("Generated summary (copy/paste)", summary, height=180)
