
import streamlit as st
import pandas as pd
import numpy as np

from db import run_df
from ai.anomaly_engine import detect_anomalies
from ai.explain_panel import render_explain_panel


def _money(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return "0"


def _monthly_series(engine, rel, f, kind: str):
    where_sql = f.get("where_sql", "1=1")
    params = dict(f.get("params", {}))

    if kind == "expense":
        value_expr = "coalesce(sum(coalesce(net_flow,0)),0)"
        extra = ""
    elif kind == "revenue":
        value_expr = "coalesce(sum(coalesce(credit_deposit,0)),0)"
        extra = "and func_code='Revenue'"
    else:
        raise ValueError("kind must be 'expense' or 'revenue'")

    sql = f"""
        select date_trunc('month',"date")::date as month,
               {value_expr} as amount
        from {rel}
        where {where_sql}
          {extra}
        group by 1
        order by 1
    """
    df = run_df(engine, sql, params, rel=rel)
    if df is None or df.empty:
        return pd.DataFrame(columns=["month","amount"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["month"] = pd.to_datetime(df["month"])
    return df


def _linear_forecast(df: pd.DataFrame, value_col: str, periods: int):
    if df is None or df.empty or len(df) < 2:
        return pd.DataFrame(columns=["month", value_col, "pred"])

    d = df.copy().sort_values("month")
    y = d[value_col].astype(float).values
    x = np.arange(len(y), dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    x_future = np.arange(len(y) + periods, dtype=float)
    yhat = slope * x_future + intercept

    start = d["month"].iloc[0]
    months = pd.date_range(start=start, periods=len(x_future), freq="MS")

    out = pd.DataFrame({"month": months})
    out[value_col] = np.concatenate([y, [np.nan]*periods])
    out["pred"] = yhat
    return out


def _sustainability(df_rev: pd.DataFrame, df_exp: pd.DataFrame, periods: int):
    r = df_rev.rename(columns={"amount":"revenue"}).set_index("month")
    e = df_exp.rename(columns={"amount":"expense"}).set_index("month")
    idx = r.index.union(e.index).sort_values()

    hist = pd.DataFrame(index=idx)
    hist["revenue"] = r["revenue"] if "revenue" in r else 0.0
    hist["expense"] = e["expense"] if "expense" in e else 0.0
    hist["revenue"] = pd.to_numeric(hist["revenue"], errors="coerce").fillna(0.0)
    hist["expense"] = pd.to_numeric(hist["expense"], errors="coerce").fillna(0.0)
    hist["ratio"] = np.where(hist["expense"] == 0, np.nan, hist["revenue"]/hist["expense"])
    hist = hist.reset_index().rename(columns={"index":"month"})

    fr = _linear_forecast(hist, "revenue", periods)
    fe = _linear_forecast(hist, "expense", periods)

    proj = pd.DataFrame({"month": fr["month"]})
    proj["revenue_pred"] = fr["pred"]
    proj["expense_pred"] = fe["pred"]
    proj["ratio_pred"] = np.where(proj["expense_pred"] == 0, np.nan, proj["revenue_pred"]/proj["expense_pred"])
    return hist, proj


def render_intelligence_cockpit(engine, f, *, rel):
    st.subheader("🧠 Intelligence Cockpit")
    st.caption("Incremental build: trends → anomalies → severity → explanations → narrative, plus forecasting & sustainability.")

    df_exp = _monthly_series(engine, rel, f, "expense").rename(columns={"amount":"expense"})
    df_rev = _monthly_series(engine, rel, f, "revenue").rename(columns={"amount":"revenue"})

    if df_exp.empty and df_rev.empty:
        st.warning("No data available under current filters.")
        return

    # KPIs
    rev_total = float(df_rev["revenue"].sum()) if not df_rev.empty else 0.0
    exp_total = float(df_exp["expense"].sum()) if not df_exp.empty else 0.0
    net_total = rev_total - exp_total

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Revenue", _money(rev_total))
    c2.metric("Total Expense", _money(exp_total))
    c3.metric("Net Result", _money(net_total))

    # Phase 1: charts
    if not df_exp.empty:
        st.markdown("### Expense trend (monthly)")
        st.line_chart(df_exp.set_index("month")["expense"])

    if not df_rev.empty:
        st.markdown("### Revenue trend (monthly)")
        st.line_chart(df_rev.set_index("month")["revenue"])

    st.divider()

    # Phase 2: anomaly detection
    st.markdown("## 🔍 AI Anomaly Panel (Expenses)")
    if df_exp.empty or len(df_exp) < 4:
        st.info("Not enough monthly expense history for anomaly detection.")
        return

    sensitivity = st.slider("Anomaly sensitivity (higher = fewer anomalies)", 2.5, 6.0, 3.5, 0.1)
    df_anom = detect_anomalies(df_exp.copy(), "expense", z=sensitivity)

    # Phase 3: severity scoring
    s = pd.to_numeric(df_anom["expense"], errors="coerce").fillna(0.0).astype(float)
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med)))
    if mad == 0:
        df_anom["severity_z"] = 0.0
    else:
        df_anom["severity_z"] = (0.6745 * (s - med) / mad).abs()
    df_anom["severity_0_100"] = 100.0 * (1.0 - np.exp(-0.6 * df_anom["severity_z"]))

    anom_count = int(df_anom["is_anomaly"].sum())
    st.metric("Anomalous months detected", anom_count)

    st.dataframe(
        df_anom[df_anom["is_anomaly"]][["month","expense","severity_0_100","severity_z"]].sort_values("month", ascending=False),
        use_container_width=True
    )

    st.divider()

    # Phase 4/5: drilldown + narrative
    try:

        render_explain_panel(engine, f, rel=rel, df_anom=df_anom)

    except Exception as e:

        st.error("Explain panel failed (breakdown query). The cockpit still works, but drilldown needs a SQL fix.")

        st.exception(e)

        return

    worst = df_anom.sort_values("severity_z", ascending=False).iloc[0]
    worst_month = worst["month"].strftime("%Y-%m")
    worst_sev = float(worst["severity_0_100"])

    summary = (
        f"Total revenue {_money(rev_total)} and total expense {_money(exp_total)} produced net {_money(net_total)}. "
        f"{anom_count} anomalous expense month(s) were detected. "
        f"The strongest anomaly occurred in {worst_month} (severity {worst_sev:.1f}/100)."
    )
    st.markdown("### 📝 Executive Summary")
    st.text_area("Copy/paste summary", summary, height=120)

    st.divider()

    # Forecast: expense prediction, revenue projection, sustainability curve
    st.markdown("## 🔮 Forecast & Sustainability")
    horizon = st.slider("Forecast horizon (months)", 3, 24, 12, 1)

    if len(df_exp) >= 2:
        fexp = _linear_forecast(df_exp, "expense", horizon)
        st.markdown("### Expense prediction (linear trend)")
        st.line_chart(fexp.set_index("month")["pred"])

    if len(df_rev) >= 2:
        frev = _linear_forecast(df_rev, "revenue", horizon)
        st.markdown("### Revenue trend projection (linear trend)")
        st.line_chart(frev.set_index("month")["pred"])

    if not df_rev.empty and not df_exp.empty:
        hist, proj = _sustainability(df_rev.rename(columns={"revenue":"amount"}), df_exp.rename(columns={"expense":"amount"}), horizon)
        st.markdown("### Sustainability curve (Revenue ÷ Expense)")
        st.line_chart(hist.set_index("month")["ratio"])
        st.caption("Ratio > 1 means revenue covers expense (sustainable for the period).")

        st.markdown("### Projected sustainability (forecast)")
        st.line_chart(proj.set_index("month")["ratio_pred"])
