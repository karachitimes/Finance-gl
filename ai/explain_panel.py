
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

from db import run_df
from utils import show_df


def _month_bounds(d: pd.Timestamp):
    d = pd.to_datetime(d).to_pydatetime().date()
    start = d.replace(day=1)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1, day=1)
    else:
        end = start.replace(month=start.month + 1, day=1)
    return start, end


def _safe_show(df: pd.DataFrame):
    try:
        show_df(df)
    except TypeError:
        st.dataframe(df)


def _breakdown(engine, rel, f, *, kind: str, dim: str, start: date, end: date, limit: int = 15):
    where_sql = f.get("where_sql", "1=1")
    params = dict(f.get("params", {}))
    params.update({"mstart": start, "mend": end})

    if kind == "expense":
        amount_expr = "coalesce(sum(coalesce(net_flow,0)),0)"
        extra_where = ""
    else:
        amount_expr = "coalesce(sum(coalesce(credit_deposit,0)),0)"
        extra_where = "and func_code='Revenue'"

    sql = f"""
        select
          coalesce(nullif(trim({dim}),''),'(blank)') as key,
          {amount_expr} as amount
        from {rel}
        where {where_sql}
          and "date" >= %(mstart)s and "date" < %(mend)s
          {extra_where}
        group by 1
        order by 2 desc
        limit {int(limit)}
    """
    df = run_df(engine, sql, params, rel=rel)
    if df is None or df.empty:
        return pd.DataFrame(columns=["key", "amount"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df


def _lift_table(df_m: pd.DataFrame, df_b: pd.DataFrame):
    m = df_m.set_index("key")["amount"] if not df_m.empty else pd.Series(dtype=float)
    b = df_b.set_index("key")["amount"] if not df_b.empty else pd.Series(dtype=float)
    keys = sorted(set(m.index).union(set(b.index)))

    out = []
    m_total = float(m.sum()) if len(m) else 0.0
    b_total = float(b.sum()) if len(b) else 0.0

    for k in keys:
        mv = float(m.get(k, 0.0))
        bv = float(b.get(k, 0.0))
        m_share = (mv / m_total) if m_total else 0.0
        b_share = (bv / b_total) if b_total else 0.0

        pct = ((mv - bv) / bv * 100.0) if bv else (100.0 if mv else 0.0)
        share_lift = (m_share - b_share) * 100.0  # percentage points

        out.append({
            "key": k,
            "month_amount": mv,
            "baseline_amount": bv,
            "lift_%": pct,
            "share_lift_pp": share_lift,
        })

    df_out = pd.DataFrame(out)
    if df_out.empty:
        return df_out
    return df_out.sort_values(["share_lift_pp", "lift_%"], ascending=False)


def _severity_score(df_monthly: pd.DataFrame, *, month: pd.Timestamp, value_col: str):
    d = df_monthly.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").fillna(0.0).astype(float)

    med = float(np.median(d[value_col])) if len(d) else 0.0
    mad = float(np.median(np.abs(d[value_col] - med))) if len(d) else 0.0

    mv = float(d.loc[d["month"] == month, value_col].iloc[0])

    if mad == 0:
        std = float(d[value_col].std(ddof=0))
        z = abs((mv - float(d[value_col].mean())) / std) if std else 0.0
    else:
        z = abs(0.6745 * (mv - med) / mad)

    score = float(100.0 * (1.0 - np.exp(-0.6 * z)))
    return score, z


def render_explain_panel(engine, f, *, rel, df_anom: pd.DataFrame):
    st.markdown("### 🧭 Why did this anomaly happen?")

    if df_anom is None or df_anom.empty or "is_anomaly" not in df_anom.columns:
        st.info("No anomaly series available.")
        return

    anoms = df_anom[df_anom["is_anomaly"]].copy()
    if anoms.empty:
        st.info("No anomalous months detected under current filters.")
        return

    anoms = anoms.sort_values("month", ascending=False)
    choices = anoms["month"].dt.strftime("%Y-%m").tolist()
    choice = st.selectbox("Select anomalous month", choices, index=0)

    month = pd.to_datetime(choice + "-01")
    mstart, mend = _month_bounds(month)

    baseline_mode = st.radio("Baseline", ["Previous 3 months", "All filtered period"], horizontal=True)
    if baseline_mode == "Previous 3 months":
        b_start = (pd.Timestamp(mstart) - pd.DateOffset(months=3)).date()
        b_end = pd.Timestamp(mstart).date()
        baseline_label = f"{b_start} → {b_end}"
    else:
        params = f.get("params", {})
        b_start = params.get("df") or (pd.Timestamp(mstart) - pd.DateOffset(months=6)).date()
        b_end = params.get("dt") or pd.Timestamp(mstart).date()
        baseline_label = f"{b_start} → {b_end}"

    sev, z = _severity_score(df_anom, month=month, value_col="expense")
    c1, c2 = st.columns(2)
    c1.metric("Anomaly severity (0–100)", f"{sev:.1f}")
    c2.metric("Robust z-score", f"{z:.2f}")
    st.caption(f"Month window: {mstart} → {mend} | Baseline: {baseline_label}")

    dim = st.selectbox("Explain by", ["head_name", "pay_to", "bank"], index=0)
    top_n = st.slider("Top N drivers", 5, 30, 12)

    df_m = _breakdown(engine, rel, f, kind="expense", dim=dim, start=mstart, end=mend, limit=top_n)
    df_b = _breakdown(engine, rel, f, kind="expense", dim=dim, start=b_start, end=b_end, limit=top_n)

    lift = _lift_table(df_m, df_b)
    if lift.empty:
        st.info("Not enough breakdown data to explain this anomaly.")
        return

    lift_top = lift.head(top_n).copy()
    lift_top["lift_%"] = lift_top["lift_%"].round(1)
    lift_top["share_lift_pp"] = lift_top["share_lift_pp"].round(2)

    st.markdown("#### Lift drivers (month vs baseline)")
    _safe_show(lift_top[["key", "month_amount", "baseline_amount", "lift_%", "share_lift_pp"]])

    top = lift_top.iloc[0]["key"] if len(lift_top) else "(n/a)"
    top2 = lift_top.iloc[1]["key"] if len(lift_top) > 1 else "(n/a)"
    sl1 = float(lift_top.iloc[0]["share_lift_pp"]) if len(lift_top) else 0.0
    sl2 = float(lift_top.iloc[1]["share_lift_pp"]) if len(lift_top) > 1 else 0.0

    narrative = (
        f"In {choice}, an expense anomaly was detected (severity {sev:.1f}/100, robust z={z:.2f}). "
        f"Compared to baseline, the biggest drivers were '{top}' (+{sl1:.2f}pp share lift) "
        f"and '{top2}' (+{sl2:.2f}pp share lift). "
        f"This indicates the anomaly was primarily driven by changes in {dim} spending concentration."
    )
    st.text_area("One-click narrative", narrative, height=120)
