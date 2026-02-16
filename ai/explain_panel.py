import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

from db import run_df


def _month_bounds(ts: pd.Timestamp):
    d = pd.to_datetime(ts).to_pydatetime().date()
    start = d.replace(day=1)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1, day=1)
    else:
        end = start.replace(month=start.month + 1, day=1)
    return start, end


def _breakdown(engine, rel, f, *, dim: str, start: date, end: date, limit: int = 15):
    """
    Safe breakdown query:
    - Whitelists allowed dimensions (prevents SQL errors/injection)
    - Quotes identifiers for Postgres/view compatibility
    - Guards LIMIT
    - Uses bound params for dates
    """

    # 🔐 Only allow known-safe dimensions → actual SQL identifiers
    DIM_MAP = {
        "head_name": '"head_name"',
        "pay_to": '"pay_to"',
        "bank": '"bank"',
    }

    dim_sql = DIM_MAP.get(dim)
    if not dim_sql:
        raise ValueError(f"Unsupported dimension: {dim}")

    where_sql = f.get("where_sql", "1=1")
    params = dict(f.get("params", {}))
    params.update({"mstart": start, "mend": end})

    # Guard limit (avoid weird values)
    try:
        limit_i = int(limit)
    except Exception:
        limit_i = 15
    limit_i = max(1, min(limit_i, 100))

    sql = f"""
        select
          coalesce(nullif(trim({dim_sql}),''),'(blank)') as key,
          coalesce(sum(coalesce(net_flow,0)),0) as amount
        from {rel}
        where {where_sql}
          and "date" >= %(mstart)s and "date" < %(mend)s
        group by 1
        order by 2 desc
        limit {limit_i}
    """

    df = run_df(engine, sql, params, rel=rel)
    if df is None or df.empty:
        return pd.DataFrame(columns=["key", "amount"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df


def _lift(df_month: pd.DataFrame, df_base: pd.DataFrame):
    m = df_month.set_index("key")["amount"] if not df_month.empty else pd.Series(dtype=float)
    b = df_base.set_index("key")["amount"] if not df_base.empty else pd.Series(dtype=float)

    keys = sorted(set(m.index).union(set(b.index)))
    m_total = float(m.sum()) if len(m) else 0.0
    b_total = float(b.sum()) if len(b) else 0.0

    out = []
    for k in keys:
        mv = float(m.get(k, 0.0))
        bv = float(b.get(k, 0.0))
        m_share = (mv / m_total) if m_total else 0.0
        b_share = (bv / b_total) if b_total else 0.0

        lift_pct = ((mv - bv) / bv * 100.0) if bv else (100.0 if mv else 0.0)
        share_lift_pp = (m_share - b_share) * 100.0

        out.append(
            {
                "key": k,
                "month_amount": mv,
                "baseline_amount": bv,
                "lift_%": lift_pct,
                "share_lift_pp": share_lift_pp,
            }
        )

    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values(["share_lift_pp", "lift_%"], ascending=False)


def _severity(df_monthly: pd.DataFrame, month: pd.Timestamp, value_col: str):
    s = pd.to_numeric(df_monthly[value_col], errors="coerce").fillna(0.0).astype(float)
    med = float(np.median(s)) if len(s) else 0.0
    mad = float(np.median(np.abs(s - med))) if len(s) else 0.0

    mv = float(df_monthly.loc[df_monthly["month"] == month, value_col].iloc[0])

    if mad == 0:
        std = float(s.std(ddof=0))
        z = abs((mv - float(s.mean())) / std) if std else 0.0
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
    else:
        params = f.get("params", {})
        b_start = params.get("df") or (pd.Timestamp(mstart) - pd.DateOffset(months=6)).date()
        b_end = params.get("dt") or pd.Timestamp(mstart).date()

    sev, z = _severity(df_anom, month, "expense")
    c1, c2 = st.columns(2)
    c1.metric("Anomaly severity (0–100)", f"{sev:.1f}")
    c2.metric("Robust z-score", f"{z:.2f}")

    dim = st.selectbox("Explain by", ["head_name", "pay_to", "bank"], index=0)
    top_n = st.slider("Top N drivers", 5, 30, 12)

    df_m = _breakdown(engine, rel, f, dim=dim, start=mstart, end=mend, limit=top_n)
    df_b = _breakdown(engine, rel, f, dim=dim, start=b_start, end=b_end, limit=top_n)

    lift = _lift(df_m, df_b)
    if lift.empty:
        st.info("Not enough breakdown data to explain this anomaly.")
        return

    lift_top = lift.head(top_n).copy()
    lift_top["lift_%"] = lift_top["lift_%"].round(1)
    lift_top["share_lift_pp"] = lift_top["share_lift_pp"].round(2)

    st.dataframe(
        lift_top[["key", "month_amount", "baseline_amount", "lift_%", "share_lift_pp"]],
        use_container_width=True,
    )

    top = lift_top.iloc[0]["key"]
    top2 = lift_top.iloc[1]["key"] if len(lift_top) > 1 else "(n/a)"
    sl1 = float(lift_top.iloc[0]["share_lift_pp"])
    sl2 = float(lift_top.iloc[1]["share_lift_pp"]) if len(lift_top) > 1 else 0.0

    narrative = (
        f"In {choice}, an expense anomaly was detected (severity {sev:.1f}/100, robust z={z:.2f}). "
        f"Compared to baseline, the biggest drivers were '{top}' (+{sl1:.2f}pp share lift) "
        f"and '{top2}' (+{sl2:.2f}pp share lift)."
    )
    st.text_area("One-click narrative", narrative, height=110)
