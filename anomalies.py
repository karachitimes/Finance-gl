import streamlit as st
import pandas as pd
import numpy as np

from db import run_df
from filters import build_where_from_ui
from semantic import (
    get_source_relation,
    pick_view,
    expense_amount_expr,
    cashflow_net_expr,
    REL_EXP,
    REL_CF,
    REL_REV,
)
from utils import show_df


def _zscore_last(df: pd.DataFrame, *, value_col: str, group_col: str, window: int) -> pd.DataFrame:
    """Compute rolling z-score for each group; return latest month rows with scores."""
    if df.empty:
        return df

    d = df.copy()
    d["month"] = pd.to_datetime(d["month"])
    d = d.sort_values([group_col, "month"]).reset_index(drop=True)

    def _calc(g: pd.DataFrame) -> pd.DataFrame:
        x = pd.to_numeric(g[value_col], errors="coerce").fillna(0.0)
        roll_mean = x.rolling(window=window, min_periods=max(2, window // 2)).mean()
        roll_std = x.rolling(window=window, min_periods=max(2, window // 2)).std(ddof=0)
        z = (x - roll_mean) / roll_std.replace(0, np.nan)
        out = g.copy()
        out["baseline_mean"] = roll_mean
        out["baseline_std"] = roll_std
        out["z_score"] = z
        return out

    d = d.groupby(group_col, group_keys=False).apply(_calc)

    # keep only latest month in the data
    latest = d["month"].max()
    latest_rows = d[d["month"] == latest].copy()
    latest_rows["abs_z"] = latest_rows["z_score"].abs()
    latest_rows = latest_rows.sort_values("abs_z", ascending=False)
    return latest_rows


def render_anomalies_tab(engine, f, *, rel: str):
    st.subheader("Anomaly Detection (Spikes / Drops / Reversals)")
    st.caption("Flags unusual movement vs rolling baseline. Uses your current date filters.")

    anomaly_kind = st.selectbox(
        "What do you want to scan?",
        [
            "Expense spikes (by Head)",
            "Expense spikes (by Bank)",
            "Expense spikes (by Payee)",
            "Revenue drops (by Head)",
            "Cashflow reversals (by Bank)",
        ],
        index=0,
    )
    window = st.slider("Baseline window (months)", min_value=3, max_value=18, value=6, step=1)
    z_thresh = st.slider("Flag threshold (|z|)", min_value=1.0, max_value=5.0, value=2.5, step=0.1)
    top_n = st.slider("Show top N", min_value=10, max_value=200, value=50, step=10)

    rel_sem = get_source_relation(engine)
    # Respect caller-provided base relation when available
    rel0 = rel or rel_sem

    where, params, _ = build_where_from_ui(
        f["df"], f["dt"], f["bank"], f["head"], f["account"], f["attribute"], f["func_code"],
        fy_label=f["fy_label"], func_override=None
    )
    where_sql = " and ".join(where) if where else "1=1"

    if anomaly_kind.startswith("Expense"):
        rele = pick_view(engine, REL_EXP, rel0)
        amt_expr = expense_amount_expr(engine, rele)
        if "by Head" in anomaly_kind:
            group_col = "head_name"
            group_label = "Head"
        elif "by Bank" in anomaly_kind:
            group_col = "bank"
            group_label = "Bank"
        else:
            group_col = "pay_to"
            group_label = "Payee"

        sql = f"""
            select date_trunc('month', "date")::date as month,
                   coalesce({group_col}, 'UNKNOWN') as grp,
                   sum({amt_expr}) as amount
            from {rele}
            where {where_sql}
            group by 1,2
            order by 1,3 desc
        """
        df = run_df(engine, sql, params, ["month", group_label, "Amount"], rel=rele)
        if df.empty:
            st.info("No rows for selected filters.")
            return
        df = df.rename(columns={group_label: "group", "Amount": "amount"})
        latest = _zscore_last(df.rename(columns={"group": "grp"}), value_col="amount", group_col="grp", window=window)
        if latest.empty:
            st.info("Not enough history in the selected date range to compute a baseline.")
            return
        latest = latest[latest["abs_z"].fillna(0) >= z_thresh].head(int(top_n))
        out = latest[["month", "grp", "amount", "baseline_mean", "baseline_std", "z_score"]].copy()
        out.columns = ["Month", group_label, "Amount", "Baseline Mean", "Baseline Std", "Z Score"]
        show_df(out, label_col=group_label)
        return

    if anomaly_kind.startswith("Revenue"):
        relr = pick_view(engine, REL_REV, rel0)
        sql = f"""
            select date_trunc('month', "date")::date as month,
                   coalesce(head_name, 'UNKNOWN') as grp,
                   sum(coalesce(credit_deposit,0)) as amount
            from {relr}
            where {where_sql}
              and coalesce(credit_deposit,0) > 0
              and coalesce(func_code,'') = 'Revenue'
            group by 1,2
            order by 1,3 desc
        """
        df = run_df(engine, sql, params, ["month", "Head", "Revenue"], rel=relr)
        if df.empty:
            st.info("No revenue rows for selected filters.")
            return
        df = df.rename(columns={"Head": "grp", "Revenue": "amount"})
        latest = _zscore_last(df, value_col="amount", group_col="grp", window=window)
        latest["drop_score"] = -latest["z_score"]
        latest = latest.sort_values("drop_score", ascending=False)
        latest = latest[latest["drop_score"].fillna(0) >= z_thresh].head(int(top_n))
        out = latest[["month", "grp", "amount", "baseline_mean", "baseline_std", "z_score"]].copy()
        out.columns = ["Month", "Head", "Revenue", "Baseline Mean", "Baseline Std", "Z Score"]
        st.caption("Revenue drops are flagged where Z Score is strongly negative.")
        show_df(out, label_col="Head")
        return

    # Cashflow reversals (by bank)
    relc = pick_view(engine, REL_CF, rel0)
    net_expr = cashflow_net_expr(engine, relc)
    sql = f"""
        select date_trunc('month', "date")::date as month,
               coalesce(bank, 'UNKNOWN') as bank,
               sum({net_expr}) as net
        from {relc}
        where {where_sql}
        group by 1,2
        order by 1,2
    """
    df = run_df(engine, sql, params, ["month", "Bank", "Net"], rel=relc)
    if df.empty:
        st.info("No cashflow rows for selected filters.")
        return
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["Bank", "month"])
    df["prev_net"] = df.groupby("Bank")["Net"].shift(1)
    df["reversal"] = np.sign(df["prev_net"].fillna(0)) != np.sign(df["Net"].fillna(0))
    latest_month = df["month"].max()
    latest = df[df["month"] == latest_month].copy()
    latest = latest[latest["reversal"]].head(int(top_n))
    if latest.empty:
        st.success("No bank cashflow reversals detected in the latest month in your selected range.")
        return
    out = latest[["month", "Bank", "prev_net", "Net"]].copy()
    out.columns = ["Month", "Bank", "Previous Month Net", "Current Month Net"]
    show_df(out, label_col="Bank")
