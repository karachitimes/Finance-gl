import streamlit as st
import pandas as pd

from db import run_df
from filters import build_where_from_ui
from semantic import (
    get_source_relation,
    pick_view,
    expense_amount_expr,
    cashflow_net_expr,
    REL_REV,
    REL_EXP,
    REL_CF,
)


def _fmt_money(x: float) -> str:
    try:
        return f"{x:,.0f} PKR"
    except Exception:
        return str(x)


def render_narrative_tab(engine, f, *, rel: str):
    st.subheader("Narrative Generator (Management Brief)")
    st.caption("Deterministic briefing from the selected filters/date range. No black-box AI needed.")

    rel_sem = get_source_relation(engine)
    rel0 = rel or rel_sem

    where, params, _ = build_where_from_ui(
        f["df"], f["dt"], f["bank"], f["head"], f["account"], f["attribute"], f["func_code"],
        fy_label=f["fy_label"], func_override=None
    )
    where_sql = " and ".join(where) if where else "1=1"

    # Revenue
    relr = pick_view(engine, REL_REV, rel0)
    sql_rev = f"""
        select date_trunc('month', "date")::date as month,
               sum(coalesce(credit_deposit,0)) as revenue
        from {relr}
        where {where_sql}
          and coalesce(credit_deposit,0) > 0
          and coalesce(func_code,'') = 'Revenue'
        group by 1
        order by 1
    """
    df_rev = run_df(engine, sql_rev, params, ["Month","Revenue"], rel=relr)

    # Expense
    rele = pick_view(engine, REL_EXP, rel0)
    exp_expr = expense_amount_expr(engine, rele)
    sql_exp = f"""
        select date_trunc('month', "date")::date as month,
               sum({exp_expr}) as expense
        from {rele}
        where {where_sql}
        group by 1
        order by 1
    """
    df_exp = run_df(engine, sql_exp, params, ["Month","Expense"], rel=rele)

    # Cashflow net
    relc = pick_view(engine, REL_CF, rel0)
    net_expr = cashflow_net_expr(engine, relc)
    sql_cf = f"""
        select sum({net_expr}) as net_cashflow
        from {relc}
        where {where_sql}
    """
    df_cf = run_df(engine, sql_cf, params, ["Net Cashflow"], rel=relc)

    # Top revenue heads
    sql_rev_heads = f"""
        select coalesce(head_name,'UNKNOWN') as head,
               sum(coalesce(credit_deposit,0)) as revenue
        from {relr}
        where {where_sql}
          and coalesce(credit_deposit,0) > 0
          and coalesce(func_code,'') = 'Revenue'
        group by 1
        order by 2 desc
        limit 5
    """
    df_rh = run_df(engine, sql_rev_heads, params, ["Head","Revenue"], rel=relr)

    # Top expense heads
    sql_exp_heads = f"""
        select coalesce(head_name,'UNKNOWN') as head,
               sum({exp_expr}) as expense
        from {rele}
        where {where_sql}
        group by 1
        order by 2 desc
        limit 5
    """
    df_eh = run_df(engine, sql_exp_heads, params, ["Head","Expense"], rel=rele)

    # Period summary
    df0 = f["df"]
    dt0 = f["dt"]
    period = f"{df0} to {dt0}"

    total_rev = float(df_rev["Revenue"].sum()) if not df_rev.empty else 0.0
    total_exp = float(df_exp["Expense"].sum()) if not df_exp.empty else 0.0
    net_result = total_rev - total_exp
    net_cf = float(df_cf.iloc[0][0]) if (df_cf is not None and not df_cf.empty) else 0.0

    # Trends: compare last month vs previous month (within selected range)
    def _mom(df: pd.DataFrame, col: str) -> tuple[float, float]:
        if df is None or df.empty or len(df) < 2:
            return 0.0, 0.0
        d = df.copy()
        d["Month"] = pd.to_datetime(d["Month"])
        d = d.sort_values("Month")
        last = float(d.iloc[-1][col])
        prev = float(d.iloc[-2][col])
        return last, (last - prev)

    last_rev, mom_rev = _mom(df_rev, "Revenue")
    last_exp, mom_exp = _mom(df_exp, "Expense")

    lines: list[str] = []
    lines.append(f"**Period:** {period}")
    lines.append(f"**Total Revenue:** {_fmt_money(total_rev)}")
    lines.append(f"**Total Expense:** {_fmt_money(total_exp)}")
    lines.append(f"**Net Result:** {_fmt_money(net_result)}")
    lines.append(f"**Net Cashflow:** {_fmt_money(net_cf)}")

    if len(df_rev) >= 2:
        direction = "up" if mom_rev >= 0 else "down"
        lines.append(f"Revenue last month: {_fmt_money(last_rev)} ({direction} {_fmt_money(abs(mom_rev))} vs prior month).")
    if len(df_exp) >= 2:
        direction = "up" if mom_exp >= 0 else "down"
        lines.append(f"Expense last month: {_fmt_money(last_exp)} ({direction} {_fmt_money(abs(mom_exp))} vs prior month).")

    if not df_rh.empty:
        top = ", ".join([f"{r['Head']} ({_fmt_money(float(r['Revenue']))})" for _, r in df_rh.iterrows()])
        lines.append(f"Top revenue heads: {top}.")

    if not df_eh.empty:
        top = ", ".join([f"{r['Head']} ({_fmt_money(float(r['Expense']))})" for _, r in df_eh.iterrows()])
        lines.append(f"Top expense heads: {top}.")

    st.markdown("\n\n".join(lines))

    st.divider()
    st.subheader("Optional: Data used")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Revenue (monthly)")
        st.dataframe(df_rev, width="stretch")
    with c2:
        st.caption("Expense (monthly)")
        st.dataframe(df_exp, width="stretch")
