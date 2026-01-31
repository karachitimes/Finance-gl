
import streamlit as st
import pandas as pd
from datetime import date

from db import run_df
from utils import show_df


def _money(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return "0"


def _aging_bucket(days: float):
    if days is None:
        return "unknown"
    d = float(days)
    if d <= 30: return "0-30"
    if d <= 60: return "31-60"
    if d <= 90: return "61-90"
    if d <= 180: return "91-180"
    return "180+"


def render_billing_tab(engine, f, *, rel):
    st.subheader("🧾 Billing Intelligence")
    st.caption("Bills issued are tracked in debit_payment; receipts are tracked in credit_deposit. Use bill_no and folio_chq_no (receipt no) for reconciliation.")

    where_sql = f.get("where_sql", "1=1")
    params = dict(f.get("params", {}))

    bill_like = st.text_input("Filter bill_no contains", value="")
    if bill_like.strip():
        where_bill = "and coalesce(bill_no,'') ilike %(bill_like)s"
        params["bill_like"] = f"%{bill_like.strip()}%"
    else:
        where_bill = ""

    sql_tot = f"""
        select
          coalesce(sum(case when coalesce(debit_payment,0) > 0 then coalesce(debit_payment,0) else 0 end),0) as bills_issued,
          coalesce(sum(case when coalesce(credit_deposit,0) > 0 then coalesce(credit_deposit,0) else 0 end),0) as receipts
        from {rel}
        where {where_sql}
        {where_bill}
    """
    dft = run_df(engine, sql_tot, params, rel=rel)
    bills = float(dft.iloc[0]["bills_issued"]) if dft is not None and not dft.empty else 0.0
    recs = float(dft.iloc[0]["receipts"]) if dft is not None and not dft.empty else 0.0

    c1,c2,c3 = st.columns(3)
    c1.metric("Bills Issued (Debit)", _money(bills))
    c2.metric("Receipts (Credit)", _money(recs))
    c3.metric("Net Outstanding (Issued - Receipts)", _money(bills - recs))

    st.divider()

    st.markdown("### Bills issued by bill_no")
    sql_by_bill = f"""
        select coalesce(nullif(trim(bill_no),''),'(blank)') as bill_no,
               coalesce(sum(case when coalesce(debit_payment,0) > 0 then coalesce(debit_payment,0) else 0 end),0) as issued,
               coalesce(sum(case when coalesce(credit_deposit,0) > 0 then coalesce(credit_deposit,0) else 0 end),0) as received,
               coalesce(sum(case when coalesce(debit_payment,0) > 0 then coalesce(debit_payment,0) else 0 end),0)
               - coalesce(sum(case when coalesce(credit_deposit,0) > 0 then coalesce(credit_deposit,0) else 0 end),0) as outstanding
        from {rel}
        where {where_sql}
        {where_bill}
        group by 1
        order by outstanding desc
        limit 200
    """
    df_bill = run_df(engine, sql_by_bill, params, rel=rel)
    show_df(df_bill)

    st.divider()

    st.markdown("### Receipts by bank (where deposited)")
    sql_bank = f"""
        select coalesce(nullif(trim(bank),''),'(blank)') as bank,
               count(*) as receipts_count,
               coalesce(sum(coalesce(credit_deposit,0)),0) as receipts_amount,
               sum(case when nullif(trim(coalesce(folio_chq_no,'')),'') is not null then 1 else 0 end) as with_receipt_no
        from {rel}
        where {where_sql}
          and coalesce(credit_deposit,0) > 0
          {where_bill}
        group by 1
        order by receipts_amount desc
        limit 50
    """
    df_bank = run_df(engine, sql_bank, params, rel=rel)
    show_df(df_bank)

    st.divider()

    st.markdown("### Outstanding aging by bill_no")
    sql_aging = f"""
        select
          coalesce(nullif(trim(bill_no),''),'(blank)') as bill_no,
          max("date")::date as last_activity_date,
          coalesce(sum(case when coalesce(debit_payment,0) > 0 then coalesce(debit_payment,0) else 0 end),0)
          - coalesce(sum(case when coalesce(credit_deposit,0) > 0 then coalesce(credit_deposit,0) else 0 end),0) as outstanding
        from {rel}
        where {where_sql}
        {where_bill}
        group by 1
        having (coalesce(sum(case when coalesce(debit_payment,0) > 0 then coalesce(debit_payment,0) else 0 end),0)
                - coalesce(sum(case when coalesce(credit_deposit,0) > 0 then coalesce(credit_deposit,0) else 0 end),0)) <> 0
        order by outstanding desc
        limit 200
    """
    df_age = run_df(engine, sql_aging, params, rel=rel)

    if df_age is None or df_age.empty:
        st.info("No outstanding bills found under current filters.")
        return

    today = date.today()
    df_age["days_since_activity"] = (pd.to_datetime(today) - pd.to_datetime(df_age["last_activity_date"])).dt.days
    df_age["aging_bucket"] = df_age["days_since_activity"].apply(_aging_bucket)

    show_df(df_age)

    st.markdown("#### Aging bucket totals")
    df_bucket = df_age.groupby("aging_bucket", as_index=False).agg(
        bills=("bill_no","count"),
        outstanding=("outstanding","sum")
    ).sort_values("aging_bucket")
    show_df(df_bucket)

    st.divider()

    st.markdown("### Receipts missing receipt no (folio_chq_no)")
    sql_missing = f"""
        select "date"::date as date, bank, account, pay_to, bill_no, folio_chq_no,
               coalesce(credit_deposit,0) as receipt_amount,
               description
        from {rel}
        where {where_sql}
          and coalesce(credit_deposit,0) > 0
          and nullif(trim(coalesce(folio_chq_no,'')),'') is null
          {where_bill}
        order by "date" desc
        limit 300
    """
    df_miss = run_df(engine, sql_missing, params, rel=rel)
    show_df(df_miss)
