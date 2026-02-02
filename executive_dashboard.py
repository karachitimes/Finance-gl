
import streamlit as st

def render_executive_dashboard(engine, f, *, rel):
    st.subheader("Executive Dashboard")
    sql = """
        SELECT 
            SUM(credit_deposit) AS total_revenue,
            SUM(net_flow) AS total_expense,
            SUM(CASE WHEN bill_no = 'Recoup' THEN (debit_payment - credit_deposit) ELSE 0 END) AS total_recoup
        FROM v_finance_semantic
        WHERE "date" BETWEEN :df AND
