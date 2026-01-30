import streamlit as st
from sqlalchemy.exc import OperationalError

from db import get_engine, test_connection
from semantic import get_source_relation
from filters import render_filter_bar, build_where_from_ui
from dashboards import render_revenue_tab, render_expense_tab, render_cashflow_tab, render_trial_balance_tab
from recoup import render_recoup_kpi_tab
from qa import render_qa_tab

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Finance Analytics System", layout="wide")
st.title("📊 Finance Analytics System")

engine = get_engine()

# -------------------------------------------------
# DB STATUS
# -------------------------------------------------
try:
    _ = test_connection(engine)
    st.success("Database connected ✅")
except OperationalError as e:
    st.error("Database connection failed")
    try:
        real_err = str(e.orig)
    except Exception:
        real_err = str(e)
    st.code(real_err)
    st.stop()

# -------------------------------------------------
# FILTER BAR
# -------------------------------------------------
f = render_filter_bar(engine)

# Build a base where_sql+params that other modules can reuse
where, params, _ = build_where_from_ui(
    f["df"], f["dt"], f["bank"], f["head"], f["account"], f["attribute"], f["func_code"],
    fy_label=f["fy_label"], func_override=None
)
where_sql = " and ".join(where) if where else "1=1"
f["where_sql"] = where_sql
f["params"] = params

rel = get_source_relation(engine)

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab_rev, tab_exp, tab_cf, tab_tb, tab_rec_kpi, tab_qa = st.tabs(
    ["Revenue","Expense","Cashflow","Trial Balance","Recoup KPIs","AI Q&A"]
)

with tab_rev:
    render_revenue_tab(engine, f)

with tab_exp:
    render_expense_tab(engine, f)

with tab_cf:
    render_cashflow_tab(engine, f)

with tab_tb:
    render_trial_balance_tab(engine, f)

with tab_rec_kpi:
    render_recoup_kpi_tab(engine, f, rel=rel)

with tab_qa:
    render_qa_tab(engine, f, rel=rel)
