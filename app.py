import streamlit as st
from sqlalchemy.exc import OperationalError

from db import get_engine, test_connection
from semantic import get_source_relation
from filters import render_filter_bar, build_where_from_ui
from dashboards import render_revenue_tab, render_expense_tab, render_cashflow_tab, render_trial_balance_tab
from recoup import render_recoup_kpi_tab
from qa import render_qa_tab
from search import render_search_tab
from revenue_intelligence import render_revenue_intelligence
from expense_intelligence import render_expense_intelligence


from forecast_engine import render_forecast_engine
from scenario_engine import render_scenario_engine
from policy_engine import render_policy_engine
from ai.ai_dashboard import render_ai_dashboard
# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="KoFHA Finance Analytics System", layout="wide")
st.title("📊 KoFHA Finance Analytics System")

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
tab_rev, tab_exp, tab_cf, tab_tb, tab_rec_kpi, \
tab_rev_intel, tab_exp_intel, tab_ai, \
tab_qa, tab_search, tab_forecast, tab_scenario, tab_policy, tab_twin = st.tabs([
    "Revenue",
    "Expense",
    "Cashflow",
    "Trial Balance",
    "Recoup KPIs",
    "Revenue Intelligence",
    "Expense Intelligence",
    "AI Intelligence",
    "AI Q&A",
    "Search",
    "Forecast Engine",
    "Scenario Engine",
    "Policy Engine"

])

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

with tab_rev_intel:
    render_revenue_intelligence(engine, f, rel=rel)

with tab_exp_intel:
    render_expense_intelligence(engine, f, rel=rel)

with tab_ai:
    render_ai_dashboard(engine, f, rel=rel)

with tab_qa:
    render_qa_tab(engine, f, rel=rel)

with tab_search:
    render_search_tab(engine, f, rel=rel)
with tab_forecast:
    render_forecast_engine(engine, f, rel=rel)

with tab_scenario:
    render_scenario_engine(engine, f, rel=rel)

with tab_policy:
    render_policy_engine(engine, f, rel=rel)
