
import streamlit as st
import pandas as pd
from db import run_df
from semantic import get_source_relation

# Simple Rule DSL Engine
# Format:
# IF <condition> THEN <action>

def render_policy_engine(engine, f, *, rel):
    st.subheader("Policy Engine")
    st.caption("Rule DSL + Compliance Automation")

    st.markdown("### Define Rule")
    rule = st.text_area("Rule (DSL)", 
        "IF func_code='Revenue' AND credit_deposit>0 AND bill_no IS NULL THEN FLAG 'Unbilled Revenue'"
    )

    if st.button("Validate Rule"):
        if "IF" in rule and "THEN" in rule:
            st.success("Rule syntax valid (basic DSL validation).")
        else:
            st.error("Invalid rule syntax. Use IF ... THEN ...")

    st.markdown("### Policy Checks (Active)")
    st.info("Future version: Rules auto-applied to semantic view with compliance tagging.")
