
import streamlit as st

def render_health_scores(engine, f, *, rel):
    st.subheader("Health Scores")
    st.metric("Organizational Health Score", 72)
    st.metric("Financial Fragility Index", 28)
