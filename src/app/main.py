"""Kelly Criterion Monte Carlo Simulator - Streamlit entry point."""

import streamlit as st

from app.state import init_session_state

st.set_page_config(
    page_title="Kelly MC Simulator",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

init_session_state()

st.title("Kelly Criterion Monte Carlo Simulator")
st.markdown(
    "Define setups with win/loss/stress scenarios, configure simulation parameters, "
    "and explore the distribution of portfolio outcomes."
)
st.markdown("Use the sidebar to navigate between **Setup Configuration** and **Simulation**.")
