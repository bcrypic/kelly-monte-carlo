"""Setup configuration page."""

import numpy as np
import streamlit as st

from app.components.setup_form import render_setup_form
from app.state import init_session_state

init_session_state()

st.header("Setup Configuration")
st.markdown("Define your Kelly Criterion setups. Each setup has a probability of being "
            "the active regime and contains win/loss/stress scenarios.")

# Add setup button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("+ Add Setup"):
        n = len(st.session_state["setups"]) + 1
        st.session_state["setups"].append({
            "name": f"Setup {n}",
            "probability": 0.0,
            "kelly_fraction": None,
            "scenarios": [
                {"name": "Win", "probability": 0.5, "return_pct": 0.10},
                {"name": "Loss", "probability": 0.4, "return_pct": -0.10},
                {"name": "Stress", "probability": 0.1, "return_pct": -0.30},
            ],
        })
        st.rerun()

with col2:
    if st.button("Load Example"):
        from app.state import DEFAULT_SETUPS
        st.session_state["setups"] = [
            {**s, "scenarios": [sc.copy() for sc in s["scenarios"]]}
            for s in DEFAULT_SETUPS
        ]
        st.rerun()

# Render each setup
for i in range(len(st.session_state["setups"])):
    if i < len(st.session_state["setups"]):
        render_setup_form(i)

# Overall validation
st.divider()
setup_prob_total = sum(s["probability"] for s in st.session_state["setups"])
if np.isclose(setup_prob_total, 1.0, atol=1e-4):
    st.success("All setup probabilities sum to 1.0. Ready to simulate.")
else:
    st.error(f"Setup probabilities sum to {setup_prob_total:.2f} (must equal 1.0)")
