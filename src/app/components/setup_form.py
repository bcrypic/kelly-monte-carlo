"""Reusable setup configuration form component."""

from __future__ import annotations

import numpy as np
import streamlit as st

from kelly_mc.kelly import compute_kelly_fraction
from kelly_mc.models import Scenario, Setup


def render_setup_form(index: int) -> None:
    """Render an editable form for a single setup."""
    setup_data = st.session_state["setups"][index]

    with st.expander(f"Setup {index + 1}: {setup_data['name']}", expanded=True):
        cols = st.columns([3, 2, 2])
        with cols[0]:
            setup_data["name"] = st.text_input(
                "Setup Name",
                value=setup_data["name"],
                key=f"setup_name_{index}",
            )
        with cols[1]:
            setup_data["probability"] = st.number_input(
                "Setup Probability",
                min_value=0.01,
                max_value=1.0,
                value=setup_data["probability"],
                step=0.05,
                format="%.2f",
                key=f"setup_prob_{index}",
            )
        with cols[2]:
            use_custom_kelly = st.checkbox(
                "Custom Kelly Fraction",
                value=setup_data.get("kelly_fraction") is not None,
                key=f"custom_kelly_{index}",
            )
            if use_custom_kelly:
                current_kelly = setup_data.get("kelly_fraction") or 0.25
                setup_data["kelly_fraction"] = st.number_input(
                    "Kelly Fraction",
                    min_value=0.01,
                    max_value=5.0,
                    value=current_kelly,
                    step=0.05,
                    format="%.2f",
                    key=f"kelly_frac_{index}",
                )
            else:
                setup_data["kelly_fraction"] = None
                st.caption("Using run default")

        st.markdown("**Scenarios**")
        for sc_idx, sc in enumerate(setup_data["scenarios"]):
            sc_cols = st.columns([2, 2, 2])
            with sc_cols[0]:
                sc["name"] = st.text_input(
                    "Name",
                    value=sc["name"],
                    key=f"sc_name_{index}_{sc_idx}",
                )
            with sc_cols[1]:
                sc["probability"] = st.number_input(
                    "Probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=sc["probability"],
                    step=0.05,
                    format="%.2f",
                    key=f"sc_prob_{index}_{sc_idx}",
                )
            with sc_cols[2]:
                sc["return_pct"] = st.number_input(
                    "Return (%)",
                    min_value=-99.0,
                    max_value=1000.0,
                    value=sc["return_pct"] * 100,
                    step=1.0,
                    format="%.1f",
                    key=f"sc_ret_{index}_{sc_idx}",
                ) / 100.0

        # Add/remove scenario buttons
        btn_cols = st.columns([1, 1, 4])
        with btn_cols[0]:
            if st.button("+ Scenario", key=f"add_sc_{index}"):
                setup_data["scenarios"].append(
                    {"name": f"Scenario {len(setup_data['scenarios']) + 1}",
                     "probability": 0.0, "return_pct": 0.0}
                )
                st.rerun()
        with btn_cols[1]:
            if len(setup_data["scenarios"]) > 2:
                if st.button("- Scenario", key=f"rm_sc_{index}"):
                    setup_data["scenarios"].pop()
                    st.rerun()

        # Validation
        sc_prob_total = sum(sc["probability"] for sc in setup_data["scenarios"])
        if not np.isclose(sc_prob_total, 1.0, atol=1e-4):
            st.warning(f"Scenario probabilities sum to {sc_prob_total:.2f} (must be 1.0)")
        else:
            # Show Kelly info
            try:
                scenarios = tuple(
                    Scenario(name=sc["name"], probability=sc["probability"],
                             return_pct=sc["return_pct"])
                    for sc in setup_data["scenarios"]
                )
                temp_setup = Setup(
                    name=setup_data["name"],
                    probability=max(setup_data["probability"], 0.01),
                    scenarios=scenarios,
                )
                kelly_info = compute_kelly_fraction(temp_setup)
                info_cols = st.columns(3)
                with info_cols[0]:
                    st.metric("Kelly Optimal f*", f"{kelly_info.optimal_fraction:.3f}")
                with info_cols[1]:
                    st.metric("Expected Return", f"{kelly_info.expected_return:.2%}")
                with info_cols[2]:
                    st.metric("Expected Log Growth", f"{kelly_info.expected_log_growth:.4f}")
            except (ValueError, Exception):
                st.error("Invalid scenario parameters")

        # Remove setup button
        if len(st.session_state["setups"]) > 1:
            if st.button(f"Remove Setup {index + 1}", key=f"rm_setup_{index}"):
                st.session_state["setups"].pop(index)
                st.rerun()
