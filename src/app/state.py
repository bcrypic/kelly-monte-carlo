"""Session state management for the Streamlit app."""

from __future__ import annotations

import streamlit as st

from kelly_mc.models import Scenario, Setup, SimulationConfig


DEFAULT_SETUPS = [
    {
        "name": "Setup A",
        "probability": 0.5,
        "kelly_fraction": None,
        "scenarios": [
            {"name": "Win", "probability": 0.6, "return_pct": 0.20},
            {"name": "Loss", "probability": 0.3, "return_pct": -0.10},
            {"name": "Stress", "probability": 0.1, "return_pct": -0.50},
        ],
    },
    {
        "name": "Setup B",
        "probability": 0.5,
        "kelly_fraction": None,
        "scenarios": [
            {"name": "Win", "probability": 0.4, "return_pct": 0.15},
            {"name": "Loss", "probability": 0.45, "return_pct": -0.08},
            {"name": "Stress", "probability": 0.15, "return_pct": -0.40},
        ],
    },
]


def init_session_state() -> None:
    """Initialize session state with defaults if not already set."""
    if "setups" not in st.session_state:
        st.session_state["setups"] = [s.copy() for s in DEFAULT_SETUPS]
        for i, s in enumerate(st.session_state["setups"]):
            st.session_state["setups"][i]["scenarios"] = [
                sc.copy() for sc in s["scenarios"]
            ]
    if "sim_result" not in st.session_state:
        st.session_state["sim_result"] = None
    if "analytics_report" not in st.session_state:
        st.session_state["analytics_report"] = None


def build_config_from_state(
    num_sims: int,
    num_periods: int,
    initial_capital: float,
    default_kelly_fraction: float,
    seed: int | None,
) -> SimulationConfig:
    """Convert mutable session state dicts into frozen dataclass config."""
    setups = []
    for s in st.session_state["setups"]:
        scenarios = tuple(
            Scenario(
                name=sc["name"],
                probability=sc["probability"],
                return_pct=sc["return_pct"],
            )
            for sc in s["scenarios"]
        )
        setups.append(
            Setup(
                name=s["name"],
                probability=s["probability"],
                scenarios=scenarios,
                kelly_fraction=s.get("kelly_fraction"),
            )
        )

    return SimulationConfig(
        setups=tuple(setups),
        num_simulations=num_sims,
        num_periods=num_periods,
        initial_capital=initial_capital,
        default_kelly_fraction=default_kelly_fraction,
        seed=seed,
    )
