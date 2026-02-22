"""Simulation and results page."""

from __future__ import annotations

import streamlit as st

from app.state import build_config_from_state, init_session_state
from kelly_mc.analytics import compute_analytics
from kelly_mc.engine import run_simulation
from kelly_mc.validators import validate_config

init_session_state()

st.header("Simulation & Results")

# Sidebar parameters
with st.sidebar:
    st.subheader("Simulation Parameters")
    num_sims = st.number_input(
        "Number of Simulations",
        min_value=100,
        max_value=100_000,
        value=10_000,
        step=1000,
    )
    num_periods = st.number_input(
        "Number of Periods",
        min_value=1,
        max_value=1000,
        value=100,
        step=10,
    )
    initial_capital = st.number_input(
        "Initial Capital",
        min_value=1.0,
        max_value=1_000_000.0,
        value=100.0,
        step=10.0,
    )
    default_kelly = st.slider(
        "Default Kelly Fraction",
        min_value=0.05,
        max_value=3.0,
        value=0.25,
        step=0.05,
    )
    use_seed = st.checkbox("Use Random Seed", value=True)
    seed = st.number_input("Seed", value=42, disabled=not use_seed) if use_seed else None

    run_button = st.button("Run Simulation", type="primary", use_container_width=True)

# Run simulation
if run_button:
    try:
        config = build_config_from_state(
            num_sims=num_sims,
            num_periods=num_periods,
            initial_capital=initial_capital,
            default_kelly_fraction=default_kelly,
            seed=int(seed) if seed is not None else None,
        )
    except ValueError as e:
        st.error(f"Configuration error: {e}")
        st.stop()

    errors = validate_config(config)
    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    with st.spinner("Running simulation..."):
        result = run_simulation(config)
        report = compute_analytics(result)
        st.session_state["sim_result"] = result
        st.session_state["analytics_report"] = report

# Display results
result = st.session_state.get("sim_result")
report = st.session_state.get("analytics_report")

if result is None or report is None:
    st.info("Configure your setups and click 'Run Simulation' to see results.")
    st.stop()

# Summary metrics
st.subheader("Summary")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("CAGR (Mean)", f"{report.cagr_mean:.2%}")
m2.metric("CAGR (Median)", f"{report.cagr_median:.2%}")
m3.metric("Sharpe Ratio", f"{report.sharpe_ratio:.3f}")
m4.metric("Max Drawdown (Avg)", f"{report.max_drawdown_mean:.2%}")
m5.metric("P(Loss)", f"{report.probability_of_loss:.1%}")
m6.metric("P(Ruin)", f"{report.probability_of_ruin:.1%}")

# Tabs for detailed views
tab_paths, tab_terminal, tab_drawdown, tab_stats, tab_kelly = st.tabs(
    ["Portfolio Paths", "Terminal Distribution", "Drawdown", "Statistics", "Kelly Analysis"]
)

with tab_paths:
    from app.components.charts import plot_portfolio_paths
    fig = plot_portfolio_paths(result)
    st.plotly_chart(fig, use_container_width=True)

with tab_terminal:
    from app.components.charts import plot_terminal_distribution
    fig = plot_terminal_distribution(result, report)
    st.plotly_chart(fig, use_container_width=True)

with tab_drawdown:
    from app.components.charts import plot_drawdown_analysis
    fig = plot_drawdown_analysis(result, report)
    st.plotly_chart(fig, use_container_width=True)

with tab_stats:
    from app.components.metrics_display import render_detailed_stats
    render_detailed_stats(report)

with tab_kelly:
    from app.components.charts import plot_kelly_curves
    fig = plot_kelly_curves(result.config, report)
    st.plotly_chart(fig, use_container_width=True)
