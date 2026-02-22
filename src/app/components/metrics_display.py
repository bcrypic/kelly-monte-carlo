"""Detailed statistics display component."""

from __future__ import annotations

import streamlit as st

from kelly_mc.models import AnalyticsReport


def render_detailed_stats(report: AnalyticsReport) -> None:
    """Render a detailed statistics breakdown."""

    st.subheader("Terminal Wealth")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean", f"{report.terminal_mean:,.2f}")
    c2.metric("Median", f"{report.terminal_median:,.2f}")
    c3.metric("Std Dev", f"{report.terminal_std:,.2f}")

    st.markdown("**Percentiles**")
    pct_cols = st.columns(len(report.terminal_percentiles))
    for col, (pct, val) in zip(pct_cols, sorted(report.terminal_percentiles.items())):
        col.metric(f"P{pct}", f"{val:,.2f}")

    st.divider()

    st.subheader("Returns")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("CAGR (Mean)", f"{report.cagr_mean:.4%}")
    r2.metric("CAGR (Median)", f"{report.cagr_median:.4%}")
    r3.metric("Volatility", f"{report.annualized_volatility:.4f}")
    r4.metric("Sharpe Ratio", f"{report.sharpe_ratio:.4f}")

    st.divider()

    st.subheader("Drawdown")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Max DD (Mean)", f"{report.max_drawdown_mean:.2%}")
    d2.metric("Max DD (Median)", f"{report.max_drawdown_median:.2%}")
    d3.metric("Max DD (Worst)", f"{report.max_drawdown_worst:.2%}")
    d4.metric("Avg Drawdown", f"{report.avg_drawdown:.2%}")
    d5.metric("Max DD Duration", f"{report.max_drawdown_duration_mean:.1f} periods")

    st.divider()

    st.subheader("Risk")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("P(Loss)", f"{report.probability_of_loss:.2%}")
    k2.metric("P(Ruin)", f"{report.probability_of_ruin:.2%}")
    k3.metric("VaR 95%", f"{report.var_95:,.2f}")
    k4.metric("CVaR 95%", f"{report.cvar_95:,.2f}")

    st.divider()

    st.subheader("Kelly Criterion by Setup")
    for info in report.kelly_info:
        with st.expander(info.setup_name):
            i1, i2, i3, i4 = st.columns(4)
            i1.metric("Optimal f*", f"{info.optimal_fraction:.4f}")
            i2.metric("Expected Return", f"{info.expected_return:.4%}")
            i3.metric("Variance", f"{info.variance:.6f}")
            i4.metric("Expected Log Growth", f"{info.expected_log_growth:.6f}")
            st.caption(info.odds_description)
