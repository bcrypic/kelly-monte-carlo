"""Plotly chart components for simulation results."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kelly_mc.kelly import compute_kelly_fraction, expected_log_growth_at
from kelly_mc.models import AnalyticsReport, SimulationConfig, SimulationResult


def plot_portfolio_paths(result: SimulationResult) -> go.Figure:
    """Fan chart of portfolio value paths with percentile bands."""
    pv = result.portfolio_values
    periods = np.arange(pv.shape[1])

    p10 = np.percentile(pv, 10, axis=0)
    p25 = np.percentile(pv, 25, axis=0)
    p50 = np.percentile(pv, 50, axis=0)
    p75 = np.percentile(pv, 75, axis=0)
    p90 = np.percentile(pv, 90, axis=0)

    fig = go.Figure()

    # 10-90 band
    fig.add_trace(go.Scatter(
        x=np.concatenate([periods, periods[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="10th-90th percentile",
    ))

    # 25-75 band
    fig.add_trace(go.Scatter(
        x=np.concatenate([periods, periods[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.25)",
        line=dict(color="rgba(255,255,255,0)"),
        name="25th-75th percentile",
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=periods, y=p50,
        mode="lines",
        line=dict(color="rgb(31, 119, 180)", width=2),
        name="Median",
    ))

    # Sample paths (thin, low opacity)
    n_sample = min(30, result.portfolio_values.shape[0])
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(pv.shape[0], n_sample, replace=False)
    for idx in sample_idx:
        fig.add_trace(go.Scatter(
            x=periods, y=pv[idx],
            mode="lines",
            line=dict(color="rgba(100, 100, 100, 0.08)", width=0.5),
            showlegend=False,
        ))

    fig.update_layout(
        title="Portfolio Value Paths",
        xaxis_title="Period",
        yaxis_title="Portfolio Value",
        hovermode="x unified",
    )
    return fig


def plot_terminal_distribution(
    result: SimulationResult, report: AnalyticsReport
) -> go.Figure:
    """Histogram of terminal wealth with percentile markers."""
    terminal = result.portfolio_values[:, -1]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=terminal,
        nbinsx=80,
        name="Terminal Wealth",
        marker_color="rgba(31, 119, 180, 0.7)",
    ))

    # Percentile lines
    for pct, color, dash in [
        (5, "red", "dash"),
        (50, "green", "solid"),
        (95, "blue", "dash"),
    ]:
        val = report.terminal_percentiles[pct]
        fig.add_vline(
            x=val, line_dash=dash, line_color=color,
            annotation_text=f"P{pct}: {val:.1f}",
        )

    # Initial capital line
    fig.add_vline(
        x=result.config.initial_capital,
        line_dash="dot",
        line_color="black",
        annotation_text="Initial",
    )

    fig.update_layout(
        title="Terminal Wealth Distribution",
        xaxis_title="Terminal Portfolio Value",
        yaxis_title="Count",
    )
    return fig


def plot_drawdown_analysis(
    result: SimulationResult, report: AnalyticsReport
) -> go.Figure:
    """Drawdown time series and distribution."""
    pv = result.portfolio_values
    running_max = np.maximum.accumulate(pv, axis=1)
    drawdowns = (pv - running_max) / running_max

    periods = np.arange(drawdowns.shape[1])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Drawdown Over Time", "Max Drawdown Distribution"),
    )

    # Left: drawdown time series with bands
    dd_median = np.percentile(drawdowns, 50, axis=0)
    dd_10 = np.percentile(drawdowns, 10, axis=0)
    dd_90 = np.percentile(drawdowns, 90, axis=0)

    fig.add_trace(go.Scatter(
        x=np.concatenate([periods, periods[::-1]]),
        y=np.concatenate([dd_90, dd_10[::-1]]),
        fill="toself",
        fillcolor="rgba(255, 65, 54, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="10th-90th DD",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=periods, y=dd_median,
        mode="lines",
        line=dict(color="red", width=2),
        name="Median Drawdown",
    ), row=1, col=1)

    # Right: max drawdown histogram
    max_dd_per_path = np.min(drawdowns, axis=1)
    fig.add_trace(go.Histogram(
        x=max_dd_per_path,
        nbinsx=60,
        name="Max Drawdown",
        marker_color="rgba(255, 65, 54, 0.7)",
    ), row=1, col=2)

    fig.update_layout(
        title="Drawdown Analysis",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Period", row=1, col=1)
    fig.update_xaxes(title_text="Max Drawdown", row=1, col=2)
    fig.update_yaxes(title_text="Drawdown", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    return fig


def plot_kelly_curves(
    config: SimulationConfig, report: AnalyticsReport
) -> go.Figure:
    """G(f) growth rate curves for each setup."""
    fig = go.Figure()

    colors = [
        "rgb(31, 119, 180)", "rgb(255, 127, 14)", "rgb(44, 160, 44)",
        "rgb(214, 39, 40)", "rgb(148, 103, 189)",
    ]

    for i, (setup, info) in enumerate(zip(config.setups, report.kelly_info)):
        color = colors[i % len(colors)]

        # Compute G(f) curve
        min_return = min(sc.return_pct for sc in setup.scenarios)
        if min_return < 0:
            f_max = min(-1.0 / min_return * 0.95, 10.0)
        else:
            f_max = 10.0

        fractions = np.linspace(0, f_max, 200)
        growth_rates = [expected_log_growth_at(setup, f) for f in fractions]

        fig.add_trace(go.Scatter(
            x=fractions, y=growth_rates,
            mode="lines",
            name=f"{setup.name} G(f)",
            line=dict(color=color, width=2),
        ))

        # Mark optimal
        if info.optimal_fraction > 0:
            fig.add_trace(go.Scatter(
                x=[info.optimal_fraction],
                y=[info.expected_log_growth],
                mode="markers",
                marker=dict(size=10, color=color, symbol="star"),
                name=f"{setup.name} f*={info.optimal_fraction:.3f}",
            ))

        # Mark chosen fraction
        effective_f = config.get_effective_kelly(setup)
        g_at_chosen = expected_log_growth_at(setup, effective_f)
        fig.add_trace(go.Scatter(
            x=[effective_f],
            y=[g_at_chosen],
            mode="markers",
            marker=dict(size=8, color=color, symbol="circle"),
            name=f"{setup.name} chosen f={effective_f:.3f}",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray")

    fig.update_layout(
        title="Kelly Growth Rate G(f) by Setup",
        xaxis_title="Kelly Fraction (f)",
        yaxis_title="Expected Log Growth G(f)",
    )
    return fig
