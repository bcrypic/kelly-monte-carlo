"""Post-simulation analytics and statistics."""

from __future__ import annotations

import numpy as np

from .kelly import compute_kelly_fraction
from .models import AnalyticsReport, SimulationResult


def compute_analytics(result: SimulationResult) -> AnalyticsReport:
    """Compute all analytics from simulation results."""
    config = result.config
    pv = result.portfolio_values  # (n_sims, n_periods + 1)
    n_periods = pv.shape[1] - 1
    initial = config.initial_capital

    # Terminal wealth
    terminal = pv[:, -1]
    percentile_keys = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    terminal_percentiles = {
        p: float(np.percentile(terminal, p)) for p in percentile_keys
    }

    # CAGR per path
    cagr_per_path = np.power(terminal / initial, 1.0 / n_periods) - 1.0

    # Volatility from log returns
    log_returns = np.log(pv[:, 1:] / pv[:, :-1])
    period_vol = np.std(log_returns, axis=1)
    annualized_vol = float(np.mean(period_vol))

    # Sharpe-like metric across all observations
    all_returns = result.period_returns.flatten()
    std_returns = float(np.std(all_returns))
    sharpe = float(np.mean(all_returns) / std_returns) if std_returns > 0 else 0.0

    # Drawdown analysis
    running_max = np.maximum.accumulate(pv, axis=1)
    drawdowns = (pv - running_max) / running_max
    max_dd_per_path = np.min(drawdowns, axis=1)

    is_in_drawdown = drawdowns < -1e-10
    dd_durations = _max_consecutive_true(is_in_drawdown)

    avg_drawdown = float(np.mean(drawdowns[is_in_drawdown])) if is_in_drawdown.any() else 0.0

    # Risk metrics
    prob_loss = float(np.mean(terminal < initial))
    ruin_threshold = initial * 0.10
    prob_ruin = float(np.mean(terminal < ruin_threshold))

    terminal_pnl = terminal - initial
    var_95 = float(np.percentile(terminal_pnl, 5))
    cvar_mask = terminal_pnl <= var_95
    cvar_95 = float(np.mean(terminal_pnl[cvar_mask])) if cvar_mask.any() else var_95

    # Kelly info per setup
    kelly_info = [compute_kelly_fraction(s) for s in config.setups]

    return AnalyticsReport(
        terminal_mean=float(np.mean(terminal)),
        terminal_median=float(np.median(terminal)),
        terminal_std=float(np.std(terminal)),
        terminal_percentiles=terminal_percentiles,
        cagr_mean=float(np.mean(cagr_per_path)),
        cagr_median=float(np.median(cagr_per_path)),
        annualized_volatility=annualized_vol,
        sharpe_ratio=sharpe,
        max_drawdown_mean=float(np.mean(max_dd_per_path)),
        max_drawdown_median=float(np.median(max_dd_per_path)),
        max_drawdown_worst=float(np.min(max_dd_per_path)),
        avg_drawdown=avg_drawdown,
        max_drawdown_duration_mean=float(np.mean(dd_durations)),
        probability_of_loss=prob_loss,
        probability_of_ruin=prob_ruin,
        var_95=var_95,
        cvar_95=cvar_95,
        kelly_info=kelly_info,
    )


def _max_consecutive_true(arr: np.ndarray) -> np.ndarray:
    """For each row in a 2D boolean array, find the longest run of True values."""
    n_rows, n_cols = arr.shape
    result = np.zeros(n_rows, dtype=np.int64)

    for i in range(n_rows):
        row = arr[i]
        if not row.any():
            continue
        d = np.diff(row.astype(np.int8))
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1
        if row[0]:
            starts = np.concatenate(([0], starts))
        if row[-1]:
            ends = np.concatenate((ends, [n_cols]))
        if len(starts) > 0 and len(ends) > 0:
            lengths = ends - starts
            result[i] = lengths.max()

    return result
