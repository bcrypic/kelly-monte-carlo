"""Vectorized Monte Carlo simulation engine."""

from __future__ import annotations

import numpy as np

from .models import SimulationConfig, SimulationResult


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Execute the full Monte Carlo simulation.

    Sampling is two-tiered:
    1. For each (sim, period) cell, sample which setup is active
    2. For each setup group, sample which scenario occurs

    Kelly fraction is resolved per-cell: if the drawn setup has a custom
    kelly_fraction, use it; otherwise use config.default_kelly_fraction.
    """
    rng = np.random.default_rng(config.seed)
    n_sims = config.num_simulations
    n_periods = config.num_periods

    # Build lookup tables
    setup_probs = np.array([s.probability for s in config.setups])
    scenario_probs = []
    scenario_returns = []
    kelly_fractions = []
    for setup in config.setups:
        scenario_probs.append(np.array([sc.probability for sc in setup.scenarios]))
        scenario_returns.append(np.array([sc.return_pct for sc in setup.scenarios]))
        kelly_fractions.append(config.get_effective_kelly(setup))

    kelly_fractions_arr = np.array(kelly_fractions)

    # Step 1: Sample setup indices for all (sim, period) cells
    setup_indices = _sample_categorical(rng, setup_probs, size=(n_sims, n_periods))

    # Step 2: Sample scenario indices and resolve returns per cell
    scenario_indices = np.empty((n_sims, n_periods), dtype=np.int64)
    period_returns = np.empty((n_sims, n_periods), dtype=np.float64)

    for setup_idx in range(len(config.setups)):
        mask = setup_indices == setup_idx
        count = mask.sum()
        if count == 0:
            continue
        sc_idx = _sample_categorical(rng, scenario_probs[setup_idx], size=(count,))
        scenario_indices[mask] = sc_idx
        period_returns[mask] = scenario_returns[setup_idx][sc_idx]

    # Step 3: Resolve effective Kelly fraction per cell
    effective_kelly = kelly_fractions_arr[setup_indices]

    # Step 4: Compute portfolio value paths
    # V_{t+1} = V_t * (1 + f * r_t)
    growth_factors = 1.0 + effective_kelly * period_returns
    cum_growth = np.cumprod(growth_factors, axis=1)

    portfolio_values = np.empty((n_sims, n_periods + 1), dtype=np.float64)
    portfolio_values[:, 0] = config.initial_capital
    portfolio_values[:, 1:] = config.initial_capital * cum_growth

    return SimulationResult(
        portfolio_values=portfolio_values,
        period_returns=period_returns,
        setup_indices=setup_indices,
        scenario_indices=scenario_indices,
        config=config,
    )


def _sample_categorical(
    rng: np.random.Generator,
    probs: np.ndarray,
    size: tuple[int, ...],
) -> np.ndarray:
    """Sample from a categorical distribution using inverse CDF (vectorized)."""
    cumprobs = np.cumsum(probs)
    cumprobs[-1] = 1.0  # avoid floating point edge case
    u = rng.random(size)
    return np.searchsorted(cumprobs, u).astype(np.int64)
