"""Data models for Kelly Criterion Monte Carlo simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Scenario:
    """A single outcome within a setup (e.g., win, loss, stress)."""

    name: str
    probability: float
    return_pct: float  # e.g., 0.20 for +20%, -0.50 for -50%

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"Probability must be in [0, 1], got {self.probability}")
        if self.return_pct <= -1.0:
            raise ValueError(f"Return must be > -100%, got {self.return_pct}")
        if not self.name.strip():
            raise ValueError("Scenario name cannot be empty")


@dataclass(frozen=True)
class Setup:
    """A Kelly Criterion setup: a named collection of scenarios with a
    probability of being the active regime."""

    name: str
    probability: float  # probability this setup is drawn
    scenarios: tuple[Scenario, ...]
    kelly_fraction: float | None = None  # per-setup override; None = use run default

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Setup name cannot be empty")
        if not 0.0 < self.probability <= 1.0:
            raise ValueError(f"Setup probability must be in (0, 1], got {self.probability}")
        if len(self.scenarios) < 2:
            raise ValueError("Setup must have at least 2 scenarios")
        total = sum(s.probability for s in self.scenarios)
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"Scenario probabilities must sum to 1.0, got {total} "
                f"for setup '{self.name}'"
            )
        if self.kelly_fraction is not None and self.kelly_fraction <= 0.0:
            raise ValueError(
                f"Per-setup kelly_fraction must be positive, got {self.kelly_fraction}"
            )


@dataclass(frozen=True)
class SimulationConfig:
    """All parameters needed to run a simulation."""

    setups: tuple[Setup, ...]
    num_simulations: int = 10_000
    num_periods: int = 100
    initial_capital: float = 100.0
    default_kelly_fraction: float = 0.25
    seed: int | None = None

    def __post_init__(self) -> None:
        if len(self.setups) == 0:
            raise ValueError("At least one setup is required")
        total = sum(s.probability for s in self.setups)
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Setup probabilities must sum to 1.0, got {total}")
        if self.num_simulations < 1:
            raise ValueError("Number of simulations must be >= 1")
        if self.num_periods < 1:
            raise ValueError("Number of periods must be >= 1")
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.default_kelly_fraction <= 0.0:
            raise ValueError(
                f"default_kelly_fraction must be positive, got {self.default_kelly_fraction}"
            )

    def get_effective_kelly(self, setup: Setup) -> float:
        """Resolve the effective Kelly fraction for a given setup."""
        if setup.kelly_fraction is not None:
            return setup.kelly_fraction
        return self.default_kelly_fraction


@dataclass
class SimulationResult:
    """Container for all simulation outputs."""

    portfolio_values: np.ndarray  # (num_simulations, num_periods + 1)
    period_returns: np.ndarray  # (num_simulations, num_periods)
    setup_indices: np.ndarray  # (num_simulations, num_periods)
    scenario_indices: np.ndarray  # (num_simulations, num_periods)
    config: SimulationConfig


@dataclass(frozen=True)
class KellyInfo:
    """Kelly Criterion analytics for a single setup."""

    setup_name: str
    optimal_fraction: float
    expected_log_growth: float
    expected_return: float
    variance: float
    edge: float
    odds_description: str


@dataclass
class AnalyticsReport:
    """Aggregated analytics computed from a SimulationResult."""

    # Terminal wealth
    terminal_mean: float
    terminal_median: float
    terminal_std: float
    terminal_percentiles: dict[int, float]

    # Returns
    cagr_mean: float
    cagr_median: float
    annualized_volatility: float
    sharpe_ratio: float

    # Drawdown
    max_drawdown_mean: float
    max_drawdown_median: float
    max_drawdown_worst: float
    avg_drawdown: float
    max_drawdown_duration_mean: float

    # Risk
    probability_of_loss: float
    probability_of_ruin: float
    var_95: float
    cvar_95: float

    # Per-setup Kelly info
    kelly_info: list[KellyInfo]
