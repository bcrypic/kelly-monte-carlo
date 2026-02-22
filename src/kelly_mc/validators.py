"""Input validation for simulation configurations."""

from __future__ import annotations

import numpy as np

from .models import Scenario, Setup, SimulationConfig


def validate_scenario(scenario: Scenario) -> list[str]:
    """Return a list of validation error messages (empty if valid)."""
    errors = []
    if not 0.0 <= scenario.probability <= 1.0:
        errors.append(f"Probability {scenario.probability} not in [0, 1]")
    if scenario.return_pct <= -1.0:
        errors.append(f"Return {scenario.return_pct} would cause total loss (must be > -100%)")
    if not scenario.name.strip():
        errors.append("Scenario name cannot be empty")
    return errors


def validate_setup(setup: Setup) -> list[str]:
    """Validate a setup and all its scenarios."""
    errors = []
    if not setup.name.strip():
        errors.append("Setup name cannot be empty")
    if len(setup.scenarios) < 2:
        errors.append("Setup must have at least 2 scenarios")

    total_prob = sum(s.probability for s in setup.scenarios)
    if not np.isclose(total_prob, 1.0, atol=1e-4):
        errors.append(f"Scenario probabilities sum to {total_prob:.4f}, must equal 1.0")

    for sc in setup.scenarios:
        sc_errors = validate_scenario(sc)
        for e in sc_errors:
            errors.append(f"[{sc.name}] {e}")

    if setup.kelly_fraction is not None and setup.kelly_fraction <= 0.0:
        errors.append(f"Per-setup kelly_fraction must be positive, got {setup.kelly_fraction}")

    return errors


def validate_config(config: SimulationConfig) -> list[str]:
    """Validate the entire simulation configuration."""
    errors = []

    if config.num_simulations < 1:
        errors.append("Number of simulations must be >= 1")
    if config.num_simulations > 1_000_000:
        errors.append("Number of simulations exceeds 1,000,000 limit")
    if config.num_periods < 1:
        errors.append("Number of periods must be >= 1")
    if config.initial_capital <= 0:
        errors.append("Initial capital must be positive")
    if config.default_kelly_fraction <= 0.0:
        errors.append(
            f"default_kelly_fraction must be positive, got {config.default_kelly_fraction}"
        )
    if len(config.setups) == 0:
        errors.append("At least one setup is required")

    total_setup_prob = sum(s.probability for s in config.setups)
    if not np.isclose(total_setup_prob, 1.0, atol=1e-4):
        errors.append(f"Setup probabilities sum to {total_setup_prob:.4f}, must equal 1.0")

    for setup in config.setups:
        setup_errors = validate_setup(setup)
        for e in setup_errors:
            errors.append(f"[{setup.name}] {e}")

    return errors
