"""Kelly Criterion optimal fraction computation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .models import KellyInfo, Setup


def compute_kelly_fraction(setup: Setup) -> KellyInfo:
    """Compute the Kelly optimal fraction for a single setup.

    For discrete scenarios with probabilities p_i and returns r_i,
    the Kelly optimal fraction f* maximizes:
        G(f) = sum_i [ p_i * ln(1 + f * r_i) ]

    Solved numerically since there is no closed-form for 3+ scenarios.
    """
    probs = np.array([s.probability for s in setup.scenarios])
    returns = np.array([s.return_pct for s in setup.scenarios])

    expected_return = float(np.dot(probs, returns))
    variance = float(np.dot(probs, (returns - expected_return) ** 2))

    # If expected return is non-positive, Kelly says don't bet
    if expected_return <= 0:
        return KellyInfo(
            setup_name=setup.name,
            optimal_fraction=0.0,
            expected_log_growth=0.0,
            expected_return=expected_return,
            variance=variance,
            edge=expected_return,
            odds_description=_build_odds_description(setup),
        )

    # Feasibility constraint: 1 + f * r_i > 0 for all i
    # For negative returns: f < -1/r_i
    min_return = returns.min()
    if min_return < 0:
        f_max = -1.0 / min_return - 1e-9
    else:
        f_max = 10.0

    def neg_expected_log_growth(f: float) -> float:
        log_terms = np.log(1.0 + f * returns)
        return -float(np.dot(probs, log_terms))

    result = minimize_scalar(
        neg_expected_log_growth,
        bounds=(0.0, f_max),
        method="bounded",
    )

    optimal_f = result.x
    expected_log_growth = -result.fun

    # If optimum is at or near zero, clamp to zero
    if optimal_f < 1e-8:
        optimal_f = 0.0
        expected_log_growth = 0.0

    return KellyInfo(
        setup_name=setup.name,
        optimal_fraction=optimal_f,
        expected_log_growth=expected_log_growth,
        expected_return=expected_return,
        variance=variance,
        edge=expected_return,
        odds_description=_build_odds_description(setup),
    )


def compute_blended_kelly(setups: tuple[Setup, ...]) -> float:
    """Compute a probability-weighted Kelly fraction across all setups."""
    weighted_sum = 0.0
    for setup in setups:
        info = compute_kelly_fraction(setup)
        weighted_sum += setup.probability * info.optimal_fraction
    return weighted_sum


def expected_log_growth_at(setup: Setup, fraction: float) -> float:
    """Compute G(f) = sum(p_i * ln(1 + f * r_i)) for a given fraction."""
    probs = np.array([s.probability for s in setup.scenarios])
    returns = np.array([s.return_pct for s in setup.scenarios])

    terms = 1.0 + fraction * returns
    if np.any(terms <= 0):
        return float("-inf")
    return float(np.dot(probs, np.log(terms)))


def _build_odds_description(setup: Setup) -> str:
    lines = []
    for sc in setup.scenarios:
        sign = "+" if sc.return_pct >= 0 else ""
        lines.append(f"{sc.name}: {sc.probability:.0%} chance of {sign}{sc.return_pct:.1%}")
    return "; ".join(lines)
