"""Tests for Kelly Criterion module."""

import numpy as np
import pytest

from kelly_mc.kelly import (
    compute_blended_kelly,
    compute_kelly_fraction,
    expected_log_growth_at,
)
from kelly_mc.models import Scenario, Setup


class TestComputeKellyFraction:
    def test_two_outcome_matches_closed_form(self):
        """For a two-outcome bet, verify against closed-form Kelly:
        f* = p/a - q/b where p = win prob, q = loss prob,
        a = loss amount, b = win amount (net odds).
        Equivalently: f* = (p*b - q*a) / (a*b) for unit bet.
        For p=0.6, win=+100% (b=1), loss=-100% (impossible, use -50%):
        f* = p - q/b = p - q*(1/odds)
        Actually: f* = (p*(1+b) - 1)/b for odds b.
        """
        # Classic Kelly: p=0.6, win doubles (b=1), lose everything isn't valid
        # Use: p=0.6, win +20%, loss -10%
        # Closed form for 2 outcomes: maximize G(f) = p*ln(1+f*w) + q*ln(1+f*l)
        # Derivative: p*w/(1+f*w) + q*l/(1+f*l) = 0
        # For p=0.6, w=0.2, q=0.4, l=-0.1:
        # 0.6*0.2/(1+0.2f) + 0.4*(-0.1)/(1-0.1f) = 0
        # 0.12/(1+0.2f) = 0.04/(1-0.1f)
        # 0.12*(1-0.1f) = 0.04*(1+0.2f)
        # 0.12 - 0.012f = 0.04 + 0.008f
        # 0.08 = 0.02f
        # f* = 4.0
        setup = Setup(
            name="TwoOutcome",
            probability=1.0,
            scenarios=(
                Scenario(name="Win", probability=0.6, return_pct=0.20),
                Scenario(name="Loss", probability=0.4, return_pct=-0.10),
            ),
        )
        info = compute_kelly_fraction(setup)
        np.testing.assert_almost_equal(info.optimal_fraction, 4.0, decimal=3)

    def test_negative_expected_value(self):
        """Setup with negative EV should return Kelly fraction of 0."""
        setup = Setup(
            name="BadBet",
            probability=1.0,
            scenarios=(
                Scenario(name="Win", probability=0.3, return_pct=0.10),
                Scenario(name="Loss", probability=0.7, return_pct=-0.10),
            ),
        )
        info = compute_kelly_fraction(setup)
        assert info.optimal_fraction == 0.0
        assert info.expected_return < 0

    def test_fair_bet(self):
        """Setup with zero expected value should return Kelly fraction near 0."""
        setup = Setup(
            name="Fair",
            probability=1.0,
            scenarios=(
                Scenario(name="Win", probability=0.5, return_pct=0.10),
                Scenario(name="Loss", probability=0.5, return_pct=-0.10),
            ),
        )
        info = compute_kelly_fraction(setup)
        assert info.optimal_fraction == 0.0

    def test_three_scenario_setup(self, basic_setup):
        """Kelly fraction should be positive for a setup with positive edge."""
        info = compute_kelly_fraction(basic_setup)
        assert info.optimal_fraction > 0
        assert info.expected_log_growth > 0
        assert info.setup_name == "Test Setup"

    def test_expected_return_computed(self, basic_setup):
        """Verify expected return matches manual calculation."""
        # Win: 0.6 * 0.20 = 0.12
        # Loss: 0.3 * (-0.10) = -0.03
        # Stress: 0.1 * (-0.50) = -0.05
        # Total: 0.04
        info = compute_kelly_fraction(basic_setup)
        np.testing.assert_almost_equal(info.expected_return, 0.04)

    def test_variance_computed(self, basic_setup):
        """Verify variance matches manual calculation."""
        # E[r] = 0.04
        # Var = 0.6*(0.20-0.04)^2 + 0.3*(-0.10-0.04)^2 + 0.1*(-0.50-0.04)^2
        #     = 0.6*0.0256 + 0.3*0.0196 + 0.1*0.2916
        #     = 0.01536 + 0.00588 + 0.02916 = 0.0504
        info = compute_kelly_fraction(basic_setup)
        np.testing.assert_almost_equal(info.variance, 0.0504, decimal=4)

    def test_odds_description(self, basic_setup):
        info = compute_kelly_fraction(basic_setup)
        assert "Win" in info.odds_description
        assert "Loss" in info.odds_description
        assert "Stress" in info.odds_description

    def test_feasibility_constraint(self):
        """Kelly fraction should not exceed 1/|min_return| (would risk ruin)."""
        setup = Setup(
            name="Risky",
            probability=1.0,
            scenarios=(
                Scenario(name="Win", probability=0.8, return_pct=0.50),
                Scenario(name="Loss", probability=0.2, return_pct=-0.90),
            ),
        )
        info = compute_kelly_fraction(setup)
        # f_max = 1/0.90 ≈ 1.111
        assert info.optimal_fraction < 1.0 / 0.90


class TestComputeBlendedKelly:
    def test_single_setup(self, basic_setup):
        """Blended Kelly with one setup should equal that setup's Kelly."""
        single_info = compute_kelly_fraction(basic_setup)
        blended = compute_blended_kelly((basic_setup,))
        np.testing.assert_almost_equal(blended, single_info.optimal_fraction, decimal=6)

    def test_two_setups_weighted(self):
        setup_a = Setup(
            name="A",
            probability=0.6,
            scenarios=(
                Scenario(name="Win", probability=0.6, return_pct=0.20),
                Scenario(name="Loss", probability=0.4, return_pct=-0.10),
            ),
        )
        setup_b = Setup(
            name="B",
            probability=0.4,
            scenarios=(
                Scenario(name="Win", probability=0.4, return_pct=0.15),
                Scenario(name="Loss", probability=0.6, return_pct=-0.10),
            ),
        )
        blended = compute_blended_kelly((setup_a, setup_b))
        kelly_a = compute_kelly_fraction(setup_a).optimal_fraction
        kelly_b = compute_kelly_fraction(setup_b).optimal_fraction
        expected = 0.6 * kelly_a + 0.4 * kelly_b
        np.testing.assert_almost_equal(blended, expected, decimal=6)


class TestExpectedLogGrowth:
    def test_at_zero(self, basic_setup):
        """G(0) should be 0 (no betting = no growth)."""
        g = expected_log_growth_at(basic_setup, 0.0)
        np.testing.assert_almost_equal(g, 0.0)

    def test_at_optimum_matches_kelly_info(self, basic_setup):
        info = compute_kelly_fraction(basic_setup)
        g = expected_log_growth_at(basic_setup, info.optimal_fraction)
        np.testing.assert_almost_equal(g, info.expected_log_growth, decimal=6)

    def test_infeasible_fraction_returns_neg_inf(self):
        """Fraction that would cause 1 + f*r <= 0 should return -inf."""
        setup = Setup(
            name="Test",
            probability=1.0,
            scenarios=(
                Scenario(name="Win", probability=0.5, return_pct=0.10),
                Scenario(name="Loss", probability=0.5, return_pct=-0.50),
            ),
        )
        # f = 3.0, worst return = -0.50: 1 + 3*(-0.50) = -0.5 < 0
        g = expected_log_growth_at(setup, 3.0)
        assert g == float("-inf")
