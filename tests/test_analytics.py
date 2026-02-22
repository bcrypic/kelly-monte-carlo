"""Tests for analytics module."""

import numpy as np
import pytest

from kelly_mc.analytics import _max_consecutive_true, compute_analytics
from kelly_mc.engine import run_simulation
from kelly_mc.models import Scenario, Setup, SimulationConfig


class TestComputeAnalytics:
    def test_report_populated(self, two_setup_config):
        result = run_simulation(two_setup_config)
        report = compute_analytics(result)

        assert report.terminal_mean > 0
        assert report.terminal_median > 0
        assert report.terminal_std >= 0
        assert len(report.terminal_percentiles) == 9
        assert len(report.kelly_info) == 2

    def test_percentile_ordering(self, two_setup_config):
        result = run_simulation(two_setup_config)
        report = compute_analytics(result)

        pcts = report.terminal_percentiles
        assert pcts[1] <= pcts[5] <= pcts[25] <= pcts[50] <= pcts[75] <= pcts[95] <= pcts[99]

    def test_flat_portfolio_zero_drawdown(self):
        """All zero returns -> no drawdown, no loss, CAGR = 0."""
        setup = Setup(
            name="Flat",
            probability=1.0,
            scenarios=(
                Scenario(name="A", probability=0.5, return_pct=0.0),
                Scenario(name="B", probability=0.5, return_pct=0.0),
            ),
        )
        config = SimulationConfig(
            setups=(setup,), num_simulations=50, num_periods=20, seed=42
        )
        result = run_simulation(config)
        report = compute_analytics(result)

        assert report.probability_of_loss == 0.0
        assert report.probability_of_ruin == 0.0
        np.testing.assert_almost_equal(report.cagr_mean, 0.0)
        np.testing.assert_almost_equal(report.max_drawdown_mean, 0.0)

    def test_always_positive_returns_no_loss(self):
        """All positive returns -> P(loss) should be 0."""
        setup = Setup(
            name="Bull",
            probability=1.0,
            scenarios=(
                Scenario(name="Small", probability=0.5, return_pct=0.05),
                Scenario(name="Big", probability=0.5, return_pct=0.10),
            ),
        )
        config = SimulationConfig(
            setups=(setup,), num_simulations=100, num_periods=20, seed=42
        )
        result = run_simulation(config)
        report = compute_analytics(result)

        assert report.probability_of_loss == 0.0
        assert report.cagr_mean > 0
        assert report.terminal_mean > config.initial_capital

    def test_drawdown_present_with_mixed_returns(self, two_setup_config):
        result = run_simulation(two_setup_config)
        report = compute_analytics(result)

        # With mixed win/loss scenarios, drawdowns should exist
        assert report.max_drawdown_worst < 0
        assert report.max_drawdown_mean < 0
        assert report.avg_drawdown < 0

    def test_var_cvar_relationship(self, two_setup_config):
        """CVaR should be <= VaR (both are losses, CVaR is worse)."""
        result = run_simulation(two_setup_config)
        report = compute_analytics(result)
        assert report.cvar_95 <= report.var_95

    def test_kelly_info_per_setup(self, two_setup_config):
        result = run_simulation(two_setup_config)
        report = compute_analytics(result)

        assert len(report.kelly_info) == len(two_setup_config.setups)
        for info, setup in zip(report.kelly_info, two_setup_config.setups):
            assert info.setup_name == setup.name

    def test_median_between_percentiles(self, two_setup_config):
        result = run_simulation(two_setup_config)
        report = compute_analytics(result)

        assert report.terminal_percentiles[25] <= report.terminal_median
        assert report.terminal_median <= report.terminal_percentiles[75]


class TestMaxConsecutiveTrue:
    def test_all_false(self):
        arr = np.zeros((3, 5), dtype=bool)
        result = _max_consecutive_true(arr)
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_all_true(self):
        arr = np.ones((2, 4), dtype=bool)
        result = _max_consecutive_true(arr)
        np.testing.assert_array_equal(result, [4, 4])

    def test_mixed(self):
        arr = np.array([
            [True, True, False, True, True, True],
            [False, True, False, False, False, False],
        ])
        result = _max_consecutive_true(arr)
        np.testing.assert_array_equal(result, [3, 1])

    def test_starts_with_true(self):
        arr = np.array([[True, True, True, False, True]])
        result = _max_consecutive_true(arr)
        np.testing.assert_array_equal(result, [3])

    def test_ends_with_true(self):
        arr = np.array([[False, True, True, True, True]])
        result = _max_consecutive_true(arr)
        np.testing.assert_array_equal(result, [4])

    def test_single_element_true(self):
        arr = np.array([[True]])
        result = _max_consecutive_true(arr)
        np.testing.assert_array_equal(result, [1])

    def test_single_element_false(self):
        arr = np.array([[False]])
        result = _max_consecutive_true(arr)
        np.testing.assert_array_equal(result, [0])
