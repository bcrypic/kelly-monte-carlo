"""Tests for the Monte Carlo simulation engine."""

import numpy as np
import pytest

from kelly_mc.engine import run_simulation
from kelly_mc.models import Scenario, Setup, SimulationConfig


class TestRunSimulation:
    def test_output_shapes(self, two_setup_config):
        result = run_simulation(two_setup_config)
        n_sims = two_setup_config.num_simulations
        n_periods = two_setup_config.num_periods

        assert result.portfolio_values.shape == (n_sims, n_periods + 1)
        assert result.period_returns.shape == (n_sims, n_periods)
        assert result.setup_indices.shape == (n_sims, n_periods)
        assert result.scenario_indices.shape == (n_sims, n_periods)

    def test_initial_capital(self, two_setup_config):
        result = run_simulation(two_setup_config)
        np.testing.assert_array_equal(
            result.portfolio_values[:, 0], two_setup_config.initial_capital
        )

    def test_deterministic_with_seed(self, two_setup_config):
        result1 = run_simulation(two_setup_config)
        result2 = run_simulation(two_setup_config)
        np.testing.assert_array_equal(result1.portfolio_values, result2.portfolio_values)
        np.testing.assert_array_equal(result1.period_returns, result2.period_returns)
        np.testing.assert_array_equal(result1.setup_indices, result2.setup_indices)

    def test_different_seeds_differ(self):
        setup = Setup(
            name="Test",
            probability=1.0,
            scenarios=(
                Scenario(name="Win", probability=0.5, return_pct=0.10),
                Scenario(name="Loss", probability=0.5, return_pct=-0.10),
            ),
        )
        config1 = SimulationConfig(setups=(setup,), num_simulations=100, num_periods=10, seed=1)
        config2 = SimulationConfig(setups=(setup,), num_simulations=100, num_periods=10, seed=2)
        r1 = run_simulation(config1)
        r2 = run_simulation(config2)
        assert not np.array_equal(r1.portfolio_values, r2.portfolio_values)

    def test_flat_portfolio_with_zero_returns(self):
        """All scenarios have 0% return -> portfolio stays flat."""
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
        np.testing.assert_array_almost_equal(
            result.portfolio_values, config.initial_capital
        )

    def test_positive_only_returns(self):
        """All scenarios positive -> portfolio should always grow."""
        setup = Setup(
            name="Bull",
            probability=1.0,
            scenarios=(
                Scenario(name="Small Win", probability=0.5, return_pct=0.05),
                Scenario(name="Big Win", probability=0.5, return_pct=0.10),
            ),
        )
        config = SimulationConfig(
            setups=(setup,), num_simulations=100, num_periods=20, seed=42
        )
        result = run_simulation(config)
        # Every terminal value should exceed initial capital
        assert np.all(result.portfolio_values[:, -1] > config.initial_capital)

    def test_setup_indices_valid(self, two_setup_config):
        result = run_simulation(two_setup_config)
        n_setups = len(two_setup_config.setups)
        assert np.all(result.setup_indices >= 0)
        assert np.all(result.setup_indices < n_setups)

    def test_scenario_indices_valid(self, two_setup_config):
        result = run_simulation(two_setup_config)
        for setup_idx, setup in enumerate(two_setup_config.setups):
            mask = result.setup_indices == setup_idx
            if mask.any():
                assert np.all(result.scenario_indices[mask] >= 0)
                assert np.all(result.scenario_indices[mask] < len(setup.scenarios))

    def test_setup_frequency_matches_probability(self):
        """With many samples, setup frequency should roughly match probability."""
        setup_a = Setup(
            name="A",
            probability=0.7,
            scenarios=(
                Scenario(name="Win", probability=0.5, return_pct=0.10),
                Scenario(name="Loss", probability=0.5, return_pct=-0.10),
            ),
        )
        setup_b = Setup(
            name="B",
            probability=0.3,
            scenarios=(
                Scenario(name="Win", probability=0.5, return_pct=0.10),
                Scenario(name="Loss", probability=0.5, return_pct=-0.10),
            ),
        )
        config = SimulationConfig(
            setups=(setup_a, setup_b),
            num_simulations=10000,
            num_periods=100,
            seed=42,
        )
        result = run_simulation(config)
        freq_a = np.mean(result.setup_indices == 0)
        assert abs(freq_a - 0.7) < 0.02  # within 2% of expected

    def test_kelly_fraction_applied(self):
        """Verify that kelly fraction affects growth correctly."""
        setup = Setup(
            name="Test",
            probability=1.0,
            scenarios=(
                Scenario(name="Certain", probability=1.0, return_pct=0.10),
                # Need 2 scenarios minimum; give second 0 probability
                Scenario(name="Never", probability=0.0, return_pct=-0.50),
            ),
        )
        # With kelly_fraction = 0.5 and return = 10%:
        # effective return = 0.5 * 0.10 = 0.05
        # After 1 period: 100 * 1.05 = 105
        config = SimulationConfig(
            setups=(setup,),
            num_simulations=10,
            num_periods=1,
            default_kelly_fraction=0.5,
            seed=42,
        )
        result = run_simulation(config)
        np.testing.assert_array_almost_equal(
            result.portfolio_values[:, 1], 105.0
        )

    def test_per_setup_kelly_override(self, mixed_kelly_config):
        """Verify that per-setup kelly override is respected."""
        result = run_simulation(mixed_kelly_config)
        config = mixed_kelly_config

        # Check that cells assigned to setup 0 (kelly=0.5) use different
        # effective returns than cells assigned to setup 1 (kelly=0.25)
        mask_0 = result.setup_indices == 0
        mask_1 = result.setup_indices == 1

        assert mask_0.any() and mask_1.any()

        # For setup 0 (kelly=0.5), effective return = 0.5 * raw_return
        # For setup 1 (kelly=0.25), effective return = 0.25 * raw_return
        # We can verify through the portfolio value evolution
        for sim in range(min(5, config.num_simulations)):
            for t in range(config.num_periods):
                raw_return = result.period_returns[sim, t]
                setup_idx = result.setup_indices[sim, t]
                effective_kelly = config.get_effective_kelly(config.setups[setup_idx])
                expected_growth = 1.0 + effective_kelly * raw_return
                actual_growth = (
                    result.portfolio_values[sim, t + 1] / result.portfolio_values[sim, t]
                )
                np.testing.assert_almost_equal(actual_growth, expected_growth, decimal=10)

    def test_single_simulation(self):
        """Engine works with num_simulations=1."""
        setup = Setup(
            name="Solo",
            probability=1.0,
            scenarios=(
                Scenario(name="Win", probability=0.5, return_pct=0.10),
                Scenario(name="Loss", probability=0.5, return_pct=-0.10),
            ),
        )
        config = SimulationConfig(
            setups=(setup,), num_simulations=1, num_periods=5, seed=42
        )
        result = run_simulation(config)
        assert result.portfolio_values.shape == (1, 6)

    def test_portfolio_values_positive(self, two_setup_config):
        """Portfolio values should remain positive (returns > -100%)."""
        result = run_simulation(two_setup_config)
        assert np.all(result.portfolio_values > 0)

    def test_config_preserved_in_result(self, two_setup_config):
        result = run_simulation(two_setup_config)
        assert result.config is two_setup_config
