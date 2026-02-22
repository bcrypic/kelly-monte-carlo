"""Tests for data models."""

import pytest

from kelly_mc.models import Scenario, Setup, SimulationConfig


class TestScenario:
    def test_valid_scenario(self):
        s = Scenario(name="Win", probability=0.6, return_pct=0.20)
        assert s.name == "Win"
        assert s.probability == 0.6
        assert s.return_pct == 0.20

    def test_probability_too_high(self):
        with pytest.raises(ValueError, match="Probability must be in"):
            Scenario(name="Win", probability=1.5, return_pct=0.10)

    def test_probability_negative(self):
        with pytest.raises(ValueError, match="Probability must be in"):
            Scenario(name="Win", probability=-0.1, return_pct=0.10)

    def test_return_total_loss(self):
        with pytest.raises(ValueError, match="Return must be > -100%"):
            Scenario(name="Wipeout", probability=0.5, return_pct=-1.0)

    def test_return_worse_than_total_loss(self):
        with pytest.raises(ValueError, match="Return must be > -100%"):
            Scenario(name="Wipeout", probability=0.5, return_pct=-1.5)

    def test_empty_name(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            Scenario(name="", probability=0.5, return_pct=0.10)

    def test_whitespace_name(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            Scenario(name="   ", probability=0.5, return_pct=0.10)

    def test_boundary_probability_zero(self):
        s = Scenario(name="Impossible", probability=0.0, return_pct=0.10)
        assert s.probability == 0.0

    def test_boundary_probability_one(self):
        s = Scenario(name="Certain", probability=1.0, return_pct=0.10)
        assert s.probability == 1.0

    def test_near_total_loss(self):
        s = Scenario(name="Almost Wiped", probability=0.1, return_pct=-0.99)
        assert s.return_pct == -0.99

    def test_frozen(self):
        s = Scenario(name="Win", probability=0.5, return_pct=0.10)
        with pytest.raises(AttributeError):
            s.probability = 0.9  # type: ignore[misc]


class TestSetup:
    def test_valid_setup(self, win_scenario, loss_scenario, stress_scenario):
        setup = Setup(
            name="Bull",
            probability=0.6,
            scenarios=(win_scenario, loss_scenario, stress_scenario),
        )
        assert setup.name == "Bull"
        assert len(setup.scenarios) == 3
        assert setup.kelly_fraction is None

    def test_custom_kelly_fraction(self, win_scenario, loss_scenario, stress_scenario):
        setup = Setup(
            name="Custom",
            probability=0.5,
            scenarios=(win_scenario, loss_scenario, stress_scenario),
            kelly_fraction=0.5,
        )
        assert setup.kelly_fraction == 0.5

    def test_invalid_kelly_fraction(self, win_scenario, loss_scenario, stress_scenario):
        with pytest.raises(ValueError, match="kelly_fraction must be positive"):
            Setup(
                name="Bad",
                probability=0.5,
                scenarios=(win_scenario, loss_scenario, stress_scenario),
                kelly_fraction=-0.1,
            )

    def test_zero_kelly_fraction(self, win_scenario, loss_scenario, stress_scenario):
        with pytest.raises(ValueError, match="kelly_fraction must be positive"):
            Setup(
                name="Bad",
                probability=0.5,
                scenarios=(win_scenario, loss_scenario, stress_scenario),
                kelly_fraction=0.0,
            )

    def test_probabilities_not_summing_to_one(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            Setup(
                name="Bad",
                probability=0.5,
                scenarios=(
                    Scenario(name="Win", probability=0.5, return_pct=0.10),
                    Scenario(name="Loss", probability=0.3, return_pct=-0.10),
                ),
            )

    def test_empty_name(self, win_scenario, loss_scenario, stress_scenario):
        with pytest.raises(ValueError, match="name cannot be empty"):
            Setup(
                name="",
                probability=0.5,
                scenarios=(win_scenario, loss_scenario, stress_scenario),
            )

    def test_zero_probability(self, win_scenario, loss_scenario, stress_scenario):
        with pytest.raises(ValueError, match="must be in"):
            Setup(
                name="Zero",
                probability=0.0,
                scenarios=(win_scenario, loss_scenario, stress_scenario),
            )

    def test_too_few_scenarios(self):
        with pytest.raises(ValueError, match="at least 2 scenarios"):
            Setup(
                name="One",
                probability=0.5,
                scenarios=(Scenario(name="Only", probability=1.0, return_pct=0.10),),
            )

    def test_frozen(self, basic_setup):
        with pytest.raises(AttributeError):
            basic_setup.name = "Changed"  # type: ignore[misc]


class TestSimulationConfig:
    def test_valid_config(self, two_setup_config):
        assert two_setup_config.num_simulations == 1000
        assert two_setup_config.initial_capital == 100.0
        assert two_setup_config.default_kelly_fraction == 0.25

    def test_setup_probs_not_summing_to_one(self, win_scenario, loss_scenario, stress_scenario):
        setup = Setup(
            name="Only",
            probability=0.5,
            scenarios=(win_scenario, loss_scenario, stress_scenario),
        )
        with pytest.raises(ValueError, match="must sum to 1.0"):
            SimulationConfig(setups=(setup,))

    def test_no_setups(self):
        with pytest.raises(ValueError, match="At least one setup"):
            SimulationConfig(setups=())

    def test_zero_simulations(self, basic_setup):
        with pytest.raises(ValueError, match="must be >= 1"):
            SimulationConfig(setups=(basic_setup,), num_simulations=0)

    def test_zero_periods(self, basic_setup):
        with pytest.raises(ValueError, match="must be >= 1"):
            SimulationConfig(setups=(basic_setup,), num_periods=0)

    def test_negative_capital(self, basic_setup):
        with pytest.raises(ValueError, match="must be positive"):
            SimulationConfig(setups=(basic_setup,), initial_capital=-100)

    def test_zero_default_kelly(self, basic_setup):
        with pytest.raises(ValueError, match="default_kelly_fraction must be positive"):
            SimulationConfig(setups=(basic_setup,), default_kelly_fraction=0.0)

    def test_get_effective_kelly_default(self, two_setup_config):
        setup = two_setup_config.setups[0]
        assert setup.kelly_fraction is None
        assert two_setup_config.get_effective_kelly(setup) == 0.25

    def test_get_effective_kelly_override(self, mixed_kelly_config):
        conservative = mixed_kelly_config.setups[0]
        default = mixed_kelly_config.setups[1]
        assert mixed_kelly_config.get_effective_kelly(conservative) == 0.5
        assert mixed_kelly_config.get_effective_kelly(default) == 0.25

    def test_defaults(self, basic_setup):
        config = SimulationConfig(setups=(basic_setup,))
        assert config.num_simulations == 10_000
        assert config.num_periods == 100
        assert config.initial_capital == 100.0
        assert config.default_kelly_fraction == 0.25
        assert config.seed is None
