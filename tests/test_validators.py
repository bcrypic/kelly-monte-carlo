"""Tests for input validation."""

import pytest

from kelly_mc.models import Scenario, Setup, SimulationConfig
from kelly_mc.validators import validate_config, validate_scenario, validate_setup


class TestValidateScenario:
    def test_valid(self, win_scenario):
        assert validate_scenario(win_scenario) == []

    def test_invalid_probability(self):
        s = Scenario.__new__(Scenario)
        object.__setattr__(s, "name", "Bad")
        object.__setattr__(s, "probability", 1.5)
        object.__setattr__(s, "return_pct", 0.10)
        errors = validate_scenario(s)
        assert any("Probability" in e for e in errors)

    def test_invalid_return(self):
        s = Scenario.__new__(Scenario)
        object.__setattr__(s, "name", "Bad")
        object.__setattr__(s, "probability", 0.5)
        object.__setattr__(s, "return_pct", -1.0)
        errors = validate_scenario(s)
        assert any("total loss" in e for e in errors)

    def test_empty_name(self):
        s = Scenario.__new__(Scenario)
        object.__setattr__(s, "name", "")
        object.__setattr__(s, "probability", 0.5)
        object.__setattr__(s, "return_pct", 0.10)
        errors = validate_scenario(s)
        assert any("empty" in e for e in errors)


class TestValidateSetup:
    def test_valid(self, basic_setup):
        assert validate_setup(basic_setup) == []

    def test_bad_scenario_probs(self):
        setup = Setup.__new__(Setup)
        object.__setattr__(setup, "name", "Bad")
        object.__setattr__(setup, "probability", 0.5)
        object.__setattr__(setup, "kelly_fraction", None)
        object.__setattr__(
            setup,
            "scenarios",
            (
                Scenario(name="A", probability=0.3, return_pct=0.10),
                Scenario(name="B", probability=0.3, return_pct=-0.10),
            ),
        )
        errors = validate_setup(setup)
        assert any("sum to" in e for e in errors)

    def test_too_few_scenarios(self):
        setup = Setup.__new__(Setup)
        object.__setattr__(setup, "name", "Bad")
        object.__setattr__(setup, "probability", 0.5)
        object.__setattr__(setup, "kelly_fraction", None)
        object.__setattr__(
            setup,
            "scenarios",
            (Scenario(name="Only", probability=1.0, return_pct=0.10),),
        )
        errors = validate_setup(setup)
        assert any("at least 2" in e for e in errors)

    def test_invalid_kelly_override(self):
        setup = Setup.__new__(Setup)
        object.__setattr__(setup, "name", "Bad")
        object.__setattr__(setup, "probability", 0.5)
        object.__setattr__(setup, "kelly_fraction", -0.5)
        object.__setattr__(
            setup,
            "scenarios",
            (
                Scenario(name="A", probability=0.5, return_pct=0.10),
                Scenario(name="B", probability=0.5, return_pct=-0.10),
            ),
        )
        errors = validate_setup(setup)
        assert any("kelly_fraction" in e for e in errors)


class TestValidateConfig:
    def test_valid(self, two_setup_config):
        assert validate_config(two_setup_config) == []

    def test_too_many_simulations(self, basic_setup):
        config = SimulationConfig.__new__(SimulationConfig)
        object.__setattr__(config, "setups", (basic_setup,))
        object.__setattr__(config, "num_simulations", 2_000_000)
        object.__setattr__(config, "num_periods", 100)
        object.__setattr__(config, "initial_capital", 100.0)
        object.__setattr__(config, "default_kelly_fraction", 0.25)
        object.__setattr__(config, "seed", None)
        errors = validate_config(config)
        assert any("1,000,000" in e for e in errors)

    def test_cascading_errors(self):
        """Validation should report errors from nested setups and scenarios."""
        setup = Setup.__new__(Setup)
        object.__setattr__(setup, "name", "")
        object.__setattr__(setup, "probability", 0.5)
        object.__setattr__(setup, "kelly_fraction", None)
        object.__setattr__(
            setup,
            "scenarios",
            (
                Scenario(name="A", probability=0.5, return_pct=0.10),
                Scenario(name="B", probability=0.5, return_pct=-0.10),
            ),
        )
        config = SimulationConfig.__new__(SimulationConfig)
        object.__setattr__(config, "setups", (setup,))
        object.__setattr__(config, "num_simulations", 100)
        object.__setattr__(config, "num_periods", 10)
        object.__setattr__(config, "initial_capital", 100.0)
        object.__setattr__(config, "default_kelly_fraction", 0.25)
        object.__setattr__(config, "seed", None)
        errors = validate_config(config)
        # Should catch setup prob not summing to 1 and empty setup name
        assert len(errors) >= 2
