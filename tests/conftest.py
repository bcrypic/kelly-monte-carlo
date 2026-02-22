"""Shared test fixtures."""

import pytest

from kelly_mc.models import Scenario, Setup, SimulationConfig


@pytest.fixture
def win_scenario():
    return Scenario(name="Win", probability=0.6, return_pct=0.20)


@pytest.fixture
def loss_scenario():
    return Scenario(name="Loss", probability=0.3, return_pct=-0.10)


@pytest.fixture
def stress_scenario():
    return Scenario(name="Stress", probability=0.1, return_pct=-0.50)


@pytest.fixture
def basic_setup(win_scenario, loss_scenario, stress_scenario):
    return Setup(
        name="Test Setup",
        probability=1.0,
        scenarios=(win_scenario, loss_scenario, stress_scenario),
    )


@pytest.fixture
def two_setup_config():
    setup_a = Setup(
        name="Bull",
        probability=0.6,
        scenarios=(
            Scenario(name="Win", probability=0.55, return_pct=0.25),
            Scenario(name="Loss", probability=0.35, return_pct=-0.12),
            Scenario(name="Stress", probability=0.10, return_pct=-0.45),
        ),
    )
    setup_b = Setup(
        name="Bear",
        probability=0.4,
        scenarios=(
            Scenario(name="Win", probability=0.30, return_pct=0.15),
            Scenario(name="Loss", probability=0.50, return_pct=-0.20),
            Scenario(name="Stress", probability=0.20, return_pct=-0.60),
        ),
    )
    return SimulationConfig(
        setups=(setup_a, setup_b),
        num_simulations=1000,
        num_periods=50,
        initial_capital=100.0,
        default_kelly_fraction=0.25,
        seed=42,
    )


@pytest.fixture
def mixed_kelly_config():
    """Config where one setup overrides the default Kelly fraction."""
    setup_a = Setup(
        name="Conservative",
        probability=0.5,
        scenarios=(
            Scenario(name="Win", probability=0.6, return_pct=0.15),
            Scenario(name="Loss", probability=0.4, return_pct=-0.10),
        ),
        kelly_fraction=0.5,  # override
    )
    setup_b = Setup(
        name="Default",
        probability=0.5,
        scenarios=(
            Scenario(name="Win", probability=0.5, return_pct=0.20),
            Scenario(name="Loss", probability=0.5, return_pct=-0.15),
        ),
        kelly_fraction=None,  # use default
    )
    return SimulationConfig(
        setups=(setup_a, setup_b),
        num_simulations=500,
        num_periods=30,
        initial_capital=100.0,
        default_kelly_fraction=0.25,
        seed=123,
    )
