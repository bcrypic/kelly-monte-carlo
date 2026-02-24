# Kelly Criterion Monte Carlo Simulator

An interactive Monte Carlo simulator that models portfolio growth under [Kelly Criterion](https://en.wikipedia.org/wiki/Kelly_criterion) position sizing. Define market regimes with probabilistic outcomes, run thousands of simulated portfolio paths, and analyze the resulting risk/return distribution.

## What This Does

The Kelly Criterion tells you the mathematically optimal fraction of your capital to wager on a bet (or invest in a position) to maximize long-term growth. For a simple coin flip with favorable odds, there's a closed-form solution. For real-world scenarios with multiple possible outcomes across different market regimes, it gets more complex.

This simulator lets you:

1. **Define market setups** (regimes) — each with a probability of occurring and its own set of outcome scenarios (e.g., "Bull Market" with 60% chance, containing Win/Loss/Crash scenarios)
2. **Compute the Kelly optimal fraction** for each setup numerically, since there's no closed-form for 3+ outcomes
3. **Run Monte Carlo simulations** — thousands of portfolio paths sampled from your defined probability distributions
4. **Analyze the results** — terminal wealth distribution, drawdowns, ruin probability, Sharpe ratio, VaR/CVaR, and more

## How It Works

### Two-Tier Sampling

Each simulated period goes through two random draws:

1. **Setup selection** — which market regime is active this period? (sampled from setup probabilities)
2. **Scenario selection** — within that regime, what outcome occurs? (sampled from scenario probabilities)

The portfolio then updates: `V(t+1) = V(t) * (1 + f * r)` where `f` is the Kelly fraction and `r` is the scenario return.

### Kelly Fraction Computation

For a setup with scenarios having probabilities `p_i` and returns `r_i`, the optimal Kelly fraction `f*` maximizes the expected log growth rate:

```
G(f) = Σ [ p_i * ln(1 + f * r_i) ]
```

This is solved numerically via `scipy.optimize.minimize_scalar` with a feasibility constraint ensuring no scenario can cause total ruin (`1 + f * r_i > 0` for all `i`).

You can use the computed optimal fraction, set a custom fraction per setup, or use a single default fraction globally (fractional Kelly is common practice to reduce variance).

### Analytics

After simulation, the engine computes:

- **Terminal wealth**: mean, median, standard deviation, percentiles (P1 through P99)
- **Returns**: CAGR (mean and median), annualized volatility, Sharpe ratio
- **Drawdowns**: max drawdown (mean/median/worst), average drawdown, max consecutive drawdown duration
- **Risk**: probability of loss, probability of ruin (<10% of initial capital), Value at Risk (95%), Conditional VaR (95%)
- **Per-setup Kelly analysis**: optimal fraction, expected return, variance, log growth rate

## Project Structure

```
kelly-monte-carlo/
├── src/
│   ├── kelly_mc/              # Core simulation library
│   │   ├── models.py          # Frozen dataclasses (Scenario, Setup, SimulationConfig, etc.)
│   │   ├── validators.py      # Input validation (returns error lists, no exceptions)
│   │   ├── engine.py          # Vectorized Monte Carlo engine (numpy)
│   │   ├── kelly.py           # Kelly Criterion optimization (scipy)
│   │   └── analytics.py       # Post-simulation statistics
│   └── app/                   # Streamlit web interface
│       ├── main.py            # Entry point
│       ├── state.py           # Session state and default setups
│       ├── components/        # UI components (forms, charts, metrics)
│       └── pages/             # Streamlit multipage (Setup Config, Simulation)
├── tests/                     # pytest suite (83 tests)
├── pyproject.toml             # Dependencies and tool config
├── Dockerfile                 # Multi-stage production container
├── docker-compose.yml         # One-command container launch
└── .dockerignore
```

## Prerequisites

- **Python 3.13+** (managed via [pyenv](https://github.com/pyenv/pyenv) recommended)
- **Docker** (optional, for containerized usage)

## Local Setup

Clone the repo and create a virtual environment:

```bash
git clone <repo-url>
cd kelly-monte-carlo

python -m venv .venv
source .venv/bin/activate
```

Install the project with dependencies:

```bash
# Runtime only
pip install -e .

# With dev tools (pytest, ruff, coverage)
pip install -e ".[dev]"
```

## Usage

### Run the App

```bash
source .venv/bin/activate
streamlit run src/app/main.py
```

Opens at `http://localhost:8501`. Two pages:

1. **Setup Configuration** — add/edit market setups and their scenarios, set probabilities and returns. The app shows real-time validation and computes the Kelly optimal fraction as you configure.
2. **Simulation** — set parameters (number of simulations, periods, initial capital, Kelly fraction, optional seed), hit Run, and explore results across five tabs: Portfolio Paths, Terminal Distribution, Drawdown Analysis, Detailed Statistics, and Kelly Growth Curves.

### Run with Docker

```bash
# Build and start
docker compose up --build

# Or build manually
docker build -t kelly-monte-carlo .
docker run -p 8501:8501 kelly-monte-carlo
```

The container runs as a non-root user with a read-only filesystem and a health check on the Streamlit endpoint.

## Tests

```bash
source .venv/bin/activate

# Run all tests
pytest

# With verbose output
pytest -v

# With coverage report
pytest --cov=kelly_mc --cov-report=term-missing
```

## Linting

```bash
source .venv/bin/activate

# Check
ruff check src/ tests/

# Auto-fix
ruff check --fix src/ tests/
```

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| Simulations | 10,000 | Number of independent portfolio paths |
| Periods | 100 | Number of time steps per path |
| Initial Capital | 100 | Starting portfolio value |
| Default Kelly Fraction | 0.25 | Fraction of capital to risk (can override per setup) |
| Seed | None | Random seed for reproducibility |

Each **setup** has:
- A name and probability of being the active regime
- Two or more **scenarios**, each with a name, probability, and return (%). Scenario probabilities within a setup must sum to 1.0. Setup probabilities across all setups must also sum to 1.0.
- An optional custom Kelly fraction override

## Example

The default configuration ships with two setups (50/50 probability):

**Setup A** — Favorable regime:
| Scenario | Probability | Return |
|---|---|---|
| Win | 60% | +20% |
| Loss | 30% | -10% |
| Stress | 10% | -50% |

**Setup B** — Less favorable regime:
| Scenario | Probability | Return |
|---|---|---|
| Win | 40% | +15% |
| Loss | 45% | -8% |
| Stress | 15% | -40% |

Running 10,000 simulations over 100 periods with a 0.25 Kelly fraction produces a distribution of terminal wealth values, drawdown profiles, and risk metrics — letting you see how fractional Kelly sizing behaves across different market environments.
