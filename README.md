# BayesOptim

**General-purpose Bayesian Optimisation with Gaussian Processes.**

Designed for sequential experiment design in science, engineering, finance, and any domain where experiments are expensive and data is sparse.

Built on [BoTorch](https://botorch.org/) and [GPyTorch](https://gpytorch.ai/).

---

## What it does

BayesOptim helps you answer: *given everything I've measured so far, which experiment should I run next?*

It fits a Gaussian Process to your observed data, uses an acquisition function to balance exploring unknown regions with exploiting promising ones, and suggests the next point(s) to evaluate.

Handles arbitrary numbers of continuous input dimensions via ARD (Automatic Relevance Determination) — the model learns which parameters actually matter.

---

## Install

```bash
pip install bayesoptim
```

Or from source:

```bash
git clone https://github.com/subarumuroi/BayesOptim
cd BayesOptim
pip install -e .
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0

---

## Quick start

```python
import torch
from bayesoptim import BayesianOptimizer, ParameterSpace

# Define your parameter space
space = ParameterSpace(bounds={
    "temperature":   (20.0, 80.0),
    "pH":            (6.0,  8.5),
    "concentration": (0.1,  10.0),
})

# Load your existing data
# X_init: tensor of shape (n_observations, n_parameters)
# Y_init: tensor of shape (n_observations,)

bo = BayesianOptimizer(space)
bo.initialize(X_init, Y_init)

# Suggest next experiment
candidates = bo.suggest()
print(candidates)  # shape: (1, 3)

# Run your experiment, get Y_new, then update
bo.update(Y_new, candidates)

# Repeat until satisfied
bo.summary()
```

---

## Human-in-the-loop workflow

```python
# Step 1: Generate initial space-filling design (no data needed)
bo = BayesianOptimizer(space)
X_initial = bo.initialize()      # returns Sobol points

# Step 2: Run those experiments in your lab / simulation / field
# ...

# Step 3: Feed back observations
import torch
Y_initial = torch.tensor([...])  # your measured values
bo.train_Y = Y_initial.reshape(-1, 1).double()

# Step 4: Suggest next experiment
bo.fit_model()
candidates = bo.suggest()

# Step 5: Update and repeat
Y_new = torch.tensor([...])
bo.update(Y_new, candidates)
```

---

## Automated loop

When you can evaluate the objective directly (simulation, benchmarking):

```python
def my_objective(X):
    # X shape: (batch, n_params) → return shape: (batch,)
    return some_function(X)

best_X, best_Y = bo.run(my_objective)
```

---

## Configuration

### Parameter space

```python
from bayesoptim import ParameterSpace

space = ParameterSpace(bounds={
    "param_1": (lower, upper),
    "param_2": (lower, upper),
    # ... any number of dimensions
})
```

### GP model settings

```python
from bayesoptim import GPConfig

gp_config = GPConfig(
    nu=2.5,              # Matérn smoothness: 0.5, 1.5, or 2.5
    fixed_noise=False,   # True = fix noise at noise_level * mean(Y)
    noise_level=0.1,     # Fractional noise (used when fixed_noise=True)
    lower_noise_bound=1e-4,
    upper_noise_bound=1.0,
)
```

### Optimisation settings

```python
from bayesoptim import OptimizationConfig

opt_config = OptimizationConfig(
    n_iterations=20,
    batch_size=1,                    # Suggest q experiments at once
    acquisition_function="qnei",     # "qnei" (default) or "ucb"
    beta=2.0,                        # UCB exploration parameter
    num_restarts=10,
    raw_samples=256,
    n_initial=10,                    # Sobol points if no data provided
    seed=42,
)
```

**Acquisition functions:**
- `qnei` — q-Noisy Expected Improvement (recommended; robust, handles batches)
- `ucb` — Upper Confidence Bound (faster; `beta` controls explore/exploit)

---

## Visualisation

```python
from bayesoptim import (
    convergence,
    acquisition_surface,
    observed_vs_predicted,
    uncertainty_over_iterations,
    full_report,
)

# Four-panel summary
fig = full_report(bo, save_path="report.png")

# Individual plots
convergence(bo.get_history())
observed_vs_predicted(bo)
acquisition_surface(bo, dim_x=0, dim_y=1)   # 2D slice, first two dims
uncertainty_over_iterations(bo)
```

---

## CLI

```bash
# Quick info
bayesoptim info

# Suggest next experiment from a CSV of existing data
bayesoptim suggest \
    --config config.json \
    --data   my_data.csv \
    --target yield \
    --batch  2

# Generate a visual report
bayesoptim report \
    --config config.json \
    --data   my_data.csv \
    --output report.png
```

**config.json format:**
```json
{
  "parameters": {
    "temperature":   [20.0, 80.0],
    "pH":            [6.0,  8.5],
    "concentration": [0.1,  10.0]
  }
}
```

**CSV format:** columns matching parameter names + target column (default: `Y`).

---

## Examples

Two annotated examples are included — convert either to a Jupyter notebook with `jupytext --to notebook <file>`.

**`examples/quickstart.py`** — 3D synthetic problem. Covers the full API: automated loop, human-in-the-loop pattern, CLI usage, and visualisation. Start here.

**`examples/highdim_example.py`** — 8D synthetic problem with 5 irrelevant dimensions. Demonstrates ARD (Automatic Relevance Determination): the kernel learns which parameters matter and which don't, reflected in per-dimension lengthscales. More representative of real scientific optimisation problems (metabolomics, materials design, hyperparameter tuning). Includes batch suggestions and a lengthscale diagnostic plot.

### Initial design sizing by dimensionality

| Dimensions | Recommended `n_initial` |
|------------|------------------------|
| 2–5        | 10                     |
| 6–10       | 2 × d  (12–20)         |
| 11–20      | 2 × d  (22–40)         |
| >20        | Consider dimensionality reduction first |

---

## Project structure

```
BayesOptim/
├── bayesoptim/
│   ├── __init__.py        # Public API
│   ├── config.py          # ParameterSpace, GPConfig, OptimizationConfig
│   ├── optimizer.py       # BayesianOptimizer (core loop)
│   ├── cli.py             # Command-line interface
│   ├── models/
│   │   └── gp_model.py    # GPModel, train_gp_model
│   └── utils/
│       └── plotting.py    # Visualisation functions
├── examples/
│   └── quickstart.py      # Annotated example (convert to .ipynb)
├── tests/
│   └── test_basic.py
├── pyproject.toml
└── README.md
```

---

## License

MIT © Subaru Ken Muroi

---

## Citation

If you use BayesOptim in published work:

```
Muroi, S.K. (2026). BayesOptim: General-purpose Bayesian Optimisation
with Gaussian Processes. https://github.com/subarumuroi/BayesOptim
```
