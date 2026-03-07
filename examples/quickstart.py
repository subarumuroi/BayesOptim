# %% [markdown]
# # BayesOptim — Quickstart Example
#
# This notebook demonstrates BayesOptim on a synthetic 3-dimensional
# objective function (a noisy Hartmann-style function scaled to realistic
# lab parameter ranges).
#
# It covers:
# 1. Defining a parameter space
# 2. Generating an initial Sobol design
# 3. Running the BO loop
# 4. Visualising results
# 5. Human-in-the-loop workflow pattern

# %% [markdown]
# ## 1. Install and import

# %%
# pip install bayesoptim  (or pip install -e . from repo root)

import torch
import numpy as np
import matplotlib.pyplot as plt

from bayesoptim import (
    BayesianOptimizer,
    ParameterSpace,
    GPConfig,
    OptimizationConfig,
    full_report,
    convergence,
    observed_vs_predicted,
)

torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ## 2. Define the parameter space
#
# Replace these with your actual parameters and their physically meaningful bounds.
# BayesOptim handles any number of continuous dimensions.

# %%
space = ParameterSpace(bounds={
    "temperature":   (20.0,  80.0),   # °C
    "pH":            (6.0,   8.5),
    "concentration": (0.1,   10.0),   # g/L
})

print(f"Parameter space: {space.dim} dimensions")
print(f"Parameters: {space.parameter_names}")

# %% [markdown]
# ## 3. Define a synthetic objective (replace with your real experiment)
#
# In a real workflow you would replace `objective_function` with your
# actual measurement — fermentation yield, material property, financial
# metric, whatever you're optimising.

# %%
def objective_function(X: torch.Tensor) -> torch.Tensor:
    """Synthetic noisy objective on 3D parameter space.

    Peak near temperature=60, pH=7.5, concentration=5.0
    Returns a scalar reward with Gaussian noise.
    """
    temp  = (X[:, 0] - 60.0) / 20.0
    ph    = (X[:, 1] - 7.5)  / 1.0
    conc  = (X[:, 2] - 5.0)  / 3.0
    signal = torch.exp(-temp**2 - ph**2 - 0.5 * conc**2)
    noise  = 0.05 * torch.randn(X.shape[0])
    return signal + noise

# %% [markdown]
# ## 4. Automated BO loop
#
# This is the simplest way to run BayesOptim when you can evaluate
# the objective function directly (benchmarking, simulation-based optimisation).

# %%
opt_config = OptimizationConfig(
    n_iterations=15,
    batch_size=1,
    acquisition_function="qnei",  # or "ucb"
    n_initial=8,
    seed=42,
)

gp_config = GPConfig(
    nu=2.5,            # Matérn smoothness — 2.5 is good default
    fixed_noise=False, # Learn noise from data
)

bo = BayesianOptimizer(
    parameter_space=space,
    gp_config=gp_config,
    opt_config=opt_config,
)

best_X, best_Y = bo.run(objective_function)

bo.summary()

# %% [markdown]
# ## 5. Visualise results

# %%
# Four-panel report
fig = full_report(bo, save_path="bayesoptim_report.png")
plt.show()
print("Report saved to bayesoptim_report.png")

# %%
# Convergence only
fig = convergence(bo.get_history(), title="Convergence on Synthetic 3D Problem")
plt.show()

# %%
# Observed vs GP posterior
bo.fit_model()  # fit GP to all data
fig = observed_vs_predicted(bo, title="GP Fit Quality")
plt.show()

# %% [markdown]
# ## 6. Human-in-the-loop pattern
#
# Use this when your experiments take time — you suggest, run the
# experiment in the lab (or field), then update and repeat.

# %%
# --- Step 1: Generate initial design ---
bo2 = BayesianOptimizer(
    parameter_space=space,
    opt_config=OptimizationConfig(n_initial=5, batch_size=2, seed=0),
)
X_initial = bo2.initialize()   # returns Sobol points, no Y yet

print("Run these experiments first:")
for i, row in enumerate(X_initial.numpy()):
    print(f"  Experiment {i+1}: " +
          ", ".join(f"{n}={v:.3f}" for n, v in zip(space.parameter_names, row)))

# --- Step 2: You run the experiments and get Y values ---
# Simulating here — replace with your actual measurements
Y_initial = objective_function(X_initial)
bo2.train_Y = Y_initial.reshape(-1, 1).double()  # inject observations

# --- Step 3: Suggest next batch ---
bo2.fit_model()
candidates = bo2.suggest()

print("\nNext experiments to run:")
for i, row in enumerate(candidates.numpy()):
    print(f"  Candidate {i+1}: " +
          ", ".join(f"{n}={v:.3f}" for n, v in zip(space.parameter_names, row)))

# --- Step 4: Run experiments, observe Y_new, update ---
Y_new = objective_function(candidates)
bo2.update(Y_new, candidates)

print(f"\nUpdated — total observations: {len(bo2.train_Y)}")
bo2.summary()

# %% [markdown]
# ## 7. CLI usage
#
# BayesOptim also ships with a command-line interface.
# Create a `config.json`:
#
# ```json
# {
#   "parameters": {
#     "temperature":   [20.0, 80.0],
#     "pH":            [6.0,  8.5],
#     "concentration": [0.1,  10.0]
#   }
# }
# ```
#
# Then from your terminal:
#
# ```bash
# # Suggest next experiment from existing data
# bayesoptim suggest --config config.json --data my_data.csv --target yield
#
# # Generate a report from existing data
# bayesoptim report --config config.json --data my_data.csv --output report.png
#
# # Quick info
# bayesoptim info
# ```
#
# The CSV should have columns matching your parameter names plus the target column.

# %% [markdown]
# ## Notes on dimensionality
#
# BayesOptim handles arbitrary numbers of input dimensions through the
# Automatic Relevance Determination (ARD) Matérn kernel, which learns
# a separate lengthscale per parameter. This means:
#
# - Parameters that matter more get shorter lengthscales (model is more
#   sensitive to changes in those dimensions)
# - Parameters that don't matter get long lengthscales (effectively ignored)
#
# Practical limits: GP scales as O(n³) with observations. Above ~500
# observations consider sparse GP approximations (not yet in v0.1).
# For high-dimensional spaces (>20 parameters), consider dimensionality
# reduction or structured priors before applying BO.
