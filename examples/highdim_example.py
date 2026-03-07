# %% [markdown]
# # BayesOptim — High-Dimensional Example (8D)
#
# This example demonstrates BayesOptim on an 8-dimensional problem
# where only 3 of the 8 parameters actually influence the outcome.
#
# This is the realistic scenario in most scientific domains:
#   - Metabolomics / synthetic biology: 10–50 media components
#   - Materials design: 6–20 process parameters
#   - Drug formulation: 8–15 excipient concentrations
#   - ML hyperparameter tuning: 5–15 hyperparameters
#
# The key demonstration is ARD (Automatic Relevance Determination):
# the Matérn kernel learns a separate lengthscale per dimension.
# Irrelevant dimensions get long lengthscales (model becomes insensitive
# to changes in those directions). This is printed after fitting and
# visible in the final lengthscale plot.

# %%
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bayesoptim import (
    BayesianOptimizer,
    ParameterSpace,
    GPConfig,
    OptimizationConfig,
    full_report,
    convergence,
)
from bayesoptim.models.gp_model import get_model_hyperparameters

torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ## 1. Define an 8-dimensional parameter space
#
# Represents a generic optimisation problem with 8 continuous parameters.
# In a real experiment these might be media concentrations, temperatures,
# process conditions, or any other controllable variables.

# %%
space = ParameterSpace(bounds={
    "param_1":  (0.0, 1.0),   # ACTIVE  — strong positive effect
    "param_2":  (0.0, 1.0),   # ACTIVE  — strong negative effect
    "param_3":  (0.0, 1.0),   # ACTIVE  — interaction with param_1
    "param_4":  (0.0, 1.0),   # inactive (noise)
    "param_5":  (0.0, 1.0),   # inactive (noise)
    "param_6":  (0.0, 1.0),   # inactive (noise)
    "param_7":  (0.0, 1.0),   # inactive (noise)
    "param_8":  (0.0, 1.0),   # inactive (noise)
})

print(f"Dimensions : {space.dim}")
print(f"Parameters : {space.parameter_names}")
print()
print("Ground truth: only param_1, param_2, param_3 matter.")
print("param_4 through param_8 are pure noise.")
print("BayesOptim should discover this through ARD lengthscales.")

# %% [markdown]
# ## 2. Define synthetic objective with known active dimensions
#
# Peak at param_1=0.8, param_2=0.2, param_3=0.7.
# Params 4–8 contribute only noise.

# %%
def objective_8d(X: torch.Tensor) -> torch.Tensor:
    """8D objective with 3 active and 5 inactive dimensions.

    True optimum near (0.8, 0.2, 0.7, *, *, *, *, *) ≈ 1.0
    """
    p1 = X[:, 0]
    p2 = X[:, 1]
    p3 = X[:, 2]
    # inactive dims contribute only noise
    noise = 0.03 * torch.randn(X.shape[0])

    signal = (
        torch.exp(-((p1 - 0.8)**2) / 0.05)   # peak at p1=0.8
        - 0.5 * (p2 - 0.2)**2                  # penalise p2 away from 0.2
        + 0.3 * torch.sin(3.14159 * p3)        # oscillation in p3
    )
    return signal + noise

# %% [markdown]
# ## 3. Run BO with a realistic initial design
#
# For 8 dimensions, a good rule of thumb is 2–3× the number of
# dimensions as initial Sobol points before starting BO.
# Here: 20 initial points, 25 BO iterations.

# %%
opt_config = OptimizationConfig(
    n_iterations=25,
    batch_size=1,
    acquisition_function="qnei",
    n_initial=20,      # 2.5× dimensionality — reasonable for 8D
    seed=42,
)

gp_config = GPConfig(
    nu=2.5,
    fixed_noise=False,
    lower_noise_bound=1e-4,
    upper_noise_bound=0.5,
)

bo = BayesianOptimizer(
    parameter_space=space,
    gp_config=gp_config,
    opt_config=opt_config,
)

print("Running 8D Bayesian Optimisation...")
print(f"Initial design: {opt_config.n_initial} Sobol points")
print(f"BO iterations:  {opt_config.n_iterations}")
print()

best_X, best_Y = bo.run(objective_8d)

bo.summary()

# %% [markdown]
# ## 4. Inspect ARD lengthscales
#
# This is the key diagnostic for high-dimensional BO.
#
# Short lengthscale → the model is sensitive to changes in that dimension
#                   → that dimension matters
# Long lengthscale  → the model is insensitive to changes in that dimension
#                   → that dimension is irrelevant
#
# We expect param_1, param_2, param_3 to have shorter lengthscales
# than param_4 through param_8.

# %%
bo.fit_model()
hp = get_model_hyperparameters(bo.model)
lengthscales = hp["lengthscale"]
names = space.parameter_names

print("\n── ARD Lengthscales ─────────────────────────────────")
print(f"  {'Parameter':<12} {'Lengthscale':>14}  {'Active?':>8}")
print(f"  {'─'*12} {'─'*14}  {'─'*8}")
for name, ls in zip(names, lengthscales):
    active = "✓ active" if ls < np.median(lengthscales) else "  noise"
    print(f"  {name:<12} {ls:>14.4f}  {active}")
print(f"  {'─'*40}")
print(f"  Median lengthscale: {np.median(lengthscales):.4f}")
print()
print("Dimensions with lengthscale < median are treated as informative.")

# %% [markdown]
# ## 5. Visualise ARD lengthscales

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#f8f9fa")

# Panel 1: Lengthscale bar chart
ax1 = axes[0]
ax1.set_facecolor("#f8f9fa")
median_ls = np.median(lengthscales)
colours = ["#1B4F72" if ls < median_ls else "#AED6F1" for ls in lengthscales]
bars = ax1.bar(names, lengthscales, color=colours, edgecolor="white", linewidth=1.2)
ax1.axhline(median_ls, color="#e74c3c", linestyle="--", linewidth=1.5,
            label=f"Median = {median_ls:.3f}")
ax1.set_xlabel("Parameter", fontsize=11)
ax1.set_ylabel("ARD Lengthscale", fontsize=11)
ax1.set_title("ARD Lengthscales\n(short = informative, long = irrelevant)",
              fontsize=12, fontweight="bold", color="#1B4F72")
ax1.legend(fontsize=9)
ax1.tick_params(axis="x", rotation=30)
# Label bars
for bar, ls in zip(bars, lengthscales):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{ls:.3f}", ha="center", va="bottom", fontsize=8, color="#2c3e50")

# Panel 2: Convergence
ax2 = axes[1]
ax2.set_facecolor("#f8f9fa")
history = bo.get_history()
iters   = [h["iteration"] + 1 for h in history if "best_Y" in h]
bests   = [h["best_Y"] for h in history if "best_Y" in h]
ax2.plot(iters, bests, color="#1B4F72", linewidth=2, marker="o", markersize=5)
ax2.fill_between(iters, min(bests), bests, alpha=0.15, color="#1B4F72")
ax2.axhline(max(bests), color="#2ecc71", linestyle="--", linewidth=1.2,
            label=f"Best = {max(bests):.4f}")
ax2.set_xlabel("BO Iteration", fontsize=11)
ax2.set_ylabel("Best Y observed", fontsize=11)
ax2.set_title("Convergence (8D problem)",
              fontsize=12, fontweight="bold", color="#1B4F72")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle("BayesOptim — 8D Example with ARD",
             fontsize=14, fontweight="bold", color="#1B4F72", y=1.02)
plt.tight_layout()
plt.savefig("highdim_ard_report.png", dpi=150, bbox_inches="tight",
            facecolor="#f8f9fa")
plt.show()
print("Saved: highdim_ard_report.png")

# %% [markdown]
# ## 6. How initial design size scales with dimensionality
#
# A common question: how many initial Sobol points do I need?
#
# Rule of thumb:
#   n_initial = max(2 * d, 10)
#
# where d is the number of dimensions. This ensures reasonable
# space coverage before BO begins.
#
# | Dimensions | Recommended n_initial |
# |------------|----------------------|
# | 2–5        | 10                   |
# | 6–10       | 2 × d (12–20)        |
# | 11–20      | 2 × d (22–40)        |
# | >20        | Consider dim reduction first |
#
# For very high-dimensional spaces (>20), ARD alone may not be enough.
# Consider:
#   - Feature selection / dimensionality reduction before BO
#   - Sparse GP approximations (planned for BayesOptim v0.2)
#   - Structured priors if domain knowledge is available

# %% [markdown]
# ## 7. Batch suggestions for parallel experiments
#
# If you can run multiple experiments simultaneously (parallel lab
# instruments, parallel simulations, compute clusters), set batch_size > 1.
# BayesOptim uses qNEI which is specifically designed for batch selection.

# %%
bo_batch = BayesianOptimizer(
    parameter_space=space,
    opt_config=OptimizationConfig(
        n_iterations=10,
        batch_size=3,       # suggest 3 experiments at once
        acquisition_function="qnei",
        n_initial=20,
        seed=1,
    )
)

# Generate initial design and evaluate
X_init = bo_batch.initialize()
Y_init = objective_8d(X_init)
bo_batch.train_Y = Y_init.reshape(-1, 1).double()

# Suggest a batch of 3
bo_batch.fit_model()
candidates = bo_batch.suggest()

print(f"\nBatch of {candidates.shape[0]} suggestions:")
for i, row in enumerate(candidates.numpy()):
    print(f"\n  Candidate {i+1}:")
    for name, val in zip(names, row):
        print(f"    {name}: {val:.4f}")

print("\nRun all 3 experiments in parallel, then call bo_batch.update(Y_new, candidates)")
