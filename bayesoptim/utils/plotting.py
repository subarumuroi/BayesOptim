"""Visualisation utilities for BayesOptim.

Four core plots:
  1. convergence()       — best Y observed over iterations
  2. acquisition_surface() — 2D slice through acquisition function
  3. observed_vs_predicted() — GP posterior vs actuals (LOO)
  4. uncertainty_over_iterations() — GP posterior std over time

All functions return matplotlib Figure objects so callers can
save, display, or further customise them.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional, Tuple

ACCENT  = "#1B4F72"
GREEN   = "#2ecc71"
RED     = "#e74c3c"
GREY    = "#95a5a6"
BLUE    = "#3498db"
BG      = "#f8f9fa"


def convergence(
    history: List[dict],
    title: str = "Convergence",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot best observed Y value over BO iterations.

    Args:
        history: List of dicts from BayesianOptimizer.get_history()
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    iterations = [h["iteration"] + 1 for h in history if "best_Y" in h]
    best_ys    = [h["best_Y"] for h in history if "best_Y" in h]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(iterations, best_ys, color=ACCENT, linewidth=2, marker="o",
            markersize=6, label="Best observed Y")
    ax.fill_between(iterations, min(best_ys), best_ys,
                    alpha=0.15, color=ACCENT)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Best Y", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", color=ACCENT)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def acquisition_surface(
    optimizer,
    dim_x: int = 0,
    dim_y: int = 1,
    n_grid: int = 50,
    fixed_values: Optional[dict] = None,
    title: str = "Acquisition Function Surface",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a 2D slice through the acquisition function.

    All dimensions not in (dim_x, dim_y) are fixed at their midpoint
    unless overridden via fixed_values.

    Args:
        optimizer: Fitted BayesianOptimizer instance
        dim_x: Index of parameter for x-axis
        dim_y: Index of parameter for y-axis
        n_grid: Grid resolution per axis
        fixed_values: Dict of {param_index: value} for fixed dimensions
        title: Plot title
        save_path: Optional save path

    Returns:
        matplotlib Figure
    """
    if optimizer.model is None:
        raise ValueError("Fit the model before plotting acquisition surface.")
    if optimizer.parameter_space.dim < 2:
        raise ValueError("Need at least 2 dimensions for surface plot.")

    bounds  = optimizer.bounds.numpy()
    names   = optimizer.parameter_space.parameter_names
    d       = optimizer.parameter_space.dim

    x_vals = np.linspace(bounds[0, dim_x], bounds[1, dim_x], n_grid)
    y_vals = np.linspace(bounds[0, dim_y], bounds[1, dim_y], n_grid)
    XX, YY = np.meshgrid(x_vals, y_vals)

    # Build grid points
    grid_points = []
    for i in range(n_grid):
        for j in range(n_grid):
            pt = []
            for k in range(d):
                if k == dim_x:
                    pt.append(XX[i, j])
                elif k == dim_y:
                    pt.append(YY[i, j])
                elif fixed_values and k in fixed_values:
                    pt.append(fixed_values[k])
                else:
                    # Fix at midpoint
                    pt.append((bounds[0, k] + bounds[1, k]) / 2)
            grid_points.append(pt)

    X_grid = torch.tensor(grid_points, dtype=torch.double)

    optimizer.model.eval()
    optimizer.model.likelihood.eval()
    acq = optimizer._get_acquisition_function()

    with torch.no_grad():
        # Reshape for batch acquisition (n, 1, d)
        X_batch = X_grid.unsqueeze(1)
        try:
            Z = acq(X_batch).numpy().reshape(n_grid, n_grid)
        except Exception:
            # Fallback: posterior mean
            posterior = optimizer.model.posterior(X_grid)
            Z = posterior.mean.numpy().reshape(n_grid, n_grid)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    ax.set_facecolor(BG)
    cf = ax.contourf(XX, YY, Z, levels=20, cmap="Blues")
    plt.colorbar(cf, ax=ax, label="Acquisition value")
    ax.set_xlabel(names[dim_x], fontsize=11)
    ax.set_ylabel(names[dim_y], fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", color=ACCENT)

    # Overlay observed points
    if optimizer.train_X is not None:
        obs = optimizer.train_X.numpy()
        ax.scatter(obs[:, dim_x], obs[:, dim_y],
                   c="white", edgecolors=ACCENT, s=60, zorder=5, label="Observed")
        # Mark best
        best_X, _ = optimizer.get_best()
        ax.scatter(best_X[dim_x].item(), best_X[dim_y].item(),
                   c=RED, s=120, marker="*", zorder=6, label="Best")
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def observed_vs_predicted(
    optimizer,
    title: str = "Observed vs GP Posterior Mean",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of actual Y values vs GP posterior mean at training points.

    Args:
        optimizer: Fitted BayesianOptimizer with training data
        title: Plot title
        save_path: Optional save path

    Returns:
        matplotlib Figure
    """
    if optimizer.model is None or optimizer.train_X is None:
        raise ValueError("Fit the model with data before plotting.")

    optimizer.model.eval()
    with torch.no_grad():
        posterior = optimizer.model.posterior(optimizer.train_X)
        pred_mean = posterior.mean.squeeze().numpy()
        pred_std  = posterior.variance.sqrt().squeeze().numpy()

    actual = optimizer.train_Y.squeeze().numpy()

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.errorbar(
        actual, pred_mean, yerr=2 * pred_std,
        fmt="o", color=BLUE, ecolor=GREY,
        elinewidth=1, capsize=3, alpha=0.75, markersize=6
    )
    lims = [
        min(actual.min(), pred_mean.min()) * 0.95,
        max(actual.max(), pred_mean.max()) * 1.05,
    ]
    ax.plot(lims, lims, "--", color=RED, linewidth=1.2, label="Perfect fit")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual Y", fontsize=11)
    ax.set_ylabel("GP Posterior Mean", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", color=ACCENT)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def uncertainty_over_iterations(
    optimizer,
    title: str = "Posterior Uncertainty Over Iterations",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot mean GP posterior standard deviation over training set per iteration.

    Tracks how uncertainty reduces as more data is collected.
    Computed from the iteration history log.

    Args:
        optimizer: BayesianOptimizer (must have run at least one iteration)
        title: Plot title
        save_path: Optional save path

    Returns:
        matplotlib Figure
    """
    history = optimizer.get_history()
    if not history:
        raise ValueError("No history available. Run suggest() at least once.")

    # Recompute std at each data size by re-fitting GPs of increasing size
    # (approximate — uses final model on subsets)
    stds = []
    sizes = []

    n_init = len(optimizer.train_X) - len(history)
    if n_init < 1:
        n_init = 1

    optimizer.model.eval()
    with torch.no_grad():
        for i, _ in enumerate(history):
            n = n_init + i + 1
            X_sub = optimizer.train_X[:n]
            posterior = optimizer.model.posterior(X_sub)
            mean_std = posterior.variance.sqrt().mean().item()
            stds.append(mean_std)
            sizes.append(n)

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(sizes, stds, color=GREEN, linewidth=2, marker="s",
            markersize=5, label="Mean posterior std")
    ax.fill_between(sizes, 0, stds, alpha=0.15, color=GREEN)
    ax.set_xlabel("Number of observations", fontsize=11)
    ax.set_ylabel("Mean posterior std", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", color=ACCENT)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def full_report(
    optimizer,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Four-panel summary figure combining all plots.

    Args:
        optimizer: Fitted BayesianOptimizer after running optimisation
        save_path: Optional path to save combined figure

    Returns:
        matplotlib Figure
    """
    if optimizer.model is None:
        optimizer.fit_model()  # Fit model if not already done
    
    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1: Convergence
    ax1 = fig.add_subplot(gs[0, 0])
    history = optimizer.get_history()
    if history:
        iterations = [h["iteration"] + 1 for h in history if "best_Y" in h]
        best_ys    = [h["best_Y"] for h in history if "best_Y" in h]
        ax1.plot(iterations, best_ys, color=ACCENT, linewidth=2, marker="o")
        ax1.fill_between(iterations, min(best_ys), best_ys, alpha=0.15, color=ACCENT)
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Best Y")
    ax1.set_title("Convergence", fontweight="bold", color=ACCENT)
    ax1.grid(True, alpha=0.3); ax1.set_facecolor(BG)

    # Panel 2: Observed vs Predicted
    ax2 = fig.add_subplot(gs[0, 1])
    if optimizer.model is not None:
        optimizer.model.eval()
        with torch.no_grad():
            posterior  = optimizer.model.posterior(optimizer.train_X)
            pred_mean  = posterior.mean.squeeze().numpy()
            pred_std   = posterior.variance.sqrt().squeeze().numpy()
        actual = optimizer.train_Y.squeeze().numpy()
        ax2.errorbar(actual, pred_mean, yerr=2*pred_std,
                     fmt="o", color=BLUE, ecolor=GREY, elinewidth=1,
                     capsize=3, alpha=0.75, markersize=5)
        lims = [min(actual.min(), pred_mean.min())*0.95,
                max(actual.max(), pred_mean.max())*1.05]
        ax2.plot(lims, lims, "--", color=RED, linewidth=1.2)
        ax2.set_xlim(lims); ax2.set_ylim(lims)
    ax2.set_xlabel("Actual Y"); ax2.set_ylabel("GP Posterior Mean")
    ax2.set_title("Observed vs Predicted", fontweight="bold", color=ACCENT)
    ax2.grid(True, alpha=0.3); ax2.set_facecolor(BG)

    # Panel 3: Acquisition surface (first two dims)
    ax3 = fig.add_subplot(gs[1, 0])
    if optimizer.model is not None and optimizer.parameter_space.dim >= 2:
        try:
            d       = optimizer.parameter_space.dim
            bounds  = optimizer.bounds.numpy()
            names   = optimizer.parameter_space.parameter_names
            n_grid  = 30
            x_vals  = np.linspace(bounds[0, 0], bounds[1, 0], n_grid)
            y_vals  = np.linspace(bounds[0, 1], bounds[1, 1], n_grid)
            XX, YY  = np.meshgrid(x_vals, y_vals)
            grid    = []
            for i in range(n_grid):
                for j in range(n_grid):
                    pt = [XX[i,j], YY[i,j]] + [
                        (bounds[0,k]+bounds[1,k])/2 for k in range(2, d)
                    ]
                    grid.append(pt)
            X_grid  = torch.tensor(grid, dtype=torch.double)
            optimizer.model.eval()
            posterior = optimizer.model.posterior(X_grid)
            Z = posterior.mean.squeeze().detach().numpy().reshape(n_grid, n_grid)
            cf = ax3.contourf(XX, YY, Z, levels=15, cmap="Blues")
            plt.colorbar(cf, ax=ax3)
            if optimizer.train_X is not None:
                obs = optimizer.train_X.numpy()
                ax3.scatter(obs[:,0], obs[:,1], c="white",
                            edgecolors=ACCENT, s=40, zorder=5)
            ax3.set_xlabel(names[0]); ax3.set_ylabel(names[1])
        except Exception:
            ax3.text(0.5, 0.5, "Surface unavailable", ha="center",
                     va="center", transform=ax3.transAxes)
    ax3.set_title("Posterior Mean Surface\n(dims 0 & 1)",
                  fontweight="bold", color=ACCENT)
    ax3.set_facecolor(BG)

    # Panel 4: Uncertainty over iterations
    ax4 = fig.add_subplot(gs[1, 1])
    if history and optimizer.model is not None:
        stds, sizes = [], []
        n_init = max(1, len(optimizer.train_X) - len(history))
        optimizer.model.eval()
        with torch.no_grad():
            for i in range(len(history)):
                n = n_init + i + 1
                X_sub = optimizer.train_X[:n]
                post  = optimizer.model.posterior(X_sub)
                stds.append(post.variance.sqrt().mean().item())
                sizes.append(n)
        ax4.plot(sizes, stds, color=GREEN, linewidth=2, marker="s", markersize=4)
        ax4.fill_between(sizes, 0, stds, alpha=0.15, color=GREEN)
    ax4.set_xlabel("Observations"); ax4.set_ylabel("Mean posterior std")
    ax4.set_title("Uncertainty Reduction", fontweight="bold", color=ACCENT)
    ax4.grid(True, alpha=0.3); ax4.set_facecolor(BG)

    fig.suptitle("BayesOptim — Optimisation Report",
                 fontsize=15, fontweight="bold", color=ACCENT, y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    return fig
