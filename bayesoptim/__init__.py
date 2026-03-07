"""BayesOptim — Bayesian Optimisation with Gaussian Processes.

General-purpose sequential experiment design for science, engineering,
finance, and any domain where experiments are expensive and data is sparse.

Quick start:
    >>> from bayesoptim import BayesianOptimizer, ParameterSpace
    >>> space = ParameterSpace({"temperature": (20.0, 80.0), "pH": (6.0, 8.5)})
    >>> bo = BayesianOptimizer(space)
    >>> bo.initialize(X_init, Y_init)
    >>> candidates = bo.suggest()
    >>> bo.update(Y_new, candidates)
"""

from .config import ParameterSpace, GPConfig, OptimizationConfig
from .optimizer import BayesianOptimizer
from .utils.plotting import (
    convergence,
    acquisition_surface,
    observed_vs_predicted,
    uncertainty_over_iterations,
    full_report,
)

__version__ = "0.1.0"
__author__  = "Subaru Ken Muroi"

__all__ = [
    "BayesianOptimizer",
    "ParameterSpace",
    "GPConfig",
    "OptimizationConfig",
    "convergence",
    "acquisition_surface",
    "observed_vs_predicted",
    "uncertainty_over_iterations",
    "full_report",
]
