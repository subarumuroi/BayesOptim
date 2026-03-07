"""Configuration dataclasses for BayesOptim.

All model and optimisation settings live here. Users can override
defaults by instantiating these classes with custom values.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
from gpytorch.priors import GammaPrior, SmoothedBoxPrior
from gpytorch.constraints import Interval, Positive


@dataclass
class ParameterSpace:
    """Defines the input parameter space for optimisation.

    Args:
        bounds: Dict mapping parameter name -> (lower, upper) bounds.
                All parameters are treated as continuous.

    Example:
        >>> space = ParameterSpace(bounds={
        ...     "temperature": (20.0, 80.0),
        ...     "pH": (6.0, 8.5),
        ...     "concentration": (0.1, 10.0),
        ... })
    """
    bounds: Dict[str, Tuple[float, float]]

    def __post_init__(self):
        for name, (lo, hi) in self.bounds.items():
            if lo >= hi:
                raise ValueError(
                    f"Parameter '{name}': lower bound {lo} must be < upper bound {hi}"
                )

    @property
    def parameter_names(self) -> List[str]:
        return list(self.bounds.keys())

    @property
    def dim(self) -> int:
        return len(self.bounds)

    def to_bounds_tensor(self) -> torch.Tensor:
        """Return [2, d] bounds tensor for BoTorch."""
        lower = [self.bounds[n][0] for n in self.parameter_names]
        upper = [self.bounds[n][1] for n in self.parameter_names]
        return torch.tensor([lower, upper], dtype=torch.double)

    def validate_point(self, x: torch.Tensor) -> bool:
        """Check whether a point lies within bounds."""
        bounds = self.to_bounds_tensor()
        return bool(torch.all(x >= bounds[0]) and torch.all(x <= bounds[1]))


@dataclass
class GPConfig:
    """Gaussian Process model configuration.

    Args:
        nu: Matérn smoothness parameter. 0.5 = exponential (rough),
            1.5 = once differentiable, 2.5 = twice differentiable (default).
        fixed_noise: If True, noise is fixed at noise_level * mean(Y).
                     If False, noise is learned during training.
        noise_level: Fractional noise level (std / mean) when fixed_noise=True.
        lower_noise_bound: Lower bound on learned noise std.
        upper_noise_bound: Upper bound on learned noise std.
        lengthscale_prior: Prior on kernel lengthscales.
        outputscale_prior: Prior on kernel output scale.
    """
    nu: float = 2.5
    fixed_noise: bool = False
    noise_level: float = 0.1
    lower_noise_bound: float = 1e-4
    upper_noise_bound: float = 1.0
    lengthscale_prior: object = field(
        default_factory=lambda: GammaPrior(3.0, 6.0)
    )
    outputscale_prior: object = field(
        default_factory=lambda: GammaPrior(2.0, 0.15)
    )
    lengthscale_constraint: object = field(
        default_factory=lambda: Positive()
    )
    outputscale_constraint: object = field(
        default_factory=lambda: Positive()
    )


@dataclass
class OptimizationConfig:
    """Bayesian Optimisation loop configuration.

    Args:
        n_iterations: Number of BO iterations to run.
        batch_size: Number of candidates to suggest per iteration (q).
        acquisition_function: One of "ucb", "qnei".
        beta: Exploration-exploitation trade-off for UCB (higher = more explore).
        num_restarts: Number of random restarts for acquisition optimisation.
        raw_samples: Number of raw samples for acquisition initialisation.
        n_initial: Number of initial random points if no data provided.
        seed: Random seed for reproducibility (None = no seed).
    """
    n_iterations: int = 20
    batch_size: int = 1
    acquisition_function: str = "qnei"
    beta: float = 2.0
    num_restarts: int = 10
    raw_samples: int = 256
    n_initial: int = 10
    seed: Optional[int] = None

    def __post_init__(self):
        valid_acq = {"ucb", "qnei"}
        if self.acquisition_function not in valid_acq:
            raise ValueError(
                f"acquisition_function must be one of {valid_acq}, "
                f"got '{self.acquisition_function}'"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be >= 1")
        if self.beta <= 0:
            raise ValueError("beta must be > 0")
