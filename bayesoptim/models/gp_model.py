"""Gaussian Process surrogate model for BayesOptim.

Extends BoTorch's SingleTaskGP with:
  - Matérn ARD kernel (separate lengthscale per dimension)
  - Flexible fixed or learned noise
  - Configurable priors and constraints
  - Clean summary printing
"""

import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.fit import fit_gpytorch_mll
from gpytorch import kernels, means, likelihoods
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval, Positive
from gpytorch.priors import SmoothedBoxPrior

from ..config import GPConfig


class GPModel(SingleTaskGP):
    """Gaussian Process model with Matérn ARD kernel.

    Supports both fixed and learned noise. Normalises inputs and
    standardises outputs internally — predictions are returned on
    the original scale.

    Args:
        train_X: Training inputs (n_samples, n_features)
        train_Y: Training outputs (n_samples, 1)
        dim: Input dimensionality
        config: GPConfig with model settings
    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        dim: int,
        config: GPConfig,
    ):
        self.config = config

        if config.fixed_noise:
            noise_variance = (config.noise_level * train_Y.mean()).pow(2)
            # Clamp to avoid zero or negative variance
            noise_variance = noise_variance.clamp(min=1e-6)
            train_Yvar = torch.full_like(train_Y, noise_variance.item())
            super().__init__(
                train_X,
                train_Y,
                train_Yvar=train_Yvar,
                outcome_transform=Standardize(m=1),
                input_transform=Normalize(d=dim),
            )
        else:
            likelihood = likelihoods.GaussianLikelihood()
            super().__init__(
                train_X,
                train_Y,
                likelihood=likelihood,
                outcome_transform=Standardize(m=1),
                input_transform=Normalize(d=dim),
            )
            lower_noise = config.lower_noise_bound ** 2
            upper_noise = config.upper_noise_bound ** 2
            self.likelihood.noise_covar.register_prior(
                "noise_prior",
                SmoothedBoxPrior(lower_noise, upper_noise),
                "raw_noise",
            )
            self.likelihood.noise_covar.register_constraint(
                "raw_noise",
                Interval(lower_noise, upper_noise),
            )

        self.mean_module = means.ConstantMean()
        matern_kernel = kernels.MaternKernel(
            nu=config.nu,
            ard_num_dims=dim,
            lengthscale_prior=config.lengthscale_prior,
            lengthscale_constraint=config.lengthscale_constraint,
        )
        self.covar_module = kernels.ScaleKernel(
            base_kernel=matern_kernel,
            outputscale_prior=config.outputscale_prior,
            outputscale_constraint=config.outputscale_constraint,
        )


def train_gp_model(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    config: GPConfig,
) -> tuple:
    """Train a GP model on provided data.

    Args:
        train_X: Inputs (n_samples, n_features)
        train_Y: Outputs (n_samples, 1)
        config: GPConfig

    Returns:
        (model, mll): Trained model and marginal log likelihood object
    """
    if train_X.shape[0] != train_Y.shape[0]:
        raise ValueError(
            f"X and Y must have same number of rows. "
            f"Got X: {train_X.shape[0]}, Y: {train_Y.shape[0]}"
        )
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)

    dim = train_X.shape[1]
    model = GPModel(train_X, train_Y, dim, config)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model, mll


def get_model_hyperparameters(model: GPModel) -> dict:
    """Extract hyperparameters from a trained GP model."""
    hyperparams = {
        "lengthscale": model.covar_module.base_kernel.lengthscale
            .detach().cpu().numpy().flatten().tolist(),
        "outputscale": float(
            model.covar_module.outputscale.detach().cpu().item()
        ),
    }
    if hasattr(model.likelihood, "noise_covar"):
        hyperparams["noise"] = float(
            model.likelihood.noise.detach().cpu().item()
        )
    return hyperparams


def print_model_summary(model: GPModel) -> None:
    """Print a readable summary of GP hyperparameters."""
    hp = get_model_hyperparameters(model)
    print("\n── GP Hyperparameters ──────────────────────────")
    print(f"  Outputscale : {hp['outputscale']:.6f}")
    ls = hp["lengthscale"]
    if len(ls) == 1:
        print(f"  Lengthscale : {float(ls[0]):.6f}")
    else:
        for i, l in enumerate(ls):
            print(f"  Lengthscale[dim {i}] : {float(l):.6f}")
    if "noise" in hp:
        print(f"  Noise var   : {hp['noise']:.6f}")
    print("────────────────────────────────────────────────")