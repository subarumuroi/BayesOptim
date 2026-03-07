"""Core Bayesian Optimisation loop for BayesOptim.

Orchestrates: GP training → acquisition function → candidate suggestion
→ user evaluation → update → repeat.

Designed for both:
  - Automated loops (objective function available)
  - Human-in-the-loop workflows (suggest → evaluate → update manually)
"""

import torch
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

from .config import OptimizationConfig, GPConfig, ParameterSpace
from .models.gp_model import train_gp_model, print_model_summary, get_model_hyperparameters


class BayesianOptimizer:
    """Bayesian Optimisation with Gaussian Process surrogate.

    Supports:
      - Continuous multi-dimensional parameter spaces
      - Batch suggestions (q > 1)
      - UCB and qNEI acquisition functions
      - Fixed or learned noise
      - Linear constraints on parameters
      - Human-in-the-loop and automated workflows

    Args:
        parameter_space: ParameterSpace defining parameter names and bounds
        gp_config: GPConfig for GP model settings
        opt_config: OptimizationConfig for BO loop settings

    Example — human-in-the-loop:
        >>> from bayesoptim import BayesianOptimizer, ParameterSpace
        >>> space = ParameterSpace({"temp": (20.0, 80.0), "pH": (6.0, 8.5)})
        >>> bo = BayesianOptimizer(space)
        >>> bo.initialize(X_init, Y_init)
        >>> candidates = bo.suggest()   # run experiment
        >>> bo.update(Y_new, candidates)

    Example — automated loop:
        >>> bo.run(objective_function)
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        gp_config: Optional[GPConfig] = None,
        opt_config: Optional[OptimizationConfig] = None,
    ):
        self.parameter_space = parameter_space
        self.gp_config = gp_config or GPConfig()
        self.opt_config = opt_config or OptimizationConfig()

        if self.opt_config.seed is not None:
            torch.manual_seed(self.opt_config.seed)
            np.random.seed(self.opt_config.seed)

        self.bounds = parameter_space.to_bounds_tensor()
        self.train_X: Optional[torch.Tensor] = None
        self.train_Y: Optional[torch.Tensor] = None
        self.model = None
        self.iteration = 0

        # History for diagnostics and plotting
        self._history: List[Dict] = []

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(
        self,
        X_init: Optional[torch.Tensor] = None,
        Y_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Initialise with existing data or generate a Sobol initial design.

        Args:
            X_init: Initial inputs (n, d). If None, generates Sobol design.
            Y_init: Initial outputs (n,) or (n, 1). Required if X_init given.

        Returns:
            X_init tensor (useful if Sobol design was generated)
        """
        if X_init is not None:
            if Y_init is None:
                raise ValueError("Y_init required when X_init is provided")
            self.train_X = X_init.double()
            self.train_Y = Y_init.reshape(-1, 1).double()
        else:
            # Generate space-filling Sobol design
            n = self.opt_config.n_initial
            sobol = draw_sobol_samples(
                bounds=self.bounds, n=n, q=1
            ).squeeze(1)
            self.train_X = sobol.double()
            print(f"Generated {n} initial Sobol points. Evaluate these before calling suggest().")

        return self.train_X

    # ── Core loop ─────────────────────────────────────────────────────────────

    def fit_model(self):
        """Train GP on current data."""
        self._check_data()
        print(f"\n=== Training GP (Iteration {self.iteration + 1}) ===")
        self.model, self.mll = train_gp_model(
            self.train_X, self.train_Y, self.gp_config
        )
        print_model_summary(self.model)
        return self.model

    def suggest(
        self,
        constraints: Optional[List[Tuple]] = None,
    ) -> torch.Tensor:
        """Suggest next batch of experiments.

        Args:
            constraints: Optional list of linear inequality constraints.
                         Each constraint is (A, b) such that A @ x <= b.
                         See BoTorch docs for full format.

        Returns:
            Tensor of shape (batch_size, n_parameters)
        """
        if self.model is None:
            self.fit_model()

        self.model.eval()
        self.model.likelihood.eval()

        print(f"\n=== Optimising Acquisition Function ===")
        acq_func = self._get_acquisition_function()

        kwargs = dict(
            acq_function=acq_func,
            bounds=self.bounds,
            q=self.opt_config.batch_size,
            num_restarts=self.opt_config.num_restarts,
            raw_samples=self.opt_config.raw_samples,
        )
        if constraints is not None:
            kwargs["inequality_constraints"] = constraints

        candidates, acq_value = optimize_acqf(**kwargs)

        print(f"Suggested {self.opt_config.batch_size} candidate(s)")
        self._log_iteration(candidates, acq_value)
        self.iteration += 1

        return candidates

    def update(
        self,
        Y_new: torch.Tensor,
        X_new: Optional[torch.Tensor] = None,
    ) -> None:
        """Update model with new observations.

        Args:
            Y_new: New output observations (batch_size,) or (batch_size, 1)
            X_new: Corresponding inputs. Must be provided unless you are
                   managing train_X manually. If None, only Y is appended
                   (use only if train_X is already up to date).
        """
        Y_new = Y_new.reshape(-1, 1).double()

        if self.train_X is None:
            raise ValueError("Call initialize() before update()")

        if X_new is not None:
            if X_new.shape[0] != Y_new.shape[0]:
                raise ValueError(
                    f"X_new and Y_new must have same number of rows. "
                    f"Got X_new: {X_new.shape[0]}, Y_new: {Y_new.shape[0]}"
                )
            self.train_X = torch.cat([self.train_X, X_new.double()], dim=0)
        self.train_Y = torch.cat([self.train_Y, Y_new], dim=0)

        # Invalidate model — will be retrained on next suggest() or fit_model()
        self.model = None

    def run(
        self,
        objective_function: Callable[[torch.Tensor], torch.Tensor],
        X_init: Optional[torch.Tensor] = None,
        Y_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Run complete automated BO loop.

        Args:
            objective_function: Callable that takes X (batch, d) and returns Y (batch,)
            X_init: Optional initial inputs
            Y_init: Optional initial outputs

        Returns:
            (best_X, best_Y): Best point and value found
        """
        # Initialise
        X_init = self.initialize(X_init, Y_init)

        if self.train_Y is None:
            print("Evaluating initial design...")
            Y_init = objective_function(self.train_X)
            self.train_Y = Y_init.reshape(-1, 1).double()

        # Main loop
        for i in range(self.opt_config.n_iterations):
            print(f"\n{'='*55}")
            print(f"  BO ITERATION {i + 1} / {self.opt_config.n_iterations}")
            print(f"{'='*55}")

            self.fit_model()
            candidates = self.suggest()
            Y_new = objective_function(candidates)
            self.update(Y_new, candidates)

            best_X, best_Y = self.get_best()
            print(f"\n  Best Y so far : {best_Y:.6f}")
            names = self.parameter_space.parameter_names
            for name, val in zip(names, best_X.numpy()):
                print(f"  {name:20s}: {val:.4f}")

        return self.get_best()

    # ── Results ───────────────────────────────────────────────────────────────

    def get_best(self) -> Tuple[torch.Tensor, float]:
        """Return best observed (X, Y) pair."""
        self._check_data()
        best_idx = torch.argmax(self.train_Y)
        return self.train_X[best_idx], float(self.train_Y[best_idx])

    def get_history(self) -> List[Dict]:
        """Return list of per-iteration logs."""
        return self._history

    def summary(self) -> None:
        """Print a summary of the optimisation run."""
        if self.train_Y is None:
            print("No data yet.")
            return
        best_X, best_Y = self.get_best()
        print(f"\n{'='*55}")
        print(f"  BayesOptim Summary")
        print(f"{'='*55}")
        print(f"  Iterations run    : {self.iteration}")
        print(f"  Total observations: {len(self.train_Y)}")
        print(f"  Best value        : {best_Y:.6f}")
        print(f"  Best parameters   :")
        for name, val in zip(self.parameter_space.parameter_names, best_X.numpy()):
            print(f"    {name:20s}: {val:.4f}")
        print(f"{'='*55}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_data(self):
        if self.train_X is None or self.train_Y is None:
            raise ValueError(
                "No training data. Call initialize() with X_init and Y_init, "
                "or run initialize() then evaluate the Sobol points."
            )

    def _get_acquisition_function(self):
        acq = self.opt_config.acquisition_function
        if acq == "ucb":
            return qUpperConfidenceBound(
                model=self.model, beta=self.opt_config.beta
            )
        elif acq == "qnei":
            return qLogNoisyExpectedImprovement(
                model=self.model, X_baseline=self.train_X
            )
        else:
            raise ValueError(f"Unknown acquisition function: {acq}")

    def _log_iteration(self, candidates: torch.Tensor, acq_value: torch.Tensor):
        entry = {
            "iteration": self.iteration,
            "candidates": candidates.detach().cpu().numpy(),
            "acq_value": float(acq_value.detach().cpu()),
        }
        if self.train_Y is not None:
            entry["best_Y"] = float(self.train_Y.max())
        self._history.append(entry)
