"""Basic tests for BayesOptim.

Run with: pytest tests/
"""

import pytest
import torch
import numpy as np
from bayesoptim import BayesianOptimizer, ParameterSpace, GPConfig, OptimizationConfig


@pytest.fixture
def simple_space():
    return ParameterSpace(bounds={
        "x": (0.0, 1.0),
        "y": (0.0, 1.0),
    })


@pytest.fixture
def small_data(simple_space):
    torch.manual_seed(0)
    X = torch.rand(8, 2, dtype=torch.double)
    Y = (-(X[:, 0] - 0.5)**2 - (X[:, 1] - 0.5)**2 + 1.0
         + 0.01 * torch.randn(8, dtype=torch.double))
    return X, Y


# ── ParameterSpace ────────────────────────────────────────────────────────────

def test_parameter_space_basic():
    space = ParameterSpace(bounds={"a": (0.0, 1.0), "b": (-1.0, 1.0)})
    assert space.dim == 2
    assert space.parameter_names == ["a", "b"]
    bounds = space.to_bounds_tensor()
    assert bounds.shape == (2, 2)


def test_parameter_space_invalid_bounds():
    with pytest.raises(ValueError):
        ParameterSpace(bounds={"a": (1.0, 0.0)})  # lower >= upper


def test_parameter_space_validate_point(simple_space):
    good = torch.tensor([0.5, 0.5], dtype=torch.double)
    bad  = torch.tensor([1.5, 0.5], dtype=torch.double)
    assert simple_space.validate_point(good)
    assert not simple_space.validate_point(bad)


# ── OptimizationConfig ────────────────────────────────────────────────────────

def test_opt_config_invalid_acq():
    with pytest.raises(ValueError):
        OptimizationConfig(acquisition_function="banana")


def test_opt_config_invalid_batch():
    with pytest.raises(ValueError):
        OptimizationConfig(batch_size=0)


# ── BayesianOptimizer — initialisation ───────────────────────────────────────

def test_initialize_with_data(simple_space, small_data):
    X, Y = small_data
    bo = BayesianOptimizer(simple_space)
    bo.initialize(X, Y)
    assert bo.train_X.shape == (8, 2)
    assert bo.train_Y.shape == (8, 1)


def test_initialize_sobol(simple_space):
    bo = BayesianOptimizer(
        simple_space,
        opt_config=OptimizationConfig(n_initial=5)
    )
    X = bo.initialize()
    assert X.shape == (5, 2)
    # All points within bounds
    bounds = simple_space.to_bounds_tensor()
    assert torch.all(X >= bounds[0])
    assert torch.all(X <= bounds[1])


def test_initialize_x_without_y_raises(simple_space, small_data):
    X, _ = small_data
    bo = BayesianOptimizer(simple_space)
    with pytest.raises(ValueError):
        bo.initialize(X_init=X, Y_init=None)


# ── BayesianOptimizer — fit and suggest ──────────────────────────────────────

def test_fit_model(simple_space, small_data):
    X, Y = small_data
    bo = BayesianOptimizer(simple_space)
    bo.initialize(X, Y)
    model = bo.fit_model()
    assert model is not None


def test_suggest_qnei(simple_space, small_data):
    X, Y = small_data
    bo = BayesianOptimizer(
        simple_space,
        opt_config=OptimizationConfig(
            batch_size=1,
            acquisition_function="qnei",
            num_restarts=2,
            raw_samples=32,
        )
    )
    bo.initialize(X, Y)
    candidates = bo.suggest()
    assert candidates.shape == (1, 2)
    # Candidates should be within bounds
    bounds = simple_space.to_bounds_tensor()
    assert torch.all(candidates >= bounds[0] - 1e-6)
    assert torch.all(candidates <= bounds[1] + 1e-6)


def test_suggest_ucb(simple_space, small_data):
    X, Y = small_data
    bo = BayesianOptimizer(
        simple_space,
        opt_config=OptimizationConfig(
            acquisition_function="ucb",
            num_restarts=2,
            raw_samples=32,
        )
    )
    bo.initialize(X, Y)
    candidates = bo.suggest()
    assert candidates.shape == (1, 2)


def test_suggest_batch(simple_space, small_data):
    X, Y = small_data
    bo = BayesianOptimizer(
        simple_space,
        opt_config=OptimizationConfig(
            batch_size=3,
            num_restarts=2,
            raw_samples=32,
        )
    )
    bo.initialize(X, Y)
    candidates = bo.suggest()
    assert candidates.shape == (3, 2)


# ── BayesianOptimizer — update and get_best ───────────────────────────────────

def test_update(simple_space, small_data):
    X, Y = small_data
    bo = BayesianOptimizer(simple_space, opt_config=OptimizationConfig(num_restarts=2, raw_samples=32))
    bo.initialize(X, Y)
    candidates = bo.suggest()
    Y_new = torch.tensor([0.9], dtype=torch.double)
    bo.update(Y_new, candidates)
    assert len(bo.train_Y) == 9


def test_get_best(simple_space, small_data):
    X, Y = small_data
    bo = BayesianOptimizer(simple_space)
    bo.initialize(X, Y)
    best_X, best_Y = bo.get_best()
    assert best_X.shape == (2,)
    assert abs(best_Y - float(Y.max())) < 1e-6


# ── Automated run ─────────────────────────────────────────────────────────────

def test_run_automated(simple_space):
    def obj(X):
        return -(((X[:, 0] - 0.5)**2) + ((X[:, 1] - 0.5)**2))

    bo = BayesianOptimizer(
        simple_space,
        opt_config=OptimizationConfig(
            n_iterations=3,
            n_initial=5,
            num_restarts=2,
            raw_samples=32,
            seed=0,
        )
    )
    best_X, best_Y = bo.run(obj)
    assert best_X.shape == (2,)
    assert isinstance(best_Y, float)
    assert len(bo.get_history()) == 3


# ── 3D space ──────────────────────────────────────────────────────────────────

def test_three_dimensional():
    space = ParameterSpace(bounds={
        "a": (0.0, 1.0),
        "b": (0.0, 1.0),
        "c": (0.0, 1.0),
    })
    torch.manual_seed(1)
    X = torch.rand(10, 3, dtype=torch.double)
    Y = X[:, 0] + X[:, 1] - X[:, 2] + 0.01 * torch.randn(10, dtype=torch.double)
    bo = BayesianOptimizer(
        space,
        opt_config=OptimizationConfig(num_restarts=2, raw_samples=32)
    )
    bo.initialize(X, Y)
    candidates = bo.suggest()
    assert candidates.shape == (1, 3)
