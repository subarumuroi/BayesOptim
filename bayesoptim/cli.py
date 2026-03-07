"""Command-line interface for BayesOptim.

Usage examples:
    bayesoptim info
    bayesoptim suggest --config config.json --data data.csv
    bayesoptim report --data data.csv --config config.json --output report.png
"""

import json
import sys
from pathlib import Path

import click
import numpy as np
import torch

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@click.group()
@click.version_option(version="0.1.0", prog_name="BayesOptim")
def main():
    """BayesOptim — Bayesian Optimisation with Gaussian Processes.

    Designed for sequential experiment design in science, engineering,
    and any domain where experiments are expensive and data is sparse.
    """
    pass


@main.command()
def info():
    """Print package info and quick-start guide."""
    click.echo("""
╔══════════════════════════════════════════════════════╗
║              BayesOptim v0.1.0                       ║
║  Bayesian Optimisation with Gaussian Processes       ║
╚══════════════════════════════════════════════════════╝

Quick start (Python API):

    from bayesoptim import BayesianOptimizer, ParameterSpace

    space = ParameterSpace({
        "temperature": (20.0, 80.0),
        "pH":          (6.0,  8.5),
    })

    bo = BayesianOptimizer(space)
    bo.initialize(X_init, Y_init)   # your existing data
    candidates = bo.suggest()       # next experiment(s)
    # ... run experiment, get Y_new ...
    bo.update(Y_new, candidates)
    bo.summary()

Acquisition functions:  ucb  |  qnei (default)
See README.md for full documentation.
    """)


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="JSON config file with parameter space and BO settings.")
@click.option("--data", "-d", required=True, type=click.Path(exists=True),
              help="CSV file with columns matching parameter names + target column.")
@click.option("--target", "-t", default="Y",
              help="Name of the target column in the CSV (default: Y).")
@click.option("--batch", "-b", default=1, type=int,
              help="Number of candidates to suggest (default: 1).")
@click.option("--acq", "-a", default="qnei",
              type=click.Choice(["ucb", "qnei"]),
              help="Acquisition function (default: qnei).")
@click.option("--output", "-o", default=None,
              help="Optional CSV path to save suggestions.")
def suggest(config, data, target, batch, acq, output):
    """Suggest next experiment(s) given existing data and config.

    CONFIG: JSON file defining the parameter space.\n
    DATA:   CSV with observed parameters and target column.

    Example config.json:\n
        {\n
          "parameters": {\n
            "temperature": [20.0, 80.0],\n
            "pH":          [6.0, 8.5]\n
          }\n
        }
    """
    from bayesoptim import BayesianOptimizer, ParameterSpace
    from bayesoptim.config import OptimizationConfig

    if not HAS_PANDAS:
        click.echo("pandas required for CLI. pip install pandas", err=True)
        sys.exit(1)

    # Load config
    with open(config) as f:
        cfg = json.load(f)

    bounds = {k: tuple(v) for k, v in cfg["parameters"].items()}
    space  = ParameterSpace(bounds=bounds)

    # Load data
    df = pd.read_csv(data)
    param_names = space.parameter_names

    missing = [p for p in param_names if p not in df.columns]
    if missing:
        click.echo(f"Missing columns in data: {missing}", err=True)
        sys.exit(1)
    if target not in df.columns:
        click.echo(f"Target column '{target}' not found in data.", err=True)
        sys.exit(1)

    X = torch.tensor(df[param_names].values, dtype=torch.double)
    Y = torch.tensor(df[target].values, dtype=torch.double)

    opt_cfg = OptimizationConfig(
        batch_size=batch,
        acquisition_function=acq,
    )
    bo = BayesianOptimizer(space, opt_config=opt_cfg)
    bo.initialize(X, Y)
    bo.fit_model()
    candidates = bo.suggest()

    # Display
    click.echo(f"\nSuggested {batch} candidate(s):")
    cand_np = candidates.numpy()
    for i, row in enumerate(cand_np):
        click.echo(f"\n  Candidate {i+1}:")
        for name, val in zip(param_names, row):
            click.echo(f"    {name:20s}: {val:.4f}")

    best_X, best_Y = bo.get_best()
    click.echo(f"\n  Current best Y: {best_Y:.6f}")

    # Save suggestions
    if output:
        out_df = pd.DataFrame(cand_np, columns=param_names)
        out_df.to_csv(output, index=False)
        click.echo(f"\nSuggestions saved to {output}")


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="JSON config file.")
@click.option("--data", "-d", required=True, type=click.Path(exists=True),
              help="CSV with parameters and target.")
@click.option("--target", "-t", default="Y", help="Target column name.")
@click.option("--output", "-o", default="bayesoptim_report.png",
              help="Output path for report figure.")
def report(config, data, target, output):
    """Generate a visual report from existing data.

    Fits a GP to the data and produces a four-panel summary figure.
    """
    from bayesoptim import BayesianOptimizer, ParameterSpace
    from bayesoptim.utils.plotting import full_report

    if not HAS_PANDAS:
        click.echo("pandas required for CLI.", err=True)
        sys.exit(1)

    with open(config) as f:
        cfg = json.load(f)

    bounds = {k: tuple(v) for k, v in cfg["parameters"].items()}
    space  = ParameterSpace(bounds=bounds)

    df = pd.read_csv(data)
    param_names = space.parameter_names
    X = torch.tensor(df[param_names].values, dtype=torch.double)
    Y = torch.tensor(df[target].values, dtype=torch.double)

    bo = BayesianOptimizer(space)
    bo.initialize(X, Y)
    bo.fit_model()

    # Simulate history for plotting
    for i in range(len(Y)):
        bo._history.append({
            "iteration": i,
            "candidates": X[i].numpy(),
            "acq_value": 0.0,
            "best_Y": float(Y[:i+1].max()),
        })

    fig = full_report(bo, save_path=output)
    click.echo(f"Report saved to {output}")
