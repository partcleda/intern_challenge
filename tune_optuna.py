#!/usr/bin/env python3
"""
Hyperparameter search for ``train_placement`` using Optuna (optional dependency).

See ``--help`` and the epilog for usage. Requires: pip install optuna
"""

from __future__ import annotations

import argparse
import sys

import torch

from placement import (
    calculate_normalized_metrics,
    generate_placement_input,
    train_placement,
)


def build_fixed_problem(num_macros: int, num_std_cells: int, seed: int):
    """Same netlist + radial spread as ``test.py`` (deterministic for ``seed``)."""
    torch.manual_seed(seed)
    cell_features, pin_features, edge_list = generate_placement_input(
        num_macros, num_std_cells
    )
    cell_features = cell_features.clone()
    total_cells = cell_features.shape[0]
    total_area = cell_features[:, 0].sum().item()
    spread_radius = (total_area**0.5) * 0.6
    angles = torch.rand(total_cells) * 2 * 3.14159
    radii = torch.rand(total_cells) * spread_radius
    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)
    return cell_features, pin_features, edge_list


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Search hyperparameters for train_placement. Minimizes normalized "
            "wirelength on one fixed netlist + initial spread; trials with overlaps "
            "get a configurable penalty."
        ),
        epilog=(
            "Install Optuna: pip install optuna\n\n"
            "Examples:\n"
            "  python tune_optuna.py --n-trials 30\n"
            "  python tune_optuna.py --n-trials 100 --num-macros 3 "
            "--num-std-cells 50 --seed 1004\n"
            "  python tune_optuna.py --n-trials 50 --storage sqlite:///optuna.db "
            "--study-name placement"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    p.add_argument(
        "--num-macros",
        type=int,
        default=3,
        help="Macros for the fixed validation instance",
    )
    p.add_argument(
        "--num-std-cells",
        type=int,
        default=50,
        help="Standard cells for the fixed validation instance",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1004,
        help="Torch seed for netlist + initial placement (match a test.py case)",
    )
    p.add_argument(
        "--study-name",
        type=str,
        default="placement_tune",
        help="Optuna study name (used with --storage)",
    )
    p.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///optuna.db (default: in-memory)",
    )
    p.add_argument(
        "--epochs-low",
        type=int,
        default=2000,
        help="Minimum num_epochs to search",
    )
    p.add_argument(
        "--epochs-high",
        type=int,
        default=10000,
        help="Maximum num_epochs to search",
    )
    p.add_argument(
        "--epochs-step",
        type=int,
        default=1000,
        help="Step for num_epochs integer suggestions",
    )
    p.add_argument(
        "--overlap-penalty",
        type=float,
        default=1e6,
        help="Base penalty when any overlap remains (adds overlap_ratio on top)",
    )
    p.add_argument(
        "--quiet-optuna",
        action="store_true",
        help="Turn off Optuna info logs",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.epochs_low > args.epochs_high or args.epochs_step < 1:
        print("Invalid epoch bounds or step.", file=sys.stderr)
        sys.exit(2)

    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError:  # pragma: no cover - optional dependency
        print(
            "Optuna is not installed. Install with:\n  pip install optuna",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.quiet_optuna:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    cell_features, pin_features, edge_list = build_fixed_problem(
        args.num_macros, args.num_std_cells, args.seed
    )

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        lambda_wl = trial.suggest_float("lambda_wirelength", 1e-3, 1.0, log=True)
        lambda_ol = trial.suggest_float("lambda_overlap", 1.0, 200.0, log=True)
        num_epochs = trial.suggest_int(
            "num_epochs",
            args.epochs_low,
            args.epochs_high,
            step=args.epochs_step,
        )
        overlap_mode = trial.suggest_categorical(
            "overlap_loss_mode",
            ["area", "fast"],
        )
        clip = trial.suggest_float("per_cell_grad_clip_norm", 1.0, 20.0, log=True)

        result = train_placement(
            cell_features,
            pin_features,
            edge_list,
            num_epochs=num_epochs,
            lr=lr,
            lambda_wirelength=lambda_wl,
            lambda_overlap=lambda_ol,
            overlap_loss_mode=overlap_mode,
            per_cell_grad_clip_norm=clip,
            verbose=False,
        )
        metrics = calculate_normalized_metrics(
            result["final_cell_features"],
            pin_features,
            edge_list,
        )
        if metrics["num_cells_with_overlaps"] > 0:
            return args.overlap_penalty + metrics["overlap_ratio"]
        return metrics["normalized_wl"]

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=bool(args.storage),
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    completed = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    print()
    print("=" * 60)
    print(f"Study: {args.study_name}  |  completed trials: {len(completed)}")
    if study.best_trial is not None:
        print(f"Best value (normalized WL or penalty): {study.best_value:.6g}")
        print("Best params:")
        for k, v in sorted(study.best_params.items()):
            print(f"  {k}: {v}")
    else:
        print("No completed trials.")
    print("=" * 60)


if __name__ == "__main__":
    main()
