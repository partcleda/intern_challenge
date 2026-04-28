from learning_rate_scheduler_util import suggest_scheduler_config


DEFAULT_OPTUNA_TUNING_CASES = [
    (2, 20, 1201),
    (3, 40, 1202),
    (4, 75, 1203),
    (10, 2000, 1210)
]


def run_optuna_search(
    args,
    *,
    get_best_device,
    seed_torch,
    generate_placement_input,
    initialize_cell_positions,
    train_placement,
    calculate_normalized_metrics,
    tuning_cases=None,
):
    """Run Optuna-based hyperparameter search."""
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is not installed. Install it with `pip install optuna` to use --optuna."
        ) from exc

    tuning_cases = tuning_cases or DEFAULT_OPTUNA_TUNING_CASES
    device = get_best_device()

    def objective(trial):
        lambda_wirelength = trial.suggest_float(
            "lambda_wirelength",
            0.1,
            10.0,
            log=True,
        )
        lambda_overlap = trial.suggest_float(
            "lambda_overlap",
            0.5,
            50.0,
            log=True,
        )
        lr = trial.suggest_float("lr", 1e-3, 3e-1, log=True)
        scheduler_name, scheduler_kwargs = suggest_scheduler_config(
            trial,
            lr=lr,
            num_epochs=args.optuna_epochs,
        )

        overlap_scores = []
        wirelength_scores = []

        for case_idx, (num_macros, num_std_cells, seed) in enumerate(
            tuning_cases,
            start=1,
        ):
            seed_torch(seed)
            cell_features, pin_features, edge_list = generate_placement_input(
                num_macros,
                num_std_cells,
                device=device,
                verbose=False,
            )
            initialize_cell_positions(cell_features)

            result = train_placement(
                cell_features,
                pin_features,
                edge_list,
                num_epochs=args.optuna_epochs,
                lr=lr,
                lambda_wirelength=lambda_wirelength,
                lambda_overlap=lambda_overlap,
                scheduler_name=scheduler_name,
                scheduler_kwargs=scheduler_kwargs,
                track_loss_history=args.track_loss_history,
                verbose=False,
                run_metadata={
                    "runner": "optuna",
                    "trial_number": trial.number,
                    "seed": seed,
                    "num_macros": num_macros,
                    "num_std_cells": num_std_cells,
                },
                early_stop_enabled=args.early_stop,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_min_delta,
                early_stop_overlap_threshold=args.early_stop_overlap_threshold,
                early_stop_zero_overlap_patience=args.early_stop_zero_overlap_patience,
                device=device,
            )

            metrics = calculate_normalized_metrics(
                result["final_cell_features"],
                pin_features,
                edge_list,
            )
            overlap_scores.append(metrics["overlap_ratio"])
            wirelength_scores.append(metrics["normalized_wl"])

            partial_score = (
                sum(overlap_scores) / len(overlap_scores) * 1000.0
                + sum(wirelength_scores) / len(wirelength_scores)
            )
            trial.report(partial_score, step=case_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        avg_overlap = sum(overlap_scores) / len(overlap_scores)
        avg_wirelength = sum(wirelength_scores) / len(wirelength_scores)
        objective_value = avg_overlap * 1000.0 + avg_wirelength

        trial.set_user_attr("avg_overlap", avg_overlap)
        trial.set_user_attr("avg_wirelength", avg_wirelength)
        return objective_value

    storage = args.optuna_storage or None
    study = optuna.create_study(
        direction="minimize",
        study_name=args.optuna_study_name,
        storage=storage,
        load_if_exists=bool(storage),
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    print("=" * 70)
    print("RUNNING OPTUNA SEARCH")
    print("=" * 70)
    print(f"Trials: {args.optuna_trials}")
    print(f"Epochs per trial: {args.optuna_epochs}")
    print(f"Tuning cases: {tuning_cases}")
    print(f"Device: {device}")

    study.optimize(objective, n_trials=args.optuna_trials)

    best_trial = study.best_trial
    print("\n" + "=" * 70)
    print("BEST OPTUNA TRIAL")
    print("=" * 70)
    print(f"Objective: {best_trial.value:.6f}")
    print(
        f"Average Overlap: {best_trial.user_attrs.get('avg_overlap', float('nan')):.6f}"
    )
    print(
        f"Average Wirelength: {best_trial.user_attrs.get('avg_wirelength', float('nan')):.6f}"
    )
    print("Best parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
