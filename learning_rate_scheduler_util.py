import torch.optim as optim

SCHEDULER_CHOICES = ("plateau", "cosine", "step", "exponential", "none")


def build_scheduler_kwargs_from_args(args):
    """Translate CLI scheduler arguments into scheduler kwargs."""
    if args.scheduler == "plateau":
        return {
            "factor": args.scheduler_factor,
            "patience": args.scheduler_patience,
        }
    if args.scheduler == "cosine":
        return {
            "eta_min": args.scheduler_eta_min,
        }
    if args.scheduler == "step":
        return {
            "step_size": args.scheduler_step_size,
            "gamma": args.scheduler_gamma,
        }
    if args.scheduler == "exponential":
        return {
            "gamma": args.scheduler_gamma,
        }
    return {}


def suggest_scheduler_config(trial, lr, num_epochs):
    """Sample a scheduler configuration for Optuna."""
    scheduler_name = trial.suggest_categorical("scheduler", list(SCHEDULER_CHOICES))
    scheduler_kwargs = {}

    if scheduler_name == "plateau":
        max_patience = max(20, min(120, max(1, num_epochs - 1)))
        scheduler_kwargs["factor"] = trial.suggest_float(
            "scheduler_factor",
            0.2,
            0.8,
        )
        scheduler_kwargs["patience"] = trial.suggest_int(
            "scheduler_patience",
            20,
            max_patience,
        )
    elif scheduler_name == "cosine":
        eta_min_ratio = trial.suggest_float(
            "scheduler_eta_min_ratio",
            1e-4,
            0.2,
            log=True,
        )
        scheduler_kwargs["eta_min"] = lr * eta_min_ratio
    elif scheduler_name == "step":
        max_step_size = max(10, num_epochs)
        scheduler_kwargs["step_size"] = trial.suggest_int(
            "scheduler_step_size",
            10,
            max_step_size,
            log=True,
        )
        scheduler_kwargs["gamma"] = trial.suggest_float(
            "scheduler_gamma",
            0.1,
            0.95,
        )
    elif scheduler_name == "exponential":
        scheduler_kwargs["gamma"] = trial.suggest_float(
            "scheduler_gamma",
            0.95,
            0.9999,
        )

    return scheduler_name, scheduler_kwargs


def create_lr_scheduler(
    optimizer,
    scheduler_name,
    num_epochs,
    scheduler_kwargs=None,
):
    """Build the requested learning-rate scheduler."""
    scheduler_kwargs = dict(scheduler_kwargs or {})

    if scheduler_name == "none":
        return None, False

    if scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=scheduler_kwargs.get("factor", 0.5),
            patience=scheduler_kwargs.get("patience", 50),
        )
        return scheduler, True

    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_epochs),
            eta_min=scheduler_kwargs.get("eta_min", 1e-4),
        )
        return scheduler, False

    if scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, scheduler_kwargs.get("step_size", 100)),
            gamma=scheduler_kwargs.get("gamma", 0.5),
        )
        return scheduler, False

    if scheduler_name == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_kwargs.get("gamma", 0.99),
        )
        return scheduler, False

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")
