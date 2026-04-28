import argparse

from benchmark_test_cases import TEST_CASES_BY_ID
from learning_rate_scheduler_util import SCHEDULER_CHOICES


def _positive_int(value):
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed_value


def parse_args():
    """Parse command line arguments for optional profiling."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile and dump results to the profile directory.",
    )
    parser.add_argument(
        "--profile-tag",
        default="",
        help="Optional tag to include in the profile output filename.",
    )
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="Enable torch.profiler trace capture during training.",
    )
    parser.add_argument(
        "--torch-profile-wait",
        type=int,
        default=1,
        help="Number of initial training steps to skip before torch profiler warmup.",
    )
    parser.add_argument(
        "--torch-profile-warmup",
        type=int,
        default=1,
        help="Number of warmup steps for torch profiler.",
    )
    parser.add_argument(
        "--torch-profile-active",
        type=int,
        default=3,
        help="Number of active recording steps for torch profiler.",
    )
    parser.add_argument(
        "--torch-profile-repeat",
        type=int,
        default=1,
        help="Number of wait/warmup/active cycles to record. Use 0 to repeat until the run ends.",
    )
    parser.add_argument(
        "--torch-profile-record-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable input-shape recording in torch profiler.",
    )
    parser.add_argument(
        "--torch-profile-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable memory tracking in torch profiler.",
    )
    parser.add_argument(
        "--torch-profile-with-stack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable stack trace capture in torch profiler.",
    )
    parser.add_argument(
        "--torch-profile-acc-events",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Accumulate profiler events across schedule cycles to avoid cycle-reset warnings.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1000,
        help="Number of optimization epochs for a regular training run.",
    )
    parser.add_argument(
        "--num-macros",
        type=int,
        default=3,
        help="Number of macro cells to generate for a placement run.",
    )
    parser.add_argument(
        "--num-std-cells",
        type=int,
        default=10,
        help="Number of standard cells to generate for a placement run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to generate and initialize the placement problem.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Torch device to use. 'auto' selects cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--test-case-id",
        type=int,
        choices=sorted(TEST_CASES_BY_ID),
        help="Optional benchmark test case to load. Overrides --num-macros, --num-std-cells, and --seed.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument(
        "--lambda-wirelength",
        type=float,
        default=3.0,
        help="Weight applied to the wirelength term.",
    )
    parser.add_argument(
        "--lambda-overlap",
        type=float,
        default=1.0,
        help="Weight applied to the overlap term.",
    )
    parser.add_argument(
        "--scheduler",
        choices=SCHEDULER_CHOICES,
        default="plateau",
        help="Learning-rate scheduler to use during training.",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=50,
        help="Patience for ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="Decay factor for ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--scheduler-eta-min",
        type=float,
        default=1e-4,
        help="Minimum learning rate for cosine annealing.",
    )
    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=100,
        help="Step size in epochs for StepLR.",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.95,
        help="Gamma decay used by StepLR and ExponentialLR.",
    )
    parser.add_argument(
        "--optuna",
        action="store_true",
        help="Run Optuna hyperparameter search instead of a single training run.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=25,
        help="Number of Optuna trials to execute.",
    )
    parser.add_argument(
        "--optuna-epochs",
        type=int,
        default=400,
        help="Number of epochs per Optuna trial.",
    )
    parser.add_argument(
        "--optuna-study-name",
        default="placement_hparam_search",
        help="Study name used by Optuna.",
    )
    parser.add_argument(
        "--optuna-storage",
        default="",
        help="Optional Optuna storage URL, for example sqlite:///optuna.db.",
    )
    parser.add_argument(
        "--track-loss-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable loss-history collection and persistence.",
    )
    parser.add_argument(
        "--track-overlap-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable per-epoch overlap-metric collection for loss tracking.",
    )
    parser.add_argument(
        "--early-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable overlap-first early stopping during training.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=_positive_int,
        default=75,
        help="Patience before stopping when overlap stops improving.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum improvement required to reset early-stop patience.",
    )
    parser.add_argument(
        "--early-stop-overlap-threshold",
        type=float,
        default=1e-4,
        help="Treat overlap below this value as effectively zero for early stopping.",
    )
    parser.add_argument(
        "--early-stop-zero-overlap-patience",
        type=_positive_int,
        default=25,
        help="Extra patience after zero-overlap is reached to keep reducing wirelength.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=4,
        help="Number of worker processes for test.py. Use 1 to run serially.",
    )
    return parser.parse_args()
