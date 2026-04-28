import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime

import torch


@dataclass(frozen=True)
class TorchProfilerConfig:
    enabled: bool = False
    wait: int = 1
    warmup: int = 1
    active: int = 3
    repeat: int = 1
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = True
    acc_events: bool = False


def build_profile_path(output_dir, profile_tag):
    """Build a timestamped cProfile output path."""
    profile_dir = os.path.join(output_dir, "profile")
    os.makedirs(profile_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_parts = ["profile"]
    if profile_tag:
        filename_parts.append(profile_tag)
    filename_parts.append(timestamp)

    return os.path.join(profile_dir, "_".join(filename_parts) + ".prof")


def build_torch_profiler_config_from_args(args):
    """Create a serializable torch profiler config from CLI args."""
    return TorchProfilerConfig(
        enabled=args.torch_profile,
        wait=args.torch_profile_wait,
        warmup=args.torch_profile_warmup,
        active=args.torch_profile_active,
        repeat=args.torch_profile_repeat,
        record_shapes=args.torch_profile_record_shapes,
        profile_memory=args.torch_profile_memory,
        with_stack=args.torch_profile_with_stack,
        acc_events=args.torch_profile_acc_events,
    )


def run_with_optional_profile(main_fn, args, output_dir):
    """Run main_fn(), optionally under cProfile."""
    if not args.profile:
        main_fn()
        return

    import cProfile

    profiler = cProfile.Profile()
    profiler.runcall(main_fn)

    profile_path = build_profile_path(output_dir, args.profile_tag)
    profiler.dump_stats(profile_path)
    print(f"\nProfile stats dumped to: {profile_path}")


class TorchProfilerSession:
    """Manage an optional torch.profiler session for a training loop."""

    def __init__(self, config, output_dir, profile_tag="", run_metadata=None):
        self.config = config or TorchProfilerConfig()
        self.output_dir = output_dir
        self.profile_tag = profile_tag
        self.run_metadata = dict(run_metadata or {})
        self.trace_dir = None
        self.summary_path = None
        self.metadata_path = None
        self._activities = None
        self._profiler = None
        self._profiler_context = None
        self._base_name = None

    def __enter__(self):
        if not self.config.enabled:
            return self

        self._base_name = self._build_base_name()
        self.trace_dir = self._build_trace_dir()
        self.summary_path = os.path.join(
            self.trace_dir,
            f"{self._base_name}_key_averages.txt",
        )
        self.metadata_path = os.path.join(
            self.trace_dir,
            f"{self._base_name}_metadata.json",
        )

        profiler_module = torch.profiler
        self._activities = [profiler_module.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            self._activities.append(profiler_module.ProfilerActivity.CUDA)

        self._profiler_context = profiler_module.profile(
            activities=self._activities,
            schedule=profiler_module.schedule(
                wait=self.config.wait,
                warmup=self.config.warmup,
                active=self.config.active,
                repeat=self.config.repeat,
            ),
            on_trace_ready=profiler_module.tensorboard_trace_handler(
                self.trace_dir,
                worker_name=self._base_name,
            ),
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            acc_events=self.config.acc_events,
        )
        self._profiler = self._profiler_context.__enter__()

        self._write_metadata()
        print(f"Torch profiler enabled. Writing traces to: {self.trace_dir}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.config.enabled:
            return False

        should_suppress = False
        if self._profiler_context is not None:
            should_suppress = self._profiler_context.__exit__(
                exc_type,
                exc_value,
                traceback,
            )
        self._write_summary()
        print(f"Torch profiler summary saved to: {self.summary_path}")
        return should_suppress

    def step(self):
        """Advance the profiler schedule by one training step."""
        if self._profiler is not None:
            self._profiler.step()

    def _build_trace_dir(self):
        profile_root = os.path.join(self.output_dir, "torch_profile")
        os.makedirs(profile_root, exist_ok=True)
        trace_dir = os.path.join(profile_root, self._base_name)
        os.makedirs(trace_dir, exist_ok=True)
        return trace_dir

    def _build_base_name(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts = ["torch_profile"]
        if self.profile_tag:
            filename_parts.append(self.profile_tag)

        runner = self.run_metadata.get("runner")
        if runner:
            filename_parts.append(str(runner))

        test_id = self.run_metadata.get("test_id")
        if test_id is not None:
            filename_parts.append(f"test_{test_id}")

        filename_parts.append(f"pid_{os.getpid()}")
        filename_parts.append(timestamp)
        return "_".join(_slugify(part) for part in filename_parts if part)

    def _write_metadata(self):
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "activities": [activity.name for activity in self._activities],
            "config": asdict(self.config),
            "run_metadata": self.run_metadata,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, sort_keys=True)

    def _write_summary(self):
        if self._profiler is None or self.summary_path is None:
            return

        sort_by = "self_cuda_time_total"
        if all(activity.name != "CUDA" for activity in self._activities):
            sort_by = "self_cpu_time_total"

        summary = self._profiler.key_averages().table(
            sort_by=sort_by,
            row_limit=50,
        )
        with open(self.summary_path, "w", encoding="utf-8") as summary_file:
            summary_file.write(summary)


def create_torch_profiler_session(
    config,
    output_dir,
    profile_tag="",
    run_metadata=None,
):
    """Return a no-op or active torch profiler session for a training loop."""
    return TorchProfilerSession(
        config=config,
        output_dir=output_dir,
        profile_tag=profile_tag,
        run_metadata=run_metadata,
    )


def _slugify(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-") or "run"
