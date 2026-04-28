import os
from datetime import datetime


def build_profile_path(output_dir, profile_tag):
    """Build a timestamped profile output path."""
    profile_dir = os.path.join(output_dir, "profile")
    os.makedirs(profile_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_parts = ["profile"]
    if profile_tag:
        filename_parts.append(profile_tag)
    filename_parts.append(timestamp)

    return os.path.join(profile_dir, "_".join(filename_parts) + ".prof")


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
