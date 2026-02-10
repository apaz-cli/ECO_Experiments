#!/usr/bin/env python3

"""
Run ECO experiment sweeps.

Discovers sweep definitions from .py files in sweeps/, generates all
combinations, and executes each via run_config.sh with CLI overrides.

Can be invoked directly or used as an interpreter via shebang:
    python run_sweep.py --sweep beta --base_config configs/scaling_law/100m_eco.toml
    ./sweeps/beta.py --base_config configs/scaling_law/100m_eco.toml

Extra args after the sweep's own flags are passed through to every run:
    python run_sweep.py --sweep debug_smoke -g -j 2 -- --training.steps 5
"""

import argparse
import importlib.util
import itertools
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# ANSI color helpers
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"
_RESET = "\033[0m"
_DIM_COLORS = [_CYAN, _YELLOW, _MAGENTA, _BLUE,]

OK = f"{_GREEN}   OK{_RESET}"
FAIL = f"{_RED} FAIL{_RESET}"
SKIP = f"{_YELLOW} SKIP{_RESET}"
START = f"{_CYAN}START{_RESET}"


def _gpu_count():
    """Detect number of GPUs via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        return max(1, len(result.stdout.strip().splitlines()))
    except FileNotFoundError:
        return 1


def load_sweeps():
    """Import all .py files in sweeps/ and collect their OPTIONS and BASE_CONFIG."""
    sweeps = {}
    sweep_dir = Path(__file__).parent / "sweeps"
    for f in sorted(sweep_dir.glob("*.py")):
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sweeps[f.stem] = {
            "options": mod.OPTIONS,
            "base_config": getattr(mod, "BASE_CONFIG", None),
        }
    return sweeps


def load_sweep_file(path):
    """Load a single sweep .py file and return (name, options, base_config)."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return path.stem, mod.OPTIONS, getattr(mod, "BASE_CONFIG", None)


def should_skip_config(current_combo, failed_combos, options):
    """
    Skip configs based on monotonicity rules: if an option is marked monotonic
    and a smaller value already failed (with all other settings identical), skip.
    """
    for failed_combo in failed_combos:
        for key, opt in options.items():
            if not opt.get("monotonic", False):
                continue

            other_dims_match = all(
                failed_combo.get(k) == current_combo.get(k)
                for k in options
                if k != key
            )
            if not other_dims_match:
                continue

            values_list = opt["values"]
            try:
                failed_idx = values_list.index(failed_combo.get(key))
                current_idx = values_list.index(current_combo.get(key))
                if failed_idx <= current_idx:
                    return True
            except (ValueError, TypeError):
                continue

    return False


def generate_variations(sweep_name, options):
    """Generate all config variations for a sweep by taking the cartesian product."""
    variations = []
    keys = list(options.keys())
    value_lists = [options[k]["values"] for k in keys]

    for combination in itertools.product(*value_lists):
        name_parts = []
        overrides = []
        combo_dict = dict(zip(keys, combination))

        for key, value in zip(keys, combination):
            opt = options[key]
            flags = opt["flags"].get(value, [])
            overrides.extend(flags)

            if isinstance(value, bool):
                name_parts.append(f"{opt['name']}{'T' if value else 'F'}")
            else:
                name_parts.append(f"{opt['name']}{value}")

        base_name = "_".join(name_parts) if name_parts else "default"
        variations.append({
            "name": f"ECO_{sweep_name}_{base_name}",
            "overrides": overrides,
            "combo": combo_dict,
        })

    return variations


def build_cmd(base_config_path, overrides, run_dir, run_name, extra_overrides=()):
    """Build the training command for a single run."""
    return [
        os.path.join(REPO_ROOT, "run_config.sh"),
        base_config_path,
        "--job.dump_folder", run_dir,
        "--job.run_name", run_name,
        "--job.description", run_name,
        *overrides,
        *extra_overrides,
    ]


def run_training(base_config_path, overrides, output_dir, run_name,
                 experiment_name, extra_overrides=(), gpu_id=None):
    """Run training for one configuration. Returns (run_name, success, elapsed, log_file)."""
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "training.log")

    cmd = build_cmd(base_config_path, overrides, run_dir, run_name, extra_overrides)

    env = os.environ.copy()
    if "NGPU" not in env:
        env["NGPU"] = "1"
    env["AIM_EXPERIMENT"] = experiment_name
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start_time = time.time()
    try:
        with open(log_file, "w") as f:
            subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
        elapsed = time.time() - start_time
        return run_name, True, elapsed, log_file
    except subprocess.CalledProcessError:
        elapsed = time.time() - start_time
        return run_name, False, elapsed, log_file


def run_sweep(variations, base_config, output_dir, experiment_name,
              options, extra_overrides, num_parallel=1, num_gpus=1):
    """Run all variations, sequentially (with monotonicity skipping) or in parallel."""
    results = {}
    timings = {}
    failed_combos = []
    lock = threading.Lock()
    pbar = tqdm(total=len(variations), desc="Progress", unit="run")

    def _record(run_name, success, elapsed, log_file):
        results[run_name] = success
        timings[run_name] = elapsed
        if success:
            pbar.write(f"  {OK}  {run_name} ({elapsed:.1f}s)")
        else:
            pbar.write(f"  {FAIL}  {run_name} ({elapsed:.1f}s) — see {log_file}")
        pbar.update(1)

    if num_parallel > 1:
        jobs_per_gpu = num_parallel // num_gpus
        gpu_q = queue.Queue()
        for i in range(num_gpus):
            for _ in range(jobs_per_gpu):
                gpu_q.put(i)

        def _run_one(variation):
            gpu_id = gpu_q.get()
            with lock:
                pbar.write(f"  {START}  {variation['name']} (gpu {gpu_id})")
            try:
                return run_training(
                    base_config, variation["overrides"], output_dir,
                    variation["name"], experiment_name, extra_overrides,
                    gpu_id=gpu_id,
                )
            finally:
                gpu_q.put(gpu_id)

        with ThreadPoolExecutor(max_workers=num_parallel) as executor:
            futures = {executor.submit(_run_one, v): v for v in variations}
            for future in as_completed(futures):
                run_name, success, elapsed, log_file = future.result()
                with lock:
                    _record(run_name, success, elapsed, log_file)
    else:
        for variation in variations:
            run_name = variation["name"]
            combo = variation["combo"]

            if should_skip_config(combo, failed_combos, options):
                pbar.write(f"  {SKIP}  {run_name} (monotonicity rule)")
                results[run_name] = "skipped"
                timings[run_name] = 0.0
                pbar.update(1)
                continue

            pbar.write(f"  {START}  {run_name}")
            pbar.set_description(f"Running {run_name}")
            _, success, elapsed, log_file = run_training(
                base_config, variation["overrides"], output_dir, run_name,
                experiment_name, extra_overrides, gpu_id=0,
            )
            _record(run_name, success, elapsed, log_file)
            if not success:
                failed_combos.append(combo)

    pbar.close()
    return results, timings


def print_summary(results, timings):
    successful = [n for n, r in results.items() if r is True]
    failed = [n for n, r in results.items() if r is False]
    skipped = [n for n, r in results.items() if r == "skipped"]

    print(f"\n{'=' * 80}")
    print(f"SUMMARY — {len(successful)}/{len(successful) + len(failed)} OK in {sum(timings.values()):.1f}s")
    if skipped:
        print(f"Skipped: {len(skipped)}")
    print(f"{'=' * 80}")

    for name in successful:
        print(f"  {OK}  {name} ({timings[name]:.1f}s)")
    for name in failed:
        print(f"  {FAIL}  {name} ({timings[name]:.1f}s)")
    for name in skipped:
        print(f"  {SKIP}  {name}")

    return len(failed) > 0


def main():
    # Detect interpreter mode: first arg is a .py sweep file (via shebang)
    argv = sys.argv[1:]
    sweep_name = None
    options = None
    default_base_config = None
    if argv and argv[0].endswith(".py") and os.path.isfile(argv[0]):
        sweep_name, options, default_base_config = load_sweep_file(argv[0])
        argv = argv[1:]

    parser = argparse.ArgumentParser(
        description="Run ECO experiment sweeps",
        epilog="Extra args after -- are passed through to every training run.",
    )
    parser.add_argument("--base_config", default=None,
                        help="Base TOML config (overrides sweep file's BASE_CONFIG)")
    if sweep_name is None:
        sweeps = load_sweeps()
        parser.add_argument("--sweep", required=True, choices=sorted(sweeps),
                            help="Sweep name")
    parser.add_argument("--output_dir", default="./outputs/sweeps",
                        help="Output directory")
    parser.add_argument("--experiment", default=None,
                        help="Aim experiment name (default: <sweep_name>_<timestamp>)")
    parser.add_argument("--gpus", "-g", type=int, nargs="?", const=0, default=None,
                        help="Number of GPUs to spread across (default: all)")
    parser.add_argument("--jobs-per-gpu", "-j", type=int, default=1,
                        help="Number of concurrent jobs per GPU (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")

    args, extra_overrides = parser.parse_known_args(argv)
    # Remove the '--' separator if it appears as the first extra argument
    if extra_overrides and extra_overrides[0] == '--':
        extra_overrides = extra_overrides[1:]

    if sweep_name is None:
        sweep_name = args.sweep
        sweep_info = sweeps[sweep_name]
        options = sweep_info["options"]
        default_base_config = sweep_info["base_config"]

    base_config = args.base_config or default_base_config
    if base_config is None:
        print("Error: no base config specified. Use --base_config or set BASE_CONFIG in the sweep file.")
        sys.exit(1)

    if args.gpus is not None:
        num_gpus = args.gpus if args.gpus != 0 else _gpu_count()
    else:
        num_gpus = 1
    num_parallel = num_gpus * args.jobs_per_gpu

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name = args.experiment or f"{sweep_name}_{timestamp}"

    if not os.path.exists(base_config):
        print(f"Error: config not found: {base_config}")
        sys.exit(1)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    variations = generate_variations(sweep_name, options)

    print(f"Base config: {base_config}")
    print(f"Sweep: {sweep_name} ({len(variations)} runs)")
    print(f"Experiment: {experiment_name}")
    print(f"GPUs: {num_gpus}, jobs/GPU: {args.jobs_per_gpu}, total workers: {num_parallel}")
    print(f"Extra overrides: {' '.join(extra_overrides or [])}")

    print(f"Output: {output_dir}\n")
    for i, var in enumerate(variations, 1):
        run_dir = os.path.join(output_dir, var["name"])
        # Colored summary of overrides only
        colored_flags = []
        for ci, key in enumerate(var["combo"]):
            flags = options[key]["flags"].get(var["combo"][key], [])
            if flags:
                color = _DIM_COLORS[ci % len(_DIM_COLORS)]
                colored_flags.append(f"{color}{' '.join(flags)}{_RESET}")
        overrides_str = ' '.join(colored_flags)
        # Full uncolored command
        cmd = build_cmd(base_config, var["overrides"], run_dir,
                        var["name"], extra_overrides)
        print(f"{_GREEN}{var['name']}{_RESET}: {overrides_str}")
        print(f"{' '.join(cmd)}")
        print()

    if args.dry_run:
        print(f"\n{'=' * 80}")
        print("DRY RUN — commands that would be executed:")
        print(f"{'=' * 80}\n")
        for var in variations:
            run_dir = os.path.join(output_dir, var["name"])
            cmd = build_cmd(base_config, var["overrides"], run_dir,
                            var["name"], extra_overrides)
            print(f"  {' '.join(cmd)}\n")
        return

    print(f"\n{'=' * 80}")
    print("Starting sweep...")
    print(f"{'=' * 80}\n")

    results, timings = run_sweep(
        variations, base_config, output_dir, experiment_name,
        options, extra_overrides, num_parallel, num_gpus,
    )

    has_failures = print_summary(results, timings)
    print(f"\nOutput: {output_dir}")

    if has_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
