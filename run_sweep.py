#!/usr/bin/env python3

"""
Run ECO experiment sweeps.

Discovers sweep definitions from .py files in sweeps/, generates all
combinations, and executes each via run_config.sh with CLI overrides.

USAGE:
    Direct invocation:
        python run_sweep.py --sweep <name> [options] [-- extra_overrides...]
    
    Shebang mode (sweep file as interpreter):
        ./sweeps/<name>.py [options] [-- extra_overrides...]

OPTIONS:
    --sweep <name>          Sweep name (file sweeps/<name>.py)
    --base_config <path>    Base TOML config (overrides sweep's BASE_CONFIG)
    --output_dir <dir>      Output directory (default: outputs/sweeps)
    --experiment <name>     Aim experiment name (default: <sweep>_<timestamp>)
    -g [N], --gpus [N]      Number of GPUs (0 = all visible, default = 1)
    -j N, --jobs-per-gpu N  Concurrent jobs per GPU (default: 1)
    --dry-run               Print commands without execution
    --sweep-log <file>      Log file for sweep output (plain text, no colors)
    --                      Pass extra arguments to every training run

SWEEP FILE FORMAT:
    Each sweep file must define:
        BASE_CONFIG = "path/to/base.toml"
        OPTIONS = {
            "key": {
                "values": [list of possible values],
                "flags": {value: [list of CLI flags], ...},
                "name": "short_name",
                "monotonic": "increasing"|"decreasing"|None (optional),
            },
            ...
        }
    See docs/sweep_configuration.md for details.

MONOTONIC SKIPPING:
    If an option is marked monotonic and a run fails, subsequent runs with
    larger (or smaller, depending on direction) values in that dimension
    (with all other settings identical) will be skipped. Only works in
    sequential execution (parallel mode records failures but does not skip).

EXAMPLES:
    python run_sweep.py --sweep beta --base_config configs/scaling_law/100m_eco.toml
    python run_sweep.py --sweep debug_smoke -g -j 2 -- --training.steps 5
    ./sweeps/beta.py --base_config configs/experiments/master.toml --gpus 2
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
import re

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Global log file for sweep output (plain text, no colors)
SWEEP_LOG_FILE = None

def sweep_print(msg, end="\n"):
    """Print message with colors to stdout, and plain text to log file if set."""
    print(msg, end=end)
    if SWEEP_LOG_FILE is not None:
        # Strip ANSI color codes
        plain = re.sub(r"\033\[[0-9;]*m", "", msg)
        SWEEP_LOG_FILE.write(plain + end)
        SWEEP_LOG_FILE.flush()

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

def ensure_tmux(session_name):
    """Re-exec inside tmux for SSH persistence (transparent to user)."""
    if os.environ.get("TMUX") or os.environ.get("ECO_NO_TMUX"):
        return  # Already in tmux or explicitly disabled

    import shutil
    if not shutil.which("tmux"):
        sweep_print("Warning: tmux not found, session will not persist after SSH disconnect")
        return

    # Deduplicate session names
    base = session_name
    i = 2
    while subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True,
    ).returncode == 0:
        session_name = f"{base}-{i}"
        i += 1

    # Re-exec inside tmux (replaces current process)
    # User sees normal output, but process persists after SSH disconnect
    cmd = [sys.executable] + sys.argv
    os.execvp("tmux", ["tmux", "new-session", "-s", session_name, "--"] + cmd)


def load_sweeps():
    """Import all .py files in sweeps/ and collect their OPTIONS and BASE_CONFIG."""
    sweeps = {}
    sweep_dir = Path(__file__).parent / "sweeps"
    for f in sorted(sweep_dir.glob("*.py")):
        spec = importlib.util.spec_from_file_location(f.stem, f)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load sweep file: {f}")
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
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load sweep file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return path.stem, mod.OPTIONS, getattr(mod, "BASE_CONFIG", None)


def should_skip_config(current_combo, failed_combos, options):
    """
    Skip configs based on monotonicity rules: if an option is marked monotonic
    and a smaller value already failed (with all other settings identical), skip.
    Direction 'increasing' means failure likelihood increases with value index;
    'decreasing' means failure likelihood decreases with value index.
    """
    for failed_combo in failed_combos:
        for key, opt in options.items():
            monotonic = opt.get("monotonic")
            if not monotonic:
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
                if monotonic == "increasing":
                    if failed_idx <= current_idx:
                        return True
                else:  # decreasing
                    if failed_idx >= current_idx:
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


def validate_options(options):
    """Validate OPTIONS dictionary structure."""
    if not isinstance(options, dict):
        raise ValueError("OPTIONS must be a dict")
    for key, opt in options.items():
        if not isinstance(opt, dict):
            raise ValueError(f"Option '{key}' must be a dict")
        if "values" not in opt:
            raise ValueError(f"Option '{key}' missing 'values' list")
        if "flags" not in opt:
            raise ValueError(f"Option '{key}' missing 'flags' dict")
        if "name" not in opt:
            raise ValueError(f"Option '{key}' missing 'name' string")
        if not isinstance(opt["values"], list):
            raise ValueError(f"Option '{key}' values must be a list")
        if not isinstance(opt["flags"], dict):
            raise ValueError(f"Option '{key}' flags must be a dict")
        # Check that each value has a corresponding flags entry
        for v in opt["values"]:
            if v not in opt["flags"]:
                raise ValueError(f"Option '{key}' missing flags for value {v}")
        # Check monotonic if present
        if "monotonic" in opt:
            monotonic = opt["monotonic"]
            if monotonic is not None and monotonic not in ("increasing", "decreasing"):
                raise ValueError(f"Option '{key}' monotonic must be 'increasing', 'decreasing', or None")


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


def _get_visible_devices():
    """Get list of visible GPU devices from CUDA_VISIBLE_DEVICES, or all GPUs."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        # Parse comma-separated list, handling ranges like "0,1,2-4,5"
        devices = []
        for part in cvd.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                devices.extend(range(int(start), int(end) + 1))
            else:
                devices.append(int(part))
        return devices
    # Default: detect via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        return [int(x) for x in result.stdout.strip().splitlines() if x.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return [0]


def run_training(base_config_path, overrides, output_dir, run_name,
                 experiment_name, extra_overrides=(), gpu_id=None):
    """Run training for one configuration. Returns (run_name, success, elapsed, log_file)."""
    run_dir = os.path.join(output_dir, experiment_name, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "training.log")

    cmd = build_cmd(base_config_path, overrides, run_dir, run_name, extra_overrides)

    env = os.environ.copy()
    if "NGPU" not in env:
        env["NGPU"] = "1"
    env["AIM_EXPERIMENT"] = experiment_name
    if gpu_id is not None:
        # Map gpu_id to actual device from CUDA_VISIBLE_DEVICES if set
        visible_devices = _get_visible_devices()
        actual_device = visible_devices[gpu_id % len(visible_devices)]
        env["CUDA_VISIBLE_DEVICES"] = str(actual_device)

    start_time = time.time()
    try:
        with open(log_file, "w", buffering=1) as f:
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


def format_eta(seconds):
    """Format seconds into human-readable ETA (e.g., '5m 30s')."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def run_sweep(variations, base_config, output_dir, experiment_name,
              options, extra_overrides, num_parallel=1, num_gpus=1):
    """Run all variations, sequentially (with monotonicity skipping) or in parallel."""
    results = {}
    timings = {}
    failed_combos = []
    lock = threading.Lock()
    completed = [0]
    total = len(variations)
    start_time = time.time()

    # Calculate max run name length for alignment
    max_name_len = max(len(v["name"]) for v in variations) if variations else 0

    def _record(run_name, success, elapsed, log_file, combo=None):
        results[run_name] = success
        timings[run_name] = elapsed
        with lock:
            completed[0] += 1
            elapsed_total = time.time() - start_time
            avg_time = elapsed_total / completed[0] if completed[0] > 0 else 0
            remaining = total - completed[0]
            eta = avg_time * remaining / num_parallel if num_parallel > 0 else 0
            progress = f"[{completed[0]}/{total}, ({elapsed:.1f}s), ETA: {format_eta(eta)}]"
            name_padded = run_name.ljust(max_name_len)
            if success:
                sweep_print(f"  {OK}  {_GREEN}{name_padded}{_RESET} ({_BLUE}{log_file}{_RESET}) {progress}")
            else:
                sweep_print(f"  {FAIL}  {_GREEN}{name_padded}{_RESET} ({_BLUE}{log_file}{_RESET}) {progress}")
                if combo is not None:
                    failed_combos.append(combo)

    if num_parallel > 1:
        jobs_per_gpu = num_parallel // num_gpus
        gpu_q = queue.Queue()
        for i in range(num_gpus):
            for _ in range(jobs_per_gpu):
                gpu_q.put(i)


        def _run_one(variation):
            gpu_id = gpu_q.get()
            run_name = variation["name"]
            run_dir = os.path.join(output_dir, experiment_name, run_name)
            log_file = os.path.join(run_dir, "training.log")
            name_padded = run_name.ljust(max_name_len)
            job_start_time = time.time()
            with lock:
                 sweep_print(f"  {START}  {_GREEN}{name_padded}{_RESET} (gpu {gpu_id}) ({_BLUE}{log_file}{_RESET})")
            try:
                result = run_training(
                    base_config, variation["overrides"], output_dir,
                    run_name, experiment_name, extra_overrides,
                    gpu_id=gpu_id,
                )
                success = result[1]
                elapsed = result[2]
            except Exception:
                success = False
                elapsed = time.time() - job_start_time
            finally:
                # Always release GPU
                gpu_q.put(gpu_id)
            
            # Record result AFTER releasing GPU
            with lock:
                _record(run_name, success, elapsed, log_file, variation["combo"])
            return run_name, success, elapsed, log_file

        with ThreadPoolExecutor(max_workers=num_parallel) as executor:
            futures = {executor.submit(_run_one, v): v for v in variations}
            for future in as_completed(futures):
                # Result already recorded in _run_one before GPU was released
                future.result()
    else:
        for variation in variations:
            run_name = variation["name"]
            combo = variation["combo"]

            if should_skip_config(combo, failed_combos, options):
                with lock:
                    completed[0] += 1
                    elapsed_total = time.time() - start_time
                    avg_time = elapsed_total / completed[0] if completed[0] > 0 else 0
                    remaining = total - completed[0]
                    eta = avg_time * remaining / num_parallel if num_parallel > 0 else 0
                    progress = f"[{completed[0]}/{total}, ETA: {format_eta(eta)}]"
                    name_padded = run_name.ljust(max_name_len)
                    sweep_print(f"  {SKIP}  {_GREEN}{name_padded}{_RESET} (monotonicity rule) {progress}")
                results[run_name] = "skipped"
                timings[run_name] = 0.0
                continue

            run_dir = os.path.join(output_dir, experiment_name, run_name)
            log_file = os.path.join(run_dir, "training.log")
            name_padded = run_name.ljust(max_name_len)
            sweep_print(f"  {START}  {_GREEN}{name_padded}{_RESET} ({_BLUE}{log_file}{_RESET})")
            _, success, elapsed, log_file = run_training(
                base_config, variation["overrides"], output_dir, run_name,
                experiment_name, extra_overrides, gpu_id=0,
            )
            _record(run_name, success, elapsed, log_file, combo)


    return results, timings


def print_summary(results, timings):
    successful = [n for n, r in results.items() if r is True]
    failed = [n for n, r in results.items() if r is False]
    skipped = [n for n, r in results.items() if r == "skipped"]

    sweep_print(f"\n{'=' * 80}")
    sweep_print(f"SUMMARY — {len(successful)}/{len(successful) + len(failed)} OK in {sum(timings.values()):.1f}s")
    if skipped:
        sweep_print(f"Skipped: {len(skipped)}")
    sweep_print(f"{'=' * 80}")

    for name in successful:
        sweep_print(f"  {OK}  {_GREEN}{name}{_RESET} ({timings[name]:.1f}s)")
    for name in failed:
        sweep_print(f"  {FAIL}  {_GREEN}{name}{_RESET} ({timings[name]:.1f}s)")
    for name in skipped:
        sweep_print(f"  {SKIP}  {_GREEN}{name}{_RESET}")

    return len(failed) > 0


def main():
    # Detect interpreter mode: first arg is a .py sweep file (via shebang)
    argv = sys.argv[1:]
    sweep_name = None
    options = None
    default_base_config = None
    sweeps = None
    if argv and argv[0].endswith(".py") and os.path.isfile(argv[0]):
        sweep_name, options, default_base_config = load_sweep_file(argv[0])
        validate_options(options)
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
    parser.add_argument("--output_dir", default=os.path.join(REPO_ROOT, "outputs", "sweeps"),
                        help="Output directory")
    parser.add_argument("--experiment", default=None,
                        help="Aim experiment name (default: <sweep_name>_<timestamp>)")
    parser.add_argument("--gpus", "-g", type=int, nargs="?", const=0, default=None,
                        help="Number of GPUs to spread across (default: all)")
    parser.add_argument("--jobs-per-gpu", "-j", type=int, default=1,
                        help="Number of concurrent jobs per GPU (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--sweep-log", type=str, default=None,
                        help="Log file to record sweep output (plain text, no colors)")

    args, extra_overrides = parser.parse_known_args(argv)
    # Open sweep log file if requested
    global SWEEP_LOG_FILE
    if args.sweep_log:
        os.makedirs(os.path.dirname(os.path.abspath(args.sweep_log)), exist_ok=True)
        SWEEP_LOG_FILE = open(args.sweep_log, "w")
    # Remove the '--' separator if it appears as the first extra argument
    if extra_overrides and extra_overrides[0] == '--':
        extra_overrides = extra_overrides[1:]

    if sweep_name is None:
        if sweeps is None:
            raise RuntimeError("Internal error: sweeps not loaded")
        sweep_name = args.sweep
        sweep_info = sweeps[sweep_name]
        options = sweep_info["options"]
        validate_options(options)
        default_base_config = sweep_info["base_config"]

    base_config = args.base_config or default_base_config
    if base_config is None:
        sweep_print("Error: no base config specified. Use --base_config or set BASE_CONFIG in the sweep file.")
        sys.exit(1)

    visible_devices = _get_visible_devices()
    visible_count = len(visible_devices)
    if visible_count == 0:
        sweep_print("Warning: no GPUs detected, defaulting to GPU 0 (CPU fallback)")
        visible_devices = [0]
        visible_count = 1
    if args.gpus is not None:
        if args.gpus == 0:
            # -g passed without value: use all visible devices
            num_gpus = visible_count
        elif args.gpus > visible_count:
            # -g N passed with more GPUs than visible: error
            sweep_print(f"Error: requested {args.gpus} GPUs but only {visible_count} visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
            sys.exit(1)
        else:
            # -g N passed: use N GPUs
            num_gpus = args.gpus
    else:
        # No -g passed: use 1 GPU (but mapped to the first visible device)
        num_gpus = min(1, visible_count)
    num_parallel = num_gpus * args.jobs_per_gpu
    if num_gpus == 0:
        sweep_print("Warning: no GPUs visible, defaulting to 1 GPU (CPU fallback)")
        num_gpus = 1
        num_parallel = num_gpus * args.jobs_per_gpu

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name = args.experiment or f"{sweep_name}_{timestamp}"

    if not os.path.exists(base_config):
        sweep_print(f"Error: config not found: {base_config}")
        sys.exit(1)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if options is None:
        raise RuntimeError("Internal error: options not loaded")

    variations = generate_variations(sweep_name, options)

    sweep_print(f"Base config: {base_config}")
    sweep_print(f"Sweep: {sweep_name} ({len(variations)} runs)")
    sweep_print(f"Experiment: {experiment_name}")
    sweep_print(f"GPUs: {num_gpus}, jobs/GPU: {args.jobs_per_gpu}, total workers: {num_parallel}")
    sweep_print(f"Extra overrides: {' '.join(extra_overrides or [])}")

    sweep_print(f"Output: {output_dir}\n")
    for i, var in enumerate(variations, 1):
        run_dir = os.path.join(output_dir, experiment_name, var["name"])
        # Colored summary of overrides only
        colored_flags = []
        assert options is not None
        for ci, key in enumerate(var["combo"]):
            flags = options[key]["flags"].get(var["combo"][key], [])
            if flags:
                color = _DIM_COLORS[ci % len(_DIM_COLORS)]
                colored_flags.append(f"{color}{' '.join(flags)}{_RESET}")
        overrides_str = ' '.join(colored_flags)
        # Full uncolored command
        cmd = build_cmd(base_config, var["overrides"], run_dir,
                        var["name"], extra_overrides)
        sweep_print(f"{_GREEN}{var['name']}{_RESET}: {overrides_str}")
        sweep_print(f"{' '.join(cmd)}")
        sweep_print("")

    if args.dry_run:
        sweep_print(f"\n{'=' * 80}")
        sweep_print(f"DRY RUN — {len(variations)} runs would be executed")
        sweep_print(f"{'=' * 80}")
        if SWEEP_LOG_FILE is not None:
            SWEEP_LOG_FILE.close()
        return

    sweep_print(f"\n{'=' * 80}")
    sweep_print(f"Starting sweep — {len(variations)} runs")
    sweep_print(f"{'=' * 80}\n")

    results, timings = run_sweep(
        variations, base_config, output_dir, experiment_name,
        options, extra_overrides, num_parallel, num_gpus,
    )

    has_failures = print_summary(results, timings)
    print(f"\nOutput: {output_dir}")
    if SWEEP_LOG_FILE is not None:
        SWEEP_LOG_FILE.close()

    if has_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
