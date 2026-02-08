#!/usr/bin/env python3

"""
Run ECO experiment sweeps.

Discovers sweep definitions from .py files in sweeps/, generates all
combinations, and executes each via run_config.sh with CLI overrides.

Can be invoked directly or used as an interpreter via shebang:
    python run_sweep.py --sweep beta --base_config configs/scaling_law/100m_eco.toml
    ./sweeps/beta.py --base_config configs/scaling_law/100m_eco.toml
"""

import argparse
import importlib.util
import itertools
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


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


def run_training(base_config_path, overrides, output_dir, run_name, experiment_name, pbar):
    """Run training for one configuration."""
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "training.log")

    cmd = [
        "./run_config.sh",
        base_config_path,
        "--job.dump_folder", run_dir,
        "--job.run_name", run_name,
        "--job.description", run_name,
        *overrides,
    ]

    env = os.environ.copy()
    if "NGPU" not in env:
        env["NGPU"] = "1"
    env["AIM_EXPERIMENT"] = experiment_name

    pbar.set_description(f"Running {run_name}")
    start_time = time.time()

    try:
        with open(log_file, "w") as f:
            subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
        elapsed = time.time() - start_time
        pbar.write(f"  OK  {run_name} ({elapsed:.1f}s)")
        return True, elapsed
    except subprocess.CalledProcessError:
        elapsed = time.time() - start_time
        pbar.write(f"  FAIL  {run_name} ({elapsed:.1f}s) — see {log_file}")
        return False, elapsed


def load_sweep_file(path):
    """Load a single sweep .py file and return (name, options, base_config)."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return path.stem, mod.OPTIONS, getattr(mod, "BASE_CONFIG", None)


def main():
    # Detect interpreter mode: first arg is a .py sweep file (via shebang)
    argv = sys.argv[1:]
    sweep_name = None
    options = None
    default_base_config = None
    if argv and argv[0].endswith(".py") and os.path.isfile(argv[0]):
        sweep_name, options, default_base_config = load_sweep_file(argv[0])
        argv = argv[1:]

    parser = argparse.ArgumentParser(description="Run ECO experiment sweeps")
    parser.add_argument("--base_config", default=None, help="Base TOML config (overrides sweep file's BASE_CONFIG)")
    if sweep_name is None:
        sweeps = load_sweeps()
        parser.add_argument("--sweep", required=True, choices=sorted(sweeps), help="Sweep name")
    parser.add_argument("--output_dir", default="./outputs/sweeps", help="Output directory")
    parser.add_argument("--experiment", default=None, help="Aim experiment name (default: <sweep_name>_<timestamp>)")
    args = parser.parse_args(argv)

    if sweep_name is None:
        sweep_name = args.sweep
        sweep_info = sweeps[sweep_name]
        options = sweep_info["options"]
        default_base_config = sweep_info["base_config"]

    base_config = args.base_config or default_base_config
    if base_config is None:
        print("Error: no base config specified. Use --base_config or set BASE_CONFIG in the sweep file.")
        sys.exit(1)

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
    print(f"Output: {output_dir}\n")
    for i, var in enumerate(variations, 1):
        ov = f": {' '.join(var['overrides'])}" if var["overrides"] else ""
        print(f"  {i}. {var['name']}{ov}")

    print(f"\n{'=' * 80}")
    print("Starting sweep...")
    print(f"{'=' * 80}\n")

    results = {}
    timings = {}
    failed_combos = []

    pbar = tqdm(variations, desc="Progress", unit="run")
    for variation in pbar:
        run_name = variation["name"]
        combo = variation["combo"]

        if should_skip_config(combo, failed_combos, options):
            pbar.write(f"  SKIP  {run_name} (monotonicity rule)")
            results[run_name] = "skipped"
            timings[run_name] = 0.0
            continue

        success, elapsed = run_training(
            base_config, variation["overrides"], output_dir, run_name,
            experiment_name, pbar,
        )
        results[run_name] = success
        timings[run_name] = elapsed
        if not success:
            failed_combos.append(combo)

    # Summary
    successful = [n for n, r in results.items() if r is True]
    failed = [n for n, r in results.items() if r is False]
    skipped = [n for n, r in results.items() if r == "skipped"]

    print(f"\n{'=' * 80}")
    print(f"SUMMARY — {len(successful)}/{len(successful) + len(failed)} OK in {sum(timings.values()):.1f}s")
    if skipped:
        print(f"Skipped: {len(skipped)}")
    print(f"{'=' * 80}")

    for name in successful:
        print(f"  OK    {name} ({timings[name]:.1f}s)")
    for name in failed:
        print(f"  FAIL  {name} ({timings[name]:.1f}s)")
    for name in skipped:
        print(f"  SKIP  {name}")

    print(f"\nOutput: {output_dir}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
