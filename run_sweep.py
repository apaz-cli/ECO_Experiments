#!/usr/bin/env python3
"""Run experiment sweeps. See docs/sweep_configuration.md for format and usage."""

import argparse
import importlib.util
import itertools
import os
import queue
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# ANSI colors
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"
_RESET = "\033[0m"
_DIM_COLORS = [_CYAN, _YELLOW, _MAGENTA, _BLUE]

# Metadata keys in a dimension spec (no dot prefix). Dot-prefixed keys are subdimensions.
_METADATA_KEYS = {"values", "flags", "name", "singular", "monotonic"}

_log_file = None


def _parse_flag_list(f, ctx):
    """Normalize a flags field to a list of CLI strings (branch/fixed dims only)."""
    if f is None:
        return []
    if isinstance(f, str):
        return [f]
    if isinstance(f, list):
        return f
    raise ValueError(f"{ctx}: flags must be str or list, got {type(f).__name__}")


def sweep_print(msg, end="\n"):
    """Print to stdout (colored) and log file (plain)."""
    print(msg, end=end, flush=True)
    if _log_file is not None:
        _log_file.write(re.sub(r"\033\[[0-9;]*m", "", msg) + end)
        _log_file.flush()


def fmt_time(s):
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.0f}m"
    return f"{int(s // 3600)}h {int((s % 3600) // 60)}m"


# ── Sweep loading ──────────────────────────────────────────────────────────


def _load_module(path):
    path = Path(path)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_sweep_file(path):
    """Load a single sweep .py file, returning a sweep-info dict."""
    mod = _load_module(path)
    return {
        "name": Path(path).stem,
        "options": mod.OPTIONS,
        "base_config": getattr(mod, "BASE_CONFIG", None),
        "exclude": getattr(mod, "EXCLUDE", None),
        "extra_flags": getattr(mod, "EXTRA_FLAGS", []),
    }


def load_sweeps():
    """Import all sweep files from sweeps/ directory."""
    return {
        f.stem: load_sweep_file(f)
        for f in sorted((Path(REPO_ROOT) / "sweeps").glob("[!_]*.py"))
    }


def validate_options(options, _ancestor_keys=None):
    """Validate and normalize OPTIONS dict.

    Each key in options must start with '.' to identify it as a dimension.
    Within a dim spec, dot-prefixed keys are subdimensions; non-dot keys are metadata.

    Three dimension types (determined by content):
      Value dim   — has 'values'; sweeps over that list with per-value flags.
      Branch dim  — no 'values', has dot-prefixed subdim keys; each subdim is a
                    mutually-exclusive branch (the dim's implicit "values").
      Fixed dim   — no 'values', no subdims; flags are always appended (one combo).

    Synthesizes _values, _flags, _sub_opts_map on each dim spec for use by
    _expand_tree and related functions.
    """
    if _ancestor_keys is None:
        _ancestor_keys = frozenset()
    current_keys = frozenset(options.keys())

    for key, opt in options.items():
        if not key.startswith("."):
            raise ValueError(
                f"Dimension key {key!r} must start with '.' to mark it as a dimension "
                f"(metadata keys inside a dim spec have no dot)"
            )
        subdim_keys = [k for k in opt if k.startswith(".")]
        for mk in opt:
            if not mk.startswith(".") and not mk.startswith("_") and mk not in _METADATA_KEYS:
                raise ValueError(f"Unknown metadata key {mk!r} in dimension {key!r}")

        m = opt.get("monotonic")
        if m is not None and m not in ("increasing", "decreasing"):
            raise ValueError(f"Dimension {key!r} monotonic must be 'increasing'|'decreasing'|None")

        has_values = "values" in opt
        has_subdims = bool(subdim_keys)

        if has_values and has_subdims:
            raise ValueError(
                f"Dimension {key!r} has both 'values' and subdimensions {subdim_keys!r}. "
                f"Use 'values' for a value dim or dot-prefixed subdim keys for a branch dim, not both."
            )

        if has_values:
            # VALUE DIM — explicit list of values
            values = opt["values"]
            flags = opt.get("flags")
            if flags is None:
                flags_dict = {v: [] for v in values}
            elif isinstance(flags, str):
                flags_dict = {v: [flags, str(v)] for v in values}
            elif isinstance(flags, dict):
                flags_dict = dict(flags)
            else:
                raise ValueError(f"Dimension {key!r} flags must be str or dict, got {type(flags).__name__}")
            for v in values:
                if v not in flags_dict:
                    raise ValueError(f"Dimension {key!r} missing flags for value {v!r}")
            opt["_values"] = list(values)
            opt["_flags"] = flags_dict
            opt["_sub_opts_map"] = {}

        elif has_subdims:
            # BRANCH DIM — subdim keys are the mutually-exclusive branches
            branch_values, flags_dict, sub_opts_map = [], {}, {}
            for sdkey in subdim_keys:
                sdspec = opt[sdkey]
                val = sdkey[1:]  # branch value name = subdim key without leading dot
                branch_values.append(val)
                flags_dict[val] = _parse_flag_list(sdspec.get("flags"), f"Branch {sdkey!r} in {key!r}")
                branch_subdims = {k: v for k, v in sdspec.items() if k.startswith(".")}
                if branch_subdims:
                    sub_opts_map[val] = branch_subdims
                    for bk in branch_subdims:
                        if bk in _ancestor_keys or bk in current_keys:
                            raise ValueError(
                                f"Subdim key {bk!r} (under branch {sdkey!r} of {key!r}) "
                                f"collides with ancestor or sibling dim"
                            )
                    validate_options(branch_subdims, _ancestor_keys | current_keys)
            opt["_values"] = branch_values
            opt["_flags"] = flags_dict
            opt["_sub_opts_map"] = sub_opts_map

        else:
            # FIXED DIM — no values, no subdims; flags always appended
            f_list = _parse_flag_list(opt.get("flags"), f"Fixed dim {key!r}")
            opt["_values"] = [None]
            opt["_flags"] = {None: f_list}
            opt["_sub_opts_map"] = {}


# ── Variation generation ───────────────────────────────────────────────────


def _make_part(nm, val):
    """Build a name part string from dim name and value, or None if no name/value."""
    if nm is None or val is None:
        return None
    if isinstance(val, bool):
        return f"{nm}{'T' if val else 'F'}"
    return f"{nm}{val}"


def _flatten_tokens(tokens):
    """Flatten a name_tokens list to a plain list of strings."""
    parts = []
    for tok in tokens:
        if isinstance(tok, list):
            parts.extend(tok)
        else:
            parts.append(tok)
    return parts


def _build_level_tokens(all_keys, vals, options, contributing_keys=(), child_tokens=()):
    """Build name tokens for one level of the expansion tree.

    contributing_keys: keys whose selected branch has child sub-dims (child_tokens
                       are attributed to them, dotted onto this level's name part).
    child_tokens:      name tokens from the recursive child expansion.
    """
    tokens = []
    for key, val in zip(all_keys, vals):
        nm = options[key].get("name", key[1:])
        part = _make_part(nm, val)
        if key in contributing_keys:
            if part is not None:
                tokens.append([part] + _flatten_tokens(child_tokens))
            else:
                # No name for this dim: pass child tokens through flat
                tokens.extend(child_tokens)
        elif part is not None:
            tokens.append(part)
    return tokens


def _expand_tree(options, combo_so_far, effective_so_far):
    """Recursively expand an options tree, yielding (combo, effective_options, name_tokens).

    options: dict with dot-prefixed dimension keys (e.g. {".treatment": {...}}).
    combo keys and effective_options keys use dim names WITHOUT the leading dot.

    name_tokens is a list of:
      - str       — a simple name part (from a non-branching dim)
      - list[str] — a dot group: [parent_part, *child_parts], joined with '.'

    Non-singular (lex) dims vary fastest in sorted order.
    Singular (diag) dims vary slowest in diagonal order.
    Branch dims expand their selected branch's sub-dims as additional dims.
    """
    lex_keys = sorted(k for k in options if not options[k].get("singular"))
    diag_keys = sorted(k for k in options if options[k].get("singular"))
    all_keys = lex_keys + diag_keys

    if not all_keys:
        yield combo_so_far, effective_so_far, []
        return

    lex_combos = (list(itertools.product(*(options[k]["_values"] for k in lex_keys)))
                  if lex_keys else [()])

    if diag_keys:
        raw = list(itertools.product(*(range(len(options[k]["_values"])) for k in diag_keys)))
        raw.sort(key=lambda idx: (sum(idx), idx))
        diag_combos = [
            tuple(options[diag_keys[i]]["_values"][j] for i, j in enumerate(idx))
            for idx in raw
        ]
    else:
        diag_combos = [()]

    for dv in diag_combos:
        for lv in lex_combos:
            vals = lv + dv
            # Combo and effective keys strip the leading dot from dim keys
            combo = {**combo_so_far, **{k[1:]: v for k, v in zip(all_keys, vals)}}
            effective = {**effective_so_far, **{k[1:]: v for k, v in options.items()}}

            # Collect sub-options from dims whose selected value has branch sub-dims
            sub_opts = {}
            contributing_keys = set()
            for key, val in zip(all_keys, vals):
                opt = options[key]
                children = opt["_sub_opts_map"].get(val, {})
                if children:
                    sub_opts.update(children)
                    contributing_keys.add(key)

            if sub_opts:
                for child_combo, child_effective, child_tokens in _expand_tree(
                    sub_opts, combo, effective
                ):
                    yield child_combo, child_effective, _build_level_tokens(
                        all_keys, vals, options, contributing_keys, child_tokens)
            else:
                yield combo, effective, _build_level_tokens(all_keys, vals, options)


def generate_variations(sweep_name, options, exclude_fn=None, extra_flags=()):
    """Generate all config variations using tree expansion.

    Singular dims vary slowest (diagonal order — advances all singular dims
    at roughly the same rate).  Non-singular dims vary fastest (lex order —
    interleaves treatments for better parallel probing).

    Branch dims expand their selected branch's sub-dims as additional dims.
    Run names use '.' to signal branch ancestry and '_' to separate peer dims.
    """
    variations = []
    for combo, effective, name_tokens in _expand_tree(options, {}, {}):
        if exclude_fn and exclude_fn(combo):
            continue
        # Flatten name tokens: dot groups → "parent.child", peers → "_"-joined
        segments = []
        for tok in name_tokens:
            if isinstance(tok, list):
                segments.append(".".join(tok))
            else:
                segments.append(tok)
        name = f"{sweep_name}_{'_'.join(segments) if segments else 'default'}"
        # Build CLI overrides from all dims in this combo
        overrides = list(extra_flags)
        for key, val in combo.items():
            if key in effective:
                overrides.extend(effective[key]["_flags"].get(val, []))
        variations.append({
            "name": name,
            "overrides": overrides,
            "combo": combo,
            "effective_options": effective,
        })
    return variations


def _treatment_key(combo, options):
    """Non-singular dims identify a treatment. Options keys have dot prefix; combo keys don't."""
    return tuple(combo[k[1:]] for k in sorted(options) if not options[k].get("singular"))


def count_expected(options):
    """Expected runs, computed recursively over the options tree.

    Singular dims contribute 1 (we only need one working value).
    Non-singular dims with no sub-options multiply by their value count.
    Non-singular dims with sub-options: non-branching values count as 1,
    branching values multiply by their subtree count.
    """
    n = 1
    for key, opt in options.items():
        if opt.get("singular"):
            continue
        n_vals = len(opt["_values"])
        sub_opts_map = opt["_sub_opts_map"]
        if not sub_opts_map:
            n *= n_vals
        else:
            branch_sum = sum(count_expected(sub) for sub in sub_opts_map.values())
            n *= (n_vals - len(sub_opts_map)) + branch_sum
    return n


# ── Skip logic ─────────────────────────────────────────────────────────────


def should_skip(combo, failed, succeeded, options):
    """Check monotonic (skip worse on failure) and singular (skip others on success).

    options keys have dot prefix; combo keys are stripped (no dot).
    """
    # Monotonic: skip values worse than a known failure (all other dims must match)
    for fc in failed:
        for key, opt in options.items():
            m = opt.get("monotonic")
            if not m:
                continue
            dim = key[1:]  # combo key (no dot)
            if not all(fc.get(k[1:]) == combo.get(k[1:]) for k in options if k != key):
                continue
            vals = opt["_values"]
            try:
                fi, ci = vals.index(fc[dim]), vals.index(combo[dim])
            except (ValueError, TypeError, KeyError):
                continue
            if (m == "increasing" and fi <= ci) or (m == "decreasing" and fi >= ci):
                return True

    # Singular: skip different values once one succeeds
    # Only require non-singular dims to match (multiple singular dims resolve independently)
    for sc in succeeded:
        for key, opt in options.items():
            if not opt.get("singular"):
                continue
            dim = key[1:]
            if not all(sc.get(k[1:]) == combo.get(k[1:])
                       for k in options if k != key and not options[k].get("singular")):
                continue
            if sc.get(dim) != combo.get(dim):
                return True

    return False


# ── Execution ──────────────────────────────────────────────────────────────


def build_cmd(base_config, overrides, run_dir, run_name, extra=()):
    """Build the run_config.sh command for a single run."""
    return [
        os.path.join(REPO_ROOT, "run_config.sh"), base_config,
        "--job.dump_folder", run_dir, "--job.run_name", run_name,
        "--job.description", run_name, *overrides, *extra,
    ]


def _visible_devices():
    """Get list of visible GPU device indices."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        devs = []
        for p in cvd.split(","):
            p = p.strip()
            if "-" in p:
                a, b = p.split("-", 1)
                devs.extend(range(int(a), int(b) + 1))
            else:
                devs.append(int(p))
        return devs
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                           capture_output=True, text=True, check=True)
        return [int(x) for x in r.stdout.strip().splitlines() if x.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return [0]


def _discover_remote_gpus(workers):
    """SSH to each worker, count GPUs. Returns [(worker, gpu_id), ...]."""
    slots = []
    for w in workers:
        try:
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", w,
                 "nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                sweep_print(f"  {_RED}WARN{_RESET}  Cannot reach {w}: {result.stderr.strip()}")
                continue
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line:
                    slots.append((w, int(line)))
        except (subprocess.TimeoutExpired, ValueError) as e:
            sweep_print(f"  {_RED}WARN{_RESET}  Cannot reach {w}: {e}")
            continue
    return slots


def _parse_workers(workers_arg):
    """Parse --workers argument: comma-separated list or @file."""
    if workers_arg.startswith("@"):
        path = workers_arg[1:]
        with open(path) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return [w.strip() for w in workers_arg.split(",") if w.strip()]


def _get_default_aim_server():
    """Get default aim server address (this machine's hostname + default port)."""
    try:
        # Get the first non-loopback IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return f"{ip}:53800"
    except Exception:
        return f"{socket.gethostname()}:53800"


def _run_one(base_config, var, output_dir, experiment, extra, gpu, log,
             worker=None, remote_dir=None, aim_server=None, sync_artifacts=False):
    """Execute one training run. Returns (var, success, elapsed, log_file).

    When worker is None, runs locally (existing behavior).
    When worker is an SSH target (e.g. 'user@host'), dispatches via SSH.
    """
    name = var["name"]
    run_dir = os.path.join(output_dir, experiment, name)
    os.makedirs(run_dir, exist_ok=True)

    tag_str = ",".join(f"{k}={v}" for k, v in var["combo"].items())

    t0 = time.time()
    try:
        if worker is None:
            # === LOCAL execution (existing code path) ===
            cmd = build_cmd(base_config, var["overrides"], run_dir, name, extra)
            env = os.environ.copy()
            env.setdefault("NGPU", "1")
            env["AIM_EXPERIMENT"] = experiment
            if tag_str:
                env["AIM_TAGS"] = tag_str
            if aim_server:
                env["AIM_REPO"] = f"aim://{aim_server}"
            visible = _visible_devices()
            env["CUDA_VISIBLE_DEVICES"] = str(visible[gpu % len(visible)])

            with open(log, "w", buffering=1) as f:
                subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=f,
                               stderr=subprocess.STDOUT, check=True)
        else:
            # === REMOTE execution via SSH ===
            remote_run_dir = os.path.join(remote_dir, "outputs", "sweeps", experiment, name)
            remote_log = os.path.join(remote_run_dir, "training.log")

            env_parts = [
                f"CUDA_VISIBLE_DEVICES={gpu}",
                "NGPU=1",
                f"AIM_EXPERIMENT={shlex.quote(experiment)}",
            ]
            if tag_str:
                env_parts.append(f"AIM_TAGS={shlex.quote(tag_str)}")
            if aim_server:
                env_parts.append(f"AIM_REPO=aim://{aim_server}")
            env_str = " ".join(env_parts)

            overrides_str = " ".join(shlex.quote(o) for o in var["overrides"])
            extra_str = " ".join(shlex.quote(e) for e in extra) if extra else ""

            remote_cmd = (
                f"mkdir -p {shlex.quote(remote_run_dir)} && "
                f"cd {shlex.quote(remote_dir)} && "
                f"{env_str} bash run_config.sh {shlex.quote(base_config)} "
                f"--job.dump_folder {shlex.quote(remote_run_dir)} "
                f"--job.run_name {shlex.quote(name)} "
                f"--job.description {shlex.quote(name)} "
                f"{overrides_str}"
            )
            if extra_str:
                remote_cmd += f" {extra_str}"
            # tee: dual-write to remote log + stream back via SSH stdout
            remote_cmd += f" 2>&1 | tee {shlex.quote(remote_log)}"

            with open(log, "w", buffering=1) as f:
                subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                     worker, remote_cmd],
                    stdout=f, stderr=subprocess.STDOUT, check=True)

            # Optional: sync artifacts (checkpoints, profiling) back to controller
            if sync_artifacts:
                subprocess.run(
                    ["rsync", "-a", "--exclude=training.log",
                     f"{worker}:{remote_run_dir}/", f"{run_dir}/"],
                    capture_output=True)

        return var, True, time.time() - t0, log
    except subprocess.CalledProcessError:
        return var, False, time.time() - t0, log


def _singular_desc(combo, options):
    """Short description of singular dim values, e.g. 'bs=64, ac=full'.

    options keys have dot prefix; combo keys are stripped (no dot).
    """
    abbrev = {"local_batch_size": "bs", "ac_mode": "ac", "compile": "comp"}
    return ", ".join(
        f"{abbrev.get(k[1:], k[1:5])}={combo[k[1:]]}"
        for k in sorted(options) if options[k].get("singular")
    )


# ── Sweep execution ────────────────────────────────────────────────────────


def run_sweep(variations, base_config, output_dir, experiment, expected, extra,
              gpu_slots, jobs_per_gpu, remote_dir=None, aim_server=None,
              sync_artifacts=False):
    """Execute all variations. Single code path for sequential and parallel.

    gpu_slots: list of (worker, gpu_id) tuples.
               worker=None means local execution.
    expected:  expected number of resolved treatments (from count_expected).
    """
    num_slots = len(gpu_slots) * jobs_per_gpu
    has_singular = any(
        o.get("singular")
        for var in variations
        for o in var["effective_options"].values()
    )

    # Shared state (protected by lock)
    results = []          # (var, success, elapsed, log_file)
    failed, succeeded = [], []
    resolved = set()      # treatment keys with at least one success
    lock = threading.Lock()
    t0 = time.time()
    pad = max((len(v["name"]) for v in variations), default=0)

    # GPU pool: each slot appears jobs_per_gpu times
    # Slots are (worker, gpu_id) tuples — worker=None for local
    gpu_q = queue.Queue()
    for worker, gpu_id in gpu_slots:
        for _ in range(jobs_per_gpu):
            gpu_q.put((worker, gpu_id))

    skipped_count = [0]  # mutable for closure access

    def _job_worker(var):
        worker, gpu = gpu_q.get()
        opts = var["effective_options"]

        # Pre-flight: re-check skip in case treatment was resolved while queued
        with lock:
            if should_skip(var["combo"], failed, succeeded, opts):
                gpu_q.put((worker, gpu))
                skipped_count[0] += 1
                return

        nm = var["name"].ljust(pad)
        sdesc = _singular_desc(var["combo"], opts) if has_singular else ""
        # Compute log path here so we can print it at START
        run_dir = os.path.join(output_dir, experiment, var["name"])
        log = os.path.join(run_dir, "training.log")
        n = 0
        while os.path.exists(log):
            n += 1
            log = os.path.join(run_dir, f"training.{n}.log")
        with lock:
            if worker:
                loc = f"{_MAGENTA}{worker}{_RESET} gpu {gpu}"
            else:
                loc = f"gpu {gpu}"
            if sdesc:
                sweep_print(f"  {_CYAN}START{_RESET}  {_GREEN}{nm}{_RESET} ({sdesc}) {loc} {_BLUE}{log}{_RESET}")
            else:
                sweep_print(f"  {_CYAN}START{_RESET}  {_GREEN}{nm}{_RESET} {loc} {_BLUE}{log}{_RESET}")

        job_t0 = time.time()
        try:
            _, ok, elapsed, log = _run_one(
                base_config, var, output_dir, experiment, extra, gpu, log,
                worker=worker, remote_dir=remote_dir, aim_server=aim_server,
                sync_artifacts=sync_artifacts)
        except Exception:
            ok, elapsed, log = False, time.time() - job_t0, "?"
        finally:
            gpu_q.put((worker, gpu))

        with lock:
            results.append((var, ok, elapsed, log))
            tk = _treatment_key(var["combo"], opts)
            (succeeded if ok else failed).append(var["combo"])
            if ok:
                resolved.add(tk)
            nr = len(resolved)

            if ok:
                tag = f"{_GREEN}   OK{_RESET}"
            elif has_singular and tk not in resolved:
                tag = f"{_YELLOW}PROBE{_RESET}"
            else:
                tag = f"{_RED} FAIL{_RESET}"
            if worker:
                loc = f"{_MAGENTA}{worker}{_RESET} gpu {gpu}"
            else:
                loc = ""
            sweep_print(f"  {tag}  {_GREEN}{nm}{_RESET} {elapsed:.1f}s [{nr}/{expected} resolved] {loc} {_BLUE}{log}{_RESET}")

        return

    def _drain_completed():
        """Process any already-completed futures to update state before skip checks."""
        for f in [f for f in futures if f.done()]:
            f.result()
            del futures[f]

    inflight = {}  # treatment_key -> most recent future for that treatment

    with ThreadPoolExecutor(max_workers=num_slots) as pool:
        futures = {}
        remaining = list(variations)
        while remaining:
            deferred = []
            for var in remaining:
                _drain_completed()

                # Defer if previous run of same treatment is still in-flight
                tk = _treatment_key(var["combo"], var["effective_options"])
                prev = inflight.get(tk)
                if prev is not None and not prev.done():
                    deferred.append(var)
                    continue

                # Wait for a slot if pool is full
                while len(futures) >= num_slots:
                    done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                    for f in done:
                        f.result()
                        del futures[f]

                # Check skip with most up-to-date state
                with lock:
                    skip = should_skip(var["combo"], failed, succeeded,
                                       var["effective_options"])
                if skip:
                    skipped_count[0] += 1
                    continue

                f = pool.submit(_job_worker, var)
                futures[f] = var
                inflight[tk] = f

            remaining = deferred
            if remaining and futures:
                done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                for f in done:
                    f.result()
                    del futures[f]

        # Drain remaining futures
        for f in list(futures):
            f.result()

    return results, skipped_count[0], time.time() - t0


# ── Summary ────────────────────────────────────────────────────────────────


def print_summary(results, skipped, elapsed):
    """Print per-treatment summary."""
    has_singular = any(
        o.get("singular")
        for var, _, _, _ in results
        for o in var["effective_options"].values()
    )

    # Group by treatment
    treatments = {}
    for var, ok, el, log in results:
        tk = _treatment_key(var["combo"], var["effective_options"])
        treatments.setdefault(tk, []).append((var, ok, el, log))

    ok_count = sum(1 for runs in treatments.values() if any(s for _, s, _, _ in runs))
    total_treatments = len(treatments)
    total_runs = len(results)
    probe_fails = sum(
        sum(1 for _, s, _, _ in runs if not s)
        for runs in treatments.values() if any(s for _, s, _, _ in runs)
    ) if has_singular else 0

    parts = [f"{total_runs} runs"]
    if probe_fails:
        parts.append(f"{probe_fails} probe failures")
    if skipped:
        parts.append(f"{skipped} skipped")

    sweep_print(f"\n{'=' * 80}")
    sweep_print(f"SUMMARY — {ok_count}/{total_treatments} OK in {fmt_time(elapsed)}"
                f" ({', '.join(parts)})")
    sweep_print(f"{'=' * 80}")

    for tk in sorted(treatments):
        runs = treatments[tk]
        successes = [(v, e, lf) for v, s, e, lf in runs if s]
        if successes:
            best_var, best_el, best_log = min(successes, key=lambda x: x[1])
            sdesc = _singular_desc(best_var["combo"], best_var["effective_options"]) if has_singular else ""
            n_probes = sum(1 for _, s, _, _ in runs if not s)
            pnote = f", {n_probes} probes" if n_probes else ""
            sweep_print(f"  {_GREEN}   OK{_RESET}  {best_var['name']:<40}"
                        f" {sdesc:<20} ({best_el:.1f}s{pnote})")
            sweep_print(f"         {_BLUE}{best_log}{_RESET}")
        else:
            last_var, _, _, last_log = runs[-1]
            sweep_print(f"  {_RED} FAIL{_RESET}  {last_var['name']:<40}"
                        f" (all {len(runs)} combos failed)")
            sweep_print(f"         {_BLUE}{last_log}{_RESET}")

    return ok_count < total_treatments


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    global _log_file

    # Shebang mode: first arg is a .py sweep file
    argv = sys.argv[1:]
    sweep_name = options = default_base = exclude_fn = sweeps = None
    extra_flags = []
    if argv and argv[0].endswith(".py") and os.path.isfile(argv[0]):
        info = load_sweep_file(argv[0])
        sweep_name, options, default_base, exclude_fn, extra_flags = (
            info["name"], info["options"], info["base_config"],
            info["exclude"], info["extra_flags"])
        validate_options(options)
        argv = argv[1:]

    parser = argparse.ArgumentParser(
        description="Run experiment sweeps",
        epilog="Extra args after -- are passed to every training run.")
    parser.add_argument("--base_config", default=None,
                        help="Base TOML config (overrides sweep file's BASE_CONFIG)")
    if sweep_name is None:
        sweeps = load_sweeps()
        parser.add_argument("--sweep", required=True, choices=sorted(sweeps),
                            help="Sweep name")
    parser.add_argument("--output_dir", default=os.path.join(REPO_ROOT, "outputs", "sweeps"),
                        help="Output directory")
    parser.add_argument("--experiment", default=None,
                        help="Aim experiment name (default: <sweep>_<timestamp>)")
    parser.add_argument("--gpus", "-g", type=int, nargs="?", const=0, default=None,
                        help="Number of GPUs (0 = all visible, default = 1)")
    parser.add_argument("--jobs-per-gpu", "-j", type=int, default=1,
                        help="Concurrent jobs per GPU (default: 1)")
    parser.add_argument("--workers", default=None,
                        help="SSH targets for remote dispatch (comma-separated or @file)")
    parser.add_argument("--remote-dir", default=None,
                        help="Repo path on workers (default: same as local)")
    parser.add_argument("--aim-server", default=None,
                        help="Aim tracking server host:port (default: auto-detect)")
    parser.add_argument("--sync-artifacts", action="store_true",
                        help="Rsync dump_folder back from workers after each job")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without execution")

    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]

    if sweep_name is None:
        info = sweeps[args.sweep]
        sweep_name, options, default_base = args.sweep, info["options"], info["base_config"]
        exclude_fn = info.get("exclude")
        extra_flags = info.get("extra_flags", [])
        validate_options(options)

    base_config = args.base_config or default_base
    if not base_config:
        sweep_print("Error: no base config. Use --base_config or set BASE_CONFIG in sweep file.")
        sys.exit(1)
    if not os.path.exists(base_config):
        sweep_print(f"Error: config not found: {base_config}")
        sys.exit(1)

    # GPU setup: build list of (worker, gpu_id) slots
    remote_dir = args.remote_dir or REPO_ROOT
    aim_server = args.aim_server

    if args.workers:
        # Remote mode: discover GPUs on each worker via SSH
        worker_list = _parse_workers(args.workers)
        sweep_print(f"Discovering GPUs on {len(worker_list)} workers...")
        gpu_slots = _discover_remote_gpus(worker_list)
        if not gpu_slots:
            sweep_print(f"{_RED}Error: no reachable workers with GPUs{_RESET}")
            sys.exit(1)
        if aim_server is None:
            aim_server = _get_default_aim_server()
        # Respect --gpus as a per-worker limit
        if args.gpus is not None and args.gpus > 0:
            limited = []
            seen = Counter()
            for w, g in gpu_slots:
                if seen[w] < args.gpus:
                    limited.append((w, g))
                    seen[w] += 1
            gpu_slots = limited
        # Show discovered topology
        worker_counts = Counter(w for w, _ in gpu_slots)
        for w, cnt in sorted(worker_counts.items()):
            sweep_print(f"  {_GREEN}OK{_RESET}    {w}: {cnt} GPUs")
        sweep_print(f"Aim server: {aim_server}")
    else:
        # Local mode: same as before
        visible = _visible_devices()
        if not visible:
            visible = [0]
        if args.gpus is None:
            num_gpus = 1
        elif args.gpus == 0:
            num_gpus = len(visible)
        elif args.gpus > len(visible):
            sweep_print(f"Error: requested {args.gpus} GPUs but only {len(visible)} visible")
            sys.exit(1)
        else:
            num_gpus = args.gpus
        gpu_slots = [(None, visible[g % len(visible)]) for g in range(num_gpus)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    experiment = args.experiment or f"{sweep_name}_{timestamp}"
    output_dir = os.path.abspath(args.output_dir)
    exp_dir = os.path.join(output_dir, experiment)
    os.makedirs(exp_dir, exist_ok=True)

    if not args.dry_run:
        _log_file = open(os.path.join(exp_dir, "sweep.log"), "w")

    variations = generate_variations(sweep_name, options, exclude_fn, extra_flags)
    expected = count_expected(options)
    n = len(variations)

    num_total_slots = len(gpu_slots) * args.jobs_per_gpu

    # Header
    sweep_print(f"Base config: {base_config}")
    if expected < n:
        sweep_print(f"Sweep: {sweep_name} ({expected} expected, {n} worst case)")
    else:
        sweep_print(f"Sweep: {sweep_name} ({n} runs)")
    sweep_print(f"Experiment: {experiment}")
    if args.workers:
        sweep_print(f"GPU slots: {len(gpu_slots)}, jobs/GPU: {args.jobs_per_gpu},"
                    f" workers: {num_total_slots}")
    else:
        sweep_print(f"GPUs: {len(gpu_slots)}, jobs/GPU: {args.jobs_per_gpu},"
                    f" workers: {num_total_slots}")
    if extra:
        sweep_print(f"Extra overrides: {' '.join(extra)}")
    if not args.dry_run:
        sweep_print(f"Sweep log: {os.path.join(exp_dir, 'sweep.log')}")
    sweep_print(f"Output: {output_dir}\n")

    # List all variations with colored flags
    for var in variations:
        colored = []
        for ci, (key, val) in enumerate(var["combo"].items()):
            flags = var["effective_options"].get(key, {}).get("_flags", {}).get(val, [])
            if flags:
                color = _DIM_COLORS[ci % len(_DIM_COLORS)]
                colored.append(f"{color}{' '.join(flags)}{_RESET}")
        sweep_print(f"{_GREEN}{var['name']}{_RESET}: {' '.join(colored)}")
        rd = os.path.join(output_dir, experiment, var["name"])
        sweep_print(f"{' '.join(build_cmd(base_config, var['overrides'], rd, var['name'], extra))}\n")

    if args.dry_run:
        sweep_print(f"\n{'=' * 80}")
        if expected < n:
            sweep_print(f"DRY RUN — {expected} expected ({n} worst case)")
        else:
            sweep_print(f"DRY RUN — {n} runs would be executed")
        sweep_print(f"{'=' * 80}")
        return

    sweep_print(f"\n{'=' * 80}")
    sweep_print("Starting sweep")
    sweep_print(f"{'=' * 80}\n")

    results, skipped, elapsed = run_sweep(
        variations, base_config, output_dir, experiment, expected, extra,
        gpu_slots, args.jobs_per_gpu, remote_dir=remote_dir,
        aim_server=aim_server, sync_artifacts=args.sync_artifacts)

    has_failures = print_summary(results, skipped, elapsed)
    sweep_print(f"\nOutput: {output_dir}")
    if _log_file:
        _log_file.close()
        print(f"Sweep log: {os.path.join(exp_dir, 'sweep.log')}")

    if has_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
