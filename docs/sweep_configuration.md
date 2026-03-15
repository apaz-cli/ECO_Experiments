# Sweep Configuration Guide

## Overview

Sweep definitions are Python files in the `sweeps/` directory that define a grid of hyperparameter combinations to run. Each sweep file is directly executable via a shebang pointing to `run_sweep.py`, or can be invoked via `python run_sweep.py --sweep <name>`.

## Sweep File Format

Every sweep file must define two top‑level variables and may optionally define an `EXCLUDE` predicate:

```python
BASE_CONFIG = "path/to/base.toml"
OPTIONS = { ... }

# Optional: filter out meaningless combinations
def EXCLUDE(combo: dict) -> bool: ...
```

### `BASE_CONFIG`

Path to a TOML configuration file that serves as the base for all runs in the sweep. The path is relative to the repository root.

### `OPTIONS` Dictionary

`OPTIONS` is a dictionary where **each key starts with `.`** to mark it as a dimension. Within a dimension spec, keys without `.` are metadata; keys with `.` are sub-dimensions.

Each dimension spec supports these metadata keys:

| Key | Type | Description |
|-----|------|-------------|
| `"name"` | `str` or `None` | Short identifier for run names (e.g., `"lr"`). Defaults to the dim key without the leading dot. `None` suppresses the name from the run name entirely. |
| `"monotonic"` | `str` or `None` | (Optional) `"increasing"`, `"decreasing"`, or `None`. Determines skip direction on failure. |
| `"singular"` | `bool` | (Optional) If `True`, skip all other values once one succeeds. Default `False`. |

There are two dimension types based on content:

#### Value Dims (explicit sweep)

Has a `"values"` list. Sweeps over those values with per-value CLI flags:

| Key | Type | Description |
|-----|------|-------------|
| `"values"` | `list` | All possible values for this dimension. |
| `"flags"` | `dict` or `str` | Mapping from each value to CLI args (`{value: [args...], ...}`) or string shorthand (see below). |

#### Branch Dims (mutually exclusive cases)

Has **no** `"values"`, but has dot-prefixed sub-dim keys. Each sub-dim is a distinct branch — when that branch is selected, its `"flags"` are applied and its own sub-dims expand:

```python
".optimizer": {
    "name": "opt",           # metadata
    ".adam": {               # branch — selected exclusively or exclusively
        "flags": ADAM_FLAGS, # applied when this branch is chosen
    },
    ".muon": {
        "flags": MUON_FLAGS,
        ".approach": {       # sub-dim: only expands within the muon branch
            "values": ["pre_ns", "frobenius", "jacobian"],
            "flags": "--eco.approach",
            "name": "a",
        },
    },
}
```

This produces:
- `(optimizer=adam, treatment=*)` — 2 combos, no `approach` dimension
- `(optimizer=muon, treatment=*, approach=*)` — 6 combos

Total: **8** (vs. 12 with a flat `EXCLUDE`-based approach).

**Run naming** uses `.` to signal ancestry and `_` to separate peer dimensions:
- Adam: `sweep_optadam_tmtbf16`
- Muon: `sweep_optmuon.apre_ns_tmtbf16` (approach dotted onto optmuon)

Branch dims support arbitrary nesting depth. Sub-dim keys are validated against ancestor and sibling dim names to prevent collisions.

### Flags Shorthand

For the common pattern of passing a flag name with each value, you can use a string shorthand instead of a dict comprehension:

```python
# Verbose (dict):
"flags": {v: ["--optimizer.lr", str(v)] for v in [1e-4, 3e-4, 1e-3]}

# Shorthand (string):
"flags": "--optimizer.lr"
```

The shorthand automatically generates `{v: [flag_string, str(v)] for v in values}`. This works for any dimension where each value is passed as an argument to the same flag.

**Example:**

```python
OPTIONS = {
    ".learning_rate": {
        "values": [1e-4, 3e-4, 1e-3],
        "flags": "--optimizer.lr",  # shorthand for {v: ["--optimizer.lr", str(v)] for v in values}
        "name": "lr",
    },
    ".local_batch_size": {
        "values": [64, 32, 16, 8],
        "flags": "--training.local_batch_size",
        "name": "lbs",
        "monotonic": "decreasing",  # try large first, go down until it fits in GPU memory
    },
}
```

## `EXCLUDE` Predicate (Static Filtering)

A sweep file may optionally define an `EXCLUDE` function to statically filter out combinations before any runs are launched:

```python
def EXCLUDE(combo: dict) -> bool:
    """Return True to exclude this combination from the sweep."""
    return combo["weight_decay"] == 0.0 and combo["include_wd_injection"] is True
```

The function receives a `combo` dict mapping dimension names (without the leading dot) to their values for a given combination. If it returns `True`, that combination is dropped from the sweep entirely — it won't appear in `--dry-run` output and won't count toward the run total.

Use `EXCLUDE` for **simple combo exclusions** — cases where one combination of existing dimensions is logically meaningless (e.g., toggling a weight-decay correction when weight decay is zero). For combinations that might **fail at runtime** (e.g., OOM), use monotonic/singular skipping instead.

**Prefer branch dims over `EXCLUDE`** when an entire dimension only exists for certain cases of another dimension — e.g. an `approach` dim that only applies to Muon runs. Branch dims cleanly express this as tree structure rather than as a filter on a flat product.

## Monotonic Skipping (Skip on Failure)

When an option has `"monotonic"` set to `"increasing"` or `"decreasing"` and a run **fails**, the sweep script will automatically skip **future runs** where that dimension's value is worse (larger or smaller, depending on direction) while **all other dimensions stay the same**.

- **Direction `"increasing"`**: failure likelihood increases with larger values. Example: if a run with `steps=1000` fails, any run with `steps=10000` (and identical other settings) will be skipped.
- **Direction `"decreasing"`**: failure likelihood increases with smaller values. Example: when tuning local batch size for GPU memory, if `local_batch_size=32` OOMs, all larger batch sizes (64, 128, etc.) will also OOM, so they can be skipped.

## Singular Skipping (Skip on Success)

When an option has `"singular": True` and a run **succeeds**, the sweep script will automatically skip **all other values** in that dimension (with all other dimensions held constant).

- **Use case**: Dimensions where you only need to find ONE working value, not explore all of them
- **Example**: Local batch size tuning - if `local_batch_size=64` succeeds, there's no need to try 32, 16, 8 (smaller is just slower)
- **Dimension ordering**: Singular dimensions are automatically sorted to vary **slowest** in the cartesian product, so other dimensions are explored first. This maximizes parallelism when using `-g` with multiple GPUs.
- **Expected run count**: The sweep displays "expected" runs treating singular dimensions as contributing only 1 value

### Combining Monotonic and Singular

These can work together! Example for local batch size:

```python
".local_batch_size": {
    "values": [512, 256, 128, 64, 32, 16, 8],
    "monotonic": "decreasing",  # if 128 OOMs, skip 256, 512
    "singular": True,            # if 64 succeeds, skip 32, 16, 8
}
```

**Important:** Both monotonic and singular skipping work **only in sequential mode** (`--gpus 1 --jobs-per-gpu 1`). In parallel mode, results are recorded but skipping does not happen dynamically because runs are launched concurrently.

## Run Naming

Each run is given a unique name constructed from its dimension values:

```
{sweep_name}_{dim1_name}{value1}_{dim2_name}{value2}_…
```

The `name` field on a dim spec provides the prefix; the value (or branch key) is appended. If `name` is omitted, the dim key without the leading dot is used. If `name` is `None`, that dim contributes nothing to the run name.

Boolean values are abbreviated as `T` (True) and `F` (False).

When branch dims are in use, the branch sub-dim's name is **dotted** onto its parent dimension's name segment, and peer dimensions are still separated by `_`:

```
{sweep_name}_opt{optimizer}.a{approach}_tmt{treatment}
```

For example, from `debug_smoke`:

- Adam run: `debug_smoke_optadam_tmtbf16`
- Muon run: `debug_smoke_optmuon.apre_ns_tmtbf16`
- Muon run: `debug_smoke_optmuon.afrobenius_tmtfp8_eco_sr`

The `.` signals "this segment is a branch of the preceding segment." For three-level nesting: `sweep_optmuon.afrobenius.subX_tmt...`

The name is used as the `--job.run_name` and `--job.description` passed to `run_config.sh`.

## Invocation Modes

### 1. Direct Sweep Runner

```bash
python run_sweep.py --sweep <name> [options] [-- extra_overrides...]
```

Example:

```bash
python run_sweep.py --sweep beta --base_config configs/scaling_law/100m_eco.toml -g 2 -j 4 -- --training.steps 1000
```

### 2. Shebang Mode (Sweep File as Interpreter)

Make the sweep file executable and run it directly:

```bash
chmod +x sweeps/beta.py
./sweeps/beta.py --base_config configs/experiments/master.toml --gpus 2
```

The shebang line (`#!/usr/bin/env mlsweep_run`) causes the sweep file to be interpreted by `run_sweep.py`. The script detects this mode, loads the `OPTIONS` from the same file, and processes the remaining command‑line arguments.

## Command‑Line Options

| Option | Description |
|--------|-------------|
| `--sweep <name>` | Name of the sweep (file `sweeps/<name>.py`). Required when not using shebang mode. |
| `--base_config <path>` | Override the sweep's `BASE_CONFIG`. |
| `--output_dir <dir>` | Directory where run outputs will be stored (default: `outputs/sweeps`). |
| `--experiment <name>` | Aim experiment name (default: `<sweep_name>_<timestamp>`). |
| `-g [N]`, `--gpus [N]` | Number of GPUs to spread across. `-g` alone means "use all visible GPUs". Default is 1. |
| `-j N`, `--jobs-per-gpu N` | Concurrent jobs per GPU (default: 1). Total parallel workers = GPUs × jobs‑per‑GPU. |
| `--dry-run` | Print the commands that would be executed, but do not run them. |
| `--` | Pass extra arguments to every training run (useful for overriding config fields). |

## GPU Management

The script respects `CUDA_VISIBLE_DEVICES`. If no GPUs are visible (or `nvidia-smi` fails), it defaults to GPU 0 (CPU fallback). GPUs are allocated to jobs in a round‑robin fashion; oversubscription is intentional and allowed.

## Output Structure

Each run creates a subdirectory:

```
{output_dir}/{experiment_name}/{run_name}/
```

Inside, `training.log` contains the stdout/stderr of that run. The script prints a colored summary to the terminal (and a plain‑text version to the `--sweep-log` file if requested).

## Example Sweep Files

### Optimizer × Treatment Sweep with Branch Dim (`sweeps/debug_smoke.py`)

Adam and Muon share the `treatment` dimension, but `approach` (ECO injection strategy)
only applies to Muon. Using a branch dim gives 55 runs instead of 88 with `EXCLUDE`.

```python
#!/usr/bin/env mlsweep_run
from _treatments import TREATMENTS, _flags

BASE_CONFIG = "configs/debug/baseline.toml"

MUON_APPROACHES = ["pre_ns", "naive_sgdm", "frobenius", "jacobian"]

OPTIONS = {
    ".optimizer": {
        "name": "opt",
        ".adam": {
            "flags": ADAM_BASE_FLAGS,
        },
        ".muon": {
            "flags": MUON_BASE_FLAGS,
            # .approach only expands within the muon branch
            ".approach": {
                "values": MUON_APPROACHES,
                "flags": {a: ["--eco.approach", a] for a in MUON_APPROACHES},
                "name": "a",
            },
        },
    },
    ".treatment": {
        "values": list(TREATMENTS.keys()),
        "flags": {name: _flags(*args) for name, args in TREATMENTS.items()},
        "name": "tmt",
    },
}
```

Produces names like:
- `debug_smoke_optadam_tmtbf16` (Adam, 11 per-treatment runs)
- `debug_smoke_optmuon.apre_ns_tmtbf16` (Muon+approach, 44 runs)

### Local Batch Size Tuning Sweep

```python
#!/usr/bin/env mlsweep_run

BASE_CONFIG = "configs/experiments/master.toml"

# Try largest first, go down until it fits in GPU memory
LOCAL_BATCH_SIZES = [512, 256, 128, 64, 32, 16, 8]

OPTIONS = {
    ".local_batch_size": {
        "values": LOCAL_BATCH_SIZES,
        "flags": "--training.local_batch_size",  # shorthand!
        "name": "lbs",
        "monotonic": "decreasing",  # if 128 OOMs, skip 256, 512
        "singular": True,            # if 64 succeeds, skip 32, 16, 8
    },
    ".learning_rate": {
        "values": [1e-4, 3e-4, 1e-3],
        "flags": "--optimizer.lr",  # shorthand!
        "name": "lr",
    },
}
```

With 7 batch sizes and 3 learning rates, this generates 21 total variations, but expects only **3 runs** (one successful batch size per learning rate) in the best case.

## Troubleshooting

- **`Error: no base config specified`** – Provide `--base_config` or define `BASE_CONFIG` in the sweep file.
- **`Dimension key '...' must start with '.'`** – All keys in `OPTIONS` (and in branch dim specs) must begin with `.`. Metadata keys inside a dim spec (name, flags, values, singular, monotonic) do not.
- **Monotonic skipping not working in parallel** – This is expected; monotonic skipping only works in sequential mode (`-g 1 -j 1`).
- **Shebang mode fails with "No such file or directory"** – Ensure `/tmp/mlsweep_shims/run_sweep.py` exists (run `./start_aim_server.sh` once to create the wrapper).

## See Also

- `run_sweep.py` – full implementation and inline documentation.
- `eco_experiments_plan.md` – overall experiment plan, including sweep philosophy.
- Existing sweep files in `sweeps/` for concrete examples.
