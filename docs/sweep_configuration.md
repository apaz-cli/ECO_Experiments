# Sweep Configuration Guide

## Overview

Sweep definitions are Python files in the `sweeps/` directory that define a grid of hyperparameter combinations to run. Each sweep file is directly executable via a shebang pointing to `run_sweep.py`, or can be invoked via `python run_sweep.py --sweep <name>`.

## Sweep File Format

Every sweep file must define two top‑level variables:

```python
BASE_CONFIG = "path/to/base.toml"
OPTIONS = { ... }
```

### `BASE_CONFIG`

Path to a TOML configuration file that serves as the base for all runs in the sweep. The path is relative to the repository root.

### `OPTIONS` Dictionary

`OPTIONS` is a dictionary where each key is a sweep dimension (a meaningful name for the dimension). Each dimension must have the following sub‑keys:

| Key | Type | Description |
|-----|------|-------------|
| `"values"` | `list` | List of all possible values for this dimension. Values can be of any Python type (int, float, bool, str, etc.) |
| `"flags"` | `dict` | Mapping from each value to a list of CLI arguments that will be passed to `run_config.sh`. The list is appended to the command line exactly as given. |
| `"name"` | `str` | Short identifier used to build run names (e.g., `"lr"` for learning rate). |
| `"monotonic"` | `str` or `None` | (Optional) Either `"increasing"`, `"decreasing"`, or `None`. Determines whether failure likelihood increases with larger values (`"increasing"`) or with smaller values (`"decreasing"`). If `None` or omitted, monotonic skipping is disabled for this dimension. |

**Example:**

```python
OPTIONS = {
    "learning_rate": {
        "values": [1e-4, 3e-4, 1e-3],
        "flags": {
            1e-4: ["--optimizer.lr", "1e-4"],
            3e-4: ["--optimizer.lr", "3e-4"],
            1e-3: ["--optimizer.lr", "1e-3"],
        },
        "name": "lr",
    },
    "local_batch_size": {
        "values": [64, 32, 16, 8],
        "flags": {
            64: ["--training.local_batch_size", "64"],
            32: ["--training.local_batch_size", "32"],
            16: ["--training.local_batch_size", "16"],
            8:  ["--training.local_batch_size", "8"],
        },
        "name": "lbs",
        "monotonic": "decreasing",  # try large first, go down until it fits in GPU memory
    },
}
```

## Monotonic Skipping

When an option has `"monotonic"` set to `"increasing"` or `"decreasing"` and a run fails, the sweep script will automatically skip **future runs** where that dimension's value is larger (or smaller, depending on direction) while **all other dimensions stay the same**.

- **Direction `"increasing"`**: failure likelihood increases with larger values. Example: if a run with `steps=1000` fails, any run with `steps=10000` (and identical other settings) will be skipped.
- **Direction `"decreasing"`**: failure likelihood increases with smaller values. Example: when tuning local batch size for GPU memory, if `local_batch_size=32` OOMs, all larger batch sizes (64, 128, etc.) will also OOM, so they can be skipped. This is the most common use case.

**Important:** Monotonic skipping works **only in sequential mode** (`--gpus 1 --jobs-per-gpu 1`). In parallel mode, failures are recorded but skipping does not happen dynamically because runs are launched concurrently. However, the script still records failed combinations, which can be useful for post‑mortem analysis.

## Run Naming

Each run is given a unique name constructed as:

```
ECO_{sweep_name}_{dim1_name}{value1}_{dim2_name}{value2}_…
```

Boolean values are abbreviated as `T` (True) and `F` (False). For example:

- `ECO_debug_smoke_ecoT_qdtfp8_srF_osdtbf16`
- `ECO_lr_bs_lr1e-3_bs256_tmtfp8_eco`

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

The shebang line (`#!/tmp/eco/run_sweep.py`) causes the sweep file to be interpreted by `run_sweep.py`. The script detects this mode, loads the `OPTIONS` from the same file, and processes the remaining command‑line arguments.

## Command‑Line Options

| Option | Description |
|--------|-------------|
| `--sweep <name>` | Name of the sweep (file `sweeps/<name>.py`). Required when not using shebang mode. |
| `--base_config <path>` | Override the sweep’s `BASE_CONFIG`. |
| `--output_dir <dir>` | Directory where run outputs will be stored (default: `outputs/sweeps`). |
| `--experiment <name>` | Aim experiment name (default: `<sweep_name>_<timestamp>`). |
| `-g [N]`, `--gpus [N]` | Number of GPUs to spread across. `-g` alone means “use all visible GPUs”. Default is 1. |
| `-j N`, `--jobs-per-gpu N` | Concurrent jobs per GPU (default: 1). Total parallel workers = GPUs × jobs‑per‑GPU. |
| `--dry-run` | Print the commands that would be executed, but do not run them. |
| `--sweep-log <file>` | Write a plain‑text log of the sweep output (without ANSI colors) to the specified file. |
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

### Simple 2×2 Sweep (`sweeps/debug_smoke.py`)

```python
#!/tmp/eco/run_sweep.py

BASE_CONFIG = "configs/debug/baseline.toml"

OPTIONS = {
    "eco_enabled": {
        "values": [False, True],
        "flags": {
            False: ["--eco.no-enabled"],
            True:  ["--eco.enabled"],
        },
        "name": "eco",
    },
    "quant_dtype": {
        "values": ["bf16", "fp8"],
        "flags": {
            "bf16": ["--eco.quant-dtype", "bf16"],
            "fp8":  ["--eco.quant-dtype", "fp8"],
        },
        "name": "qdt",
    },
}
```

### Local Batch Size Tuning Sweep (`sweeps/local_bs_tune.py`)

```python
#!/tmp/eco/run_sweep.py

BASE_CONFIG = "configs/experiments/master.toml"

# Try largest first, go down until it fits in GPU memory
LOCAL_BATCH_SIZES = [128, 64, 32, 16, 8]

OPTIONS = {
    "local_batch_size": {
        "values": LOCAL_BATCH_SIZES,
        "flags": {
            lbs: ["--training.local_batch_size", str(lbs)]
            for lbs in LOCAL_BATCH_SIZES
        },
        "name": "lbs",
        "monotonic": "decreasing",  # if 64 OOMs, 128 will too
    },
}
```

## Troubleshooting

- **`Error: no base config specified`** – Provide `--base_config` or define `BASE_CONFIG` in the sweep file.
- **`Option '...' missing 'values' list`** – The `OPTIONS` dictionary is malformed; run `validate_options()` on your sweep file to check.
- **Monotonic skipping not working in parallel** – This is expected; monotonic skipping only works in sequential mode (`-g 1 -j 1`).
- **Shebang mode fails with “No such file or directory”** – Ensure `/tmp/eco/run_sweep.py` exists (run `./start_aim_server.sh` once to create the wrapper).

## See Also

- `run_sweep.py` – full implementation and inline documentation.
- `eco_experiments_plan.md` – overall experiment plan, including sweep philosophy.
- Existing sweep files in `sweeps/` for concrete examples.
