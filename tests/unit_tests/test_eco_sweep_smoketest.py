"""
Smoke test: runs all 16 ECO config combinations (2x2x2x2) on the debug model
for 5 steps each, asserting none crash.

This is a functional test that invokes torchrun via run_config.sh, so it
requires a GPU and the full training stack. Run with:

    pytest tests/unit_tests/test_sweep_smoke.py -v
"""

import os
import subprocess
import tempfile

import pytest

# Import sweep infrastructure
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from run_sweep import generate_variations, load_sweep_file

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SWEEP_FILE = os.path.join(REPO_ROOT, "sweeps", "debug_smoke.py")

# Extra overrides for testing: 5 steps, no logging
TEST_OVERRIDES = [
    "--training.steps", "5",
    "--metrics.no-enable-aim",
    "--metrics.no-enable-tensorboard",
]


def _run_config(base_config, overrides, run_dir, run_name):
    """Run a single training config via run_config.sh. Returns (success, log)."""
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "training.log")

    cmd = [
        os.path.join(REPO_ROOT, "run_config.sh"),
        base_config,
        "--job.dump_folder", run_dir,
        *overrides,
        *TEST_OVERRIDES,
    ]

    env = os.environ.copy()
    env["NGPU"] = "1"

    with open(log_file, "w") as f:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    with open(log_file) as f:
        log_contents = f.read()

    return result.returncode == 0, log_contents


# Load sweep and generate all 16 variations
_sweep_name, _options, _base_config = load_sweep_file(SWEEP_FILE)
_variations = generate_variations(_sweep_name, _options)


@pytest.mark.parametrize(
    "variation",
    _variations,
    ids=[v["name"] for v in _variations],
)
def test_config_runs_without_error(variation, tmp_path):
    run_name = variation["name"]
    run_dir = str(tmp_path / run_name)

    success, log = _run_config(_base_config, variation["overrides"], run_dir, run_name)

    assert success, f"{run_name} failed.\nLog tail:\n{log[-2000:]}"
