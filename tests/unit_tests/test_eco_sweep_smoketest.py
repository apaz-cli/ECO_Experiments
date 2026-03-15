"""
Smoke test: runs all 16 ECO config combinations on the debug model
for 5 steps each in parallel, asserting none crash.

This is a functional test that invokes mlsweep_run → torchrun, so it
requires a GPU and the full training stack. Run with:

    pytest tests/unit_tests/test_eco_sweep_smoketest.py -v
"""

import os
import re
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_all_configs_run(tmp_path):
    mlsweep_run = os.path.join(os.path.dirname(sys.executable), "mlsweep_run")
    cmd = [
        mlsweep_run,
        os.path.join(REPO_ROOT, "sweeps", "debug_smoke.py"),
        "--output_dir", str(tmp_path),
        "-g",
        "-j", "60",
        "--",
        "--training.steps", "5",
        "--metrics.no-enable-exp",
        "--metrics.no-enable-tensorboard",
    ]

    result = subprocess.run(
        cmd, cwd=REPO_ROOT,
        capture_output=True, text=True,
        timeout=1800,
    )

    # Print output for visibility in pytest -v
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, (
        f"Sweep failed (exit {result.returncode}).\n"
        f"stdout tail:\n{result.stdout[-3000:]}\n"
        f"stderr tail:\n{result.stderr[-1000:]}"
    )

    # Verify all runs passed by checking "N/N OK" in output
    match = re.search(r"(\d+)/(\d+) OK", result.stdout)
    assert match and match.group(1) == match.group(2), (
        f"Not all runs passed.\nstdout tail:\n{result.stdout[-3000:]}"
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_path:
        test_all_configs_run(tmp_path)
    print("PASSED")
