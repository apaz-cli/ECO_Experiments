#!/usr/bin/env mlsweep_run
# BF16 optimizer state dtype sweep
# Dimensions: BF16 treatments × speed(LBS, AC)
# Speed options are singular — resolved in the first few runs, then locked in.
#
# Fixed token budget per run: steps are adjusted per batch size so that
# every run sees the same number of tokens (default: 100*N tokens).
# Warmup is always 10% of steps, matching the paper.
#
# Treatments:
#   bf16_nomw      — BF16 forward, no master weights, FP32 optimizer states
#   bf16_eco       — BF16 forward + ECO, no master weights, FP32 optimizer states
#   bf16_optim     — BF16 forward, no master weights, BF16 optimizer states
#   bf16_optim_eco — BF16 forward + ECO, no master weights, BF16 optimizer states
#
# Default model is 30M (from master.toml). Override for other sizes:
#   python run_sweep.py --sweep bf16_optim -- --model.flavor 430M

from _common import SPEED_OPTIONS
from _treatments import TREATMENTS, _flags

COMMAND = ["bash", "run_config.sh", "configs/experiments/master.toml"]

# Select the four BF16 treatments
SELECTED_TREATMENTS = {
    "bf16_nomw": TREATMENTS["bf16_nomw"],
    "bf16_eco": TREATMENTS["bf16_eco"],
    "bf16_optim": TREATMENTS["bf16_optim"],
    "bf16_optim_eco": TREATMENTS["bf16_optim_eco"],
}

OPTIONS = {
    **SPEED_OPTIONS,
    ".treatment": {
        "values": list(SELECTED_TREATMENTS.keys()),
        "flags": {
            name: _flags(*args) for name, args in SELECTED_TREATMENTS.items()
        },
        "name": "tmt",
    },
}