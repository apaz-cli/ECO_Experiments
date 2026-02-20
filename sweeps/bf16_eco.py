#!/tmp/eco/run_sweep.py
# BF16 vs BF16+ECO comparison sweep
# Dimensions: bf16 vs bf16_eco × speed(LBS, AC)
# Speed options are singular — resolved in the first few runs, then locked in.
#
# Fixed token budget per run: steps are adjusted per batch size so that
# every run sees the same number of tokens (default: 100*N tokens).
# Warmup is always 10% of steps, matching the paper.
#
# Treatments:
#   bf16      — BF16 forward, FP32 master weights (reference baseline)
#   bf16_eco  — BF16 forward + ECO, no master weights
#
# Default model is 30M (from master.toml). Override for other sizes:
#   python run_sweep.py --sweep bf16_eco -- --model.flavor 430M

from _common import SPEED_OPTIONS
from _treatments import TREATMENTS, _flags

BASE_CONFIG = "configs/experiments/master.toml"

# Select only the two BF16 treatments
SELECTED_TREATMENTS = {
    "bf16": TREATMENTS["bf16"],
    "bf16_eco": TREATMENTS["bf16_eco"],
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