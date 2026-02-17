#!/tmp/eco/run_sweep.py
# FP8 weights + BF16 optimizer states sweep.
# Tests whether BF16 optimizer states compose with FP8+ECO (the main ECO setting).
#
# Dimensions: 4 treatments × speed(LBS, AC)
#
# Treatments:
#   fp8_eco_sr             — FP8 weights + ECO + SR, FP32 optimizer states (9 bytes/param)
#   fp8_eco_sr_bf16optim   — FP8 weights + ECO + SR, BF16 optimizer states (5 bytes/param)
#   fp8_sr                 — FP8 weights + SR, no ECO, FP32 optimizer states
#   fp8_sr_bf16optim       — FP8 weights + SR, no ECO, BF16 optimizer states
#
# Default model is 30M (from master.toml). Override for other sizes:
#   python run_sweep.py --sweep fp8_optim -- --model.flavor 430M

from _common import SPEED_OPTIONS
from _treatments import _flags

BASE_CONFIG = "configs/experiments/master.toml"

# (eco_enabled, quant_dtype, master_weights_dtype, optim_state_dtype, stochastic_rounding, activation_quant)
SELECTED = {
    "fp8_eco_sr":             (True,  "fp8", None, "fp32", True, True),
    "fp8_eco_sr_bf16optim":   (True,  "fp8", None, "bf16", True, True),
    "fp8_sr":                 (False, "fp8", None, "fp32", True, True),
    "fp8_sr_bf16optim":       (False, "fp8", None, "bf16", True, True),
}

OPTIONS = {
    **SPEED_OPTIONS,
    "treatment": {
        "values": list(SELECTED.keys()),
        "flags": {name: _flags(*args) for name, args in SELECTED.items()},
        "name": "tmt",
    },
}
