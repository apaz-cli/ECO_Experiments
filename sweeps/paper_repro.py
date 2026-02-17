#!/tmp/eco/run_sweep.py
# LR and batch size sweep for finding a good base config.
#
# Dimensions: treatment × speed(LBS, AC)
# Speed options are singular — resolved in the first few runs, then locked in.
#
# Fixed token budget per run: steps are adjusted per batch size so that
# every run sees the same number of tokens (default: 100*N tokens).
# Warmup is always 10% of steps, matching the paper.
#
# Treatments (paper §4.1):
#
#   With master weights (FP32 accumulation):
#     bf16           — BF16 forward, FP32 master weights (reference baseline)
#     fp8_mw_rtn     — FP8 forward (RTN), FP32 master weights
#     fp8_mw_sr      — FP8 forward (SR weights, RTN activations), FP32 MW
#
#   Without master weights (FP8 accumulation):
#     fp8_rtn        — FP8 forward (RTN), no master weights
#     fp8_sr         — FP8 forward (SR weights, RTN activations), no MW
#     fp8_eco_rtn    — FP8 forward (RTN) + ECO, no MW
#     fp8_eco_sr     — FP8 forward (SR weights, RTN activations) + ECO, no MW
#
# Default model is 30M (from master.toml). Override for other sizes:
#   python run_sweep.py --sweep paper_repro -- --model.flavor 430M

from _common import SPEED_OPTIONS
from _treatments import PAPER_TREATMENTS, _flags

BASE_CONFIG = "configs/experiments/master.toml"

OPTIONS = {
    **SPEED_OPTIONS,
    "treatment": {
        "values": list(PAPER_TREATMENTS.keys()),
        "flags": {
            name: _flags(*args) for name, args in PAPER_TREATMENTS.items()
        },
        "name": "tmt",
    },
}
