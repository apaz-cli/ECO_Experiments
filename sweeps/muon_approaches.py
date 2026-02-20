#!/tmp/eco/run_sweep.py
# Section 5: Muon ECO Approach Comparison
# Dimensions: fp8_eco_sr treatment × 4 ECO approaches (pre_ns, naive_sgdm, frobenius, jacobian) × speed(LBS, AC)
# Tests all four ECO injection strategies for Muon optimizer with FP8+SR base configuration.
#
# Base treatment: fp8_eco_sr (ECO enabled, FP8 weights, no master weights, FP32 optimizer states,
#                              stochastic rounding, FP8 activations)
# Overrides: optimizer.name = "ECOMuon" (already in muon.toml), eco.approach = <approach>
#
# Default model is 100M (from muon.toml). Override for other sizes:
#   python run_sweep.py --sweep muon_approaches -- --model.flavor 430M
from _common import SPEED_OPTIONS
from _treatments import TREATMENTS, _flags

BASE_CONFIG = "configs/experiments/muon.toml"
EXTRA_FLAGS = _flags(*TREATMENTS["fp8_eco_sr"])

APPROACHES = ["pre_ns", "naive_sgdm", "frobenius", "jacobian"]
OPTIONS = {
    **SPEED_OPTIONS,
    ".eco_approach": {
        "values": APPROACHES,
        "flags": "--eco.approach",
        "name": "a",
    },
}
