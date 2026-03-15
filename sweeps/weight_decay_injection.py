#!/usr/bin/env mlsweep_run
# Weight-decay factor injection sweep.
# Compare ECO injection with/without the (1 - ηλ) weight-decay factor.
# Based on leloykun's derivation: https://leloykun.github.io/ponder/eco/
#
# Dimensions:
#   - weight_decay (0.0, 0.1) to test factor effect
#   - include_weight_decay_in_injection (False, True)
#
# Fixed parameters:
#   - Muon optimizer with approach = jacobian (exact Jacobian-based injection)
#   - Model: 100M (default)
#
# When weight_decay = 0.0, the factor (1 - ηλ) = 1 regardless of flag.
# When weight_decay = 0.1, factor ≈ 0.9997 (η = 3e-4) — a tiny correction.
# The flag toggles whether this factor is included in the injection coefficient.
#
# To run a quick comparison with fewer steps, add extra overrides:
#   -- --training.steps 100 --validation.enable true --validation.freq 50
#
# Example command:
#   python run_sweep.py --sweep weight_decay_injection -- --training.steps 100

COMMAND = ["bash", "run_config.sh", "configs/experiments/muon.toml"]

def EXCLUDE(combo):
    """Skip weight-decay injection when weight_decay is 0 (factor is always 1)."""
    return combo["weight_decay"] == 0.0 and combo["include_weight_decay_in_injection"] == "true"

OPTIONS = {
    ".approach": {
        "flags": {"jacobian": ["--eco.approach", "jacobian"]},
        "name": "app",
    },
    ".weight_decay": {
        "flags": {
            0.0: ["--optimizer.muon.weight_decay", "0.0"],
            0.1: ["--optimizer.muon.weight_decay", "0.1"],
        },
        "name": "wd",
    },
    ".include_weight_decay_in_injection": {
        "flags": {
            "false": ["--eco.no-include_weight_decay_in_injection"],
            "true": ["--eco.include_weight_decay_in_injection"],
        },
        "name": "wdinj",
    },
}
