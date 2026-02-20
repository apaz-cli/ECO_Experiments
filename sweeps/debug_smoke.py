#!/tmp/eco/run_sweep.py
# Smoke test: one run per treatment on debugmodel, for both Adam and Muon.
# Validates that all treatment configs (paper + extra) run without blowing up.
#
# Muon runs sweep all 4 ECO approaches. Adam runs don't use eco.approach —
# the "approach" dim is nested as a sub-option under optimizer="muon", so it
# only appears in Muon combos (55 total: 11 Adam + 44 Muon).
#
# Note: treatments with activation_quant=True set --eco.activation-dtype fp8
# but baseline.toml has no model.converters, so activation quant is a no-op.
#
# Learning rates (NanoGPT speedrun style):
#   - Muon LR: 0.02 (for 2D weight matrices)
#   - Adam LR: 3e-4 (for 1D params in Muon runs, and all params in Adam runs)
from _treatments import TREATMENTS, _flags

BASE_CONFIG = "configs/debug/baseline.toml"

WD = "0.1"
MUON_LR = "0.02"
ADAM_LR = "3e-4"

MUON_BASE_FLAGS = [
    "--optimizer.name", "ECOMuon",
    "--optimizer.muon.lr", MUON_LR,
    "--optimizer.adamw.lr", ADAM_LR,
    "--optimizer.muon.weight_decay", WD,
    "--optimizer.adamw.weight_decay", WD,
]

ADAM_BASE_FLAGS = [
    "--optimizer.adamw.lr", ADAM_LR,
]

MUON_APPROACHES = ["pre_ns", "naive_sgdm", "frobenius", "jacobian"]

OPTIONS = {
    ".optimizer": {
        "name": "opt",
        ".adam": {
            "flags": ADAM_BASE_FLAGS,
        },
        ".muon": {
            "flags": MUON_BASE_FLAGS,
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
