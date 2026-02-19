#!/tmp/eco/run_sweep.py
# Smoke test: one run per treatment on debugmodel, for both Adam and Muon.
# Validates that all treatment configs (paper + extra) run without blowing up.
#
# Muon runs sweep all 4 ECO approaches. Adam runs only run once (approach is
# irrelevant for Adam — EXCLUDE filters out redundant Adam×approach combos).
#
# Note: treatments with activation_quant=True set --eco.activation-dtype fp8
# but baseline.toml has no model.converters, so activation quant is a no-op.
#
# Learning rates (NanoGPT speedrun style):
#   - Muon LR: 0.02 (for 2D weight matrices)
#   - Adam LR: 3e-4 (for 1D params in Muon runs, and all params in Adam runs)
from _treatments import TREATMENTS, _flags

BASE_CONFIG = "configs/debug/baseline.toml"

MUON_BASE_FLAGS = [
    "--optimizer.name", "ECOMuon",
    "--optimizer.muon.lr", "0.02",
    "--optimizer.muon.weight_decay", "0.1",
    "--optimizer.adamw.lr", "3e-4",  # Adam LR for 1D params (embeddings, biases, norms)
]

ADAM_BASE_FLAGS = [
    "--optimizer.adamw.lr", "3e-4",  # Same LR as Adam params in Muon runs
]

OPTIMIZERS = {
    "adam": ADAM_BASE_FLAGS,
    "muon": MUON_BASE_FLAGS,
}

MUON_APPROACHES = ["pre_ns", "naive_sgdm", "frobenius", "jacobian"]

OPTIONS = {
    "optimizer": {
        "values": list(OPTIMIZERS.keys()),
        "flags": OPTIMIZERS,
        "name": "opt",
    },
    "approach": {
        "values": MUON_APPROACHES,
        "flags": {a: ["--eco.approach", a] for a in MUON_APPROACHES},
        "name": "a",
    },
    "treatment": {
        "values": list(TREATMENTS.keys()),
        "flags": {
            name: _flags(*args) for name, args in TREATMENTS.items()
        },
        "name": "tmt",
    },
}


def EXCLUDE(combo):
    # Adam doesn't use eco.approach — only run it with the first approach value
    # to avoid 4x redundant identical Adam runs.
    if combo["optimizer"] == "adam" and combo["approach"] != MUON_APPROACHES[0]:
        return True
    return False
