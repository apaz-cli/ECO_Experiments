#!/tmp/eco/run_sweep.py
# Smoke test: one run per treatment on debugmodel, for both Adam and Muon.
# Validates that all treatment configs (paper + extra) run without blowing up.
#
# Note: treatments with activation_quant=True set --eco.activation-dtype fp8
# but baseline.toml has no model.converters, so activation quant is a no-op.
from _treatments import TREATMENTS, _flags

BASE_CONFIG = "configs/debug/baseline.toml"

MUON_FLAGS = [
    "--optimizer.name", "ECOMuon",
    "--optimizer.muon.lr", "0.02",
    "--optimizer.muon.weight_decay", "0.1",
    "--optimizer.adamw.lr", "3e-3",
    "--eco.approach", "frobenius",
]

OPTIMIZERS = {
    "adam": [],
    "muon": MUON_FLAGS,
}

OPTIONS = {
    "optimizer": {
        "values": list(OPTIMIZERS.keys()),
        "flags": OPTIMIZERS,
        "name": "opt",
    },
    "treatment": {
        "values": list(TREATMENTS.keys()),
        "flags": {
            name: _flags(*args) for name, args in TREATMENTS.items()
        },
        "name": "tmt",
    },
}
