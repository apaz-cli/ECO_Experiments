#!/tmp/eco/run_sweep.py
# Muon Table 1 reproduction: all 7 paper treatments run twice — once with Adam
# (from master.toml) and once with Muon (from muon.toml) — for direct comparison.
#
# The ECO approach for Muon is fixed to the winning approach from the
# muon_approaches sweep.
#
# Default model is 100M. Override for other sizes:
#   python run_sweep.py --sweep muon_repro -- --model.flavor 430M

from _common import SPEED_OPTIONS
from _treatments import PAPER_TREATMENTS, _flags

BASE_CONFIG = "configs/experiments/master.toml"

# TODO: update to winning approach from muon_approaches sweep
# lr, beta1, beta2, eps inherited from master.toml (shared with Adam fallback for 1D params).
# momentum is Muon-only (default 0.95). weight_decay typically 0 for Muon.
MUON_FLAGS = [
    "--optimizer.name", "ECOMuon",
    "--optimizer.muon.lr", "3e-4",
    "--optimizer.muon.weight_decay", "0.1",
    "--optimizer.adamw.lr", "3e-3",
    "--eco.approach", "frobenius",
]

OPTIMIZERS = {
    "adam": [],
    "muon": MUON_FLAGS,
}

OPTIONS = {
    **SPEED_OPTIONS,
    "optimizer": {
        "values": list(OPTIMIZERS.keys()),
        "flags": OPTIMIZERS,
        "name": "opt",
    },
    "treatment": {
        "values": list(PAPER_TREATMENTS.keys()),
        "flags": {
            name: _flags(*args) for name, args in PAPER_TREATMENTS.items()
        },
        "name": "tmt",
    },
}
