#!/usr/bin/env mlsweep_run
# Muon Table 1 reproduction: all 7 paper treatments run twice — once with Adam
# (from master.toml) and once with Muon (from muon.toml) — for direct comparison.
#
# The ECO approach for Muon is fixed to the winning approach from the
# muon_approaches sweep.
#
# Learning rates (NanoGPT speedrun style):
#   - Muon LR: 0.02 (for 2D weight matrices)
#   - Adam LR: 3e-4 (for 1D params in Muon runs, and all params in Adam runs)
#
# Default model is 100M. Override for other sizes:
#   python run_sweep.py --sweep muon_repro -- --model.flavor 430M

from _common import SPEED_OPTIONS
from _treatments import PAPER_TREATMENTS, _flags

COMMAND = ["bash", "run_config.sh", "configs/experiments/master.toml"]

# lr, beta1, beta2, eps inherited from master.toml (shared with Adam fallback for 1D params).
# momentum is Muon-only (default 0.95). weight_decay typically 0 for Muon.
MUON_FLAGS = [
    "--optimizer.name", "ECOMuon",
    "--optimizer.muon.lr", "0.02",      # NanoGPT style
    "--optimizer.muon.weight_decay", "0.1",
    "--optimizer.adamw.lr", "3e-4",     # Adam LR for 1D params
    "--eco.approach", "frobenius",
]

ADAM_FLAGS = [
    "--optimizer.adamw.lr", "3e-4",     # Same LR as Adam params in Muon runs
]

OPTIMIZERS = {
    "adam": ADAM_FLAGS,
    "muon": MUON_FLAGS,
}

OPTIONS = {
    **SPEED_OPTIONS,
    ".optimizer": {
        "values": list(OPTIMIZERS.keys()),
        "flags": OPTIMIZERS,
        "name": "opt",
    },
    ".treatment": {
        "values": list(PAPER_TREATMENTS.keys()),
        "flags": {
            name: _flags(*args) for name, args in PAPER_TREATMENTS.items()
        },
        "name": "tmt",
    },
}
