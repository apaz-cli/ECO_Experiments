#!/usr/bin/env mlsweep_run
from _treatments import TREATMENTS, _flags

COMMAND = ["bash", "run_config.sh", "configs/debug/baseline.toml"]

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
    "--optimizer.name", "ECOAdamW",
    "--optimizer.adamw.lr", ADAM_LR,
    "--optimizer.adamw.weight_decay", WD,
]

MUON_APPROACHES = ["pre_ns", "naive_sgdm", "frobenius", "jacobian"]

OPTIONS = {
    ".eco": {
        "flags": {"true": ["--eco.enabled"], "false": ["--eco.no-enabled"]},
        "name": "eco",
    },
    ".quant_dtype": {
        "values": ["bf16", "fp8"],
        "flags": "--eco.quant-dtype",
        "name": "qd",
    },
    ".master_weights_dtype": {
        "values": [None, "bf16", "fp32"],
        "flags": "--eco.master-weights-dtype",
        "name": "mw",
    },
    ".optim_dtype": {
        "values": ["bf16", "fp32"],
        "flags": "--eco.optim-state-dtype",
        "name": "od",
    },
    ".activation_dtype": {
        "values": ["bf16", "fp8"],
        "flags": "--eco.activation-dtype",
        "name": "ad",
    },
    ".stochastic_rounding": {
        "flags": {"true": ["--eco.stochastic-rounding"], "false": ["--eco.no-stochastic-rounding"]},
        "name": "sr",
    },
    ".optimizer": {
        "name": "opt",
        ".adam": {
            "flags": ADAM_BASE_FLAGS,
        },
        ".muon": {
            "flags": MUON_BASE_FLAGS,
            ".approach": {
                "flags": {a: ["--eco.approach", a] for a in MUON_APPROACHES},
                "name": "a",
            },
        },
    },
}
