#!/tmp/eco/run_sweep.py
# Smoke test: 2x2x2x2 = 16 runs on debugmodel.
# Validates that all ECO config combos run without blowing up.
BASE_CONFIG = "configs/debug/baseline.toml"
ECO_OFF = ["--optimizer.name", "AdamW", "--eco.no-enabled"]
ECO_ON = ["--optimizer.name", "ECOAdamW", "--eco.enabled"]

OPTIONS = {
    "eco_enabled": {
        "values": [False, True],
        "flags": {False: ECO_OFF, True: ECO_ON},
        "name": "eco",
    },
    "stochastic_rounding": {
        "values": [False, True],
        "flags": {
            False: ["--eco.no-stochastic-rounding"],
            True: ["--eco.stochastic-rounding"],
        },
        "name": "sr",
    },
    "optim_state_dtype": {
        "values": ["fp32", "bf16"],
        "flags": {
            "fp32": ["--eco.optim-state-dtype", "fp32"],
            "bf16": ["--eco.optim-state-dtype", "bf16"],
        },
        "name": "osdt",
    },
    "optim_compute_dtype": {
        "values": ["fp32", "bf16"],
        "flags": {
            "fp32": ["--eco.optim-compute-dtype", "fp32"],
            "bf16": ["--eco.optim-compute-dtype", "bf16"],
        },
        "name": "ocdt",
    },
}
