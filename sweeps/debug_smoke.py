#!/tmp/eco/run_sweep.py
# Smoke test: 2x2x2x2x2 = 32 runs on debugmodel.
# Validates that all ECO config combos run without blowing up.
BASE_CONFIG = "configs/debug/baseline.toml"

OPTIONS = {
    "eco_enabled": {
        "values": [False, True],
        "flags": {
            False: ["--eco.no-enabled"],
            True: ["--eco.enabled"],
        },
        "name": "eco",
    },
    "quant_dtype": {
        "values": ["bf16", "fp8"],
        "flags": {
            "bf16": ["--eco.quant-dtype", "bf16"],
            "fp8": ["--eco.quant-dtype", "fp8"],
        },
        "name": "qdt",
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
