#!/tmp/eco/run_sweep.py
# Section 3: BF16 optimizer states
# Dimensions: ±ECO × FP32/BF16 optimizer states (4 runs)
BASE_CONFIG = "configs/experiments/master.toml"
ECO_OFF = ["--eco.no-enabled", "--eco.quant-dtype", "bf16"]
ECO_ON = ["--eco.enabled", "--eco.quant-dtype", "fp8"]

OPTIONS = {
    "eco_enabled": {
        "values": [False, True],
        "flags": {False: ECO_OFF, True: ECO_ON},
        "name": "eco",
    },
    "optim_state_dtype": {
        "values": ["fp32", "bf16"],
        "flags": {
            "fp32": ["--eco.optim-state-dtype", "fp32"],
            "bf16": ["--eco.optim-state-dtype", "bf16"],
        },
        "name": "osdt",
    },
}
