#!/tmp/eco/run_sweep.py
BASE_CONFIG = "configs/debug/baseline.toml"
ECO_OFF = ["--optimizer.name", "AdamW", "--eco.no-enabled"]
ECO_ON = ["--optimizer.name", "ECOAdamW", "--eco.enabled"]

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
        "name": "opt",
    },
}
