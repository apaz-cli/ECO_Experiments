#!/tmp/eco/run_sweep.py
# Section 2: ECO vs Baseline comparison
# Dimensions: ±ECO (2 runs)
BASE_CONFIG = "configs/experiments/master.toml"
ECO_OFF = ["--eco.no-enabled", "--eco.quant-dtype", "bf16", "--eco.activation-dtype", "none", "--eco.no-stochastic-rounding"]
ECO_ON = ["--eco.enabled", "--eco.quant-dtype", "fp8", "--eco.activation-dtype", "fp8", "--eco.no-stochastic-rounding"]

OPTIONS = {
    "eco_enabled": {
        "values": [False, True],
        "flags": {False: ECO_OFF, True: ECO_ON},
        "name": "eco",
    },
}
