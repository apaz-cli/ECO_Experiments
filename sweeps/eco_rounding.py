#!/tmp/eco/run_sweep.py
# Section 2: ECO with stochastic rounding ablation
# Dimensions: ±ECO × ±stochastic rounding (4 runs)
BASE_CONFIG = "configs/experiments/master.toml"
ECO_OFF = ["--eco.no-enabled", "--eco.quant-dtype", "bf16"]
ECO_ON = ["--eco.enabled", "--eco.quant-dtype", "fp8"]

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
}
