#!/tmp/eco/run_sweep.py
# Section 4: Adam β₁/β₂ sensitivity analysis
# Dimensions: β₁(5) × β₂(4) × ±ECO (40 runs)
BASE_CONFIG = "configs/experiments/master.toml"
ECO_OFF = ["--eco.no-enabled", "--eco.quant-dtype", "bf16"]
ECO_ON = ["--eco.enabled", "--eco.quant-dtype", "fp8"]

BETA1_VALUES = [0.8, 0.85, 0.9, 0.95, 0.99]
BETA2_VALUES = [0.95, 0.98, 0.99, 0.999]

OPTIONS = {
    "beta1": {
        "values": BETA1_VALUES,
        "flags": {v: ["--optimizer.beta1", str(v)] for v in BETA1_VALUES},
        "name": "b1",
    },
    "beta2": {
        "values": BETA2_VALUES,
        "flags": {v: ["--optimizer.beta2", str(v)] for v in BETA2_VALUES},
        "name": "b2",
    },
    "eco_enabled": {
        "values": [False, True],
        "flags": {False: ECO_OFF, True: ECO_ON},
        "name": "eco",
    },
}
