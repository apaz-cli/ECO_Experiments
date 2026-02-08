#!/tmp/eco/run_sweep.py
BASE_CONFIG = "configs/debug/baseline.toml"
ECO_OFF = ["--optimizer.name", "AdamW", "--eco.no-enabled"]
ECO_ON = ["--optimizer.name", "ECOAdamW", "--eco.enabled"]

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
