#!/tmp/eco/run_sweep.py
ECO_OFF = ["--eco.enabled=false", "--model.converters", ""]
ECO_ON = ["--eco.enabled=true", "--model.converters", "quantize.linear.float8,eco"]

BETA1_VALUES = [0.8, 0.85, 0.9, 0.95, 0.99]
BETA2_VALUES = [0.95, 0.98, 0.99, 0.999]

OPTIONS = {
    "beta1": {
        "values": BETA1_VALUES,
        "flags": {v: [f"--optimizer.beta1={v}"] for v in BETA1_VALUES},
        "name": "b1",
    },
    "beta2": {
        "values": BETA2_VALUES,
        "flags": {v: [f"--optimizer.beta2={v}"] for v in BETA2_VALUES},
        "name": "b2",
    },
    "eco_enabled": {
        "values": [False, True],
        "flags": {False: ECO_OFF, True: ECO_ON},
        "name": "eco",
    },
}
