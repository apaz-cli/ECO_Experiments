#!/tmp/eco/run_sweep.py
# Section 4: Adam β₁ sensitivity analysis
# The ECO injection coefficient (1 − 1/β₁) depends directly on β₁.
# β₂ is fixed at 0.98 (paper's choice, line 305).
# Compare fp8_eco_sr against bf16 baseline at each β₁ to see if ECO shifts the optimum.
from _treatments import TREATMENTS, _flags

BASE_CONFIG = "configs/experiments/master.toml"

SELECTED = {
    "bf16": TREATMENTS["bf16"],
    "fp8_eco_sr": TREATMENTS["fp8_eco_sr"],
}

OPTIONS = {
    "treatment": {
        "values": list(SELECTED.keys()),
        "flags": {name: _flags(*args) for name, args in SELECTED.items()},
        "name": "tmt",
    },
    "beta1": {
        "values": [0.8, 0.85, 0.9, 0.95, 0.99],
        "flags": "--optimizer.beta1",
        "name": "b1",
    },
}
