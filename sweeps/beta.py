#!/tmp/eco/run_sweep.py
# Section 4: Adam β₁ sensitivity analysis
# The ECO injection coefficient (1 − 1/β₁) depends directly on β₁.
# β₂ is fixed at 0.98 (paper's choice, line 305).
from _treatments import TREATMENTS, _flags

BASE_CONFIG = "configs/experiments/master.toml"
EXTRA_FLAGS = _flags(*TREATMENTS["fp8_eco_sr"])

OPTIONS = {
    "beta1": {
        "values": [0.8, 0.85, 0.9, 0.95, 0.99],
        "flags": "--optimizer.beta1",
        "name": "b1",
    },
}
