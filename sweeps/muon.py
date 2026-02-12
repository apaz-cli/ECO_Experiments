#!/tmp/eco/run_sweep.py
# Section 5: Muon ECO Approach Comparison
# Dimensions: 4 ECO approaches (pre_ns, naive_sgdm, frobenius, jacobian)
# Tests all four ECO injection strategies for Muon optimizer

BASE_CONFIG = "configs/experiments/muon.toml"

APPROACHES = ["pre_ns", "naive_sgdm", "frobenius", "jacobian"]

OPTIONS = {
    "eco_approach": {
        "values": APPROACHES,
        "flags": {
            approach: ["--eco.approach", approach]
            for approach in APPROACHES
        },
        "name": "approach",
    },
}
