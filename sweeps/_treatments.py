#!/tmp/eco/run_sweep.py
# Common treatment definitions for the ECO paper experiments.
#
# Treatments (paper §4.1):
#
#   With master weights (FP32 accumulation):
#     bf16           — BF16 forward, FP32 master weights, FP32 optimizer states (reference baseline)
#     fp8_mw_rtn     — FP8 forward (RTN), FP32 master weights, FP32 optimizer states
#     fp8_mw_sr      — FP8 forward (SR weights, RTN activations), FP32 MW, FP32 optimizer states
#
#   Without master weights (FP8 accumulation):
#     fp8_rtn        — FP8 forward (RTN), no master weights, FP32 optimizer states
#     fp8_sr         — FP8 forward (SR weights, RTN activations), no MW, FP32 optimizer states
#     fp8_eco_rtn    — FP8 forward (RTN) + ECO, no MW, FP32 optimizer states
#     fp8_eco_sr     — FP8 forward (SR weights, RTN activations) + ECO, no MW, FP32 optimizer states
#
# Extra treatments (not in paper):
#   bf16_nomw       — BF16 forward, no master weights, FP32 optimizer states
#   bf16_eco        — BF16 forward + ECO, no master weights, FP32 optimizer states
#   bf16_optim      — BF16 forward, no master weights, BF16 optimizer states
#   bf16_optim_eco  — BF16 forward + ECO, no master weights, BF16 optimizer states

# (eco_enabled, quant_dtype, master_weights_dtype, optim_state_dtype, stochastic_rounding, activation_quant)

# Paper treatments (§4.1)
PAPER_TREATMENTS = {
    "bf16":           (False, "bf16", "fp32",  "fp32", False, False),  # BF16 baseline (reference)
    "fp8_mw_rtn":     (False, "fp8",  "fp32",  "fp32", False, True),   # FP8 w/ MW + RTN
    "fp8_mw_sr":      (False, "fp8",  "fp32",  "fp32", True,  True),   # FP8 w/ MW + SR
    "fp8_rtn":        (False, "fp8",  None,    "fp32", False, True),   # FP8 w/o MW + RTN
    "fp8_sr":         (False, "fp8",  None,    "fp32", True,  True),   # FP8 w/o MW + SR
    "fp8_eco_rtn":    (True,  "fp8",  None,    "fp32", False, True),   # FP8 w/o MW ECO + RTN
    "fp8_eco_sr":     (True,  "fp8",  None,    "fp32", True,  True),   # FP8 w/o MW ECO + SR (main paper method)
}

# Extra treatments (not in paper)
EXTRA_TREATMENTS = {
    "bf16_nomw":      (False, "bf16", None,    "fp32", False, False),  # BF16 baseline (reference without master weights)
    "bf16_eco":       (True,  "bf16", None,    "fp32", False, False),  # BF16 + ECO
    "bf16_optim":     (False, "bf16", None,    "bf16", False, False),  # BF16 baseline + Optimizer state in bf16, no master weights
    "bf16_optim_eco": (True,  "bf16", None,    "bf16", False, False),  # BF16 baseline + Optimizer state in bf16 + ECO
}

# All treatments
TREATMENTS = {**PAPER_TREATMENTS, **EXTRA_TREATMENTS}

def _flags(eco, quant_dtype, master_weights_dtype, optim_state_dtype, stochastic_rounding, activation_quant):
    """Build CLI flags for a treatment.

    master.toml already has model.converters = ["activation_quant"] and
    eco.activation_dtype = "fp8".  The converter is a no-op when
    activation_dtype = "none", so BF16 treatments just set that flag.
    """
    flags = []
    flags += ["--eco.enabled" if eco else "--eco.no-enabled"]
    flags += ["--eco.quant-dtype", quant_dtype]
    if master_weights_dtype is not None:
        flags += ["--eco.master-weights-dtype", master_weights_dtype]
    flags += ["--eco.optim-state-dtype", optim_state_dtype]
    flags += ["--eco.stochastic-rounding" if stochastic_rounding else "--eco.no-stochastic-rounding"]
    flags += ["--eco.activation-dtype", "fp8" if activation_quant else "none"]
    return flags
