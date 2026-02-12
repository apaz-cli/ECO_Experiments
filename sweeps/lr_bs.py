#!/tmp/eco/run_sweep.py
# LR and batch size sweep for finding a good base config.
#
# Dimensions: LR(7) × BS(3) × treatment(4) = 84 runs
#
# Fixed token budget per run: steps are adjusted per batch size so that
# every run sees the same number of tokens (500M ≈ 5% of full 100M training).
# Warmup is always 10% of steps, matching the paper.
#
# Treatments:
#   bf16       — BF16 simulated quant, no ECO (baseline)
#   bf16_eco   — BF16 simulated quant + ECO (control: ECO on tiny rounding error)
#   fp8        — FP8 simulated quant, no ECO (naive quantization)
#   fp8_eco    — FP8 simulated quant + ECO (the main method from the paper)
#
# Default model is 100M (from master.toml). Override for other sizes:
#   python run_sweep.py --sweep lr_bs -- --model.flavor 430M
#
# local_batch_size stays at 64 (master.toml default). This works for 1 GPU.
# For multi-GPU, override: -- --training.local_batch_size <N>

BASE_CONFIG = "configs/experiments/master.toml"

SEQ_LEN = 512
TOKEN_BUDGET = 1_000_000_000  # 1B tokens

# --- Batch sizes and their derived steps/warmup ---
BATCH_SIZES = [128, 256, 512]
BS_META = {}
for _bs in BATCH_SIZES:
    _steps = round(TOKEN_BUDGET / (_bs * SEQ_LEN))
    _warmup = round(_steps * 0.1)
    BS_META[_bs] = (_steps, _warmup)

# --- LR: ~3.5 decades, half-decade spacing ---
LR_VALUES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

# --- Treatments: (eco_enabled, quant_dtype) ---
TREATMENTS = {
    "bf16":     (False, "bf16"),
    "bf16_eco": (True,  "bf16"),
    "fp8":      (False, "fp8"),
    "fp8_eco":  (True,  "fp8"),
}

OPTIONS = {
    "lr": {
        "values": LR_VALUES,
        "flags": {v: ["--optimizer.lr", str(v)] for v in LR_VALUES},
        "name": "lr",
    },
    "batch_size": {
        "values": BATCH_SIZES,
        "flags": {
            bs: [
                "--training.global_batch_size", str(bs),
                "--training.steps", str(BS_META[bs][0]),
                "--lr_scheduler.warmup_steps", str(BS_META[bs][1]),
            ]
            for bs in BATCH_SIZES
        },
        "name": "bs",
    },
    "treatment": {
        "values": list(TREATMENTS.keys()),
        "flags": {
            name: [
                "--eco.enabled" if eco else "--eco.no-enabled",
                "--eco.quant-dtype", qdt,
            ]
            for name, (eco, qdt) in TREATMENTS.items()
        },
        "name": "tmt",
    },
}
