# ECO Configuration Knobs

## Overview

This document lists all configuration parameters for ECO experiments, categorized by component.

## 1. Training Hyperparameters (`[training]`)

| Parameter | Type | Default | Description | Paper Value |
|-----------|------|---------|-------------|-------------|
| `local_batch_size` | int | 8 | Per-device batch size | 512 (global) |
| `global_batch_size` | int | -1 | Global batch size (auto) | 512 |
| `seq_len` | int | 2048 | Sequence length | 512 |
| `max_norm` | float | 1.0 | Gradient clipping norm | 1.0 |
| `steps` | int | 10000 | Training steps | Varies (100N tokens) |
| `dtype` | "bfloat16" \| "float32" | "float32" | Training dtype | BF16 (mixed) |
| `mixed_precision_param` | "bfloat16" \| "float32" | "bfloat16" | Mixed precision param dtype | BF16 |
| `mixed_precision_reduce` | "float32" | "float32" | Mixed precision reduce dtype | FP32 |

**Note**: Paper uses batch size 512 sequences, seq_len 512, gradient clipping 1.0.

## 2. Optimizer Hyperparameters (`[optimizer]`)

| Parameter | Type | Default | Description | Paper Value |
|-----------|------|---------|-------------|-------------|
| `name` | str | "AdamW" | Optimizer name | "AdamW" |
| `lr` | float | 8e-4 | Learning rate | 1.2e-3 × 50M/N_params |
| `beta1` | float | 0.9 | Adam β₁ | 0.9 |
| `beta2` | float | 0.95 | Adam β₂ | 0.98 |
| `momentum` | float | 0.95 | Muon momentum (ECOMuon only) | 0.95 |
| `eps` | float | 1e-8 | Epsilon | 1e-9 |
| `weight_decay` | float | 0.1 | Weight decay | 0.1 |
| `implementation` | "for-loop" \| "foreach" \| "fused" | "fused" | Optimizer implementation | N/A |

**Note**: Paper uses AdamW with (β₁, β₂, ε) = (0.9, 0.98, 1e-9), weight_decay=0.1.

## 3. Learning Rate Scheduler (`[lr_scheduler]`)

| Parameter | Type | Default | Description | Paper Value |
|-----------|------|---------|-------------|-------------|
| `warmup_steps` | int | 200 | Linear warmup steps | 10% of total |
| `decay_type` | "linear" \| "sqrt" \| "cosine" | "linear" | LR decay type | "cosine" |
| `min_lr_factor` | float | 0.0 | Minimum LR ratio | 0.1 |
| `decay_ratio` | float \| None | None | Warmup-Stable-Decay ratio | N/A |
| `total_steps` | int \| None | None | Override total steps | training.steps |

**Note**: Paper uses linear warmup (10% steps), cosine decay to 0.1× peak.

## 4. ECO Configuration (`[eco]`)

| Parameter | Type | Default | Description | Paper Value |
|-----------|------|---------|-------------|-------------|
| `enabled` | bool | False | Enable ECO injection | True (ECO runs) |
| `stochastic_rounding` | bool | False | Use stochastic rounding | True (ECO best) |
| `optim_state_dtype` | "fp32" \| "bf16" \| "fp16" | "fp32" | Optimizer states dtype | "fp32" |
| `optim_compute_dtype` | "fp32" \| "bf16" \| "fp16" | "fp32" | Compute dtype | "fp32" |
| `quantize_weights` | bool | True | Simulate quantized storage | True |
| `quant_dtype` | "fp8" \| "bf16" | "bf16" | Weight quantization dtype | "fp8" |
| `activation_dtype` | "none" \| "fp8" | "none" | Activation quantization | "fp8" (paper) |
| `master_weights_dtype` | None \| "fp32" \| "bf16" | None | Master weights dtype | None (ECO) / "fp32" (baseline) |
| `approach` | "pre_ns" \| "naive_sgdm" \| "frobenius" \| "jacobian" | "frobenius" | ECOMuon injection approach | N/A |
| `heuristic_log_freq` | int | 1 | Heuristic diagnostics frequency | 1 |

**Key ECO Knobs**:
- `master_weights_dtype`: Controls presence of master weights
  - `None`: Pure ECO (no master weights)
  - `"fp32"`: Traditional quantized training (baseline)
  - `"bf16"`: Intermediate memory
- `quant_dtype`: Weight quantization format
  - `"fp8"`: FP8 E4M3 (main ECO format)
  - `"bf16"`: BF16 (baseline/control)
- `stochastic_rounding`: Critical for ECO performance (paper recommends True)

## 5. Model Configuration (`[model]`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "llama3" | Model architecture |
| `flavor` | str | "debugmodel" | Model size (50M, 100M, 430M, etc.) |
| `hf_assets_path` | str | "./tests/assets/tokenizer" | Tokenizer path |
| `converters` | list[str] | [] | Model converters (e.g., "float8") |

## 6. Parallelism (`[parallelism]`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_parallel_replicate_degree` | int | 1 | DDP replication degree |
| `data_parallel_shard_degree` | int | -1 | FSDP sharding degree |
| `tensor_parallel_degree` | int | 1 | Tensor parallelism degree |
| `pipeline_parallel_degree` | int | 1 | Pipeline parallelism degree |

## 7. Experimental Dimensions (Sweeps)

### Existing Sweep Dimensions:

1. **LR & Batch Size** (`lr_bs.py`):
   - Learning Rate: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
   - Batch Size: [128, 256, 512]
   - Treatment: 7 variations (see below)
   - Local Batch Size: [64, 32, 16, 8, 4, 2, 1] (singular)

2. **β₁/β₂ Sensitivity** (`beta.py`):
   - β₁: [0.8, 0.85, 0.9, 0.95, 0.99]
   - β₂: [0.95, 0.98, 0.99, 0.999]
   - ECO enabled: [True, False]

3. **BF16 Optimizer States** (`bf16_optim.py`):
   - ECO enabled: [True, False]
   - Optimizer state dtype: ["fp32", "bf16"]

4. **Stochastic Rounding** (`eco_rounding.py`):
   - ECO enabled: [True, False]
   - Stochastic rounding: [True, False]

5. **Muon Approaches** (`muon.py`):
   - ECO approach: ["pre_ns", "naive_sgdm", "frobenius", "jacobian"]

### Treatment Combinations:

From `lr_bs.py` (updated):
1. `bf16`: BF16 simulated quant, no ECO (BF16 baseline, no master weights)
2. `bf16_eco`: BF16 simulated quant + ECO (control)
3. `fp8`: FP8 simulated quant, no ECO (naive quantization, no master weights)
4. `fp8_eco`: FP8 simulated quant + ECO (main method, RTN)
5. `fp8_eco_sr`: FP8 simulated quant + ECO with stochastic rounding (paper best)
6. `fp8_mw_rtn`: FP32 master weights + FP8 RTN baseline (traditional quantized training)
7. `fp8_mw_sr`: FP32 master weights + FP8 SR baseline

## 8. Critical Configuration Gaps

### Missing Baselines (Now Added):
- ✅ `master_weights_fp8_rtn.toml`: FP32 master weights + FP8 RTN
- ✅ `master_weights_fp8_sr.toml`: FP32 master weights + FP8 SR
- ❓ Pure BF16 without quantization simulation (`quantize_weights = false`)

### Paper Mismatches Fixed:
- ✅ Weight decay: 0 → 0.1 (paper uses 0.1)
- ✅ Token budget clarification: 50N vs paper 100N (due to batch size 256 vs 512)
- ✅ Stochastic rounding defaults updated for ECO configs

### Remaining Mismatches:
- Batch size: 256 vs paper 512 (affects token budget)
- β₂ default: 0.95 vs paper 0.98 (config default, but master.toml sets 0.98)
- ε default: 1e-8 vs paper 1e-9 (config default, but master.toml sets 1e-9)
- Activation quantization: "none" vs paper "fp8" (can be enabled via `activation_dtype = "fp8"`)

## 9. Recommended Baseline Configurations

### 1. BF16 Baseline (No Master Weights)
```toml
[eco]
enabled = false
quantize_weights = true  # or false for pure BF16
quant_dtype = "bf16"
master_weights_dtype = None
```

### 2. FP8 Baseline (With Master Weights)
```toml
[eco]
enabled = false
quantize_weights = true
quant_dtype = "fp8"
master_weights_dtype = "fp32"
activation_dtype = "fp8"  # matches paper
stochastic_rounding = false  # or true for SR baseline
```

### 3. ECO (No Master Weights)
```toml
[eco]
enabled = true
quantize_weights = true
quant_dtype = "fp8"
master_weights_dtype = None
activation_dtype = "fp8"  # matches paper
stochastic_rounding = true  # recommended for ECO
```

## 10. Validation Checklist

Before running experiments:
1. [ ] Weight decay = 0.1 (paper value)
2. [ ] β₁ = 0.9, β₂ = 0.98, ε = 1e-9 (paper values)
3. [ ] Gradient clipping = 1.0 (paper value)
4. [ ] LR schedule: 10% warmup, cosine decay to 0.1× peak
5. [ ] Token budget calculation matches intended N tokens
6. [ ] Master weights baseline present for comparison
7. [ ] ECO runs with stochastic rounding enabled
8. [ ] Activation quantization enabled if matching paper ("fp8")

## 11. Sweep Design Considerations

When designing sweeps, consider these independent dimensions:

1. **Master weights**: None vs FP32 vs BF16
2. **Weight quantization**: FP8 vs BF16
3. **Activation quantization**: None vs FP8
4. **ECO injection**: Enabled vs Disabled
5. **Rounding**: Stochastic vs Round-to-nearest
6. **Optimizer states dtype**: FP32 vs BF16 vs FP16
7. **Optimizer**: AdamW vs Muon
8. **ECO approach** (Muon only): pre_ns, naive_sgdm, frobenius, jacobian
9. **Hyperparameters**: LR, β₁, β₂, batch size

