# Quantization Guide for ECO Experiments

## Table of Contents
1. [Overview](#1-overview)
2. [QuantizedTensor: The Core Abstraction](#2-quantizedtensor-the-core-abstraction)
3. [ECOAdamW: How the Optimizer Handles Quantized Weights](#3-ecoadamw-how-the-optimizer-handles-quantized-weights)
4. [Dtype Options Reference](#4-dtype-options-reference)
5. [Experiment Configurations](#5-experiment-configurations)
6. [Sweep Files](#6-sweep-files)
7. [Critical Issues To Fix](#7-critical-issues-to-fix)
8. [WARNING: Never Use quantize.linear.float8](#8-warning-never-use-quantizelinearfloat8)
9. [Future: NVFP4 / MXFP4](#9-future-nvfp4--mxfp4)

---

## 1. Overview

ECO experiments use **two separate quantization systems** — make sure you understand the
difference:

| System | What it does | Used by ECO? |
|--------|-------------|-------------|
| **QuantizedTensor** (`torchtitan/components/quantized_tensor.py`) | Custom tensor subclass. Stores weights in FP8 E4M3 with a scale factor. Dequantizes transparently for PyTorch ops via `__torch_dispatch__`. | **YES** — this is our system |
| **quantize.linear.float8** (torchao's `Float8Linear`) | Replaces `nn.Linear` with torchao's `Float8Linear`. Keeps FP32 master weights internally. FP8 only for matmul. | **NO** — never use this for ECO. See [Section 8](#8-warning-never-use-quantizelinearfloat8) |

The ECO approach is fundamentally different from standard FP8 training:
- **Standard FP8 (torchao):** FP32 master weights + FP8 forward matmul. Memory: 14B/param.
- **ECO:** FP8 stored weights (no master copy) + FP32 optimizer states + error injection. Memory: 9B/param.

---

## 2. QuantizedTensor: The Core Abstraction

**File:** `torchtitan/components/quantized_tensor.py`

`QuantizedTensor` is a `torch.Tensor` subclass created via `_make_wrapper_subclass`. It stores:

| Field | Description |
|-------|-------------|
| `_data` | The quantized data (e.g., `torch.float8_e4m3fn`) |
| `_scale` | Per-tensor scale factor (symmetric quantization: `amax / fp8_max`) |
| `_zero_point` | Optional, unused (we use symmetric quantization) |
| `_quant_dtype` | Storage dtype (`torch.float8_e4m3fn`) |
| `_compute_dtype` | Presentation dtype (`torch.bfloat16` or `torch.float32`) |
| `_axiswise_dim` | Optional per-channel quantization axis |

### Quantization scheme

Symmetric tensorwise scaling:
```
scale = amax(abs(tensor)) / fp8_max
quantized = round(tensor / scale)  →  stored as float8_e4m3fn
dequantized = quantized.to(compute_dtype) * scale
```

**E4M3 format:** 4 exponent bits, 3 mantissa bits. Range ±448, precision ~1/16 (~6%).
Stochastic rounding adds uniform noise in `[-0.5 ulp, +0.5 ulp]` before rounding, making
`E[round(x)] = x` (unbiased, satisfies paper Assumption 3.2).

### Transparent dispatch

All PyTorch operations on a QuantizedTensor auto-dequantize via `__torch_dispatch__`:
```python
qt = QuantizedTensor.quantize(tensor, torch.float8_e4m3fn)
result = qt + other  # auto-dequantizes qt to compute_dtype, then adds
```

### DTensor/FSDP2 compatibility

`QuantizedTensor` implements `__tensor_flatten__` / `__tensor_unflatten__` for
serialization, which is required for DTensor wrapping and FSDP2 checkpointing.

### Usage

```python
from torchtitan.components.quantized_tensor import QuantizedTensor

# Quantize a tensor
qt = QuantizedTensor.quantize(fp32_tensor, torch.float8_e4m3fn)

# Dequantize
fp32_back = qt.dequantize()  # returns compute_dtype tensor

# Check
from torchtitan.components.quantized_tensor import is_quantized_tensor
is_quantized_tensor(qt)  # True
```

---

## 3. ECOAdamW: How the Optimizer Handles Weights

**File:** `torchtitan/components/eco_optimizer.py`

ECOAdamW has three code paths, selected by priority:

### 1. `quantize_weights=True` and `param.dim() >= 2` → `_step_simulated_quant()`

**This is the primary path for real training.** Applies a quant→dequant round-trip
on regular FP32/BF16 parameters after the Adam update, simulating FP8 storage without
requiring QuantizedTensor params or autograd changes.

```
1. Clone param to optim_compute_dtype (FP32)
2. Adam update in FP32 → θ̃
3. Quant→dequant round-trip → θ̂ (FP8-representable value)
4. If eco_enabled: error e = θ̃ − θ̂, inject into momentum
5. Store θ̂ back into param.data
```

### 2. `isinstance(param, QuantizedTensor)` → `_step_quantized()`

Legacy/test path for params that are actual QuantizedTensor instances:

```
1. Dequantize FP8 weights → upcast to optim_compute_dtype (FP32)
2. Adam update in FP32 → θ̃
3. Requantize θ̃ → θ̂ (FP8, optional stochastic rounding)
4. Compute error: e = θ̃ − θ̂
5. If eco_enabled: inject into momentum
6. Store θ̂ back into the QuantizedTensor (update _data and _scale)
```

### 3. All other params → `_step_regular()`

Standard AdamW. No quantization, no ECO injection. Used for:
- Biases, layer norms, embeddings (dim < 2)
- All params when `quantize_weights=False`

### Four meaningful configurations

| `quantize_weights` | `eco_enabled` | Behavior |
|---|---|---|
| `false` | `false` | Pure AdamW (BF16 baseline) |
| `true` | `false` | FP8 baseline (quant error, no compensation) |
| `true` | `true` | **FP8 ECO** (the main experiment) |
| `false` | `true` | ECO on FP32 (ablation, injection is ~no-op) |

---

## 4. Dtype Options Reference

### ECO config flags (`[eco]` section / `--eco.*` CLI)

| Flag | Values | Default | Effect |
|------|--------|---------|--------|
| `--eco.enabled` / `--eco.no-enabled` | bool | `false` | Enable ECO injection with ECOAdamW |
| `--eco.quantize-weights` / `--eco.no-quantize-weights` | bool | `false` | Simulate FP8 weight storage via quant→dequant round-trip |
| `--eco.quant-dtype` | `fp8_e4m3`, `fp8_e5m2` | `fp8_e4m3` | Quantization dtype for simulated weight storage |
| `--eco.optim-state-dtype` | `fp32`, `bf16`, `fp16` | `fp32` | Storage dtype for m and v |
| `--eco.optim-compute-dtype` | `fp32`, `bf16`, `fp16` | `fp32` | Compute dtype for Adam update + error |
| `--eco.stochastic-rounding` / `--eco.no-stochastic-rounding` | bool | `false` | Stochastic vs RTN rounding |
| `--eco.heuristic-log-freq` | int | `100` | Steps between heuristic logging (0=off) |

### Memory per parameter

| Configuration | Weights | m | v | Total | Notes |
|--------------|---------|---|---|-------|-------|
| BF16 baseline | 2B (BF16) + 4B (FP32 MW) | 4B | 4B | 14B | Standard mixed-precision |
| FP8 + master weights | 1B (FP8) + 4B (FP32 MW) | 4B | 4B | 13B | QAT baseline |
| FP8 ECO (FP32 opt) | 1B (FP8) | 4B | 4B | **9B** | Paper default |
| FP8 ECO (BF16 opt) | 1B (FP8) | 2B | 2B | **5B** | Maximum compression |

### Why FP32 compute is the default

The error `e = θ̃ − θ̂` is the difference between two very close values (differ by ~6%
relative for E4M3). In BF16 (7 mantissa bits), this subtraction would leave only ~2-3
significant bits — catastrophic cancellation. FP32 (23 mantissa bits) retains ~17
significant bits.

The `optim_state_dtype` and `optim_compute_dtype` can differ. A useful combination is
BF16 storage with FP32 compute: m and v are stored in BF16 (2B each), but upcasted to
FP32 for the Adam math, then downcasted back to BF16. The error computation always
happens in `optim_compute_dtype`.

---

## 5. Experiment Configurations

### 5.1 BF16 Baseline

Standard mixed-precision training. No quantization, no ECO.

```bash
# Flags
--optimizer.name AdamW
# (no eco flags needed, eco.enabled defaults to false)
```

**Status: WORKS.** This is standard torchtitan training.

**Memory:** 14B/param (BF16 weight + FP32 master weight in optimizer + FP32 m + FP32 v)

### 5.2 FP8 Baseline (Master Weights, No ECO)

Standard QAT: keep FP32 master weights, quantize to FP8 only for forward matmul.
This is the paper's "FP8 w/ MW" condition.

```bash
# PROPOSED flags (requires implementation — see Section 7)
--optimizer.name ECOAdamW
--eco.no-enabled
--eco.master-weights          # NEW FLAG NEEDED
--eco.stochastic-rounding     # or --eco.no-stochastic-rounding for RTN
```

**Status: NOT YET IMPLEMENTED.** See [Section 7.4](#74-master-weights-mode).

ECOAdamW currently has no "master weights" mode. Without it, this experiment cannot be
run without `quantize.linear.float8` (which we do not use).

**Memory:** 13B/param (FP8 weight + FP32 master weight + FP32 m + FP32 v)

### 5.3 FP8 ECO

ECO: FP8 stored weights (simulated via quant→dequant round-trip), no master copy,
error injection into momentum. This is the paper's "FP8 w/o MW ECO" condition.

```bash
--optimizer.name ECOAdamW
--eco.enabled
--eco.quantize-weights
--eco.stochastic-rounding     # or --eco.no-stochastic-rounding for RTN
--eco.optim-state-dtype fp32
--eco.optim-compute-dtype fp32
```

**Status: WORKS.** Uses simulated quantization inside ECOAdamW — weight matrices
(dim >= 2) undergo a quant→dequant round-trip after each Adam update, producing
identical training dynamics to true FP8 storage.

**Memory:** 9B/param effective (FP8-representable weights + FP32 m + FP32 v).
Note: weights are stored as FP32 tensors holding FP8-representable values.

### 5.4 FP8 Baseline + BF16 Optimizer State

Same as 5.2 but with BF16 m and v.

```bash
# PROPOSED flags
--optimizer.name ECOAdamW
--eco.no-enabled
--eco.master-weights          # NEW FLAG NEEDED
--eco.optim-state-dtype bf16
--eco.optim-compute-dtype fp32
--eco.stochastic-rounding
```

**Status: NOT YET IMPLEMENTED.** Depends on master weights mode (Section 7.4).

**Memory:** 9B/param (FP8 weight + FP32 master weight + BF16 m + BF16 v)

### 5.5 FP8 ECO + BF16 Optimizer State

ECO with maximum memory compression.

```bash
--optimizer.name ECOAdamW
--eco.enabled
--eco.quantize-weights
--eco.optim-state-dtype bf16
--eco.optim-compute-dtype fp32
--eco.stochastic-rounding
```

**Status: WORKS.** Uses simulated quantization with BF16 optimizer states.

**Memory:** 5B/param effective (FP8-representable weights + BF16 m + BF16 v)

---

## 6. Sweep Files

### Current sweep flag patterns

All sweeps in `sweeps/` use:
```python
ECO_OFF = ["--optimizer.name", "AdamW", "--eco.no-enabled", "--eco.no-quantize-weights"]
ECO_ON  = ["--optimizer.name", "ECOAdamW", "--eco.enabled", "--eco.quantize-weights"]
```

The `--eco.quantize-weights` flag activates simulated FP8 quantization inside ECOAdamW.
ECO_OFF uses plain AdamW with no quantization (BF16 baseline).

---

## 7. Issues and Status

### 7.1 ~~No mechanism to convert model weights to QuantizedTensor~~ FIXED

**Resolved** via simulated quantization (`quantize_weights` flag). Instead of converting
model params to QuantizedTensor (which has autograd and FSDP composition issues),
ECOAdamW applies a quant→dequant round-trip on regular FP32 params after each Adam
update. This produces identical training dynamics without requiring any model converter.

### 7.2 ~~isinstance check: nn.Parameter vs QuantizedTensor~~ BYPASSED

No longer the critical path. The `_step_simulated_quant()` path uses
`self._quantize_weights and param.dim() >= 2` instead of isinstance checks.
The `_step_quantized()` path is retained for unit tests with manual QuantizedTensor params.

### 7.3 ~~eco_config passthrough bug in train.py~~ FIXED

Fixed: `train.py` now passes `eco_config` whenever `optimizer.name == "ECOAdamW"`,
regardless of `eco.enabled`.

### 7.4 Master weights mode (future)

For the "FP8 baseline with master weights" experiment, we need ECOAdamW to optionally
keep a FP32 copy of the weight in the optimizer state dict. Not yet implemented.

---

## 8. WARNING: Never Use quantize.linear.float8

**Do NOT use the `quantize.linear.float8` model converter for ECO experiments.**

`quantize.linear.float8` is **torchtitan's built-in** FP8 training converter (located at
`torchtitan/components/quantization/float8.py`). It wraps torchao's `Float8Linear` and
is part of torchtitan's standard quantization infrastructure — it is NOT our code. It:
- Replaces `nn.Linear` with torchao's `Float8Linear`
- Keeps FP32 master weights internally
- Does FP8 only for the matmul kernel
- Has its own scaling logic (dynamic, delayed, tensorwise/rowwise)
- Is designed for training speedup, NOT for memory reduction
- Is registered via `--model.converters quantize.linear.float8`

This is **completely orthogonal** to ECO:
- ECO eliminates master weights; Float8Linear keeps them
- ECO uses our custom QuantizedTensor for storage; Float8Linear uses FP32 parameters
- ECO injects error into momentum; Float8Linear has no optimizer interaction
- ECO's quantization scheme is simple symmetric tensorwise; Float8Linear has
  multiple configurable recipes

Using both together would be contradictory — Float8Linear's master weights defeat
the purpose of ECO. The two systems solve different problems and should not be mixed.

**We do not use torchtitan's model converters for ECO experiments.** Our quantization
is handled entirely by `QuantizedTensor` + `ECOAdamW`, which is our own implementation
independent of torchtitan's `quantize.linear.float8` / `quantize.linear.mx` system.

The `--model.converters quantize.linear.float8` flag should **never** appear in any
ECO experiment config. If you see it in old configs or documentation, remove it.

---

## 9. Future: NVFP4 / MXFP4

### Motivation

FP8 E4M3 gives 1 byte/weight. FP4 (4-bit) would give 0.5 bytes/weight, further
reducing memory:

| Configuration | Weights | m (FP32) | v (FP32) | Total |
|--------------|---------|----------|----------|-------|
| FP8 ECO | 1 B | 4 B | 4 B | 9 B |
| FP4 ECO (FP32 opt) | 0.5 B | 4 B | 4 B | 8.5 B |
| FP4 ECO (BF16 opt) | 0.5 B | 2 B | 2 B | 4.5 B |

The main question is whether the larger quantization error at 4-bit can still be
effectively compensated by ECO. The paper's theory (Theorem 3.5) shows a noise floor
proportional to σ² (quantization variance), which is larger at 4-bit. But ECO's
injection should still help vs naive 4-bit training.

### FP4 formats

| Format | Exponent | Mantissa | Range | Notes |
|--------|----------|----------|-------|-------|
| NVFP4 (E2M1) | 2 | 1 | ±6 | NVIDIA's native FP4, supported on Blackwell |
| MXFP4 (E2M1 + shared exponent) | 2+8 | 1 | wider via block scaling | Microscaling standard |
| NF4 | — | — | 4-bit normal float | QLoRA-style, non-uniform grid |

### Implementation plan: qutlass

[qutlass](https://github.com/IST-DASLab/qutlass) provides CUDA kernels for FP4 matmul
on NVIDIA GPUs. Integration plan:

1. **Add FP4 quant/dequant to QuantizedTensor:**
   - QuantizedTensor currently only supports dtypes known to PyTorch (`torch.float8_e4m3fn`, etc.)
   - FP4 doesn't have a native `torch.dtype`, so we'd need a custom representation:
     pack two FP4 values into a `torch.uint8`, with a separate scale tensor
   - Add `quant_format` field to QuantizedTensor (e.g., `"fp8_e4m3"`, `"nvfp4_e2m1"`, `"mxfp4"`)
   - Implement `_quantize_fp4()` and `_dequantize_fp4()` methods

2. **Add new config option for quant dtype:**
   - Add `--eco.quant-dtype` flag: `fp8` (default), `nvfp4`, `mxfp4`
   - The ECO config dataclass needs:
     ```python
     quant_dtype: str = "fp8"
     """Quantization format for weights: 'fp8' (E4M3), 'nvfp4' (E2M1), 'mxfp4'"""
     ```

3. **ECOAdamW changes:**
   - `_requantize()` needs to dispatch on quant format
   - Stochastic rounding ULP calculation changes (1 mantissa bit for FP4 vs 3 for FP8)
   - Error magnitude will be ~16x larger (4-bit vs 8-bit), so injection coefficient
     may need tuning

4. **Forward pass integration:**
   - For FP4 matmul: use qutlass kernels
   - The QuantizedTensor dispatch (`__torch_dispatch__`) currently dequantizes for all ops.
     For matmul, we should dispatch to qutlass instead of dequantizing.

5. **Install qutlass:**
   ```bash
   pip install qutlass  # or build from source
   # Requires: CUDA 12.x, SM 89+ (Ada/Hopper/Blackwell)
   ```

### Proposed config flags for FP4 experiments

```bash
# FP4 ECO (once implemented)
--optimizer.name ECOAdamW
--eco.enabled
--eco.quant-dtype nvfp4        # NEW FLAG
--eco.stochastic-rounding
--eco.optim-state-dtype fp32
--eco.optim-compute-dtype fp32

# FP4 ECO + BF16 optimizer state
--optimizer.name ECOAdamW
--eco.enabled
--eco.quant-dtype nvfp4
--eco.stochastic-rounding
--eco.optim-state-dtype bf16
--eco.optim-compute-dtype fp32
```

### What needs to happen first

Before FP4 support:
1. Fix the QuantizedTensor → model integration (Section 7.1, 7.2)
2. Verify FP8 ECO works end-to-end on real models
3. Then extend QuantizedTensor with FP4 support
4. Add qutlass dependency and matmul dispatch
