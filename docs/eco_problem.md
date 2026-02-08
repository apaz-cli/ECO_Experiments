# ECO Optimizer State Dtype Problem

## Context

ECO eliminates master weights by storing model parameters in FP8 and injecting
quantization error into Adam's momentum buffer. The next memory win is shrinking
optimizer states (exp_avg, exp_avg_sq) from FP32 to BF16, cutting optimizer memory
in half.

| Configuration               | Weights | m   | v   | Total |
|------------------------------|---------|-----|-----|-------|
| Standard BF16 training (MW)  | 6B      | 4B  | 4B  | 14B   |
| ECO (FP8 weights, FP32 opt)  | 1B      | 4B  | 4B  | 9B    |
| ECO + BF16 opt               | 1B      | 2B  | 2B  | 5B    |

(Bytes per parameter.)

## Goal

Optimizer states in BF16 from the start. No FP32 states at any point during
training.

## The Problem

PyTorch's Adam (all implementations: fused, foreach, for-loop) creates optimizer
states lazily on the first `step()` call, matching the parameter's dtype:

```python
# Inside Adam._init_group():
if len(state) == 0:
    state["step"] = torch.tensor(0.0)
    state["exp_avg"] = torch.zeros_like(p)       # inherits param dtype
    state["exp_avg_sq"] = torch.zeros_like(p)     # inherits param dtype
```

If params are FP8, states would be FP8 (terrible precision for momentum/variance).
If params are FP32/BF16, states are FP32/BF16 — no way to decouple.

### Pre-populating states is rejected

All three Adam implementations validate that states match param dtype:

```
# fused, foreach:
"Tensors of the same index must be on the same device and the same dtype
 except `step` tensors that can be CPU and float32/64 notwithstanding"

# for-loop:
"expected dtype c10::BFloat16 for `end` but got dtype float"
(lerp_ rejects mismatched operand dtypes)
```

Tested: pre-populating `state[p]` with BF16 tensors when `p` is FP32 fails
on `optimizer.step()` for all three implementations.

### All-BF16 works, but only if everything matches

For-loop Adam works when params, grads, and states are all BF16:

```python
p = torch.randn(4, 4, dtype=torch.bfloat16, requires_grad=True)
opt = torch.optim.AdamW([p], lr=1e-3, fused=False, foreach=False)
opt.state[p] = {
    'step': torch.tensor(0.0),
    'exp_avg': torch.zeros_like(p),       # BF16
    'exp_avg_sq': torch.zeros_like(p),    # BF16
}
p.grad = torch.randn_like(p)             # BF16
opt.step()  # works
```

The constraint is: **param dtype == grad dtype == state dtype**.

## ECO's Situation

With ECO:
- **Params**: FP8 (actual storage — the whole point of ECO)
- **Grads**: Higher precision (BF16), since backward computes in the forward dtype
- **States**: Want BF16

This is a three-way dtype mismatch: FP8 params, BF16 grads, BF16 states.

Adam requires all three to match. ECO needs them to differ.

## Possible Solutions

### 1. One-shot post-step hook

Register a `register_step_post_hook` that casts states to BF16 after the first
step, then removes itself. States start in whatever dtype Adam creates (matching
param dtype), get cast once, then stay BF16 via in-place ops on subsequent steps.

**Problem**: First step runs with wrong-dtype states. Subsequent in-place ops
(`lerp_`, `addcmul_`) may still fail if param/grad dtype doesn't match state dtype.

### 2. Subclass AdamW

Override state initialization to force BF16 states regardless of param dtype.
Also need to handle the in-place update operations that mix param/grad/state dtypes.

**Problem**: Significant maintenance burden. Must track upstream Adam changes.

### 3. Present BF16 views to optimizer

Give the optimizer BF16 "views" or copies of the FP8 params. The optimizer sees
BF16 params, creates BF16 states, computes BF16 updates. Then apply the updates
back to the FP8 params.

**Problem**: Breaks the standard optimizer-param relationship. Needs careful
wiring to make `param.grad` land on the right tensor.

### 4. Use a dtype-aware optimizer

Some libraries (e.g., bitsandbytes, torchao) provide optimizers that natively
support mixed-precision states. If `torch.optim` gains an `optim_dtype` parameter
upstream, this becomes trivial.

**Problem**: External dependency or waiting for upstream support.

### 5. Cast everything to BF16 before optimizer step

Before `optimizer.step()`, cast params and grads to BF16. After step, cast params
back to FP8. States naturally stay BF16.

**Problem**: Extra copies every step. The param cast to BF16 is a temporary
allocation. But this is what ECO simulation mode already does (params are BF16,
round-tripped through FP8 after each step).

## Current Status

ECO is currently in **simulation mode**: params stay in BF16, are round-tripped
through FP8 after each optimizer step. Optimizer states are FP32 (matching the
BF16 param dtype — actually they're FP32 because that's what Adam creates for
FP32 params... need to verify what happens with BF16 params and fused Adam).

The `optim_dtype` config field and casting logic are not yet implemented. The
sweep definition `sweeps/bf16_optim.py` references `--eco.optim_dtype` which
does not exist.
