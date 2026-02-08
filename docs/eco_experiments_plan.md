# ECO Experiments: Implementation Plan v2

## Table of Contents
0. [Infrastructure Setup](#0-infrastructure-setup)
1. [Base ECO Implementation](#1-base-eco-implementation)
2. [Scaling to Llama 3 8B](#2-scaling-to-llama-3-8b)
3. [BF16 Optimizer States](#3-bf16-optimizer-states)
4. [Adam β₁/β₂ Sensitivity](#4-adam-β1β2-sensitivity)
5. [ECO for Muon](#5-eco-for-muon)
6. [Implementation Roadmap](#6-implementation-roadmap)

---

## 0. Infrastructure Setup

Before any ECO work, two components need to be ported from `/mnt/skraid0/torchtitan/`.

### 0.1 AIM Logger

The ECO repo (`/home/apaz/git/ECO_Experiments/`) currently has only TensorBoard and WandB
loggers (`torchtitan/components/metrics.py`). The other torchtitan repo at
`/mnt/skraid0/torchtitan/` has an `AimLogger` class (lines 167–239 of
`torchtitan/components/metrics.py`) that we will use for all experiments.

**What to port:**
- `AimLogger` class from `/mnt/skraid0/torchtitan/torchtitan/components/metrics.py:167-239`
  — wraps `aim.Run`, supports run resume via `run_hash`, tracks hparams, uses shared
  `.aim` repo.
- `enable_aim: bool = False` config field from
  `/mnt/skraid0/torchtitan/torchtitan/config/job_config.py:95`.
- Updates to `_build_metric_logger()` to wire in AIM
  (`/mnt/skraid0/torchtitan/torchtitan/components/metrics.py:324-421`) — this version
  also supports a `LoggerContainer` that runs multiple loggers, and accepts an
  `aim_run_hash` parameter for checkpoint resume.
- The `MetricsProcessor.initialize_logger()` method
  (`/mnt/skraid0/torchtitan/torchtitan/components/metrics.py:590-606`) — defers logger
  creation until after checkpoint loading, so resumed runs don't create throwaway AIM
  runs.
- `/mnt/skraid0/torchtitan/start_aim_server.sh` — convenience script for the AIM UI.

### 0.2 TOML Shebang Runner

The other torchtitan repo has a pattern where TOML config files are directly executable.
A shebang on line 1 points to a runner script:

```toml
#!/path/to/run_config.sh

[job]
dump_folder = "./outputs"
...
```

TOML treats `#!...` as a comment, so parsers ignore it. The OS treats it as a shebang,
so `chmod +x myconfig.toml && ./myconfig.toml` launches training with that config.

The runner script is `/mnt/skraid0/torchtitan/run_config.sh` — a minimal wrapper that
takes `$1` as the config file path and passes the rest as CLI overrides to torchrun:

```bash
#!/bin/bash
set -ex
CONFIG_FILE="$1"
shift
NGPU=${NGPU:-"1"}
export LOG_RANK=${LOG_RANK:-0}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
  -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} "$@"
```

**What to do:**
- Copy `run_config.sh` into the ECO repo root.
- All ECO experiment TOML configs should include the shebang `#!/tmp/eco/run_config.sh`.
  The `start_aim_server.sh` script generates wrapper scripts in `/tmp/eco/` that
  forward to the repo's actual `run_config.sh`. This avoids hardcoding the repo path
  in every config and lets configs be portable across checkouts.
- Run `./start_aim_server.sh` once (or just its wrapper-generation portion) to
  create `/tmp/eco/run_config.sh` and `/tmp/eco/run_sweep.py` before executing
  any configs directly.
- The existing `run_train.sh` (which uses `CONFIG_FILE` as an env var) is kept for
  backward compatibility and for the sweep script (which sets `CONFIG_FILE` explicitly).

### 0.3 Experiment Sweep Script

`/mnt/skraid0/torchtitan/collect_ablation_traces.py` provides a framework for running
multiple training configurations. It generates combinations from a dict of options,
executes each via `./run_train.sh` with CLI overrides, and has smart skipping via
monotonicity rules (if a smaller batch size OOMs, skip larger ones).

We adapt this script for ECO experiments:
- Replace the profiling-oriented options (`use_liger_loss`, `use_flex_attn`, `ac_mode`,
  etc.) with ECO-relevant dimensions: `eco_enabled`, `rounding_mode`, `optimizer_type`,
  `beta1`, `beta2`, `model_size`, etc.
- Remove the `--profiling.kill_after_profile` flag — we want full training runs.
- Add Aim integration: pass `--job.description` and `--job.run_name` for each
  variation so runs are organized in the Aim UI.
- Keep the monotonicity-based skip logic (useful for batch size sweeps).

---

## 1. Base ECO Implementation

### 1.1 Algorithm

ECO (Algorithm 3 from the paper, line 206–218 of `ECO_Paper.tex`): after each Adam step,

```
θ̂_{t+1} = quantize(θ̃_{t+1})
e_{t+1} = θ̃_{t+1} − θ̂_{t+1}
m̂_{t+1} = m̃_{t+1} + [(1 − β₁^t) / η] · (1 − 1/β₁) · (√(v_{t+1}/(1 − β₂^t)) + ε) ⊙ e_{t+1}
```

The second moment v is not modified. The injection is element-wise.

### 1.2 Quantization

Use the same FP8 setup that torchtitan already provides through its `Float8LinearConverter`
(`torchtitan/components/quantization/float8.py`). This uses torchao's
`convert_to_float8_training`, which replaces `nn.Linear` modules with `Float8Linear`. The
format is E4M3 (`torch.float8_e4m3fn`). Scaling is dynamic, either tensorwise (default)
or rowwise (via the `recipe_name` config). See `torchtitan/config/job_config.py` lines
~730–740 for recipe options: `"tensorwise"`, `"rowwise"`, `"rowwise_with_gw_hp"`.

Torchtitan's Float8 setup handles the forward-pass quantization (weights and activations
are cast to FP8 for matmuls). For ECO, we additionally need to simulate quantized weight
*storage*: after each optimizer step, round-trip the parameter through FP8 (quantize then
dequantize back to the training dtype). The difference between pre- and post-roundtrip is
the quantization error e.

This "simulation mode" produces the same numerical trajectory as true FP8 storage but
doesn't physically save memory — the parameters remain in the training dtype (BF16).
This is sufficient for measuring ECO's effect on convergence. Actual memory savings
follow analytically (see Section 3). The quantization format and rounding used for this
round-trip should be the same as what Float8LinearConverter uses for the forward pass, so
the stored weights and the compute weights are on the same grid.

Which layers are quantized is determined by the Float8LinearConverter's filter function
(`torchtitan/components/quantization/float8.py:104-138`), controlled by the
`filter_fqns` config. The paper quantizes only linear layers in transformer blocks,
excluding embeddings and the output head (paper line 443: "For quantized runs, we apply
the method only to the linear layers within transformer blocks, excluding the embedding
and output layers"). Torchtitan's Float8 config already supports this via `filter_fqns`.
ECO applies wherever quantization applies — there is no separate notion of "ECO scope."

### 1.3 Hook Point

torchtitan's model converters (`torchtitan/protocols/model_converter.py`) provide a
`post_optimizer_hook(model)` that fires after each optimizer step (wired in
`torchtitan/train.py:287-291` via `register_step_post_hook`). The existing
`Float8LinearConverter` uses this hook to precompute FP8 scales.

ECO needs access to both the model parameters and the optimizer state (to inject into the
momentum buffer `exp_avg`). The current `post_optimizer_hook` signature only receives the
model. Two options:

1. **Extend the post_optimizer_hook signature** to also pass the optimizer. This is a
   small change to `ModelConvertersContainer.post_optimizer_hook` and the lambda in
   `train.py:287-291`.
2. **Register a separate hook** on the optimizer directly in `train.py`, outside the
   model converter system.

Option 1 is cleaner since ECO is conceptually a model converter (it modifies parameter
storage). The implementation should keep ECO's hook ordered *after* Float8's hook (if
both are active) since Float8's hook precomputes scales that ECO's quantization may use.

### 1.4 Heuristic Validation Logging

The paper's key approximation is e_t ≈ e_{t+1} (consecutive quantization errors are
similar; paper lines 155–163). The approximation error in the momentum injection is
exactly (1/η)·(e_t − e_{t+1}) (derivable from the exact formula in Appendix A, paper
lines 583–592). We validate this heuristic in every experiment.

**Metrics (per-layer, aggregated to model-wide min/mean/max):**

| Metric | Formula | Meaning |
|--------|---------|---------|
| Relative norm ratio | ‖e_{t+1}‖ / ‖e_t‖ | Magnitude stability (paper's metric, Fig. 3) |
| Cosine similarity | cos(e_t, e_{t+1}) | Direction stability (paper's metric, Fig. 3) |
| Relative diff norm | ‖e_t − e_{t+1}‖ / ‖e_t‖ | Direct approximation quality |
| Injection-relative error | (1/η)·‖e_t − e_{t+1}‖ / ‖m_t‖ | Approximation error vs momentum — **the metric that directly measures whether the heuristic is adequate** |

**Implementation:** Every `heuristic_log_freq` steps (configurable), stash a clone of
e_t. On the very next step, compute e_{t+1}, evaluate all metrics, log them via the AIM
logger, free the buffer. Cost: one transient copy of the quantized parameters, held for
exactly 1 step per logging interval.

### 1.5 Files to Create/Modify

**New files:**
- `torchtitan/components/eco.py` — ECO logic: quantize-error-inject cycle, heuristic
  validation, configuration.
- `torchtitan/models/llama3/train_configs/llama3_eco_*.toml` — experiment configs.

**Modified files:**
- `torchtitan/components/metrics.py` — Port AIM logger (Section 0.1).
- `torchtitan/config/job_config.py` — Add `[eco]` config section and `enable_aim` field.
- `torchtitan/protocols/model_converter.py` — Extend `post_optimizer_hook` signature to
  receive optimizer (if using option 1 from Section 1.3).
- `torchtitan/train.py` — Wire up ECO; update `post_optimizer_hook` call.

**Ported/adapted files:**
- `run_config.sh` — Copied from `/mnt/skraid0/torchtitan/run_config.sh` (Section 0.2).
  All new TOML configs include `#!/tmp/eco/run_config.sh` as line 1 (see Section 0.2).
- `collect_eco_sweeps.py` — Adapted from
  `/mnt/skraid0/torchtitan/collect_ablation_traces.py` (Section 0.3).
- `start_aim_server.sh` — Copied from `/mnt/skraid0/torchtitan/start_aim_server.sh`.

### 1.6 Baseline

Every experiment compares:
- **BF16 baseline** (standard training, no quantization, no ECO)
- **ECO** (FP8 weight quantization with error injection, no master weights)

The BF16 baseline uses torchtitan's default mixed-precision setup: parameters in the
training dtype, BF16 matmuls via FSDP's `mixed_precision_param`.

---

## 2. Scaling to Llama 3 8B

### 2.1 Motivation

The paper tests dense models up to 1B (Gemma-3, paper line 466) and MoE up to 2.1B
(paper line 480). 8B is the standard scale for current open-weight models. Demonstrating
ECO at 8B is the single most impactful experiment.

### 2.2 Setup

**Model:** Llama 3 8B — torchtitan already has this config:
`TransformerModelArgs(dim=4096, n_layers=32, n_heads=32, ...)`, defined in
`torchtitan/models/llama3/__init__.py`.

**Hyperparameters (matching the paper where possible, paper lines 438–442):**
- AdamW, β₁=0.9, β₂=0.98, ε=10⁻⁹
- Weight decay 0.1, gradient clipping 1.0
- LR: linear warmup (10% of steps), cosine decay to 0.1× peak
- Dataset: C4 (matching the paper; paper line 436)

**Parallelism:** FSDP2 across available GPUs. torchtitan configs for 8B already exist
at `torchtitan/models/llama3/train_configs/`.

### 2.3 Scaling Law Replication

Before the 8B run, replicate the paper's scaling law study (30M–800M, Table 1 on paper
line 373) to validate the ECO implementation. This requires creating small Llama-style
model configs. The paper follows the QUEST architecture (paper line 434: "following
\citet{quest}"); exact dims need to be matched. Each model trains on 100N tokens
(paper line 436).

### 2.4 What to Measure

- Validation loss curves and final validation loss.
- Heuristic validation metrics (Section 1.4) — does e_t ≈ e_{t+1} hold at 8B?
- Training stability (any divergence).
- Peak memory (`torch.cuda.max_memory_allocated`).

---

## 3. BF16 Optimizer States

### 3.1 Motivation

ECO eliminates master weights. Quantizing optimizer states to BF16 eliminates another
large memory cost. BF16 has the same 8-bit exponent as FP32 (same dynamic range), with
7 mantissa bits instead of 23 (paper Section 2, "Optimizer State Quantization," line
111–112, notes this is a separate and complementary line of work).

Combined memory per parameter:

| Configuration | Weights | m | v | Total |
|---------------|---------|---|---|-------|
| Standard BF16 training (MW=FP32) | 2B + 4B = 6B | 4B | 4B | 14B |
| ECO (FP8 weights, FP32 opt) | 1B | 4B | 4B | 9B |
| ECO + BF16 opt | 1B | 2B | 2B | 5B |

(Bytes per parameter. MW = master weights. "1B" for FP8 weights includes scale overhead.)

### 3.2 Setup

**Four runs** at 430M and/or 1B:

| Run | ECO | Optimizer dtype |
|-----|-----|----------------|
| BF16 baseline | No | FP32 |
| BF16 optim only | No | BF16 |
| ECO + FP32 optim | Yes | FP32 |
| ECO + BF16 optim | Yes | BF16 |

This is a 2×2 design: ±ECO × ±BF16 optimizer states.

### 3.3 Implementation

After the optimizer step, cast optimizer state tensors to the target dtype:
```python
state['exp_avg'] = state['exp_avg'].to(target_dtype)      # m
state['exp_avg_sq'] = state['exp_avg_sq'].to(target_dtype) # v
```

The ECO injection is computed in FP32, then the result is stored in whatever dtype m is.
If the injection Δm is too small relative to existing m values, BF16 addition will lose
it. To monitor this, log `|Δm| / |m|` as a histogram — if consistently below ~2⁻⁷
(≈0.008, BF16's mantissa precision), the injection is being silently dropped.

### 3.4 Baseline

The 2×2 design is self-controlled. The key comparison is:
- Does adding ECO to BF16-optimizer training help, hurt, or do nothing?
- Does BF16 optimizer degrade ECO quality?

---

## 4. Adam β₁/β₂ Sensitivity

### 4.1 Motivation

ECO's injection coefficient depends directly on β₁ (paper Algorithm 3, line 216):

```
injection ∝ (1 − 1/β₁)
```

Evaluating this factor at different β₁:

| β₁ | (1 − 1/β₁) | Injection strength |
|----|-------------|-------------------|
| 0.80 | −0.250 | Strong |
| 0.85 | −0.176 | |
| 0.90 | −0.111 | Standard (paper's choice) |
| 0.95 | −0.053 | Weak |
| 0.99 | −0.010 | Very weak |

Meanwhile, the theoretical SGDM noise floor is L²σ²/(1−β²) (paper Theorem 3.5,
line 307–311):

| β | 1/(1−β²) |
|---|----------|
| 0.80 | 2.78 |
| 0.90 | 5.26 |
| 0.95 | 10.26 |
| 0.99 | 50.25 |

Higher β₁ → weaker injection AND higher noise floor. This predicts ECO may prefer lower
β₁ than standard training, potentially shifting the optimal hyperparameters.

### 4.2 Setup

**Model:** 100M and 430M (small enough for a sweep).

**Sweep grid:**
- β₁ ∈ {0.8, 0.85, 0.9, 0.95, 0.99}
- β₂ ∈ {0.95, 0.98, 0.99, 0.999}

At each (β₁, β₂): run both baseline and ECO.
Total: 5 × 4 × 2 = 40 runs at 100M.

Use the adapted sweep script (Section 0.2) to generate and launch all configs.

### 4.3 What to Measure

- **Validation loss** at each (β₁, β₂) → 2D heatmaps for baseline vs ECO.
- **Optimal (β₁, β₂)** for each method — does ECO shift the optimum?
- **Sensitivity contours** — is ECO more or less sensitive to β choice?
- **Injection magnitude** |Δm|/|m| at each β₁.
- **Heuristic validation** at each β₁.

---

## 5. ECO for Muon

### 5.1 Muon Background

Muon replaces Adam's per-element adaptive step size with spectral normalization of the
momentum via Newton-Schulz (NS) iterations. The user states that `torch.optim.Muon` is
now available in PyTorch nightly. torchtitan currently only supports Adam/AdamW
(`torchtitan/components/optimizer.py:314-317`), so we add Muon to the
`optimizer_classes` dict and handle any Muon-specific kwargs (NS steps, etc.).

Muon's update for a weight matrix W (following the standard implementation in
kellerjordan/modded-nanogpt, which torch.optim.Muon standardizes):

```
m_{t+1} = β · m_t + g_t                     # momentum (no (1−β) scaling)
X₀ = m_{t+1} / ‖m_{t+1}‖_F                 # normalize
Xₖ₊₁ = Xₖ · (aI + bXₖᵀXₖ + c(XₖᵀXₖ)²)   # NS iteration, 5 steps
u_{t+1} ≈ UVᵀ                               # polar factor of m_{t+1}
W_{t+1} = W_t − η · u_{t+1}                 # update
```

Muon applies only to 2D parameters (weight matrices). For 1D parameters (biases,
norms, embeddings), a standard Adam optimizer is used.

### 5.2 Derivation

**Step 1: SGDM ECO recap.** For SGDM (θ ← θ − η·m), the paper derives the exact
injection (Appendix A, paper lines 583–592) by constructing implicit master weights
θ* = θ̂ + e and implicit momentum m* = m̂ + (1/(ηβ))·e, and showing these follow
the standard SGDM recurrence. The approximate (memory-free) injection is:

```
Δm = (1/η)(1 − 1/β) · e
```

This works because the update θ ← θ − η·m is **linear** in m: injecting Δm into m
produces a correction of −η·Δm in θ.

**Step 2: Why Muon breaks the SGDM derivation.** For Muon, the update is
θ ← θ − η·NS(m), which is **nonlinear** in m. The virtual sequence construction
requires θ* − C·m* to follow gradient descent, but NS prevents the error terms from
canceling cleanly (the cancellation in the SGDM proof relies on linearity; see paper
lines 706–726 where the e_{t+1} coefficient vanishes *because* the update is linear in m).

**Step 3: What exact compensation would require.** For the implicit master weights to
match, we need (from the same logic as the SGDM proof):

```
NS(m*_{t+1}) = NS(m̃_{t+1}) + e_t/η
```

i.e., the orthogonalized implicit momentum must differ from the ECO orthogonalized
momentum by exactly e_t/η. Linearizing NS around m̃:

```
NS(m̃ + Δm) ≈ NS(m̃) + J_NS(m̃) · Δm
```

where J_NS is the Fréchet derivative of the polar factor map. So we need:

```
J_NS(m̃) · Δm = e_t/η
```

This requires inverting J_NS, which is problematic:
- J_NS is **singular**: perturbations to m that only scale its singular values (without
  rotating singular vectors) produce zero change in NS(m), because the polar factor UVᵀ
  is invariant to singular value scaling. So J_NS has a nontrivial null space.
- The full Fréchet derivative of the polar factor involves the singular value
  decomposition and is expensive to compute.

**Step 4: Practical approximation.** NS begins by normalizing m → m/‖m‖_F, then
iterates toward UVᵀ. For the first-order effect, NS maps a perturbation Δm to
approximately Δm/‖m‖_F (after projecting out the component along m's dominant singular
directions, which NS is insensitive to). Since the quantization error e is effectively
random with respect to m's singular structure, most of e lies outside the null space of
J_NS, and the effective gain is ≈ 1/‖m‖_F.

To achieve the SGDM-equivalent correction in the update direction, we compensate for
this gain by multiplying by ‖m‖_F:

```
Δm = (‖m̃_{t+1}‖_F / η)(1 − 1/β) · e_{t+1}
```

**This is the proposed ECO-Muon injection formula.** The ‖m‖_F factor plays the
analogous role to Adam's (√v + ε) factor — it converts from update-space back to
momentum-space, accounting for how the optimizer transforms momentum before applying it
as an update.

Computing ‖m‖_F is a single reduction (negligible cost).

**Step 5: Limitations of this derivation.**
- NS is not a simple division by ‖m‖_F — it does full spectral normalization. The
  ‖m‖_F scaling is a scalar approximation of a matrix-valued operation.
- Components of e aligned with the dominant singular vectors of m will be suppressed by
  NS regardless of the scaling. For low-rank momentum (which is typical), this means a
  small fraction of the error correction is lost each step.
- The e_t ≈ e_{t+1} approximation adds further error on top.

Whether these approximations are tolerable is an empirical question — hence the
experiment.

### 5.3 Experimental Setup

**Model:** 100M and 430M.

**Conditions:**

| Run | Optimizer | ECO | Formula |
|-----|-----------|-----|---------|
| Muon baseline | Muon | No | — |
| Muon naive (no MW, no ECO) | Muon | No | — |
| Muon ECO-A (naive SGDM) | Muon | Yes | Δm = (1/η)(1−1/β)·e |
| Muon ECO-B (Frobenius) | Muon | Yes | Δm = (‖m‖_F/η)(1−1/β)·e |
| Adam baseline | Adam | No | — |
| Adam ECO | Adam | Yes | Paper formula (Alg. 3) |

ECO-A is the control: what happens if we just apply the SGDM formula and ignore NS.
ECO-B is the derived formula. Comparing A vs B isolates the effect of the ‖m‖_F scaling.

### 5.4 Implementation

**Adding Muon to torchtitan:**
- Add `"Muon": torch.optim.Muon` to `optimizer_classes` in
  `torchtitan/components/optimizer.py:314-317`.
- Add Muon-specific config fields to `torchtitan/config/job_config.py` (momentum, NS
  steps, etc.).
- Muon uses Adam for 1D params (biases, norms) — handle this in the optimizer builder
  by splitting parameters by dimensionality.

**ECO-Muon injection:** Add `eco_inject_muon_a()` and `eco_inject_muon_b()` to
`torchtitan/components/eco.py`. The Muon optimizer state has `momentum_buffer` (not
`exp_avg`/`exp_avg_sq` like Adam). The injection targets this buffer.

### 5.5 What to Measure

- **Primary:** Does ECO-B prevent the 1/η divergence that naive master-weight removal
  causes? (Compare "Muon naive" vs "Muon ECO-B".)
- **Secondary:** How does ECO-B compare to ECO-A? Does the ‖m‖_F scaling matter?
- **Heuristic validation:** Same 4 metrics as Section 1.4. Additionally, log ‖m‖_F per
  layer over training to understand the scaling factor's dynamics.
- **Cross-optimizer:** How does the best Muon+ECO compare to Adam+ECO?

---

## 6. Implementation Roadmap

### Phase 1: Infrastructure + Base ECO
Everything else depends on this.

1. Port AIM logger and adapt sweep script (Section 0).
2. Implement `torchtitan/components/eco.py`: quantize → error → inject cycle for Adam,
   heuristic validation logging, configuration.
3. Wire ECO into `train.py` and `job_config.py`.
4. **Validation:** Replicate the paper's 30M–800M scaling results (Table 1) to confirm
   correctness. ECO+SR at 100M should produce a validation loss near 2.9888 (paper
   Table 1, line 387).

### Phase 2: Scale Up
5. Create Llama 3 8B ECO configs. Run 8B baseline + ECO. Analyze heuristic validation
   at 8B.

### Phase 3: BF16 Optimizer + β Sweep + Muon (parallelizable)
These three experiments are independent and can proceed in parallel.

6. Implement optimizer state dtype control. Run the 4 conditions at 430M/1B.
7. Run β₁/β₂ sweep at 100M using the sweep script.
8. Add Muon to torchtitan. Validate Muon baseline. Implement ECO-A and ECO-B. Run
   comparison at 100M/430M.

### Notes

- **MXFP4:** torchtitan currently supports only MXFP8 for training
  (`torchtitan/components/quantization/mx.py`; `torchtitan/config/job_config.py` only
  lists `mxfp8_cublas` and `mxfp8_cublas_rceil` recipes). MXFP4 appears only for
  checkpoint loading, not training (`torchtitan/models/gpt_oss/model/state_dict_adapter.py:56`).
  torchao may add MXFP4 training support in the future, at which point ECO + MXFP4
  becomes a natural experiment.
- **FP8 all-gather with FSDP:** If weights are already stored in FP8 (no master
  weights), FSDP all-gather communicates the FP8 data directly — this is just standard
  FSDP, not a special "quantized all-gather" mode. No separate experiment needed.
- **Gradient accumulation:** ECO operates per optimizer step. GA changes how gradients
  are computed but not how the optimizer step works. No special interaction expected.
