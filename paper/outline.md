# Paper Outline: Error Compensating Muon Does Scale.

Cite: Nikdan et al. "ECO: Quantized Training without Full-Precision Master Weights"
(arXiv:2601.22101). Cite: leloykun (https://leloykun.github.io/ponder/eco/) for the
weight decay correction and Jacobian-based ECO-Muon derivation.

## 1. Introduction
- ECO eliminates master weights via momentum error injection — proven up to 2.1B
- Muon is increasingly the optimizer of choice for LLM training
- We scale ECO further: 8B with Adam, Muon+ECO, and compose with BF16 optimizer states
- Contributions: 8B validation, ECO for Muon (4 injection strategies), BF16 states,
  β₁ sensitivity analysis

## 2. Background
- ECO recap (injection formula, convergence guarantees, 9 bytes/param)
- Muon recap (momentum + Newton-Schulz, why people use it)

## 3. Four Injection Strategies for Muon
- SGDM ECO recap: Δm = (1/η)(1 − 1/β)·e works because θ ← θ − η·m is linear in m
- Muon's update θ ← θ − η·NS(m) is nonlinear — the SGDM derivation doesn't directly apply
- **Pre-NS injection** (simplest): inject into momentum before NS, let NS process the
  error naturally. Math doesn't exactly match the derivation, but if NS is approximately
  linear locally it may be close enough.
- **Naive SGDM** (control): apply SGDM formula directly, ignore NS entirely. Measures
  how much the nonlinearity actually matters.
- **Frobenius scaling**: Δm = (‖m‖_F/η)(1 − 1/β)·e. NS normalizes by ‖m‖_F, so
  perturbations are attenuated by ~1/‖m‖_F. Compensate by scaling up. Analogous to
  Adam's √v+ε scaling.
- **Jacobian-based** (exact): solve J_NS·Δm = e/η via CG with finite-difference Jacobian.
  Expensive (extra NS evals per CG step) but mathematically exact. Upper bound on
  achievable performance.

## 4. Scaling Experiments
- Reproduce paper's 30M–800M scaling laws (validates implementation)
  - All 7 paper treatments: bf16, fp8±MW±SR, fp8+ECO±SR
  - Sweep: `paper_repro.py` at each model size
- Same table for Muon (validates ECO+Muon at multiple scales)
  - Sweep: `muon_repro.py` at each model size (winning approach from §5)
- Extend to Llama 3 8B (beyond paper's 1B/2.1B)
- Heuristic validation: does e_t ≈ e_{t+1} hold at scale?

## 5. Muon ECO Results
- Which injection strategy works best?
- Sweep: `muon_approaches.py` (4 approaches × speed)

## 6. Ablations
- β₁ sensitivity: does ECO shift the optimal β₁?
  - Sweep: `beta.py` (bf16 vs fp8_eco_sr × β₁ ∈ {0.8, 0.85, 0.9, 0.95, 0.99})
- FP8 + BF16 optimizer states: 9 → 5 bytes/param, does ECO compose with cheap states?
  - Sweep: `fp8_optim.py` (±ECO × FP32/BF16 states, all FP8 weights)
- BF16 + BF16 optimizer states: does ECO help even without FP8?
  - Sweep: `bf16_optim.py` (±ECO × FP32/BF16 states, all BF16 weights)
- BF16 weights + ECO: ECO on BF16 rounding error, no FP8
  - Sweep: `bf16_eco.py` (bf16 vs bf16+ECO)

**TODO:** Decide what to do about eval loss.

## 7. Conclusion
