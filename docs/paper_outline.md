# Paper Outline: Extending ECO to Modern Optimizers

## 1. Introduction
- ECO eliminates master weights via momentum error injection (recap)
- ECO was derived for SGDM/Adam — both linear in momentum
- Modern optimizers (Muon) use nonlinear momentum transforms (Newton-Schulz)
- This paper: extend ECO to nonlinear optimizers, validate at 8B scale, explore
  further memory reduction via BF16 optimizer states

## 2. Background
- ECO recap (Algorithm 3, injection formula, convergence result)
- Muon: momentum + Newton-Schulz spectral normalization
- Why Muon breaks ECO's derivation (nonlinearity in m)

## 3. ECO for Muon
- The problem: NS(m + Δm) ≠ NS(m) + NS(Δm)
- Four approaches: pre-NS, naive SGDM, Frobenius scaling, Jacobian-based
- Results: which approaches work, comparison to Adam+ECO
- Sweep: `muon_approaches.py` (4 approaches × speed)

## 4. Scaling Experiments
- Reproduce paper's 30M–800M scaling laws (Table 1 replication)
  - All 7 paper treatments: bf16, fp8±MW±SR, fp8+ECO±SR
  - Sweep: `paper_repro.py` at each model size
- Extend to Llama 3 8B (beyond paper's 1B/2.1B)
- Heuristic validation: does e_t ≈ e_{t+1} hold at scale?

## 5. Ablations
- β₁ sensitivity: injection strength vs noise floor tradeoff
  - Sweep: `beta.py` (β₁ ∈ {0.8, 0.85, 0.9, 0.95, 0.99}, β₂ = 0.98 fixed)
- BF16 optimizer states: 9 → 5 bytes/param, does ECO still work?
  - Sweep: `bf16_optim.py` (±ECO × FP32/BF16 states)
- BF16 weights + ECO: ECO on BF16 rounding error, no FP8
  - Sweep: `bf16_eco.py` (bf16 vs bf16+ECO)

## 6. Conclusion
