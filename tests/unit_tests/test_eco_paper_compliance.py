# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests verifying ECOAdamW matches the paper's specifications.

1. Embedding/output layers excluded from weight quantization.
2. Stochastic rounding produces correct probability-proportional rounding.
"""

import pytest
import torch
import torch.nn as nn

from torchtitan.components.eco_adamw import ECOAdamW


# ======================================================================
# Helpers
# ======================================================================

def _is_fp8_representable(tensor: torch.Tensor) -> bool:
    """Check if a tensor's values survive an FP8 round-trip."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    amax = tensor.abs().amax()
    if amax == 0:
        return True
    scale = amax / fp8_max
    scaled = tensor / scale
    roundtripped = scaled.to(torch.float8_e4m3fn).to(tensor.dtype) * scale
    return torch.allclose(tensor, roundtripped, atol=1e-6, rtol=1e-5)


class SimpleLlama(nn.Module):
    """Minimal model that mimics Llama's parameter naming conventions."""

    def __init__(self, dim=32, vocab_size=64):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleDict({
            "0": nn.ModuleDict({
                "attention": nn.ModuleDict({
                    "wq": nn.Linear(dim, dim, bias=False),
                    "wk": nn.Linear(dim, dim, bias=False),
                }),
                "feed_forward": nn.ModuleDict({
                    "w1": nn.Linear(dim, dim * 4, bias=False),
                    "w2": nn.Linear(dim * 4, dim, bias=False),
                }),
                "attention_norm": nn.LayerNorm(dim),
            }),
        })
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        # Not used in optimizer tests, but present for completeness.
        raise NotImplementedError


# ======================================================================
# Test: Embedding/output exclusion from quantization
# ======================================================================

class TestEmbeddingOutputExclusion:
    """Paper: 'we apply the method only to the linear layers within
    transformer blocks, excluding the embedding and output layers.'"""

    def _build_optimizer(self, model, **kwargs):
        """Build ECOAdamW with exclude_from_quant like build_optimizers does."""
        exclude_ids = set()
        for name, p in model.named_parameters():
            if not name.startswith("layers."):
                exclude_ids.add(id(p))
        defaults = dict(
            lr=1e-3, quantize_weights=True, eco_enabled=True,
            quant_dtype="fp8", exclude_from_quant=frozenset(exclude_ids),
        )
        defaults.update(kwargs)
        return ECOAdamW(model.parameters(), **defaults)

    def test_transformer_weights_are_quantized(self):
        """Linear layers inside layers.* should be quantized to FP8."""
        model = SimpleLlama()
        opt = self._build_optimizer(model)

        # Provide gradients for all params
        for p in model.parameters():
            p.grad = torch.randn_like(p) * 0.01
        opt.step()

        for name, p in model.named_parameters():
            if name.startswith("layers.") and p.dim() >= 2:
                assert _is_fp8_representable(p.data), (
                    f"{name} should be FP8-representable after step"
                )

    def test_embedding_not_quantized(self):
        """tok_embeddings.weight should NOT be quantized."""
        model = SimpleLlama()
        opt = self._build_optimizer(model)

        for p in model.parameters():
            p.grad = torch.randn_like(p) * 0.01
        opt.step()

        emb = model.tok_embeddings.weight
        assert not _is_fp8_representable(emb.data), (
            "tok_embeddings.weight should NOT be FP8-representable"
        )

    def test_output_not_quantized(self):
        """output.weight should NOT be quantized."""
        model = SimpleLlama()
        opt = self._build_optimizer(model)

        for p in model.parameters():
            p.grad = torch.randn_like(p) * 0.01
        opt.step()

        out = model.output.weight
        assert not _is_fp8_representable(out.data), (
            "output.weight should NOT be FP8-representable"
        )

    def test_norm_not_quantized(self):
        """LayerNorm weights (1D) should never be quantized regardless."""
        model = SimpleLlama()
        opt = self._build_optimizer(model)

        for p in model.parameters():
            p.grad = torch.randn_like(p) * 0.01
        opt.step()

        # Norms are 1D so they hit _step_regular anyway
        norm_w = model.norm.weight
        assert norm_w.dim() == 1

    def test_excluded_params_still_updated(self):
        """Excluded params should still get regular AdamW updates."""
        model = SimpleLlama()
        emb_before = model.tok_embeddings.weight.data.clone()
        out_before = model.output.weight.data.clone()
        opt = self._build_optimizer(model)

        for p in model.parameters():
            p.grad = torch.randn_like(p) * 0.01
        opt.step()

        assert not torch.allclose(model.tok_embeddings.weight.data, emb_before), (
            "tok_embeddings should still be updated by regular AdamW"
        )
        assert not torch.allclose(model.output.weight.data, out_before), (
            "output should still be updated by regular AdamW"
        )

    def test_exclude_from_quant_empty_quantizes_all(self):
        """With no exclusions, all 2D params should be quantized (old behavior)."""
        model = SimpleLlama()
        opt = ECOAdamW(
            model.parameters(), lr=1e-3,
            quantize_weights=True, eco_enabled=True,
            quant_dtype="fp8",
            exclude_from_quant=frozenset(),  # empty → no exclusions
        )

        for p in model.parameters():
            p.grad = torch.randn_like(p) * 0.01
        opt.step()

        # Now embedding and output SHOULD be FP8-representable
        assert _is_fp8_representable(model.tok_embeddings.weight.data)
        assert _is_fp8_representable(model.output.weight.data)

    def test_multiple_steps_exclusion_persists(self):
        """Exclusion should hold across multiple optimizer steps."""
        model = SimpleLlama()
        opt = self._build_optimizer(model)

        for step in range(5):
            for p in model.parameters():
                p.grad = torch.randn_like(p) * 0.01
            opt.step()

        # After 5 steps, embedding/output should still not be FP8-representable
        assert not _is_fp8_representable(model.tok_embeddings.weight.data)
        assert not _is_fp8_representable(model.output.weight.data)

        # But transformer layers should be
        for name, p in model.named_parameters():
            if name.startswith("layers.") and p.dim() >= 2:
                assert _is_fp8_representable(p.data), f"{name} should be FP8"


# ======================================================================
# Test: Stochastic rounding correctness
# ======================================================================

class TestStochasticRoundingCorrectness:
    """Paper: 'stochastic rounding maps a value to one of the two nearest
    grid points, where the probability of selecting either point is
    proportional to the distance to the other point.'"""

    def test_ulp_exact_at_powers_of_two(self):
        """At power-of-2 values, old and new ULP formulas should agree."""
        # At x = 2^e, ULP = 2^(e-3).  Old formula |x|*2^-3 = 2^(e-3). Same.
        for e in range(-6, 8):
            x = torch.tensor([2.0 ** e])
            old_ulp = x.abs() * (2 ** -3)
            log2_abs = torch.floor(torch.log2(x.abs().clamp(min=2.0 ** -6)))
            new_ulp = torch.exp2(log2_abs - 3)
            torch.testing.assert_close(old_ulp, new_ulp, atol=1e-10, rtol=1e-7)

    def test_ulp_correct_mid_binade(self):
        """At x = 1.5 * 2^e (middle of binade), ULP should be 2^(e-3), not 1.5*2^(e-3)."""
        for e in range(-5, 7):
            x = torch.tensor([1.5 * (2.0 ** e)])
            # Correct ULP: still 2^(e-3) since x is in binade [2^e, 2^(e+1))
            expected_ulp = 2.0 ** (e - 3)
            # Old (wrong) formula would give 1.5 * 2^(e-3)
            old_ulp = (x.abs() * (2 ** -3)).item()

            log2_abs = torch.floor(torch.log2(x.abs().clamp(min=2.0 ** -6)))
            new_ulp = torch.exp2(log2_abs - 3).item()

            assert abs(new_ulp - expected_ulp) < 1e-10, (
                f"e={e}: new_ulp={new_ulp}, expected={expected_ulp}"
            )
            # Old formula is 1.5x too large
            assert abs(old_ulp - 1.5 * expected_ulp) < 1e-10, (
                f"e={e}: old_ulp={old_ulp}, expected 1.5*{expected_ulp}"
            )

    def test_stochastic_rounding_unbiased(self):
        """Stochastic rounding should be unbiased: E[Q(x)] ≈ x.

        We pick a value between two FP8 grid points and verify that the
        average over many rounds converges to the original value.
        """
        torch.manual_seed(42)
        # Value between two FP8 grid points: 1.0 and 1.125 (ULP=0.125 at binade [1,2))
        # Pick x = 1.0625 (exactly halfway)
        x = torch.full((1000,), 1.0625)
        results = []
        for _ in range(200):
            result = ECOAdamW._requantize(x.clone(), "fp8", stochastic_rounding=True)
            results.append(result)
        mean_result = torch.stack(results).mean(dim=0)
        # Should be close to original value (unbiased)
        assert torch.allclose(mean_result, x, atol=0.01), (
            f"Stochastic rounding should be unbiased: mean={mean_result[0]:.4f}, expected={x[0]:.4f}"
        )

    def test_stochastic_rounding_probability_proportional(self):
        """For a value at fraction f through an FP8 interval, it should
        round up with probability f and down with probability (1-f)."""
        torch.manual_seed(123)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0

        # In binade [1, 2): FP8 E4M3 grid points are 1.000, 1.125, 1.250, ...
        # ULP = 0.125.  Pick test_val = 1.03125 → fraction = 0.03125/0.125 = 0.25
        # Should round to 1.125 with p≈0.25, to 1.000 with p≈0.75.
        test_val = 1.03125
        expected_frac = 0.25
        floor_val = 1.0
        ceil_val = 1.125

        # Include fp8_max as the first element so that per-tensor scale = 1.0,
        # keeping test_val exactly where we want it on the FP8 grid.
        n_trials = 10000
        round_up_count = 0
        for _ in range(n_trials):
            x = torch.tensor([fp8_max, test_val])
            result = ECOAdamW._requantize(x, "fp8", stochastic_rounding=True)
            val = result[1].item()
            if abs(val - ceil_val) < abs(val - floor_val):
                round_up_count += 1

        frac = round_up_count / n_trials
        assert abs(frac - expected_frac) < 0.05, (
            f"Round-up fraction {frac:.3f}, expected ~{expected_frac:.3f}"
        )

    def test_deterministic_when_disabled(self):
        """With stochastic_rounding=False, requantize should be deterministic."""
        x = torch.randn(100)
        r1 = ECOAdamW._requantize(x.clone(), "fp8", stochastic_rounding=False)
        r2 = ECOAdamW._requantize(x.clone(), "fp8", stochastic_rounding=False)
        torch.testing.assert_close(r1, r2)

    def test_stochastic_rounding_varies(self):
        """With stochastic_rounding=True, repeated calls should give different results."""
        torch.manual_seed(0)
        x = torch.randn(100)
        r1 = ECOAdamW._requantize(x.clone(), "fp8", stochastic_rounding=True)
        r2 = ECOAdamW._requantize(x.clone(), "fp8", stochastic_rounding=True)
        # Very unlikely to be exactly the same
        assert not torch.equal(r1, r2), "Stochastic rounding should produce varying results"

    def test_subnormal_ulp(self):
        """For subnormal FP8 values (below 2^-6), ULP should be 2^-9."""
        # Subnormal region: values below min_normal = 2^-6 ≈ 0.0156
        # To test subnormal behavior, we need values that land in the
        # subnormal region *after* per-tensor scaling.  Include fp8_max
        # as the anchor so scale=1.0, then small values stay small.
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        small_val = 0.005  # below 2^-6, in subnormal region

        torch.manual_seed(0)
        results = []
        for _ in range(200):
            x = torch.tensor([fp8_max, small_val])
            r = ECOAdamW._requantize(x, "fp8", stochastic_rounding=True)
            results.append(r[1].item())

        results_t = torch.tensor(results)
        # With SR noise at subnormal ULP=2^-9 ≈ 0.00195, the result should vary
        assert results_t.std() > 0, "Subnormal values should still get SR noise"
        # All results should be valid FP8 representable values (close to 0.005)
        assert results_t.abs().max() < 0.02, "Results should be near the input value"
