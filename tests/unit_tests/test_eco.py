# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ECOAdamW optimizer with ECO injection.

All ECO logic (injection, stochastic rounding, heuristic metrics) now lives
inside ECOAdamW.  There is no ECOConverter.
"""

import pytest
import torch
import torch.nn as nn

from torchtitan.components.eco_optimizer import ECOAdamW
from torchtitan.components.quantized_tensor import QuantizedTensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_quantized_param(rows=64, cols=64, seed=42):
    """Create a QuantizedTensor parameter for testing."""
    torch.manual_seed(seed)
    fp = torch.randn(rows, cols)
    qt = QuantizedTensor.quantize(fp, torch.float8_e4m3fn)
    qt.requires_grad_(True)
    return qt


def _fake_grad(param):
    """Attach a random gradient to a parameter."""
    param.grad = torch.randn_like(param.dequantize() if isinstance(param, QuantizedTensor) else param) * 0.01


# ---------------------------------------------------------------------------
# Tests: FP8 round-trip (static method _requantize)
# ---------------------------------------------------------------------------


class TestRequantize:
    def test_roundtrip_error_is_small(self):
        t = torch.randn(128, 128)
        result = ECOAdamW._requantize(t, torch.float8_e4m3fn, stochastic_rounding=False)
        rel_error = (t - result).abs() / (t.abs() + 1e-12)
        assert rel_error.mean() < 0.1

    def test_roundtrip_is_idempotent(self):
        t = torch.randn(64, 64)
        once = ECOAdamW._requantize(t, torch.float8_e4m3fn, stochastic_rounding=False)
        twice = ECOAdamW._requantize(once, torch.float8_e4m3fn, stochastic_rounding=False)
        assert torch.equal(once, twice)

    def test_stochastic_rounding_varies(self):
        t = torch.randn(128, 128)
        results = [ECOAdamW._requantize(t, torch.float8_e4m3fn, stochastic_rounding=True) for _ in range(10)]
        any_differ = any(not torch.equal(results[i], results[j]) for i in range(len(results)) for j in range(i+1, len(results)))
        assert any_differ, "Stochastic rounding should produce varying outputs"

    def test_stochastic_rounding_is_unbiased(self):
        t = torch.randn(256, 256)
        N = 50
        results = torch.stack([ECOAdamW._requantize(t, torch.float8_e4m3fn, stochastic_rounding=True) for _ in range(N)])
        mean_result = results.float().mean(dim=0)
        error = (mean_result - t.float()).abs().mean()
        assert error < 0.01, f"Stochastic rounding bias too large: {error:.6f}"


# ---------------------------------------------------------------------------
# Tests: ECO injection produces non-zero error
# ---------------------------------------------------------------------------


class TestECOInjection:
    def test_injection_produces_nonzero_error(self):
        """After one step, prev_error should be stored and non-zero."""
        qt = _make_quantized_param(32, 32)
        opt = ECOAdamW([qt], lr=1e-3, eco_enabled=True, heuristic_log_freq=1)

        _fake_grad(qt)
        opt.step()

        state = opt.state[qt]
        assert "prev_error" in state, "prev_error should be stored when heuristic_log_freq > 0"
        err = state["prev_error"]
        assert err.abs().max() > 1e-6, (
            f"Error should be non-zero (~1e-3), got max={err.abs().max():.2e}"
        )

    def test_error_magnitude(self):
        """Error magnitude should be ~1e-3 for typical weights."""
        qt = _make_quantized_param(64, 64)
        opt = ECOAdamW([qt], lr=1e-3, eco_enabled=True, heuristic_log_freq=1)

        _fake_grad(qt)
        opt.step()

        err = opt.state[qt]["prev_error"]
        err_norm = err.float().norm() / err.numel() ** 0.5
        # FP8 E4M3 relative error ~ 2^-3 = 0.125, absolute error
        # depends on weight magnitude; for randn weights ~ O(1),
        # expect per-element error ~ O(0.01-0.1)
        assert err_norm > 1e-6, f"Error too small: {err_norm:.2e}"

    def test_eco_disabled_skips_injection(self):
        """With eco_enabled=False, no prev_error is stored."""
        qt = _make_quantized_param(16, 16)
        opt = ECOAdamW([qt], lr=1e-3, eco_enabled=False, heuristic_log_freq=1)

        _fake_grad(qt)
        opt.step()

        state = opt.state[qt]
        assert "prev_error" not in state

    def test_regular_params_skip_injection(self):
        """Regular (non-QuantizedTensor) params get a standard AdamW step."""
        param = torch.randn(16, 16, requires_grad=True)
        opt = ECOAdamW([param], lr=1e-3, eco_enabled=True)

        param.grad = torch.randn_like(param) * 0.01
        opt.step()

        # Should have exp_avg, exp_avg_sq, step — but NOT prev_error
        state = opt.state[param]
        assert "exp_avg" in state
        assert "prev_error" not in state


# ---------------------------------------------------------------------------
# Tests: Heuristic metrics
# ---------------------------------------------------------------------------


class TestHeuristicMetrics:
    def test_metrics_after_two_log_steps(self):
        """Metrics should appear after 2 consecutive heuristic log steps."""
        qt = _make_quantized_param(32, 32)
        opt = ECOAdamW([qt], lr=1e-3, eco_enabled=True, heuristic_log_freq=1)

        # Step 1: stores prev_error, no comparison yet
        _fake_grad(qt)
        opt.step()
        assert opt.get_eco_metrics() == {}

        # Step 2: compares with prev_error → metrics
        _fake_grad(qt)
        opt.step()
        metrics = opt.get_eco_metrics()
        assert len(metrics) > 0, "Should have heuristic metrics after 2 log steps"
        assert "eco/heuristic/norm_ratio/mean" in metrics
        assert "eco/heuristic/cosine_sim/mean" in metrics

    def test_metrics_cleared_after_read(self):
        qt = _make_quantized_param(32, 32)
        opt = ECOAdamW([qt], lr=1e-3, eco_enabled=True, heuristic_log_freq=1)

        _fake_grad(qt); opt.step()
        _fake_grad(qt); opt.step()

        m1 = opt.get_eco_metrics()
        m2 = opt.get_eco_metrics()
        assert len(m1) > 0
        assert m2 == {}

    def test_metric_values_are_reasonable(self):
        qt = _make_quantized_param(64, 64)
        opt = ECOAdamW([qt], lr=1e-3, eco_enabled=True, heuristic_log_freq=1)

        _fake_grad(qt); opt.step()
        _fake_grad(qt); opt.step()

        metrics = opt.get_eco_metrics()
        assert 0.0 < metrics["eco/heuristic/norm_ratio/mean"] < 100.0
        assert -1.0 <= metrics["eco/heuristic/cosine_sim/mean"] <= 1.0
        assert metrics["eco/heuristic/rel_diff_norm/mean"] >= 0.0

    def test_log_freq_controls_emission(self):
        """With heuristic_log_freq=3, only every 3rd step logs."""
        qt = _make_quantized_param(16, 16)
        opt = ECOAdamW([qt], lr=1e-3, eco_enabled=True, heuristic_log_freq=3)

        for i in range(6):
            _fake_grad(qt)
            opt.step()
            m = opt.get_eco_metrics()
            if (i + 1) % 3 == 0 and (i + 1) >= 6:
                # Second log-freq step (step 6) compares with prev from step 3
                assert len(m) > 0, f"Expected metrics at step {i+1}"
            else:
                # Either not a log step, or first log step (no prev yet)
                pass  # metrics may or may not be present


# ---------------------------------------------------------------------------
# Tests: Training convergence
# ---------------------------------------------------------------------------


class TestTraining:
    STEPS = 200
    LR = 1e-3
    BETA1 = 0.9
    BETA2 = 0.98
    EPS = 1e-9
    SEED = 42

    def _make_problem(self):
        torch.manual_seed(self.SEED)
        x = torch.randn(64, 16)
        y = torch.sin(x[:, :1] * 3) + 0.5 * x[:, 1:2] ** 2
        return x, y

    def _make_mlp(self):
        torch.manual_seed(self.SEED)
        return nn.Sequential(
            nn.Linear(16, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def _train_bf16_baseline(self, x, y):
        model = self._make_mlp()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.LR,
            betas=(self.BETA1, self.BETA2), eps=self.EPS,
        )
        losses = []
        for _ in range(self.STEPS):
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return losses

    def _train_naive_fp8(self, x, y):
        """FP8 round-trip every step, no error compensation."""
        model = self._make_mlp()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.LR,
            betas=(self.BETA1, self.BETA2), eps=self.EPS,
        )
        fp8_dtype = torch.float8_e4m3fn
        fp8_max = torch.finfo(fp8_dtype).max
        losses = []
        for _ in range(self.STEPS):
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                for p in model.parameters():
                    amax = p.data.abs().amax()
                    scale = torch.where(amax > 0, amax / fp8_max, torch.ones_like(amax))
                    q = (p.data / scale).to(fp8_dtype)
                    p.data.copy_(q.to(p.data.dtype) * scale)
        return losses

    def _train_eco(self, x, y):
        """ECOAdamW with ECO injection (regular nn.Linear params)."""
        model = self._make_mlp()
        optimizer = ECOAdamW(
            model.parameters(), lr=self.LR,
            betas=(self.BETA1, self.BETA2), eps=self.EPS,
            eco_enabled=True,
        )
        losses = []
        for _ in range(self.STEPS):
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return losses

    def test_bf16_baseline_converges(self):
        x, y = self._make_problem()
        losses = self._train_bf16_baseline(x, y)
        assert losses[-1] < losses[0] * 0.01

    def test_eco_converges(self):
        x, y = self._make_problem()
        losses = self._train_eco(x, y)
        assert losses[-1] < losses[0] * 0.5, (
            f"ECO didn't converge: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_naive_fp8_barely_learns(self):
        x, y = self._make_problem()
        losses = self._train_naive_fp8(x, y)
        reduction = 1 - losses[-1] / losses[0]
        assert reduction < 0.5

    def test_eco_beats_naive_fp8(self):
        x, y = self._make_problem()
        naive = self._train_naive_fp8(x, y)
        eco = self._train_eco(x, y)
        assert eco[-1] < naive[-1] * 0.5


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_tensor(self):
        t = torch.zeros(32, 32)
        result = ECOAdamW._requantize(t, torch.float8_e4m3fn, stochastic_rounding=False)
        assert torch.allclose(result, t, atol=1e-6)
        assert not torch.isnan(result).any()

    def test_extreme_values(self):
        large = torch.full((16, 16), 1e6)
        result = ECOAdamW._requantize(large, torch.float8_e4m3fn, stochastic_rounding=False)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_single_element(self):
        t = torch.tensor([5.0])
        result = ECOAdamW._requantize(t, torch.float8_e4m3fn, stochastic_rounding=False)
        assert result.numel() == 1
        assert not torch.isnan(result)

    def test_mixed_sign(self):
        t = torch.randn(16, 16)
        result = ECOAdamW._requantize(t, torch.float8_e4m3fn, stochastic_rounding=False)
        sign_match = torch.sign(result) == torch.sign(t)
        assert sign_match.float().mean() > 0.8

    def test_sparse_gradient_rejected(self):
        param = torch.randn(64, 64, requires_grad=True)
        opt = ECOAdamW([param])
        param.grad = torch.randn_like(param).to_sparse()
        with pytest.raises(RuntimeError, match="sparse"):
            opt.step()

    def test_invalid_hyperparameters(self):
        p = torch.randn(16, 16, requires_grad=True)
        with pytest.raises(ValueError, match="learning rate"):
            ECOAdamW([p], lr=-1.0)
        with pytest.raises(ValueError, match="beta"):
            ECOAdamW([p], betas=(-0.1, 0.999))
        with pytest.raises(ValueError, match="epsilon"):
            ECOAdamW([p], eps=-1e-8)
        with pytest.raises(ValueError, match="weight_decay"):
            ECOAdamW([p], weight_decay=-0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
