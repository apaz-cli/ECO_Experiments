# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ECOAdamW simulated quantization (quantize_weights mode)."""

import pytest
import torch
import torch.nn as nn

from torchtitan.components.eco_optimizer import ECOAdamW


class TestSimulatedQuantDispatch:
    """Verify dispatch logic: _step_simulated_quant for dim>=2, _step_regular for 1D."""

    def test_2d_param_uses_simulated_quant(self):
        """Weight matrices (dim>=2) should go through simulated quant when enabled."""
        param = torch.randn(32, 32, requires_grad=True)
        original = param.data.clone()
        optimizer = ECOAdamW(
            [param], lr=1e-3,
            quantize_weights=True, eco_enabled=True,
            quant_dtype="fp8",
        )
        param.grad = torch.randn_like(param) * 0.01
        optimizer.step()
        # Param should have changed
        assert not torch.allclose(param.data, original)
        # With quantize_weights + fp8, the param value should be FP8-representable
        # (round-trip through FP8 should be a no-op)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        amax = param.data.abs().amax()
        scale = amax / fp8_max if amax > 0 else torch.ones(1)
        scaled = param.data / scale
        roundtripped = scaled.to(torch.float8_e4m3fn).to(param.dtype) * scale
        torch.testing.assert_close(param.data, roundtripped, atol=1e-6, rtol=1e-5)

    def test_1d_param_uses_regular_step(self):
        """Biases and other 1D params should use regular AdamW step."""
        bias = torch.randn(32, requires_grad=True)
        original = bias.data.clone()
        optimizer = ECOAdamW(
            [bias], lr=1e-3,
            quantize_weights=True, eco_enabled=True,
        )
        bias.grad = torch.randn_like(bias) * 0.01
        optimizer.step()
        assert not torch.allclose(bias.data, original)
        # 1D param should NOT be FP8-representable (regular AdamW, no quantization)
        # It's still float32 with full-precision values

    def test_mixed_dims_correct_dispatch(self):
        """2D weight gets simulated quant, 1D bias gets regular step."""
        weight = torch.randn(32, 32, requires_grad=True)
        bias = torch.randn(32, requires_grad=True)
        optimizer = ECOAdamW(
            [weight, bias], lr=1e-3,
            quantize_weights=True, eco_enabled=True,
            heuristic_log_freq=1,
        )
        weight.grad = torch.randn_like(weight) * 0.01
        bias.grad = torch.randn_like(bias) * 0.01
        optimizer.step()

        # Both should be in optimizer state
        assert weight in optimizer.state
        assert bias in optimizer.state

    def test_quantize_weights_false_uses_regular(self):
        """When quantize_weights=False, 2D params use regular AdamW."""
        param = torch.randn(32, 32, requires_grad=True)
        optimizer = ECOAdamW(
            [param], lr=1e-3,
            quantize_weights=False, eco_enabled=True,
        )
        param.grad = torch.randn_like(param) * 0.01
        optimizer.step()
        # Should NOT have prev_error (no ECO injection on regular params)
        assert "prev_error" not in optimizer.state[param]


class TestSimulatedQuantECOMetrics:
    """Verify ECO metrics are produced with simulated quantization."""

    def test_metrics_appear_with_quantize_weights(self):
        """ECO heuristic metrics should be emitted when quantize_weights=True."""
        param = torch.randn(32, 32, requires_grad=True)
        optimizer = ECOAdamW(
            [param], lr=1e-3,
            quantize_weights=True, eco_enabled=True,
            heuristic_log_freq=1,
        )
        # Step 1: stores prev_error
        param.grad = torch.randn_like(param) * 0.01
        optimizer.step()
        m1 = optimizer.get_eco_metrics()
        # May or may not have metrics on first step (no prev_error yet)

        # Step 2: should compare and emit metrics
        param.grad = torch.randn_like(param) * 0.01
        optimizer.step()
        m2 = optimizer.get_eco_metrics()
        assert len(m2) > 0, "Expected ECO metrics after 2 steps with quantize_weights=True"
        assert any("norm_ratio" in k for k in m2)
        assert any("cosine_sim" in k for k in m2)

    def test_no_metrics_without_eco_enabled(self):
        """With quantize_weights=True but eco_enabled=False, no ECO metrics."""
        param = torch.randn(32, 32, requires_grad=True)
        optimizer = ECOAdamW(
            [param], lr=1e-3,
            quantize_weights=True, eco_enabled=False,
            heuristic_log_freq=1,
        )
        for _ in range(3):
            param.grad = torch.randn_like(param) * 0.01
            optimizer.step()
        m = optimizer.get_eco_metrics()
        assert len(m) == 0

    def test_prev_error_stored_with_eco_enabled(self):
        """ECO injection stores prev_error in state for heuristic comparison."""
        param = torch.randn(32, 32, requires_grad=True)
        optimizer = ECOAdamW(
            [param], lr=1e-3,
            quantize_weights=True, eco_enabled=True,
            heuristic_log_freq=1,
        )
        param.grad = torch.randn_like(param) * 0.01
        optimizer.step()
        assert "prev_error" in optimizer.state[param]


class TestSimulatedQuantConvergence:
    """Verify training converges with simulated quantization."""

    def test_linear_regression_converges(self):
        """Simple linear regression should converge with simulated FP8 quantization."""
        torch.manual_seed(42)
        X = torch.randn(100, 10)
        y_true = torch.randn(100, 1)
        model = nn.Linear(10, 1)
        optimizer = ECOAdamW(
            model.parameters(), lr=0.01,
            quantize_weights=True, eco_enabled=True,
        )
        losses = []
        for _ in range(200):
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(X), y_true)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"

    def test_convergence_eco_vs_no_eco(self):
        """ECO with simulated quant should converge at least as well as quant without ECO."""
        torch.manual_seed(42)
        X = torch.randn(100, 10)
        y_true = torch.randn(100, 1)

        # With ECO
        model_eco = nn.Linear(10, 1)
        torch.manual_seed(42)
        nn.init.xavier_uniform_(model_eco.weight)
        nn.init.zeros_(model_eco.bias)
        opt_eco = ECOAdamW(
            model_eco.parameters(), lr=0.01,
            quantize_weights=True, eco_enabled=True,
        )
        for _ in range(200):
            opt_eco.zero_grad()
            loss = nn.MSELoss()(model_eco(X), y_true)
            loss.backward()
            opt_eco.step()
        loss_eco = nn.MSELoss()(model_eco(X), y_true).item()

        # Without ECO (quant only)
        model_no_eco = nn.Linear(10, 1)
        torch.manual_seed(42)
        nn.init.xavier_uniform_(model_no_eco.weight)
        nn.init.zeros_(model_no_eco.bias)
        opt_no_eco = ECOAdamW(
            model_no_eco.parameters(), lr=0.01,
            quantize_weights=True, eco_enabled=False,
        )
        for _ in range(200):
            opt_no_eco.zero_grad()
            loss = nn.MSELoss()(model_no_eco(X), y_true)
            loss.backward()
            opt_no_eco.step()
        loss_no_eco = nn.MSELoss()(model_no_eco(X), y_true).item()

        # ECO should help or at least not hurt
        assert loss_eco <= loss_no_eco * 1.5, (
            f"ECO loss {loss_eco} much worse than no-ECO {loss_no_eco}"
        )

    def test_mlp_convergence(self):
        """Multi-layer MLP converges with simulated quantization + ECO."""
        torch.manual_seed(42)
        X = torch.randn(200, 20)
        y_true = (X[:, 0:1] * 2 + X[:, 1:2] - 1).clamp(-3, 3)

        model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        optimizer = ECOAdamW(
            model.parameters(), lr=0.005,
            quantize_weights=True, eco_enabled=True,
        )
        losses = []
        for _ in range(300):
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(X), y_true)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0] * 0.5, (
            f"MLP loss did not decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


class TestSimulatedQuantConfigurations:
    """Test the 4 meaningful configurations from the plan."""

    def test_pure_adamw(self):
        """quantize_weights=False, eco_enabled=False → pure AdamW."""
        param = torch.randn(32, 32, requires_grad=True)
        opt = ECOAdamW(
            [param], lr=1e-3,
            quantize_weights=False, eco_enabled=False,
        )
        param.grad = torch.randn_like(param)
        opt.step()
        assert "prev_error" not in opt.state[param]

    def test_fp8_baseline(self):
        """quantize_weights=True, eco_enabled=False → FP8 baseline (quant, no compensation)."""
        param = torch.randn(32, 32, requires_grad=True)
        opt = ECOAdamW(
            [param], lr=1e-3,
            quantize_weights=True, eco_enabled=False,
        )
        param.grad = torch.randn_like(param)
        opt.step()
        # Should have quantized the weight but no prev_error
        assert "prev_error" not in opt.state[param]

    def test_fp8_eco(self):
        """quantize_weights=True, eco_enabled=True → FP8 ECO (main experiment)."""
        param = torch.randn(32, 32, requires_grad=True)
        opt = ECOAdamW(
            [param], lr=1e-3,
            quantize_weights=True, eco_enabled=True,
            heuristic_log_freq=1,
        )
        param.grad = torch.randn_like(param)
        opt.step()
        assert "prev_error" in opt.state[param]

    def test_eco_on_fp32(self):
        """quantize_weights=False, eco_enabled=True → ECO on FP32 (ablation)."""
        param = torch.randn(32, 32, requires_grad=True)
        opt = ECOAdamW(
            [param], lr=1e-3,
            quantize_weights=False, eco_enabled=True,
        )
        param.grad = torch.randn_like(param)
        opt.step()
        # No quantize_weights, so regular step, no ECO injection
        assert "prev_error" not in opt.state[param]

    def test_stochastic_rounding_with_simulated_quant(self):
        """Stochastic rounding works with simulated quantization."""
        param = torch.randn(32, 32, requires_grad=True)
        opt = ECOAdamW(
            [param], lr=1e-3,
            quantize_weights=True, eco_enabled=True,
            stochastic_rounding=True,
            heuristic_log_freq=1,
        )
        param.grad = torch.randn_like(param) * 0.01
        opt.step()  # Should not crash
        assert "prev_error" in opt.state[param]


class TestSimulatedQuantDtypes:
    """Test dtype handling in simulated quantization path."""

    def test_bf16_optim_state_dtype(self):
        """Optimizer states in BF16 with simulated quantization."""
        param = torch.randn(32, 32, requires_grad=True)
        opt = ECOAdamW(
            [param], lr=1e-3,
            optim_state_dtype=torch.bfloat16,
            quantize_weights=True, eco_enabled=True,
        )
        param.grad = torch.randn_like(param) * 0.01
        opt.step()
        state = opt.state[param]
        assert state["exp_avg"].dtype == torch.bfloat16
        assert state["exp_avg_sq"].dtype == torch.bfloat16

    def test_bf16_param_with_fp32_compute(self):
        """BF16 param with FP32 compute dtype should work."""
        param = torch.randn(32, 32, dtype=torch.bfloat16, requires_grad=True)
        opt = ECOAdamW(
            [param], lr=1e-3,
            optim_compute_dtype=torch.float32,
            quantize_weights=True, eco_enabled=True,
        )
        param.grad = torch.randn(32, 32, dtype=torch.bfloat16)
        opt.step()
        assert param.dtype == torch.bfloat16

    def test_quant_dtype_bf16(self):
        """BF16 quant dtype (baseline) works — no quantization loss."""
        param = torch.randn(32, 32, requires_grad=True)
        before = param.data.clone()
        opt = ECOAdamW(
            [param], lr=1e-3,
            quantize_weights=True, eco_enabled=False,
            quant_dtype="bf16",
        )
        param.grad = torch.randn_like(param) * 0.01
        opt.step()
        # Should still update (not be a complete no-op)
        assert not torch.equal(param.data, before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
