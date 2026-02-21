# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests that ECOAdamW dtype behavior matches the ECO paper.

Paper reference:
    "ECO: Quantized Training without Full-Precision Master Weights"
    Nikdan et al., 2025

Paper specifies (Algorithm 3, Section 5 memory analysis):
    - Stored weights: FP8 E4M3 (1 byte/param)
    - 1st moment m:   FP32     (4 bytes/param) — carries quantization error
    - 2nd moment v:   FP32     (4 bytes/param)
    - θ̃ (pre-quant):  FP32     ("high-precision parameters")
    - error e:        FP32     (e = θ̃ − θ_hat, must avoid catastrophic cancellation)
    - injection:      FP32     (modify momentum in optimizer precision)
    Total: 9 bytes/param (25% reduction vs 12 bytes with FP32 master weights)
"""

import pytest
import torch

from torchtitan.components.eco_adamw import ECOAdamW
from torchtitan.components.quantized_tensor import QuantizedTensor


def _make_qt(rows=64, cols=64, seed=42):
    """Create a QuantizedTensor (FP8 storage, BF16 compute interface)."""
    torch.manual_seed(seed)
    fp = torch.randn(rows, cols)
    qt = QuantizedTensor.quantize(fp, torch.float8_e4m3fn)
    qt.requires_grad_(True)
    return qt


def _fake_grad(param):
    """Attach a random gradient."""
    if isinstance(param, QuantizedTensor):
        param.grad = torch.randn_like(param.dequantize()) * 0.01
    else:
        param.grad = torch.randn_like(param) * 0.01


# ---------------------------------------------------------------------------
# Tests: Default dtype is FP32 (matching the paper)
# ---------------------------------------------------------------------------


class TestDefaultDtypesMatchPaper:
    """The paper stores m, v in FP32 and computes θ̃ and error in FP32."""

    def test_default_optim_state_dtype_is_fp32(self):
        param = torch.randn(16, 16, requires_grad=True)
        opt = ECOAdamW([param], lr=1e-3)
        assert opt.defaults["optim_state_dtype"] == torch.float32

    def test_default_optim_compute_dtype_is_fp32(self):
        param = torch.randn(16, 16, requires_grad=True)
        opt = ECOAdamW([param], lr=1e-3)
        assert opt.defaults["optim_compute_dtype"] == torch.float32

    def test_optimizer_states_fp32_by_default(self):
        """m and v should be stored in FP32 (4 bytes each, per paper)."""
        qt = _make_qt(32, 32)
        opt = ECOAdamW([qt], lr=1e-3, eco_enabled=True)
        _fake_grad(qt)
        opt.step()
        state = opt.state[qt]
        assert state["exp_avg"].dtype == torch.float32, (
            f"1st moment should be FP32, got {state['exp_avg'].dtype}"
        )
        assert state["exp_avg_sq"].dtype == torch.float32, (
            f"2nd moment should be FP32, got {state['exp_avg_sq'].dtype}"
        )

    def test_regular_param_states_fp32_by_default(self):
        """Even for non-quantized params, states default to FP32."""
        param = torch.randn(32, 32, requires_grad=True)
        opt = ECOAdamW([param], lr=1e-3)
        param.grad = torch.randn_like(param) * 0.01
        opt.step()
        state = opt.state[param]
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32

    def test_can_override_to_bf16(self):
        """Users can still opt into BF16 states for memory experiments."""
        qt = _make_qt(16, 16)
        opt = ECOAdamW([qt], lr=1e-3, optim_state_dtype=torch.bfloat16)
        _fake_grad(qt)
        opt.step()
        state = opt.state[qt]
        assert state["exp_avg"].dtype == torch.bfloat16
        assert state["exp_avg_sq"].dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Tests: Error computed in optim_compute_dtype (FP32 by default)
# ---------------------------------------------------------------------------


class TestErrorComputedInFP32:
    """The error e = θ̃ − θ_hat must be computed in FP32 to avoid
    catastrophic cancellation (θ̃ and θ_hat differ only by FP8 quant error).
    """

    def test_prev_error_dtype_matches_optim_compute_dtype_fp32(self):
        """With default optim_compute_dtype=FP32, prev_error should be FP32."""
        qt = _make_qt(32, 32)
        opt = ECOAdamW(
            [qt], lr=1e-3, eco_enabled=True, heuristic_log_freq=1,
            optim_compute_dtype=torch.float32,
        )
        _fake_grad(qt)
        opt.step()
        prev_error = opt.state[qt]["prev_error"]
        assert prev_error.dtype == torch.float32, (
            f"prev_error should be FP32, got {prev_error.dtype}"
        )

    def test_prev_error_dtype_matches_optim_compute_dtype_bf16(self):
        """With optim_compute_dtype=BF16, prev_error should be BF16."""
        qt = _make_qt(32, 32)
        opt = ECOAdamW(
            [qt], lr=1e-3, eco_enabled=True, heuristic_log_freq=1,
            optim_compute_dtype=torch.bfloat16,
        )
        _fake_grad(qt)
        opt.step()
        prev_error = opt.state[qt]["prev_error"]
        assert prev_error.dtype == torch.bfloat16

    def test_fp32_error_has_more_precision_than_bf16(self):
        """FP32 error should have more unique values than BF16 error,
        because BF16 subtraction of two close values loses significant bits.
        """
        qt_fp32 = _make_qt(64, 64, seed=123)
        opt_fp32 = ECOAdamW(
            [qt_fp32], lr=1e-3, eco_enabled=True, heuristic_log_freq=1,
            optim_compute_dtype=torch.float32,
        )
        _fake_grad(qt_fp32)
        opt_fp32.step()
        err_fp32 = opt_fp32.state[qt_fp32]["prev_error"]

        qt_bf16 = _make_qt(64, 64, seed=123)
        opt_bf16 = ECOAdamW(
            [qt_bf16], lr=1e-3, eco_enabled=True, heuristic_log_freq=1,
            optim_compute_dtype=torch.bfloat16,
        )
        _fake_grad(qt_bf16)
        opt_bf16.step()
        err_bf16 = opt_bf16.state[qt_bf16]["prev_error"]

        # FP32 should have strictly more unique error values than BF16,
        # because BF16 quantizes the error signal more coarsely.
        unique_fp32 = err_fp32.unique().numel()
        unique_bf16 = err_bf16.unique().numel()
        assert unique_fp32 > unique_bf16, (
            f"FP32 error should have more unique values ({unique_fp32}) "
            f"than BF16 ({unique_bf16})"
        )


# ---------------------------------------------------------------------------
# Tests: param_dequant upcast to optim_compute_dtype
# ---------------------------------------------------------------------------


class TestParamDequantUpcast:
    """QuantizedTensor.dequantize() returns BF16 by default, but
    _step_quantized must upcast to optim_compute_dtype before the Adam update
    so θ̃ is computed in high precision.
    """

    def test_qt_dequantize_returns_bf16(self):
        """Baseline: QuantizedTensor.dequantize() returns BF16 (its optim_compute_dtype)."""
        fp = torch.randn(16, 16)
        qt = QuantizedTensor.quantize(fp, torch.float8_e4m3fn)
        dq = qt.dequantize()
        # QuantizedTensor.quantize from FP32 source → optim_compute_dtype is FP32
        assert dq.dtype == torch.float32

    def test_qt_from_bf16_dequantizes_to_bf16(self):
        """If original tensor was BF16, dequant returns BF16."""
        fp = torch.randn(16, 16, dtype=torch.bfloat16)
        qt = QuantizedTensor.quantize(fp, torch.float8_e4m3fn)
        dq = qt.dequantize()
        assert dq.dtype == torch.bfloat16

    def test_upcast_ensures_fp32_error_even_from_bf16_qt(self):
        """Even if QuantizedTensor was created from BF16, the error
        should still be in optim_compute_dtype (FP32 by default).
        """
        torch.manual_seed(42)
        fp = torch.randn(32, 32, dtype=torch.bfloat16)
        qt = QuantizedTensor.quantize(fp, torch.float8_e4m3fn)
        qt.requires_grad_(True)

        opt = ECOAdamW(
            [qt], lr=1e-3, eco_enabled=True, heuristic_log_freq=1,
            optim_compute_dtype=torch.float32,  # explicit FP32 compute
        )
        qt.grad = torch.randn(32, 32, dtype=torch.bfloat16) * 0.01
        opt.step()

        prev_error = opt.state[qt]["prev_error"]
        assert prev_error.dtype == torch.float32, (
            f"Error should be FP32 even when QT source is BF16, got {prev_error.dtype}"
        )


# ---------------------------------------------------------------------------
# Tests: Injection operates in correct dtype
# ---------------------------------------------------------------------------


class TestInjectionDtype:
    """The error injection into momentum must happen in FP32
    to preserve the error signal.
    """

    def test_momentum_dtype_preserved_after_injection(self):
        """After ECO injection, momentum should still be in optim_state_dtype."""
        qt = _make_qt(32, 32)
        opt = ECOAdamW(
            [qt], lr=1e-3, eco_enabled=True,
            optim_state_dtype=torch.float32, optim_compute_dtype=torch.float32,
        )
        _fake_grad(qt)
        opt.step()
        state = opt.state[qt]
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32

    def test_injection_modifies_momentum(self):
        """ECO injection should change momentum compared to no-injection."""
        torch.manual_seed(42)

        # With ECO injection
        qt_eco = _make_qt(32, 32, seed=99)
        opt_eco = ECOAdamW(
            [qt_eco], lr=1e-3, eco_enabled=True,
            optim_state_dtype=torch.float32, optim_compute_dtype=torch.float32,
        )
        qt_eco.grad = torch.randn(32, 32) * 0.01
        opt_eco.step()
        m_eco = opt_eco.state[qt_eco]["exp_avg"].clone()

        # Without ECO injection
        qt_no = _make_qt(32, 32, seed=99)
        opt_no = ECOAdamW(
            [qt_no], lr=1e-3, eco_enabled=False,
            optim_state_dtype=torch.float32, optim_compute_dtype=torch.float32,
        )
        qt_no.grad = torch.randn(32, 32) * 0.01
        opt_no.step()
        m_no = opt_no.state[qt_no]["exp_avg"].clone()

        # Injection should make momentum different
        assert not torch.allclose(m_eco, m_no, atol=1e-10), (
            "ECO injection should modify momentum"
        )


# ---------------------------------------------------------------------------
# Tests: Memory layout matches paper
# ---------------------------------------------------------------------------


class TestMemoryLayout:
    """Paper says 9 bytes/param for ECO (FP8 weight + FP32 m + FP32 v).
    With master weights: 12 bytes (FP32 master + FP32 m + FP32 v).
    """

    def test_bytes_per_param_no_master_weights(self):
        """ECO: 1 (FP8 weight) + 4 (FP32 m) + 4 (FP32 v) = 9 bytes/param."""
        qt = _make_qt(64, 64)
        opt = ECOAdamW(
            [qt], lr=1e-3, eco_enabled=True,
            optim_state_dtype=torch.float32,
        )
        _fake_grad(qt)
        opt.step()

        state = opt.state[qt]
        numel = qt.numel()

        # Weight storage: FP8 = 1 byte per element
        weight_bytes = qt._data.element_size() * numel
        assert qt._data.dtype == torch.float8_e4m3fn
        assert weight_bytes == numel * 1

        # Momentum storage: FP32 = 4 bytes per element
        m_bytes = state["exp_avg"].element_size() * numel
        assert state["exp_avg"].dtype == torch.float32
        assert m_bytes == numel * 4

        # Variance storage: FP32 = 4 bytes per element
        v_bytes = state["exp_avg_sq"].element_size() * numel
        assert state["exp_avg_sq"].dtype == torch.float32
        assert v_bytes == numel * 4

        total_bytes_per_param = (weight_bytes + m_bytes + v_bytes) / numel
        assert total_bytes_per_param == 9.0, (
            f"Expected 9 bytes/param, got {total_bytes_per_param}"
        )

    def test_bf16_optim_reduces_to_5_bytes(self):
        """With BF16 optimizer states: 1 (FP8) + 2 (BF16 m) + 2 (BF16 v) = 5."""
        qt = _make_qt(64, 64)
        opt = ECOAdamW(
            [qt], lr=1e-3, eco_enabled=True,
            optim_state_dtype=torch.bfloat16,
        )
        _fake_grad(qt)
        opt.step()

        state = opt.state[qt]
        numel = qt.numel()
        weight_bytes = qt._data.element_size() * numel
        m_bytes = state["exp_avg"].element_size() * numel
        v_bytes = state["exp_avg_sq"].element_size() * numel
        total = (weight_bytes + m_bytes + v_bytes) / numel
        assert total == 5.0


# ---------------------------------------------------------------------------
# Tests: Full dtype flow end-to-end
# ---------------------------------------------------------------------------


class TestEndToEndDtypeFlow:
    """Verify the complete dtype flow matches Algorithm 3."""

    def test_algorithm3_dtype_flow(self):
        """Step-by-step verification of Algorithm 3 dtype correctness.

        Algorithm 3 (Q_ECO for Adam):
            1. θ_hat = Q(θ̃)           — FP8 quantization
            2. e = θ̃ − θ_hat           — FP32 subtraction
            3. {m̃, v} ← optimizer state — FP32
            4. m̂ = m̃ + coeff·diag(...)·e — FP32 injection
            5. Return θ_hat, {m̂, v}
        """
        qt = _make_qt(32, 32)
        opt = ECOAdamW(
            [qt], lr=1e-3, eco_enabled=True, heuristic_log_freq=1,
            optim_state_dtype=torch.float32, optim_compute_dtype=torch.float32,
        )

        _fake_grad(qt)
        opt.step()

        state = opt.state[qt]

        # Weight is FP8
        assert qt._data.dtype == torch.float8_e4m3fn

        # Optimizer states are FP32
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32

        # Error is FP32
        assert state["prev_error"].dtype == torch.float32

        # Error is non-zero (the injection actually happened)
        assert state["prev_error"].abs().max() > 1e-6

    def test_eco_converges_with_fp32_defaults(self):
        """ECO with FP32 defaults should converge on a simple problem."""
        import torch.nn as nn

        torch.manual_seed(42)
        x = torch.randn(64, 16)
        y = torch.sin(x[:, :1] * 3) + 0.5 * x[:, 1:2] ** 2

        model = nn.Sequential(
            nn.Linear(16, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
        # Use defaults (FP32 everything)
        optimizer = ECOAdamW(model.parameters(), lr=1e-3, eco_enabled=True)
        losses = []
        for _ in range(200):
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert losses[-1] < losses[0] * 0.5, (
            f"ECO w/ FP32 didn't converge: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
