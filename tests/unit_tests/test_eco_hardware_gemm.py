# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for hardware FP8 GEMM dispatch and HardwareQuantLinear.

GPU-only tests (scaled_gemm, hardware GEMM forward) are skipped on CPU.
CPU-only tests verify the dequantized fallback path.
"""

import pytest
import torch
import torch.nn as nn

from torchtitan.components.quantized_tensor import QuantizedTensor
from torchtitan.components.quant_gemm import (
    _HAS_FP8_GEMM,
    _HARDWARE_GEMM_DTYPES,
    quantize_fp8_rowwise,
    quantize_fp8_tensorwise,
    scaled_gemm,
    HardwareQuantLinearFn,
    HardwareQuantLinear,
)
from torchtitan.components.activation_quant import ActivationQuantLinear, ActivationQuantConverter
from torchtitan.components.eco_optimizer import ECOAdamW


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
requires_fp8_gpu = pytest.mark.skipif(
    not _HAS_FP8_GEMM, reason="FP8 GEMM requires compute capability >= 8.9"
)


# ---------------------------------------------------------------------------
# quantize_fp8_rowwise
# ---------------------------------------------------------------------------

class TestQuantizeFp8Rowwise:
    def test_output_dtype(self):
        x = torch.randn(8, 16)
        fp8, scale = quantize_fp8_rowwise(x)
        assert fp8.dtype == torch.float8_e4m3fn
        assert fp8.shape == (8, 16)

    def test_scale_shape(self):
        x = torch.randn(8, 16)
        fp8, scale = quantize_fp8_rowwise(x)
        assert scale.shape == (8, 1)

    def test_per_row_scaling(self):
        """Each row should be scaled independently."""
        # Two rows with very different magnitudes
        x = torch.zeros(2, 8)
        x[0] = torch.ones(8)       # small values
        x[1] = torch.ones(8) * 100 # large values
        fp8, scale = quantize_fp8_rowwise(x)
        # Scales should differ
        assert not torch.allclose(scale[0], scale[1])

    def test_3d_input_shape(self):
        """Should handle (batch, seq, features) input."""
        x = torch.randn(2, 4, 16)
        fp8, scale = quantize_fp8_rowwise(x)
        assert fp8.shape == (8, 16)   # flattened
        assert scale.shape == (8, 1)


# ---------------------------------------------------------------------------
# quantize_fp8_tensorwise
# ---------------------------------------------------------------------------

class TestQuantizeFp8Tensorwise:
    def test_output_dtype(self):
        x = torch.randn(8, 16)
        fp8, scale = quantize_fp8_tensorwise(x)
        assert fp8.dtype == torch.float8_e4m3fn
        assert fp8.shape == x.shape

    def test_scale_is_scalar(self):
        x = torch.randn(8, 16)
        fp8, scale = quantize_fp8_tensorwise(x)
        assert scale.dim() == 0

    def test_matches_eco_adamw_requantize_formula(self):
        """Scale formula should match ECOAdamW._requantize for fp8."""
        x = torch.randn(16, 16, dtype=torch.float32)
        fp8, scale = quantize_fp8_tensorwise(x)

        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        amax = x.abs().amax()
        expected_scale = torch.where(amax > 0, amax / fp8_max, torch.ones_like(amax))
        torch.testing.assert_close(scale, expected_scale)

    def test_zero_tensor_no_nan(self):
        x = torch.zeros(8, 8)
        fp8, scale = quantize_fp8_tensorwise(x)
        assert not torch.isnan(scale)
        assert scale.item() == 1.0  # falls back to 1.0


# ---------------------------------------------------------------------------
# scaled_gemm
# ---------------------------------------------------------------------------

class TestScaledGemm:
    @requires_fp8_gpu
    def test_fp8_gemm_close_to_bf16(self):
        """FP8 GEMM (TensorWise scaling) result should be close to BF16 reference."""
        device = "cuda"
        M, K, N = 32, 64, 32

        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)

        # Use TensorWise scaling: both scale_a and scale_b must be (1,) singletons.
        # cuBLASLt requires mat1 row-major and mat2 column-major (Fortran order).
        # b_fp8.T.contiguous().T gives (K, N) Fortran-contiguous layout.
        a_fp8, a_scale = quantize_fp8_tensorwise(a)
        b_fp8, b_scale = quantize_fp8_tensorwise(b)
        a_scale_1d = a_scale.reshape(1)
        b_scale_1d = b_scale.reshape(1)

        # b_col: (K, N) Fortran-contiguous — required by cuBLASLt for mat2
        b_fp8_col = b_fp8.T.contiguous().T

        # FP8 GEMM: a_fp8 (M, K) @ b_fp8_col (K, N) → (M, N)
        out_fp8 = scaled_gemm(a_fp8, a_scale_1d, b_fp8_col, b_scale_1d, "fp8", torch.bfloat16)

        # BF16 reference: dequantize both and matmul
        a_deq = a_fp8.to(torch.float32) * a_scale
        b_deq = b_fp8.to(torch.float32) * b_scale  # values same as b_fp8_col
        ref = (a_deq @ b_deq).to(torch.bfloat16)

        # FP8 GEMM should be numerically close (not exact due to accumulation order)
        torch.testing.assert_close(out_fp8, ref, atol=0.5, rtol=0.1)

    def test_nvfp4_raises(self):
        a = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
        b = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
        s = torch.tensor(1.0)
        with pytest.raises(NotImplementedError, match="nvfp4"):
            scaled_gemm(a, s, b, s, "nvfp4", torch.bfloat16)

    def test_unknown_dtype_raises(self):
        a = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
        b = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
        s = torch.tensor(1.0)
        with pytest.raises(ValueError, match="unsupported"):
            scaled_gemm(a, s, b, s, "int8", torch.bfloat16)


# ---------------------------------------------------------------------------
# HardwareQuantLinear
# ---------------------------------------------------------------------------

class TestHardwareQuantLinear:
    def test_weight_is_quantized_tensor(self):
        """Weight parameter should be stored as QuantizedTensor."""
        original = nn.Linear(16, 32, bias=False)
        layer = HardwareQuantLinear(original, "fp8")
        assert isinstance(layer.weight, QuantizedTensor)

    def test_weight_data_is_fp8(self):
        """_data should be fp8 E4M3."""
        original = nn.Linear(16, 32, bias=False)
        layer = HardwareQuantLinear(original, "fp8")
        assert layer.weight._data.dtype == torch.float8_e4m3fn

    def test_weight_appears_in_parameters(self):
        """QuantizedTensor weight must be visible to module.parameters()."""
        original = nn.Linear(16, 32, bias=False)
        layer = HardwareQuantLinear(original, "fp8")
        params = list(layer.parameters())
        # weight (QuantizedTensor) must be present
        assert any(isinstance(p, QuantizedTensor) for p in params)

    def test_bias_preserved(self):
        original = nn.Linear(16, 32, bias=True)
        layer = HardwareQuantLinear(original, "fp8")
        assert layer.bias is original.bias  # same object

    def test_forward_cpu_fallback(self):
        """Forward pass on CPU should use dequantized BF16 fallback without error."""
        original = nn.Linear(16, 32, bias=True)
        layer = HardwareQuantLinear(original, "fp8")

        x = torch.randn(4, 16)
        out = layer(x)
        assert out.shape == (4, 32)
        assert not torch.isnan(out).any()

    def test_forward_close_to_original(self):
        """Forward output should be numerically close to dequantized nn.Linear."""
        torch.manual_seed(42)
        original = nn.Linear(64, 32, bias=True)
        layer = HardwareQuantLinear(original, "fp8")

        x = torch.randn(8, 64)

        # Reference: nn.Linear with dequantized weight
        weight_deq = layer.weight.dequantize()
        ref = nn.functional.linear(x, weight_deq, original.bias)
        out = layer(x)

        # Should be close — both use same dequantized weight on CPU
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    def test_backward_grads_computed(self):
        """Backward pass must produce non-None gradients with correct shapes."""
        original = nn.Linear(16, 32, bias=True)
        layer = HardwareQuantLinear(original, "fp8")

        x = torch.randn(4, 16, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == (32, 16)
        assert layer.bias.grad is not None
        assert layer.bias.grad.shape == (32,)

    def test_invalid_dtype_raises(self):
        original = nn.Linear(16, 32)
        with pytest.raises(ValueError, match="fp8"):
            HardwareQuantLinear(original, "bf16")

    def test_extra_repr(self):
        original = nn.Linear(16, 32, bias=True)
        layer = HardwareQuantLinear(original, "fp8")
        repr_str = layer.extra_repr()
        assert "fp8" in repr_str
        assert "16" in repr_str
        assert "32" in repr_str


# ---------------------------------------------------------------------------
# _HARDWARE_GEMM_DTYPES content
# ---------------------------------------------------------------------------

class TestHardwareGemmDtypes:
    def test_fp8_e4m3fn_in_set(self):
        assert torch.float8_e4m3fn in _HARDWARE_GEMM_DTYPES

    def test_bfloat16_not_in_set(self):
        assert torch.bfloat16 not in _HARDWARE_GEMM_DTYPES


# ---------------------------------------------------------------------------
# ActivationQuantConverter dispatch
# ---------------------------------------------------------------------------

class FakeJobConfig:
    """Minimal config stub for ActivationQuantConverter tests."""
    class eco:
        activation_dtype = "fp8"


class FakeParallelDims:
    pass


class TestActivationQuantConverterDispatch:
    def _make_model(self):
        """Two-layer model with an inner 'layers.' linear."""
        model = nn.Sequential()
        inner = nn.Linear(16, 32, bias=False)
        # Wrap in a named container so FQN starts with 'layers.'
        layers_container = nn.ModuleDict({"fc": inner})
        model.add_module("layers", layers_container)
        return model, inner

    def test_fp8_dtype_uses_hardware_quant_linear(self):
        """FP8 activation dtype → HardwareQuantLinear."""

        class _JobConfig:
            class eco:
                activation_dtype = "fp8"

        converter = ActivationQuantConverter(_JobConfig(), FakeParallelDims())
        model = nn.Module()
        layers = nn.Module()
        layers.fc = nn.Linear(16, 32)
        model.layers = layers

        converter.convert(model)
        assert isinstance(model.layers.fc, HardwareQuantLinear)

    def test_non_fp8_dtype_uses_activation_quant_linear(self):
        """Non-FP8 (or 'none') activation dtype → ActivationQuantLinear or no-op."""

        class _JobConfig:
            class eco:
                activation_dtype = "none"

        converter = ActivationQuantConverter(_JobConfig(), FakeParallelDims())
        model = nn.Module()
        layers = nn.Module()
        layers.fc = nn.Linear(16, 32)
        model.layers = layers

        converter.convert(model)
        # 'none' → no conversion
        assert isinstance(model.layers.fc, nn.Linear)
        assert not isinstance(model.layers.fc, (HardwareQuantLinear, ActivationQuantLinear))

    def test_non_layers_prefix_not_converted(self):
        """Modules outside 'layers.' prefix should not be converted."""

        class _JobConfig:
            class eco:
                activation_dtype = "fp8"

        converter = ActivationQuantConverter(_JobConfig(), FakeParallelDims())
        model = nn.Module()
        model.output_proj = nn.Linear(16, 32)  # not under 'layers.'

        converter.convert(model)
        assert isinstance(model.output_proj, nn.Linear)
        assert not isinstance(model.output_proj, HardwareQuantLinear)


# ---------------------------------------------------------------------------
# ECOAdamW dispatch order — QuantizedTensor goes to _step_quantized
# ---------------------------------------------------------------------------

class TestECOAdamWDispatchOrder:
    def test_quantized_tensor_not_sent_to_simulated_quant(self):
        """QuantizedTensor param must use _step_quantized, not _step_simulated_quant.

        After the dispatch-order fix, isinstance(param, QuantizedTensor) is checked
        BEFORE quantize_weights.  We verify by checking that the QuantizedTensor
        _data bytes are updated (i.e. _step_quantized ran) rather than the plain
        data being replaced (which _step_simulated_quant would do).
        """
        fp = torch.randn(16, 16)
        qt = QuantizedTensor.quantize(fp, torch.float8_e4m3fn)
        qt.requires_grad_(True)

        optimizer = ECOAdamW(
            [qt], lr=1e-3,
            quantize_weights=True,   # would route plain params to _simulated_quant
            eco_enabled=True,
            quant_dtype="fp8",
        )
        qt.grad = torch.randn_like(qt.dequantize()) * 0.01
        original_data = qt._data.clone()
        optimizer.step()

        # _data bytes should have changed (step_quantized updated in-place)
        assert qt._data.dtype == torch.float8_e4m3fn
        # Data has been updated
        assert not torch.all(qt._data == original_data)

    def test_regular_param_still_uses_simulated_quant(self):
        """Plain 2D params with quantize_weights=True should still use simulated quant."""
        param = torch.randn(16, 16, requires_grad=True)
        optimizer = ECOAdamW(
            [param], lr=1e-3,
            quantize_weights=True, eco_enabled=True, quant_dtype="fp8",
        )
        param.grad = torch.randn_like(param) * 0.01
        optimizer.step()

        # Param should be FP8-representable after simulated quant
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        amax = param.data.abs().amax()
        scale = amax / fp8_max if amax > 0 else torch.ones(1)
        scaled = param.data / scale
        roundtripped = scaled.to(torch.float8_e4m3fn).to(param.dtype) * scale
        torch.testing.assert_close(param.data, roundtripped, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# _step_quantized with master weights
# ---------------------------------------------------------------------------

class TestStepQuantizedMasterWeights:
    def test_master_weights_initialized_for_qt_param(self):
        """With master_weights_dtype set, optimizer state should have master_weights."""
        fp = torch.randn(16, 16)
        qt = QuantizedTensor.quantize(fp, torch.float8_e4m3fn)
        qt.requires_grad_(True)

        optimizer = ECOAdamW(
            [qt], lr=1e-3,
            master_weights_dtype=torch.float32,
            eco_enabled=True,
        )
        qt.grad = torch.randn_like(qt.dequantize()) * 0.01
        optimizer.step()

        state = optimizer.state[qt]
        assert "master_weights" in state
        assert state["master_weights"].dtype == torch.float32
        assert state["master_weights"].shape == qt.shape

    def test_master_weights_updated_after_step(self):
        """After step, master_weights should differ from initial dequantized value."""
        torch.manual_seed(0)
        fp = torch.randn(16, 16)
        qt = QuantizedTensor.quantize(fp, torch.float8_e4m3fn)
        qt.requires_grad_(True)

        initial_deq = qt.dequantize().clone()

        optimizer = ECOAdamW(
            [qt], lr=1e-2,
            master_weights_dtype=torch.float32,
            eco_enabled=True,
        )
        qt.grad = torch.randn_like(qt.dequantize()) * 0.1
        optimizer.step()

        state = optimizer.state[qt]
        # Master weights should have moved from the initial values
        assert not torch.allclose(state["master_weights"], initial_deq, atol=1e-6)

    def test_fp8_data_updated_after_step(self):
        """FP8 _data must be updated regardless of master_weights mode."""
        fp = torch.randn(16, 16)
        qt = QuantizedTensor.quantize(fp, torch.float8_e4m3fn)
        qt.requires_grad_(True)

        original_data = qt._data.clone()
        optimizer = ECOAdamW(
            [qt], lr=1e-2,
            master_weights_dtype=torch.float32,
            eco_enabled=True,
        )
        qt.grad = torch.randn_like(qt.dequantize()) * 0.1
        optimizer.step()

        assert qt._data.dtype == torch.float8_e4m3fn
        assert not torch.all(qt._data == original_data)


# ---------------------------------------------------------------------------
# Full FP8 training step (forward → backward → optimizer)
# ---------------------------------------------------------------------------

class TestFullFp8TrainingStep:
    def test_loss_decreases_over_steps(self):
        """Model with HardwareQuantLinear weight should optimize (loss decreases)."""
        torch.manual_seed(42)

        # Build a small model with HardwareQuantLinear
        original = nn.Linear(32, 16, bias=True)
        layer = HardwareQuantLinear(original, "fp8")

        optimizer = ECOAdamW(
            list(layer.parameters()),
            lr=1e-2,
            eco_enabled=True,
        )

        x = torch.randn(8, 32)
        target = torch.randn(8, 16)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            out = layer(x)
            loss = ((out - target) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (not necessarily monotone due to quantization)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_quantized_tensor_weight_persists_after_step(self):
        """QuantizedTensor weight type should be preserved across optimizer steps."""
        original = nn.Linear(16, 8, bias=False)
        layer = HardwareQuantLinear(original, "fp8")

        optimizer = ECOAdamW(list(layer.parameters()), lr=1e-3, eco_enabled=True)

        x = torch.randn(4, 16)
        out = layer(x)
        out.sum().backward()
        optimizer.step()

        # Weight must still be a QuantizedTensor with FP8 _data
        assert isinstance(layer.weight, QuantizedTensor)
        assert layer.weight._data.dtype == torch.float8_e4m3fn
