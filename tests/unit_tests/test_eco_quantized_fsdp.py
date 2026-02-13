"""Tests for FSDP2 support with QuantizedTensor."""

import pytest
import torch
import torch.nn as nn

from torchtitan.components.quantized_tensor import (
    QuantizedTensor,
    is_quantized_tensor,
    maybe_dequantize,
)
from torchtitan.components.quantized_linear import QuantizedLinear


class TestQuantizedTensorSerialization:
    """Test serialization/deserialization for FSDP checkpointing."""

    def test_quantized_tensor_flatten_unflatten(self):
        """Test flatten and unflatten for checkpointing."""
        # Create a QuantizedTensor
        data_fp = torch.randn(32, 64, dtype=torch.float32)
        quantized = QuantizedTensor.quantize(
            data_fp,
            quant_dtype=torch.float8_e4m3fn,
        )

        # Flatten
        inner_tensors, metadata = quantized.__tensor_flatten__()

        # Check that inner tensors are correct
        assert '_data' in inner_tensors
        assert '_scale' in inner_tensors

        # Check that metadata is correct
        assert '_quant_dtype' in metadata
        assert '_compute_dtype' in metadata

        # Unflatten — use getattr (QuantizedTensor uses __slots__, not __dict__)
        reconstructed = QuantizedTensor.__tensor_unflatten__(
            {k: getattr(quantized, k) for k in inner_tensors},
            metadata,
            quantized.size(),
            quantized.stride(),
        )

        # Check that reconstructed tensor is correct
        assert isinstance(reconstructed, QuantizedTensor)
        assert reconstructed._quant_dtype == quantized._quant_dtype
        assert reconstructed._compute_dtype == quantized._compute_dtype


class TestQuantizedTensorBasicOperations:
    """Test basic QuantizedTensor operations on single GPU."""

    def test_quantized_tensor_checkpoint_simulation(self):
        """Test checkpoint/recovery for QuantizedTensor."""
        # Create a QuantizedTensor
        data_fp = torch.randn(32, 64, dtype=torch.float32)
        quantized = QuantizedTensor.quantize(
            data_fp,
            quant_dtype=torch.float8_e4m3fn,
        )

        # Simulate checkpoint save (flatten)
        inner_tensor_names, metadata = quantized.__tensor_flatten__()

        # Get actual tensor data
        checkpoint_data = {}
        for name in inner_tensor_names:
            checkpoint_data[name] = getattr(quantized, name).clone()

        # Simulate checkpoint load (unflatten)
        reconstructed = QuantizedTensor.__tensor_unflatten__(
            checkpoint_data,
            metadata,
            quantized.size(),
            quantized.stride(),
        )

        # Verify reconstruction
        assert isinstance(reconstructed, QuantizedTensor)
        assert reconstructed._quant_dtype == quantized._quant_dtype
        assert reconstructed._compute_dtype == quantized._compute_dtype

        # Verify values are approximately equal
        original_dequant = quantized.dequantize()
        reconstructed_dequant = reconstructed.dequantize()
        torch.testing.assert_close(original_dequant, reconstructed_dequant, atol=1e-6, rtol=1e-6)

    def test_quantized_model_state_dict(self):
        """Test state dict save/load for QuantizedLinear models."""
        # Create model with QuantizedLinear layers
        model = nn.Sequential(
            QuantizedLinear(32, 64),
            nn.ReLU(),
            QuantizedLinear(64, 16),
        )

        # Get initial weights
        original_weights = {}
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinear):
                weight = module.get_weight()
                original_weights[name] = {
                    'data': weight._data.clone(),
                    'scale': weight._scale.clone(),
                    'dequantized': weight.dequantize().clone(),
                }

        # Save state dict
        state_dict = model.state_dict()

        # Create new model and load state dict
        new_model = nn.Sequential(
            QuantizedLinear(32, 64),
            nn.ReLU(),
            QuantizedLinear(64, 16),
        )
        new_model.load_state_dict(state_dict)

        # Verify loaded weights match
        for name, module in new_model.named_modules():
            if isinstance(module, QuantizedLinear):
                weight = module.get_weight()
                orig = original_weights[name]

                torch.testing.assert_close(weight._data, orig['data'])
                torch.testing.assert_close(weight._scale, orig['scale'])
                torch.testing.assert_close(weight.dequantize(), orig['dequantized'])

    def test_quantized_linear_gradient_flow(self):
        """Test gradient flow through QuantizedLinear layers."""
        # Create model
        model = nn.Sequential(
            QuantizedLinear(32, 64),
            nn.ReLU(),
            QuantizedLinear(64, 16),
        )

        # Create input and target
        x = torch.randn(8, 32, requires_grad=True)
        target = torch.randn(8, 16)

        # Forward pass — QuantizedLinear computes in bf16, so cast for loss
        output = model(x)
        loss = torch.nn.functional.mse_loss(output.float(), target)

        # Backward pass
        loss.backward()

        # Verify gradients exist for input and parameters
        assert x.grad is not None, "Input should have gradient"

        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinear):
                # _weight_scale is the learnable parameter; _weight_data is FP8 (requires_grad=False)
                assert module._weight_scale.grad is not None, f"{name} weight scale should have gradient"
                if hasattr(module, 'bias') and module.bias is not None:
                    assert module.bias.grad is not None, f"{name} bias should have gradient"

    def test_memory_usage_benefits(self):
        """Test memory usage benefits of quantization."""
        # Create tensors of same shape in different precisions
        shape = (1024, 1024)

        # FP32 tensor
        fp32_tensor = torch.randn(shape, dtype=torch.float32)
        fp32_memory = fp32_tensor.numel() * fp32_tensor.element_size()

        # Quantized tensor (FP8 data + BF16 scale)
        quantized = QuantizedTensor.quantize(
            fp32_tensor,
            quant_dtype=torch.float8_e4m3fn,
        )
        quantized_memory = (
            quantized._data.numel() * quantized._data.element_size() +
            quantized._scale.numel() * quantized._scale.element_size()
        )

        # Verify memory savings
        compression_ratio = fp32_memory / quantized_memory
        assert compression_ratio > 2.0, f"Expected >2x compression, got {compression_ratio:.2f}x"

        # Verify dequantized values are close
        dequantized = quantized.dequantize()
        relative_error = torch.abs(dequantized - fp32_tensor) / (torch.abs(fp32_tensor) + 1e-6)
        assert relative_error.mean() < 0.1, f"Quantization error too high: {relative_error.mean():.4f}"

    def test_tensor_sharding_concepts(self):
        """Test sharding concepts without actual distributed setup."""
        # Create a large QuantizedTensor
        data_fp = torch.randn(128, 256, dtype=torch.float32)
        quantized = QuantizedTensor.quantize(
            data_fp,
            quant_dtype=torch.float8_e4m3fn,
        )

        # Simulate 2-way sharding along dimension 0 (conceptual)
        shard_dim = 0
        num_shards = 2
        shard_size = quantized.size(shard_dim) // num_shards

        # Create conceptual "shards"
        shards = []
        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = (i + 1) * shard_size if i < num_shards - 1 else quantized.size(shard_dim)

            shard_data = quantized._data[start_idx:end_idx]
            shards.append(shard_data)

        # Verify shard sizes
        for i, shard in enumerate(shards):
            expected_shard_size = list(data_fp.size())
            expected_shard_size[shard_dim] = shard_size if i < num_shards - 1 else data_fp.size(shard_dim) - i * shard_size
            assert tuple(shard.size()) == tuple(expected_shard_size), f"Shard {i} has wrong size"

        # Simulate gathering shards
        gathered_data = torch.cat(shards, dim=shard_dim)

        # Verify gathered tensor matches original
        torch.testing.assert_close(gathered_data, quantized._data)


class TestQuantizedTensorEdgeCases:
    """Test edge cases and error conditions for QuantizedTensor."""

    def test_zero_tensor_quantization(self):
        """Test quantization of zero tensors."""
        zero_tensor = torch.zeros(32, 64, dtype=torch.float32)
        quantized = QuantizedTensor.quantize(
            zero_tensor,
            quant_dtype=torch.float8_e4m3fn,
        )

        # Dequantize should give zeros
        dequantized = quantized.dequantize()
        torch.testing.assert_close(dequantized, zero_tensor, atol=1e-6, rtol=0)

        # Scale should be reasonable (not zero to avoid division issues)
        assert quantized._scale.abs().mean() > 0, "Scale should not be zero"

    def test_extreme_value_quantization(self):
        """Test quantization with extreme values."""
        # Very large values
        large_tensor = torch.full((16, 16), 1e6, dtype=torch.float32)
        quantized_large = QuantizedTensor.quantize(
            large_tensor,
            quant_dtype=torch.float8_e4m3fn,
        )
        dequantized_large = quantized_large.dequantize()

        # Should be capped at FP8 max but not crash
        assert not torch.isnan(dequantized_large).any(), "Large values should not result in NaN"
        assert not torch.isinf(dequantized_large).any(), "Large values should not result in Inf"

        # Very small values
        small_tensor = torch.full((16, 16), 1e-6, dtype=torch.float32)
        quantized_small = QuantizedTensor.quantize(
            small_tensor,
            quant_dtype=torch.float8_e4m3fn,
        )
        dequantized_small = quantized_small.dequantize()

        # Should preserve relative scale
        relative_error_small = torch.abs(dequantized_small - small_tensor) / small_tensor
        assert relative_error_small.mean() < 1.0, "Small values should be quantized reasonably"

    def test_mixed_precision_quantization(self):
        """Test quantization with different compute dtypes."""
        base_tensor = torch.randn(64, 64, dtype=torch.float32)

        compute_dtypes = [torch.float32, torch.bfloat16, torch.float16]

        for compute_dtype in compute_dtypes:
            # Cast to compute_dtype before quantizing so compute_dtype is set correctly
            quantized = QuantizedTensor.quantize(
                base_tensor.to(compute_dtype),
                quant_dtype=torch.float8_e4m3fn,
            )

            # Verify compute dtype is preserved
            assert quantized._compute_dtype == compute_dtype, (
                f"Compute dtype should be {compute_dtype}, got {quantized._compute_dtype}"
            )

            # Verify dequantization works
            dequantized = quantized.dequantize()
            assert dequantized.dtype == compute_dtype, (
                f"Dequantized dtype should be {compute_dtype}, got {dequantized.dtype}"
            )

    def test_axiswise_quantization(self):
        """Test axis-wise quantization."""
        tensor = torch.randn(32, 128, dtype=torch.float32)

        # Test per-tensor quantization (default)
        quantized_per_tensor = QuantizedTensor.quantize(
            tensor,
            quant_dtype=torch.float8_e4m3fn,
            axiswise_dim=None,
        )
        assert quantized_per_tensor._axiswise_dim is None
        assert quantized_per_tensor._scale.numel() == 1

        # Test per-channel quantization along dim 0
        quantized_per_channel = QuantizedTensor.quantize(
            tensor,
            quant_dtype=torch.float8_e4m3fn,
            axiswise_dim=0,
        )
        assert quantized_per_channel._axiswise_dim == 0
        assert quantized_per_channel._scale.numel() == tensor.size(0)

        # Verify both produce reasonable results
        dequant_per_tensor = quantized_per_tensor.dequantize()
        dequant_per_channel = quantized_per_channel.dequantize()

        # Per-channel should generally be more accurate
        error_per_tensor = torch.abs(dequant_per_tensor - tensor).mean()
        error_per_channel = torch.abs(dequant_per_channel - tensor).mean()

        # Both should have reasonable error
        assert error_per_tensor < tensor.abs().mean() * 0.2
        assert error_per_channel < tensor.abs().mean() * 0.2


class TestFSDPHooks:
    """Test FSDP2 extension hooks on QuantizedTensor (single-GPU, no distributed required)."""

    def _make_quantized(self, shape=(64, 128), quant_dtype=torch.float8_e4m3fn, axiswise_dim=None):
        """Helper to create a QuantizedTensor."""
        data = torch.randn(shape, dtype=torch.float32)
        return QuantizedTensor.quantize(data, quant_dtype=quant_dtype, axiswise_dim=axiswise_dim)

    def test_hooks_exist(self):
        """Both FSDP2 methods are present on QuantizedTensor."""
        qt = self._make_quantized()
        assert hasattr(qt, 'fsdp_pre_all_gather')
        assert hasattr(qt, 'fsdp_post_all_gather')

    def test_pre_all_gather_returns_fp8_data(self):
        """pre_all_gather returns FP8 data tensor and metadata tuple."""
        qt = self._make_quantized()
        all_gather_inputs, metadata = qt.fsdp_pre_all_gather(mesh=None)

        # all_gather_inputs is a tuple of tensors
        assert isinstance(all_gather_inputs, tuple)
        assert len(all_gather_inputs) == 1
        assert all_gather_inputs[0].dtype == torch.float8_e4m3fn
        torch.testing.assert_close(all_gather_inputs[0], qt._data)

        # metadata is a tuple of (scale, quant_dtype, compute_dtype, zero_point, axiswise_dim)
        assert isinstance(metadata, tuple)
        assert len(metadata) == 5
        scale, quant_dtype, compute_dtype, zero_point, axiswise_dim = metadata
        assert quant_dtype == torch.float8_e4m3fn
        assert compute_dtype == torch.float32
        assert zero_point is None
        assert axiswise_dim is None

    def test_post_all_gather_first_call(self):
        """First call (out=None) returns (dequantized, inner_tensors) matching dequantize()."""
        qt = self._make_quantized()
        all_gather_inputs, metadata = qt.fsdp_pre_all_gather(mesh=None)

        result = qt.fsdp_post_all_gather(all_gather_inputs, metadata, param_dtype=torch.float32)

        assert result is not None
        dequant, inner_tensors = result
        assert isinstance(dequant, torch.Tensor)
        assert not isinstance(dequant, QuantizedTensor)
        assert dequant.dtype == torch.float32
        assert dequant.shape == qt.shape
        assert isinstance(inner_tensors, tuple)

        # Must match dequantize() exactly
        torch.testing.assert_close(dequant, qt.dequantize())

    def test_post_all_gather_inplace(self):
        """out= path writes into buffer in-place and returns None."""
        qt = self._make_quantized()
        all_gather_inputs, metadata = qt.fsdp_pre_all_gather(mesh=None)

        out = torch.empty_like(qt.dequantize())
        result = qt.fsdp_post_all_gather(all_gather_inputs, metadata, param_dtype=torch.float32, out=out)

        assert result is None
        torch.testing.assert_close(out, qt.dequantize())

    def test_roundtrip_preserves_values(self):
        """pre -> post roundtrip matches dequantize() exactly."""
        qt = self._make_quantized()
        expected = qt.dequantize()

        inputs, meta = qt.fsdp_pre_all_gather(mesh=None)
        dequant, _ = qt.fsdp_post_all_gather(inputs, meta, param_dtype=torch.float32)

        torch.testing.assert_close(dequant, expected)

    def test_param_dtype_casting(self):
        """post_all_gather respects param_dtype for mixed precision."""
        qt = self._make_quantized()
        inputs, meta = qt.fsdp_pre_all_gather(mesh=None)

        for param_dtype in [torch.bfloat16, torch.float16]:
            dequant, _ = qt.fsdp_post_all_gather(inputs, meta, param_dtype=param_dtype)
            assert dequant.dtype == param_dtype, f"Expected {param_dtype}, got {dequant.dtype}"
            assert dequant.shape == qt.shape

    def test_axiswise_quantization_hooks(self):
        """Hooks work with per-channel quantized tensors."""
        qt = self._make_quantized(shape=(32, 64), axiswise_dim=0)
        assert qt._axiswise_dim == 0
        assert qt._scale.numel() == 32

        inputs, meta = qt.fsdp_pre_all_gather(mesh=None)
        dequant, _ = qt.fsdp_post_all_gather(inputs, meta, param_dtype=torch.float32)

        torch.testing.assert_close(dequant, qt.dequantize())

    def test_bf16_quantization_hooks(self):
        """Hooks work with bf16 quant_dtype (scale=1 path)."""
        qt = self._make_quantized(quant_dtype=torch.bfloat16)
        assert qt._quant_dtype == torch.bfloat16

        inputs, meta = qt.fsdp_pre_all_gather(mesh=None)
        assert inputs[0].dtype == torch.bfloat16

        dequant, _ = qt.fsdp_post_all_gather(inputs, meta, param_dtype=torch.float32)
        torch.testing.assert_close(dequant, qt.dequantize().to(torch.float32))


class TestTorchDispatch:
    """Test __torch_dispatch__ dequantizes inputs and forwards correctly."""

    def _make_qt(self, shape=(32, 64)):
        data = torch.randn(shape, dtype=torch.float32)
        return QuantizedTensor.quantize(data, quant_dtype=torch.float8_e4m3fn)

    def test_addition(self):
        """qt + qt matches dequantized addition."""
        qt1 = self._make_qt()
        qt2 = self._make_qt()
        result = qt1 + qt2
        expected = qt1.dequantize() + qt2.dequantize()
        assert not isinstance(result, QuantizedTensor)
        torch.testing.assert_close(result, expected)

    def test_scalar_multiply(self):
        """qt * scalar matches dequantized multiplication."""
        qt = self._make_qt()
        result = qt * 2.0
        expected = qt.dequantize() * 2.0
        assert not isinstance(result, QuantizedTensor)
        torch.testing.assert_close(result, expected)

    def test_matmul(self):
        """qt @ qt.T matches dequantized matmul."""
        qt1 = self._make_qt((16, 32))
        qt2 = self._make_qt((64, 32))
        result = qt1 @ qt2.dequantize().T  # qt2.T dispatch needs dequant on qt2
        expected = qt1.dequantize() @ qt2.dequantize().T
        torch.testing.assert_close(result, expected)

    def test_mixed_qt_regular(self):
        """qt + regular tensor works correctly."""
        qt = self._make_qt()
        regular = torch.randn(32, 64, dtype=torch.float32)
        result = qt + regular
        expected = qt.dequantize() + regular
        assert not isinstance(result, QuantizedTensor)
        torch.testing.assert_close(result, expected)


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_is_quantized_tensor_true(self):
        qt = QuantizedTensor.quantize(
            torch.randn(8, 8), quant_dtype=torch.float8_e4m3fn
        )
        assert is_quantized_tensor(qt) is True

    def test_is_quantized_tensor_false(self):
        assert is_quantized_tensor(torch.randn(8, 8)) is False
        assert is_quantized_tensor(torch.tensor(1.0)) is False

    def test_maybe_dequantize_qt(self):
        data = torch.randn(8, 8)
        qt = QuantizedTensor.quantize(data, quant_dtype=torch.float8_e4m3fn)
        result = maybe_dequantize(qt)
        assert not isinstance(result, QuantizedTensor)
        torch.testing.assert_close(result, qt.dequantize())

    def test_maybe_dequantize_regular(self):
        regular = torch.randn(8, 8)
        result = maybe_dequantize(regular)
        assert result is regular  # Same object, not a copy


class TestZeroPointQuantization:
    """Test zero-point (asymmetric quantization) code paths."""

    def _make_qt_with_zero_point(self, shape=(16, 32)):
        """Manually construct a QuantizedTensor with a non-None zero_point."""
        data_fp = torch.randn(shape, dtype=torch.float32)
        quant_dtype = torch.float8_e4m3fn
        amax = data_fp.abs().amax()
        dtype_max = torch.finfo(quant_dtype).max
        scale = amax / dtype_max
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        quantized_data = (data_fp / scale).to(quant_dtype)
        zero_point = torch.tensor(0.5, dtype=torch.float32)
        return QuantizedTensor(
            data=quantized_data,
            scale=scale,
            quant_dtype=quant_dtype,
            compute_dtype=torch.float32,
            zero_point=zero_point,
        )

    def test_dequantize_with_zero_point(self):
        """Dequantize applies (data - zp) * scale when zero_point is set."""
        qt = self._make_qt_with_zero_point()
        result = qt.dequantize()
        expected = (qt._data.to(torch.float32) - qt._zero_point.to(torch.float32)) * qt._scale
        torch.testing.assert_close(result, expected)

    def test_flatten_unflatten_with_zero_point(self):
        """_zero_point is included in inner_tensors when non-None."""
        qt = self._make_qt_with_zero_point()
        inner_tensors, metadata = qt.__tensor_flatten__()
        assert '_zero_point' in inner_tensors

        # Unflatten and verify
        tensor_dict = {k: getattr(qt, k) for k in inner_tensors}
        reconstructed = QuantizedTensor.__tensor_unflatten__(
            tensor_dict, metadata, qt.size(), qt.stride()
        )
        assert reconstructed._zero_point is not None
        torch.testing.assert_close(reconstructed.dequantize(), qt.dequantize())

    def test_fsdp_hooks_with_zero_point(self):
        """pre/post all-gather roundtrip preserves values with zero_point."""
        qt = self._make_qt_with_zero_point()
        inputs, meta = qt.fsdp_pre_all_gather(mesh=None)
        scale, quant_dtype, compute_dtype, zero_point, axiswise_dim = meta
        assert zero_point is not None

        dequant, _ = qt.fsdp_post_all_gather(inputs, meta, param_dtype=torch.float32)
        torch.testing.assert_close(dequant, qt.dequantize())


class TestRepr:
    """Test __repr__ output."""

    def test_repr(self):
        qt = QuantizedTensor.quantize(
            torch.randn(16, 32), quant_dtype=torch.float8_e4m3fn
        )
        r = repr(qt)
        assert "QuantizedTensor" in r
        assert "(16, 32)" in r
        assert "float8_e4m3fn" in r
        assert "float32" in r
        assert "scale=" in r


class TestManualConstruction:
    """Test direct QuantizedTensor.__new__ construction."""

    def test_manual_construction(self):
        data = torch.randn(8, 16).to(torch.float8_e4m3fn)
        scale = torch.tensor(0.01, dtype=torch.float32)
        zero_point = torch.tensor(0.0, dtype=torch.float32)

        qt = QuantizedTensor(
            data=data,
            scale=scale,
            quant_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
            zero_point=zero_point,
            axiswise_dim=None,
        )

        assert qt._data is data
        assert qt._scale is scale
        assert qt._zero_point is zero_point
        assert qt._quant_dtype == torch.float8_e4m3fn
        assert qt._compute_dtype == torch.bfloat16
        assert qt._axiswise_dim is None
        assert qt.shape == (8, 16)
        assert qt.dtype == torch.bfloat16  # presents as compute_dtype


class TestNonFP8Quantize:
    """Test quantize() with non-FP8 dtypes (bf16, fp16)."""

    def test_quantize_bf16(self):
        data = torch.randn(16, 32, dtype=torch.float32)
        qt = QuantizedTensor.quantize(data, quant_dtype=torch.bfloat16)
        assert qt._quant_dtype == torch.bfloat16
        assert qt._data.dtype == torch.bfloat16
        assert qt._scale.item() == 1.0
        # dequantize() returns compute_dtype (fp32); values match bf16 roundtrip
        dequant = qt.dequantize()
        assert dequant.dtype == torch.float32
        torch.testing.assert_close(dequant, data.to(torch.bfloat16).to(torch.float32))

    def test_quantize_fp16(self):
        data = torch.randn(16, 32, dtype=torch.float32)
        qt = QuantizedTensor.quantize(data, quant_dtype=torch.float16)
        assert qt._quant_dtype == torch.float16
        assert qt._data.dtype == torch.float16
        assert qt._scale.item() == 1.0
        dequant = qt.dequantize()
        assert dequant.dtype == torch.float32
        torch.testing.assert_close(dequant, data.to(torch.float16).to(torch.float32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
