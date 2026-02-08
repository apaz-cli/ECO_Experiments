"""Tests for FSDP2 support with QuantizedTensor."""

import pytest
import torch
import torch.nn as nn

from torchtitan.components.quantized_tensor import QuantizedTensor
from torchtitan.components.quantized_linear import QuantizedLinear
from torchtitan.distributed.quantized_dtensor import (
    QuantizedDTensor,
    shard_quantized_tensor,
    gather_quantized_tensor,
    prepare_quantized_model_for_fsdp,
)


def skip_if_no_dist():
    """Skip test if torch.distributed is not available."""
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")


def get_world_size():
    """Get world size for distributed tests."""
    if not torch.distributed.is_available():
        return 1
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


class TestQuantizedDTensor:
    """Test QuantizedDTensor functionality."""
    
    def test_quantized_dtensor_creation(self):
        """Test creating QuantizedDTensor from QuantizedTensor."""
        skip_if_no_dist()
        
        # Create a simple QuantizedTensor
        data_fp = torch.randn(32, 64, dtype=torch.float32)
        quantized = QuantizedTensor.quantize(
            data_fp,
            quant_dtype=torch.float8_e4m3fn,
        )
        
        # For single GPU, create a trivial mesh
        world_size = get_world_size()
        if world_size < 2:
            pytest.skip("Need at least 2 GPUs for sharding test")
        
        # Create device mesh
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        mesh = torch.distributed.device_mesh.init_device_mesh(device_type, (world_size,))
        
        # Shard the quantized tensor
        sharded = shard_quantized_tensor(quantized, mesh, dim=0)
        
        assert isinstance(sharded, QuantizedDTensor)
    
    def test_quantized_dtensor_dequantize(self):
        """Test dequantizing a QuantizedDTensor."""
        skip_if_no_dist()
        
        world_size = get_world_size()
        if world_size < 2:
            pytest.skip("Need at least 2 GPUs for sharding test")
        
        # Create a simple QuantizedTensor
        data_fp = torch.randn(32, 64, dtype=torch.float32)
        quantized = QuantizedTensor.quantize(
            data_fp,
            quant_dtype=torch.float8_e4m3fn,
        )
        
        # Create device mesh
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        mesh = torch.distributed.device_mesh.init_device_mesh(device_type, (world_size,))
        
        # Shard and dequantize
        sharded = shard_quantized_tensor(quantized, mesh, dim=0)
        dequantized = sharded.dequantize()
        
        assert dequantized.dtype == torch.float32
    
    def test_quantized_dtensor_gather(self):
        """Test gathering a sharded QuantizedDTensor."""
        skip_if_no_dist()
        
        world_size = get_world_size()
        if world_size < 2:
            pytest.skip("Need at least 2 GPUs for sharding test")
        
        # Create a simple QuantizedTensor
        data_fp = torch.randn(32, 64, dtype=torch.float32)
        quantized = QuantizedTensor.quantize(
            data_fp,
            quant_dtype=torch.float8_e4m3fn,
        )
        
        # Create device mesh
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        mesh = torch.distributed.device_mesh.init_device_mesh(device_type, (world_size,))
        
        # Shard and gather
        sharded = shard_quantized_tensor(quantized, mesh, dim=0)
        gathered = gather_quantized_tensor(sharded, mesh, dim=0)
        
        assert isinstance(gathered, QuantizedTensor)


class TestQuantizedLinearFSDP:
    """Test QuantizedLinear with FSDP2."""
    
    def test_prepare_model_for_fsdp(self):
        """Test preparing a model with QuantizedLinear for FSDP."""
        skip_if_no_dist()
        
        world_size = get_world_size()
        if world_size < 2:
            pytest.skip("Need at least 2 GPUs for FSDP test")
        
        # Create a simple model with QuantizedLinear
        model = nn.Sequential(
            QuantizedLinear(64, 32),
            nn.ReLU(),
            QuantizedLinear(32, 16),
        )
        
        # Create device mesh
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        mesh = torch.distributed.device_mesh.init_device_mesh(device_type, (world_size,))
        
        # Prepare for FSDP
        prepared_model = prepare_quantized_model_for_fsdp(model, mesh)
        
        # Check that the model structure is preserved
        assert isinstance(prepared_model, nn.Sequential)
        assert len(prepared_model) == 3
    
    def test_quantized_linear_forward_with_fsdp(self):
        """Test forward pass with FSDP-wrapped QuantizedLinear."""
        skip_if_no_dist()
        
        world_size = get_world_size()
        if world_size < 2:
            pytest.skip("Need at least 2 GPUs for FSDP test")
        
        # Create a simple model with QuantizedLinear
        model = nn.Sequential(
            QuantizedLinear(64, 32),
            nn.ReLU(),
            QuantizedLinear(32, 16),
        )
        
        # Create device mesh
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        mesh = torch.distributed.device_mesh.init_device_mesh(device_type, (world_size,))
        
        # Prepare for FSDP
        prepared_model = prepare_quantized_model_for_fsdp(model, mesh)
        
        # Test forward pass
        x = torch.randn(4, 64)
        output = prepared_model(x)
        
        assert output.shape == (4, 16)


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
        
        # Unflatten
        reconstructed = QuantizedTensor.__tensor_unflatten__(
            {k: quantized.__dict__[k] for k in inner_tensors},
            metadata,
            quantized.size(),
            quantized.stride(),
        )
        
        # Check that reconstructed tensor is correct
        assert isinstance(reconstructed, QuantizedTensor)
        assert reconstructed._quant_dtype == quantized._quant_dtype
        assert reconstructed._compute_dtype == quantized._compute_dtype
    
    def test_quantized_dtensor_flatten_unflatten(self):
        """Test flatten and unflatten for QuantizedDTensor."""
        skip_if_no_dist()
        
        world_size = get_world_size()
        if world_size < 2:
            pytest.skip("Need at least 2 GPUs for sharding test")
        
        # Create a simple QuantizedTensor
        data_fp = torch.randn(32, 64, dtype=torch.float32)
        quantized = QuantizedTensor.quantize(
            data_fp,
            quant_dtype=torch.float8_e4m3fn,
        )
        
        # Create device mesh
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        mesh = torch.distributed.device_mesh.init_device_mesh(device_type, (world_size,))
        
        # Shard the quantized tensor
        sharded = shard_quantized_tensor(quantized, mesh, dim=0)
        
        # Flatten
        inner_tensors, metadata = sharded.__tensor_flatten__()
        
        # Check that inner tensors are correct
        assert '_dtensor_data' in inner_tensors
        assert '_scale' in inner_tensors
        
        # Check that metadata is correct
        assert '_quant_dtype' in metadata
        assert '_compute_dtype' in metadata
        
        # Unflatten
        reconstructed = QuantizedDTensor.__tensor_unflatten__(
            {k: sharded.__dict__[k] for k in inner_tensors},
            metadata,
            sharded.size(),
            sharded.stride(),
        )
        
        # Check that reconstructed tensor is correct
        assert isinstance(reconstructed, QuantizedDTensor)


class TestQuantizedTensorDtypeOptions:
    """Test different dtype configurations for FSDP2."""
    
    @pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn])
    @pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_dtype_combinations(self, quant_dtype, compute_dtype):
        """Test that different dtype combinations work correctly."""
        skip_if_no_dist()
        
        world_size = get_world_size()
        if world_size < 2:
            pytest.skip("Need at least 2 GPUs for sharding test")
        
        # Create a QuantizedTensor with specific dtypes
        data_fp = torch.randn(32, 64, dtype=compute_dtype)
        quantized = QuantizedTensor.quantize(
            data_fp,
            quant_dtype=quant_dtype,
        )
        
        # Create device mesh
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        mesh = torch.distributed.device_mesh.init_device_mesh(device_type, (world_size,))
        
        # Shard and dequantize
        sharded = shard_quantized_tensor(quantized, mesh, dim=0)
        dequantized = sharded.dequantize()
        
        assert dequantized.dtype == compute_dtype


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
        
        # Forward pass
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Verify gradients exist for input and parameters
        assert x.grad is not None, "Input should have gradient"
        
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinear):
                assert module._weight_data.grad is not None, f"{name} weight should have gradient"
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
        # Re-quantize gathered data to verify reconstruction
        reconstructed_quantized = QuantizedTensor.quantize(
            quantized.dequantize(),
            quant_dtype=quantized._quant_dtype,
            axiswise_dim=quantized._axiswise_dim,
        )
        torch.testing.assert_close(reconstructed_quantized._data, quantized._data, atol=1e-5)


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
            quantized = QuantizedTensor.quantize(
                base_tensor,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
