# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for activation quantization functionality."""

import pytest
import torch
import torch.nn as nn

from torchtitan.config import JobConfig
from torchtitan.components.quantization.quantized import QuantizedLinearConverter
from torchtitan.components.quantized_linear import QuantizedLinear
from torchtitan.distributed import ParallelDims


def test_activation_dtype_config_parsing():
    """Test that activation_dtype config is parsed correctly."""
    from torchtitan.config import JobConfig

    # Test default
    config = JobConfig()
    assert config.eco.activation_dtype == "none"

    # Test setting values via dataclass
    from torchtitan.config.job_config import ECO
    eco_fp8 = ECO(activation_dtype="fp8")
    assert eco_fp8.activation_dtype == "fp8"


def test_converter_creates_quantized_linear_with_activation_dtype():
    """Test that QuantizedLinearConverter passes activation_dtype to QuantizedLinear."""
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )

    # Create mock JobConfig with activation_dtype
    class MockECO:
        activation_dtype = "fp8"

    class MockJobConfig:
        class model:
            converters = []
        eco = MockECO()

    # Create converter
    job_config = MockJobConfig()
    parallel_dims = ParallelDims()  # Use defaults
    converter = QuantizedLinearConverter(job_config, parallel_dims)

    # Convert model
    converter.convert(model)

    # Check that layers were converted
    assert isinstance(model[0], QuantizedLinear)
    assert isinstance(model[2], QuantizedLinear)

    # Check activation_dtype was set
    assert model[0].activation_quant_dtype == torch.float8_e4m3fn
    assert model[2].activation_quant_dtype == torch.float8_e4m3fn


def test_quantized_linear_forward_with_activation_quant():
    """Test that QuantizedLinear.forward() quantizes activations when requested."""
    torch.manual_seed(42)

    # Create QuantizedLinear with activation quantization
    layer = QuantizedLinear(
        in_features=16,
        out_features=8,
        activation_quant_dtype=torch.float8_e4m3fn,
        compute_dtype=torch.bfloat16,
    )

    # Create input
    x = torch.randn(4, 16, dtype=torch.bfloat16)

    # Forward pass (should quantize activations internally)
    output = layer(x)

    # Check output shape and dtype
    assert output.shape == (4, 8)
    assert output.dtype == torch.bfloat16


def test_quantized_linear_forward_without_activation_quant():
    """Test that QuantizedLinear works without activation quantization."""
    torch.manual_seed(42)

    # Create QuantizedLinear WITHOUT activation quantization
    layer = QuantizedLinear(
        in_features=16,
        out_features=8,
        activation_quant_dtype=None,  # No activation quant
        compute_dtype=torch.bfloat16,
    )

    # Create input
    x = torch.randn(4, 16, dtype=torch.bfloat16)

    # Forward pass
    output = layer(x)

    # Check output
    assert output.shape == (4, 8)
    assert output.dtype == torch.bfloat16


def test_activation_quant_convergence():
    """Test that a simple model converges with activation quantization."""
    torch.manual_seed(42)

    # Create simple model
    model = nn.Sequential(
        QuantizedLinear(
            16, 32,
            activation_quant_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        ),
        nn.ReLU(),
        QuantizedLinear(
            32, 1,
            activation_quant_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )
    )

    # Create toy data
    x = torch.randn(64, 16)
    y = torch.sum(x[:, :4], dim=1, keepdim=True)  # Simple target

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []

    for _ in range(100):
        optimizer.zero_grad()
        pred = model(x.to(torch.bfloat16))
        loss = nn.functional.mse_loss(pred, y.to(torch.bfloat16))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    # Check convergence
    assert losses[-1] < losses[0] * 0.5, (
        f"Model should converge: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )


def test_converter_none_activation_dtype():
    """Test that converter works with activation_dtype='none'."""
    model = nn.Sequential(nn.Linear(10, 10))

    class MockECO:
        activation_dtype = "none"

    class MockJobConfig:
        class model:
            converters = []
        eco = MockECO()

    job_config = MockJobConfig()
    parallel_dims = ParallelDims()
    converter = QuantizedLinearConverter(job_config, parallel_dims)

    converter.convert(model)

    # Should still convert to QuantizedLinear, but without activation quant
    assert isinstance(model[0], QuantizedLinear)
    assert model[0].activation_quant_dtype is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
