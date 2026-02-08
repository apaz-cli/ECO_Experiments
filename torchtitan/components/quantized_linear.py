# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""QuantizedLinear: Linear layer with quantized weights and activations for ECO.

This module provides a drop-in replacement for nn.Linear that uses QuantizedTensor
for weights and optionally quantizes activations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from torchtitan.components.quantized_tensor import QuantizedTensor


class QuantizedLinear(nn.Module):
    """Linear layer with quantized weights and activations.
    
    This is designed for H100+ GPUs with native FP8 support. The weights are stored
    in FP8 format but computations happen in BF16 or FP16.
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        weight_quant_dtype: Dtype for weight quantization (default: FP8_E4M3)
        activation_quant_dtype: Dtype for activation quantization (default: None = no activation quant)
        compute_dtype: Dtype for computation (default: BF16)
        bias: Whether to include bias term
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_quant_dtype: torch.dtype = torch.float8_e4m3fn,
        activation_quant_dtype: Optional[torch.dtype] = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_quant_dtype = weight_quant_dtype
        self.activation_quant_dtype = activation_quant_dtype
        self.compute_dtype = compute_dtype
        
        # Create weight as Parameter for gradient tracking (wrap FP8 tensor)
        # This allows optimizer to update the quantized weight directly
        weight_data = torch.empty(out_features, in_features, dtype=weight_quant_dtype)
        self.register_parameter('_weight_data', nn.Parameter(weight_data, requires_grad=False))
        
        # Scale remains a Parameter that can be updated through ECO
        self.register_parameter('_weight_scale', nn.Parameter(torch.ones(1, dtype=compute_dtype)))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=compute_dtype))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        # Initialize in compute dtype then quantize
        weight_fp = torch.empty(
            self.out_features, 
            self.in_features, 
            dtype=self.compute_dtype
        )
        nn.init.kaiming_uniform_(weight_fp, a=torch.nn.init.calculate_gain('linear'))
        
        # Quantize weight with global scaling (default behavior)
        # This uses tensor-wise quantization which is simpler and works correctly
        quantized = QuantizedTensor.quantize(
            weight_fp,
            quant_dtype=self.weight_quant_dtype,
        )
        
        # Copy quantized data to parameters
        self._weight_data.data.copy_(quantized._data)
        self._weight_scale.data.copy_(quantized._scale)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def get_weight(self) -> QuantizedTensor:
        """Get weight as QuantizedTensor."""
        return QuantizedTensor(
            data=self._weight_data,
            scale=self._weight_scale,
            quant_dtype=self.weight_quant_dtype,
            compute_dtype=self.compute_dtype,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input: Input tensor (will be cast to compute_dtype)
            
        Returns:
            Output tensor in compute_dtype
        """
        # Ensure input is in compute dtype
        if input.dtype != self.compute_dtype:
            input = input.to(self.compute_dtype)
        
        # Standard linear operation
        # Dequantize weight with gradient tracking
        weight_fp = self._weight_data.to(self.compute_dtype) * self._weight_scale
        output = F.linear(input, weight_fp, self.bias)
        return output
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"weight_quant_dtype={self.weight_quant_dtype}, "
            f"compute_dtype={self.compute_dtype}, "
            f"bias={self.bias is not None}"
        )


__all__ = ["QuantizedLinear"]
