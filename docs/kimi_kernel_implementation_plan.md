# ECO Kernel Implementation Plan

## Overview

Implement native FP4/FP8 quantized kernels for ECO training to perform actual computation in low precision rather than just storage. Target: Blackwell architecture (B100/B200) with native FP4 support.

## Current State Analysis

**Problem**: `QuantizedTensor.__torch_dispatch__` currently dequantizes to BF16 for all operations:
```python
# Current behavior (torchtitan/components/quantized_tensor.py:205-226)
@classmethod
def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    def unwrap(t):
        if isinstance(t, QuantizedTensor):
            return t.dequantize()  # <-- Upcasts to BF16!
        return t
    args_dequant = tree_map(unwrap, args)
    return func(*args_dequant, **kwargs)
```

**Goal**: Intercept matmul operations and execute them in native FP4/FP8 precision on Blackwell tensor cores.

## Phase 1: Foundation (Week 1)

### 1.1 FP4 Format Support
**File**: `torchtitan/components/quantization/fp4_utils.py`

```python
"""FP4 (E2M1) format utilities for Blackwell."""

import torch
import numpy as np
from typing import Tuple

# FP4 E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
FP4_E2M1_MAX = 6.0  # Maximum representable value
FP4_E2M1_MIN = -6.0
FP4_E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5,  # Positive values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5,  # Negative values
    -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)

def pack_fp4(tensor: torch.Tensor) -> torch.Tensor:
    """Pack FP4 values: 2 values per uint8 byte.
    
    High nibble (bits 7-4) = first value
    Low nibble (bits 3-0) = second value
    """
    assert tensor.dtype == torch.uint8
    assert tensor.shape[-1] % 2 == 0
    
    # Reshape to pairs
    shape = tensor.shape
    tensor = tensor.reshape(-1, 2)
    
    # Pack: (val0 << 4) | val1
    packed = (tensor[:, 0] << 4) | tensor[:, 1]
    
    return packed.reshape(shape[:-1] + (shape[-1] // 2,))

def unpack_fp4(packed: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """Unpack uint8 bytes to FP4 values (as uint8 indices)."""
    assert packed.dtype == torch.uint8
    
    # Unpack nibbles
    val0 = (packed >> 4) & 0x0F  # High nibble
    val1 = packed & 0x0F         # Low nibble
    
    # Interleave
    unpacked = torch.stack([val0, val1], dim=-1).reshape(shape)
    
    return unpacked

def fp4_to_fp32(indices: torch.Tensor) -> torch.Tensor:
    """Convert FP4 indices (0-15) to FP32 values using lookup table."""
    return FP4_E2M1_TABLE[indices.long()]

def fp32_to_fp4(values: torch.Tensor, stochastic: bool = False) -> torch.Tensor:
    """Convert FP32 values to FP4 indices (0-15).
    
    Args:
        values: FP32 tensor to quantize
        stochastic: Use stochastic rounding for unbiased quantization
        
    Returns:
        uint8 tensor with FP4 indices
    """
    # Clamp to representable range
    values = torch.clamp(values, FP4_E2M1_MIN, FP4_E2M1_MAX)
    
    # Find nearest FP4 value
    # For deterministic: use torch.bucketize or searchsorted
    # For stochastic: add noise proportional to quantization gap
    
    if stochastic:
        # Compute quantization gaps for each value
        # This requires finding the nearest FP4 values above and below
        pass
    
    # Find nearest index
    diffs = values.unsqueeze(-1) - FP4_E2M1_TABLE
    indices = torch.argmin(torch.abs(diffs), dim=-1).to(torch.uint8)
    
    return indices

def quantize_fp4(tensor: torch.Tensor, scale: torch.Tensor, 
                 stochastic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP32/BF16 tensor to FP4 with scaling.
    
    Returns:
        (packed_uint8, scale)
    """
    # Scale tensor
    scaled = tensor / scale
    
    # Convert to FP4 indices
    indices = fp32_to_fp4(scaled, stochastic)
    
    # Pack if needed
    if indices.shape[-1] % 2 == 0:
        packed = pack_fp4(indices)
        return packed, scale
    else:
        return indices, scale

def dequantize_fp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP4 (packed or unpacked) to FP32."""
    if packed.dtype == torch.uint8 and packed.shape[-1] * 2 > packed.numel():
        # Packed format
        shape = packed.shape[:-1] + (packed.shape[-1] * 2,)
        indices = unpack_fp4(packed, shape)
    else:
        indices = packed
    
    # Lookup and scale
    values = fp4_to_fp32(indices)
    return values * scale
```

### 1.2 Dynamic Quantization Kernel
**File**: `torchtitan/kernels/quantization/dynamic_quantize.py`

```python
"""Dynamic quantization kernels for FP4/FP8."""

import torch
import triton
import triton.language as tl
from typing import Tuple, Literal


@triton.jit
def _dynamic_quantize_fp4_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    n_elements,
    per_row: tl.constexpr,
    row_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Dynamic quantization to FP4.
    
    Args:
        input_ptr: Input tensor pointer (BF16/FP16)
        output_ptr: Output tensor pointer (uint8, packed)
        scale_ptr: Scale tensor pointer
        n_elements: Total elements
        per_row: Whether to compute per-row scales
        row_stride: Stride between rows
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    if per_row:
        # Compute per-row amax
        row_id = offsets // row_stride
        # Use atomic max to compute row-wise amax
        local_amax = tl.max(tl.abs(x))
        # Store scale (computed outside or via separate kernel)
    else:
        # Per-tensor: use precomputed scale
        scale = tl.load(scale_ptr)
        
        # Quantize: scale, then convert to FP4
        scaled = x / scale
        
        # FP4 quantization (simplified - full impl needs FP4 conversion)
        # For now, quantize to 4-bit range
        fp4_max = 6.0  # E2M1 max
        quantized = tl.clamp(scaled, -fp4_max, fp4_max)
        
        # Store (packing handled separately)
        tl.store(output_ptr + offsets, quantized, mask=mask)


def dynamic_quantize_fp4(
    tensor: torch.Tensor,
    scaling_mode: Literal["per_tensor", "per_row", "block"] = "per_tensor",
    stochastic_rounding: bool = False,
    block_size: Tuple[int, int] = (128, 128),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize tensor to FP4.
    
    Args:
        tensor: Input tensor (BF16/FP16/FP32)
        scaling_mode: "per_tensor", "per_row", or "block"
        stochastic_rounding: Use stochastic rounding
        block_size: Block dimensions for block-wise scaling
        
    Returns:
        (quantized_packed, scale)
    """
    assert tensor.is_cuda
    
    orig_shape = tensor.shape
    tensor_flat = tensor.view(-1)
    n_elements = tensor_flat.numel()
    
    # Compute scales
    if scaling_mode == "per_tensor":
        amax = tensor.abs().max()
        scale = amax / 6.0  # FP4 max
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    elif scaling_mode == "per_row":
        # Compute per-row amax
        amax = tensor.abs().amax(dim=-1, keepdim=True)
        scale = amax / 6.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    elif scaling_mode == "block":
        # Block-wise quantization (for Blackwell)
        # Reshape into blocks and compute per-block scales
        pass
    
    # Allocate output (packed: half the size)
    output = torch.empty(
        (n_elements + 1) // 2,  # Round up for packing
        device=tensor.device,
        dtype=torch.uint8
    )
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _dynamic_quantize_fp4_kernel[grid](
        tensor_flat,
        output,
        scale,
        n_elements,
        per_row=(scaling_mode == "per_row"),
        row_stride=orig_shape[-1] if len(orig_shape) > 0 else 1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output, scale
```

### 1.3 Dequantization Kernel
**File**: `torchtitan/kernels/quantization/dequantize.py`

```python
"""Dequantization kernels for FP4/FP8."""

import torch
import triton
import triton.language as tl


@triton.jit
def _dequantize_fp4_kernel(
    input_ptr,      # Packed uint8
    scale_ptr,
    output_ptr,     # BF16/FP16
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Dequantize FP4 packed data to high precision."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load packed byte (contains 2 FP4 values)
    byte_idx = offsets // 2
    byte = tl.load(input_ptr + byte_idx, mask=mask)
    
    # Extract nibble
    is_high = (offsets % 2) == 0
    nibble = tl.where(is_high, (byte >> 4) & 0x0F, byte & 0x0F)
    
    # Convert nibble (0-15) to FP32 using FP4 E2M1 table
    # This is a simplified version - full version uses lookup
    sign = (nibble >> 3) & 0x1
    exponent = (nibble >> 1) & 0x3
    mantissa = nibble & 0x1
    
    # E2M1 decoding
    # value = (-1)^sign * 2^(exponent - 1) * (1 + mantissa * 0.5)
    # Special cases: exponent=0 -> subnormal
    
    # Simplified: use if-else for now (Triton supports this)
    val = tl.zeros_like(nibble, dtype=tl.float32)
    
    # This would be a lookup table in practice
    # For now, compute directly
    
    # Load scale and apply
    scale = tl.load(scale_ptr)
    dequant = val * scale
    
    tl.store(output_ptr + offsets, dequant, mask=mask)


def dequantize_fp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
    output_shape: torch.Size = None,
) -> torch.Tensor:
    """Dequantize FP4 packed tensor.
    
    Args:
        packed: Packed uint8 tensor (2 FP4 values per byte)
        scale: Scale factor(s)
        output_dtype: Target dtype (BF16/FP16/FP32)
        output_shape: Original shape (before packing)
        
    Returns:
        Dequantized tensor
    """
    assert packed.dtype == torch.uint8
    
    # Compute output size
    n_elements = packed.numel() * 2
    if output_shape is not None:
        n_elements = min(n_elements, output_shape.numel())
    
    output = torch.empty(n_elements, device=packed.device, dtype=output_dtype)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _dequantize_fp4_kernel[grid](
        packed,
        scale,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    if output_shape is not None:
        output = output.reshape(output_shape)
    
    return output
```

## Phase 2: Core Matmul Kernels (Week 2-3)

### 2.1 FP4 × BF16 GEMM (Forward)
**File**: `torchtitan/kernels/gemm/fp4_gemm.py`

```python
"""FP4 GEMM kernels for Blackwell."""

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _fp4_gemm_forward_kernel(
    a_ptr,          # Activation [M, K] BF16
    b_ptr,          # Weight [K, N] FP4 packed
    b_scale_ptr,    # Weight scale [N] or scalar
    c_ptr,          # Output [M, N] BF16
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """FP4 GEMM: C = A @ B^T
    
    A: [M, K] BF16 activation
    B: [K, N] FP4 packed weight (transposed during unpack)
    C: [M, N] BF16 output
    """
    # Map program ids to output blocks
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Grouping for better L2 cache
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Offsets
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Accumulator in FP32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Load scales
    b_scale = tl.load(b_scale_ptr + offs_bn, mask=offs_bn < N, other=1.0)
    
    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load activation block
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        
        # Load weight block (FP4 packed)
        # Each uint8 contains 2 FP4 values, so we load BLOCK_K // 2 bytes
        # Then unpack to get BLOCK_K values
        b_packed = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        
        # Unpack FP4: each byte -> 2 values
        # High nibble and low nibble
        b_high = (b_packed >> 4) & 0x0F
        b_low = b_packed & 0x0F
        
        # Interleave high and low to get full block
        # This requires careful indexing
        
        # Convert FP4 nibbles to FP32
        # Use lookup or direct computation
        b_high_f32 = _fp4_to_f32(b_high)
        b_low_f32 = _fp4_to_f32(b_low)
        
        # Apply scales
        b_high_f32 = b_high_f32 * b_scale[None, :]
        b_low_f32 = b_low_f32 * b_scale[None, :]
        
        # Matrix multiply using tensor cores
        # Note: need to handle the interleaved nature properly
        accumulator += tl.dot(a, b_high_f32)  # Simplified
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Store output
    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _fp4_to_f32(nibble):
    """Convert FP4 nibble to FP32."""
    # FP4 E2M1 lookup
    # This is a placeholder - real impl uses tl.where or inline LUT
    sign = (nibble >> 3) & 0x1
    exp = (nibble >> 1) & 0x3
    mant = nibble & 0x1
    
    # Compute value
    # exp_bias = 1 for E2M1
    # value = (-1)^sign * 2^(exp-1) * (1 + mant * 0.5)
    
    # Simplified: use precomputed table approach
    # For now, return placeholder
    return nibble.to(tl.float32)  # Placeholder


def fp4_gemm_forward(
    weight_packed: torch.Tensor,  # [K, N//2] uint8
    weight_scale: torch.Tensor,   # [N] or scalar
    activation: torch.Tensor,     # [M, K] BF16
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FP4 GEMM forward pass.
    
    Args:
        weight_packed: Packed FP4 weights [K, N//2] (2 values per byte)
        weight_scale: Per-column or per-tensor scale [N] or []
        activation: Input activation [M, K] BF16
        output_dtype: Output dtype (BF16/FP16)
        
    Returns:
        Output tensor [M, N]
    """
    assert weight_packed.dtype == torch.uint8
    assert activation.dtype in [torch.bfloat16, torch.float16]
    
    M, K = activation.shape
    K_w, N_packed = weight_packed.shape
    N = N_packed * 2
    
    assert K == K_w, f"Activation K ({K}) != Weight K ({K_w})"
    
    # Allocate output
    output = torch.empty(M, N, device=activation.device, dtype=output_dtype)
    
    # Grid configuration
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 128
    GROUP_M = 8
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    _fp4_gemm_forward_kernel[grid](
        activation, weight_packed, weight_scale, output,
        M, N, K,
        activation.stride(0), activation.stride(1),
        weight_packed.stride(0), weight_packed.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )
    
    return output
```

### 2.2 Backward Pass Kernels
**File**: `torchtitan/kernels/gemm/fp4_gemm_backward.py`

```python
"""FP4 GEMM backward pass kernels."""

import torch
from typing import Tuple


def fp4_gemm_backward(
    grad_output: torch.Tensor,      # [M, N] BF16
    weight_packed: torch.Tensor,    # [K, N//2] uint8
    weight_scale: torch.Tensor,     # [N]
    activation: torch.Tensor,       # [M, K] BF16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP4 GEMM backward pass.
    
    Returns:
        (grad_input [M, K] BF16, grad_weight [K, N] FP32)
    """
    # grad_input = grad_output @ weight
    # Since weight is FP4, we dequantize for this pass
    # Or use specialized kernel if available
    
    from torchtitan.components.quantization.fp4_utils import dequantize_fp4
    
    # Dequantize weight to BF16 for backward
    weight_bf16 = dequantize_fp4(weight_packed, weight_scale, 
                                  output_shape=torch.Size([weight_packed.shape[0], 
                                                          weight_packed.shape[1] * 2]))
    
    # grad_input: [M, N] @ [N, K] -> [M, K]
    grad_input = torch.matmul(grad_output, weight_bf16.t())
    
    # grad_weight: [K, M] @ [M, N] -> [K, N]
    # Accumulate in FP32 for stability
    grad_weight = torch.matmul(activation.t(), grad_output)
    
    return grad_input, grad_weight
```

### 2.3 High-Level Quantized Linear
**File**: `torchtitan/kernels/gemm/quantized_linear.py`

```python
"""High-level quantized linear interface."""

import torch
import torch.nn.functional as F
from typing import Optional
from torchtitan.components.quantized_tensor import QuantizedTensor


def quantized_linear_forward(
    input: torch.Tensor,
    weight_quantized: QuantizedTensor,
    bias: Optional[torch.Tensor] = None,
    quantize_activation: bool = True,
    activation_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Native quantized linear forward.
    
    This replaces the current dequantize-then-matmul approach
    with native low-precision matmul.
    """
    from torchtitan.kernels.gemm.fp4_gemm import fp4_gemm_forward
    from torchtitan.kernels.quantization.dynamic_quantize import dynamic_quantize_fp4
    
    # Get weight data
    if isinstance(weight_quantized, QuantizedTensor):
        weight_packed = weight_quantized._data
        weight_scale = weight_quantized._scale
        weight_dtype = weight_quantized._quant_dtype
    else:
        raise TypeError(f"Expected QuantizedTensor, got {type(weight_quantized)}")
    
    # Optionally quantize activation
    if quantize_activation and activation_dtype is not None:
        if activation_dtype == torch.float8_e4m3fn:
            # Use existing FP8 quantization
            pass
        elif str(activation_dtype) == "fp4":
            # Quantize to FP4
            input_packed, input_scale = dynamic_quantize_fp4(input)
            # Call FP4×FP4 kernel (if available)
            # Otherwise: dequantize input, use FP4×BF16 kernel
        else:
            # No quantization
            pass
    
    # Dispatch to appropriate kernel based on dtype
    if "float8" in str(weight_dtype):
        # Use FP8 kernel (cutlass/cublas)
        pass
    elif str(weight_dtype) == "fp4":
        # Use FP4 kernel
        output = fp4_gemm_forward(
            weight_packed,
            weight_scale,
            input,
        )
    else:
        # Fallback: dequantize
        output = F.linear(input, weight_quantized.dequantize(), bias)
        return output
    
    # Add bias
    if bias is not None:
        output = output + bias.unsqueeze(0)
    
    return output
```

## Phase 3: ECO Integration (Week 4)

### 3.1 Quantization Error Kernel
**File**: `torchtitan/kernels/eco/compute_error.py`

```python
"""ECO quantization error computation."""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _compute_quantization_error_kernel(
    theta_tilde_ptr,    # [N] FP32 - post-Adam weight
    theta_hat_ptr,      # [N//2] uint8 - packed quantized
    scale_ptr,          # scale
    error_ptr,          # [N] FP32 - output error
    n_elements,
    stochastic: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute quantization error: e = θ̃ - Q(θ̃) in FP32.
    
    This must be done in FP32 to avoid catastrophic cancellation
    when the quantization error is small compared to θ̃.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load θ̃ in FP32
    theta_tilde = tl.load(theta_tilde_ptr + offsets, mask=mask)
    
    # Load quantized value and dequantize
    byte_idx = offsets // 2
    is_high = (offsets % 2) == 0
    byte = tl.load(theta_hat_ptr + byte_idx, mask=mask)
    nibble = tl.where(is_high, (byte >> 4) & 0x0F, byte & 0x0F)
    
    # Convert FP4 to FP32
    theta_hat = _fp4_to_f32_triton(nibble)  # Needs implementation
    
    # Apply scale
    scale = tl.load(scale_ptr)
    theta_hat = theta_hat * scale
    
    # Compute error: e = θ̃ - θ_hat
    error = theta_tilde - theta_hat
    
    # Store error
    tl.store(error_ptr + offsets, error, mask=mask)


def compute_quantization_error(
    theta_tilde: torch.Tensor,
    quant_dtype: torch.dtype,
    stochastic_rounding: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute quantization and error for ECO.
    
    Critical: Must use FP32 for error computation to avoid
    catastrophic cancellation.
    
    Args:
        theta_tilde: Post-Adam weight in FP32 [N]
        quant_dtype: Target quantization dtype (FP4/FP8)
        stochastic_rounding: Use stochastic rounding
        
    Returns:
        (theta_hat_packed [N//2 uint8], 
         theta_hat_scale [],
         error [N FP32])
    """
    assert theta_tilde.dtype == torch.float32
    
    # First: quantize to get Q(θ̃)
    from torchtitan.components.quantization.fp4_utils import quantize_fp4
    
    theta_hat_packed, scale = quantize_fp4(
        theta_tilde,
        stochastic=stochastic_rounding
    )
    
    # Compute error: e = θ̃ - θ_hat
    n_elements = theta_tilde.numel()
    error = torch.empty_like(theta_tilde)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _compute_quantization_error_kernel[grid](
        theta_tilde,
        theta_hat_packed,
        scale,
        error,
        n_elements,
        stochastic=stochastic_rounding,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return theta_hat_packed, scale, error
```

### 3.2 Update QuantizedTensor Dispatch
**File**: `torchtitan/components/quantized_tensor.py`

```python
# Add to existing file

@classmethod
def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    """Dispatch PyTorch operations with native kernel support."""
    if kwargs is None:
        kwargs = {}
    
    # Check for matmul operations we can handle natively
    if func == torch.ops.aten.mm.default:
        return dispatch_quantized_mm(args[0], args[1])
    elif func == torch.ops.aten.linear.default:
        return dispatch_quantized_linear(*args, **kwargs)
    elif func == torch.ops.aten.addmm.default:
        # bias + input @ weight
        return dispatch_quantized_addmm(*args, **kwargs)
    
    # Fallback: dequantize and run
    def unwrap(t):
        if isinstance(t, QuantizedTensor):
            return t.dequantize()
        return t
    
    args_dequant = tree_map(unwrap, args)
    kwargs_dequant = tree_map(unwrap, kwargs)
    
    return func(*args_dequant, **kwargs_dequant)


def dispatch_quantized_mm(a, b):
    """Dispatch matrix multiplication with quantized tensors."""
    from torchtitan.kernels.gemm.quantized_linear import quantized_linear_forward
    
    # Determine which operand is quantized
    a_quant = isinstance(a, QuantizedTensor)
    b_quant = isinstance(b, QuantizedTensor)
    
    if not a_quant and not b_quant:
        # Neither quantized - shouldn't reach here
        return torch.mm(a, b)
    
    if a_quant and not b_quant:
        # A is quantized: compute A @ B^T then transpose
        # This is less common in transformers
        result = quantized_linear_forward(
            b.t(),  # Transpose to get [K, N]
            a,      # Quantized tensor
        )
        return result.t()
    
    if b_quant and not a_quant:
        # B is quantized: standard case (input @ weight^T)
        return quantized_linear_forward(a, b)
    
    # Both quantized - use specialized kernel or dequantize
    # For now, dequantize one operand
    return quantized_linear_forward(a, b)


def dispatch_quantized_linear(input, weight, bias=None):
    """Dispatch linear operation."""
    from torchtitan.kernels.gemm.quantized_linear import quantized_linear_forward
    
    if isinstance(weight, QuantizedTensor):
        return quantized_linear_forward(input, weight, bias)
    else:
        # Standard linear
        return F.linear(input, weight, bias)
```

### 3.3 Update QuantizedLinear
**File**: `torchtitan/components/quantized_linear.py`

```python
# Update forward method

def forward(self, input: torch.Tensor) -> torch.Tensor:
    """Forward pass with native quantized matmul."""
    # Ensure input is in compute dtype
    if input.dtype != self.compute_dtype:
        input = input.to(self.compute_dtype)
    
    # Use native quantized linear kernel
    from torchtitan.kernels.gemm.quantized_linear import quantized_linear_forward
    
    output = quantized_linear_forward(
        input,
        self.get_weight(),
        bias=self.bias,
        quantize_activation=(self.activation_quant_dtype is not None),
        activation_dtype=self.activation_quant_dtype,
    )
    
    return output
```

## Phase 4: Testing & Validation (Week 5-6)

### 4.1 Unit Tests
**File**: `tests/unit_tests/test_fp4_kernels.py`

```python
"""Tests for FP4 kernels."""

import torch
import pytest
from torchtitan.components.quantization.fp4_utils import (
    pack_fp4, unpack_fp4, fp4_to_fp32, fp32_to_fp4
)
from torchtitan.kernels.quantization.dynamic_quantize import dynamic_quantize_fp4
from torchtitan.kernels.gemm.fp4_gemm import fp4_gemm_forward


def test_fp4_pack_unpack():
    """Test FP4 packing and unpacking."""
    # Create test data
    indices = torch.arange(16, dtype=torch.uint8)
    tensor = indices.repeat(2)  # 32 elements
    
    # Pack
    packed = pack_fp4(tensor)
    assert packed.shape[0] == tensor.shape[0] // 2
    assert packed.dtype == torch.uint8
    
    # Unpack
    unpacked = unpack_fp4(packed, tensor.shape)
    assert torch.equal(tensor, unpacked)


def test_fp4_quantization_dequantization():
    """Test round-trip quantization."""
    torch.manual_seed(42)
    tensor = torch.randn(128, 256, device='cuda', dtype=torch.bfloat16)
    
    # Quantize
    packed, scale = dynamic_quantize_fp4(tensor)
    
    # Dequantize
    from torchtitan.kernels.quantization.dequantize import dequantize_fp4
    recovered = dequantize_fp4(packed, scale, output_dtype=torch.bfloat16)
    
    # Check reconstruction quality
    mse = ((tensor - recovered) ** 2).mean()
    assert mse < 0.1  # Loose bound for FP4


def test_fp4_gemm_numerics():
    """Test FP4 GEMM against reference BF16 implementation."""
    torch.manual_seed(42)
    M, K, N = 256, 512, 384
    
    # Create test tensors
    activation = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
    
    # Quantize weight to FP4
    from torchtitan.components.quantization.fp4_utils import quantize_fp4
    weight_packed, weight_scale = quantize_fp4(weight)
    
    # FP4 GEMM
    output_fp4 = fp4_gemm_forward(weight_packed, weight_scale, activation)
    
    # Reference BF16 GEMM
    output_ref = torch.matmul(activation, weight)
    
    # Check numerical accuracy
    rel_error = (output_fp4 - output_ref).abs().mean() / output_ref.abs().mean()
    assert rel_error < 0.1  # Within 10% relative error


def test_eco_error_computation():
    """Test ECO quantization error computation."""
    from torchtitan.kernels.eco.compute_error import compute_quantization_error
    
    torch.manual_seed(42)
    theta_tilde = torch.randn(1024, device='cuda', dtype=torch.float32)
    
    # Compute quantization and error
    theta_hat_packed, scale, error = compute_quantization_error(
        theta_tilde,
        quant_dtype=torch.float8_e4m3fn,  # Use FP8 for easier testing
        stochastic_rounding=False,
    )
    
    # Verify: theta_tilde ≈ theta_hat + error
    from torchtitan.kernels.quantization.dequantize import dequantize_fp4
    theta_hat = dequantize_fp4(theta_hat_packed, scale, 
                                output_dtype=torch.float32,
                                output_shape=theta_tilde.shape)
    
    reconstruction = theta_hat + error
    max_diff = (theta_tilde - reconstruction).abs().max()
    assert max_diff < 1e-5  # Should be exact in FP32
```

### 4.2 Integration Tests
**File**: `tests/integration_tests/test_eco_quantized_training.py`

```python
"""Integration tests for ECO with quantized kernels."""

import torch
from torchtitan.components.quantized_linear import QuantizedLinear
from torchtitan.components.eco_optimizer import ECOAdamW


def test_quantized_linear_training_step():
    """Test a full training step with quantized linear layers."""
    torch.manual_seed(42)
    
    # Create model
    layer = QuantizedLinear(256, 512, weight_quant_dtype=torch.float8_e4m3fn).cuda()
    
    # Create optimizer
    optimizer = ECOAdamW(layer.parameters(), lr=1e-3)
    
    # Forward pass
    input = torch.randn(32, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    output = layer(input)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    # Verify weights are still quantized
    assert isinstance(layer.get_weight(), QuantizedTensor)
    print("Training step completed successfully")


def test_loss_curve_convergence():
    """Test that training converges similarly to BF16 baseline."""
    # This would be a longer test comparing loss curves
    pass
```

## Phase 5: Optimization (Week 7+)

### 5.1 Cutlass Migration
When Triton kernels are functional, migrate to Cutlass for:
- Better Blackwell tensor core utilization
- FP4-specific optimizations
- Fused operations (e.g., GEMM + activation)

### 5.2 Additional Optimizations
- Block-wise quantization for better accuracy
- Dynamic scaling updates
- Gradient accumulation optimization
- Mixed-precision training support

## File Structure

```
torchtitan/kernels/
├── __init__.py
├── quantization/
│   ├── __init__.py
│   ├── fp4_utils.py              # FP4 format support
│   ├── dynamic_quantize.py       # Dynamic quantization kernels
│   └── dequantize.py             # Dequantization kernels
├── gemm/
│   ├── __init__.py
│   ├── fp4_gemm.py               # FP4 GEMM forward
│   ├── fp4_gemm_backward.py      # FP4 GEMM backward
│   └── quantized_linear.py       # High-level quantized linear
└── eco/
    ├── __init__.py
    └── compute_error.py          # ECO error computation
```

## Success Criteria

1. ✅ All linear layers use native FP4/FP8 matmul (no upcast to BF16)
2. ✅ Training loss matches BF16 baseline within 1%
3. ✅ Memory usage reduced by ~50% for weights (FP4 vs BF16)
4. ✅ Throughput within 10% of BF16 (or better with tensor cores)
5. ✅ ECO error compensation works correctly with quantized weights
6. ✅ Gradient checking passes
7. ✅ Convergence on small models (30M-100M params)

## Dependencies

- PyTorch 2.4+ (with Blackwell support)
- Triton 3.0+
- CUDA 12.4+
- (Optional) Cutlass for optimized kernels

## Notes

- FP4 format is NVIDIA E2M1 (2-bit exponent, 1-bit mantissa)
- Blackwell has native FP4 tensor core support
- Accumulators should remain FP32 for numerical stability
- Stochastic rounding is critical for ECO convergence
- Error computation must be in FP32 to prevent catastrophic cancellation
