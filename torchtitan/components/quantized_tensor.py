"""QuantizedTensor: A tensor subclass for quantized training with FP8/BF16/FP16 support.

This module provides a universal quantized tensor that can store data in low precision
(FP8_E4M3, FP8_E5M2) while presenting a high-precision interface (BF16/FP16) to PyTorch.
Works seamlessly with DTensor/FSDP2 for distributed training.
"""

from typing import Dict, Optional, Tuple, Any
import torch
from torch.utils._pytree import tree_map


class QuantizedTensor(torch.Tensor):
    """A tensor subclass for quantized storage with high-precision compute.
    
    Stores data in quantized format (FP8, INT8, etc.) while presenting as high-precision
    (BF16/FP16/FP32) to PyTorch operations. Automatically dequantizes when needed.
    
    Attributes:
        _data: The underlying quantized data tensor
        _scale: Per-tensor or per-channel scale factor
        _zero_point: Zero point for asymmetric quantization (optional)
        _quant_dtype: The storage dtype (e.g., torch.float8_e4m3fn)
        _compute_dtype: The compute dtype presented to PyTorch (e.g., torch.bfloat16)
        _axiswise_dim: For channel-wise quantization, which axis to scale across
    """
    
    _data: torch.Tensor
    _scale: torch.Tensor
    _zero_point: Optional[torch.Tensor]
    _quant_dtype: torch.dtype
    _compute_dtype: torch.dtype
    _axiswise_dim: Optional[int]
    
    __slots__ = ['_data', '_scale', '_zero_point', '_quant_dtype', '_compute_dtype', '_axiswise_dim']
    
    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        quant_dtype: torch.dtype,
        compute_dtype: torch.dtype,
        zero_point: Optional[torch.Tensor] = None,
        axiswise_dim: Optional[int] = None,
    ) -> "QuantizedTensor":
        """Create a new QuantizedTensor.
        
        Args:
            data: The quantized data tensor (already in quant_dtype)
            scale: Scale factor for dequantization
            quant_dtype: The storage dtype (e.g., torch.float8_e4m3fn)
            compute_dtype: The dtype presented to PyTorch operations
            zero_point: Zero point for asymmetric quantization
            axiswise_dim: For channel-wise quantization
        """
        # Create wrapper subclass that presents as compute_dtype
        kwargs = {
            'device': data.device,
            'dtype': compute_dtype,  # Present as compute dtype
            'layout': data.layout,
            'requires_grad': data.requires_grad,
        }
        
        # Create the wrapper
        wrapper = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            **kwargs
        )
        
        # Store the quantized data and metadata
        wrapper._data = data
        wrapper._scale = scale
        wrapper._zero_point = zero_point
        wrapper._quant_dtype = quant_dtype
        wrapper._compute_dtype = compute_dtype
        wrapper._axiswise_dim = axiswise_dim
        
        return wrapper
    
    def dequantize(self) -> torch.Tensor:
        """Dequantize to compute dtype.
        
        Returns:
            Dequantized tensor in compute_dtype
        """
        # Dequantize: (data - zero_point) * scale
        if self._zero_point is not None:
            data = self._data.to(self._compute_dtype) - self._zero_point.to(self._compute_dtype)
        else:
            data = self._data.to(self._compute_dtype)
        
        # Apply scale (broadcast if needed)
        if self._axiswise_dim is not None:
            # Channel-wise scaling
            scale = self._scale.view(*[1 if i != self._axiswise_dim else -1 
                                        for i in range(self.ndim)])
        else:
            scale = self._scale
            
        return data * scale
    
    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        quant_dtype: torch.dtype,
        axiswise_dim: Optional[int] = None,
    ) -> "QuantizedTensor":
        """Quantize a high-precision tensor to QuantizedTensor.
        
        Args:
            tensor: Input tensor in high precision
            quant_dtype: Target quantization dtype
            axiswise_dim: For channel-wise quantization
            
        Returns:
            QuantizedTensor with quantized storage
        """
        # Compute scale and zero_point
        if axiswise_dim is not None:
            # Channel-wise quantization
            dims = list(range(tensor.ndim))
            dims.remove(axiswise_dim)
            amax = tensor.abs().amax(dim=dims, keepdim=True)
        else:
            # Tensor-wise quantization
            amax = tensor.abs().amax()
        
        # For FP8: per-tensor scaling required (limited dynamic range)
        # For standard dtypes (bf16, fp16): cast directly, scale = 1.0
        if quant_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            dtype_max = torch.finfo(quant_dtype).max
            scale = amax / dtype_max
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            quantized = (tensor / scale).to(quant_dtype)
        else:
            scale = torch.ones(1, device=tensor.device, dtype=tensor.dtype)
            quantized = tensor.to(quant_dtype)
        
        return cls(
            data=quantized,
            scale=scale.squeeze() if axiswise_dim is not None else scale,
            quant_dtype=quant_dtype,
            compute_dtype=tensor.dtype,
            zero_point=None,  # Symmetric quantization
            axiswise_dim=axiswise_dim,
        )
    
    def __repr__(self) -> str:
        return (
            f"QuantizedTensor(shape={tuple(self.shape)}, "
            f"quant_dtype={self._quant_dtype}, "
            f"compute_dtype={self._compute_dtype}, "
            f"scale={self._scale.shape})"
        )
    
    # Serialization support for DTensor/FSDP
    def __tensor_flatten__(self) -> Tuple[list, Dict[str, Any]]:
        """Flatten for serialization (DTensor/FSDP checkpointing).
        
        Returns:
            (inner_tensors, metadata)
        """
        inner_tensors = ['_data', '_scale']
        if self._zero_point is not None:
            inner_tensors.append('_zero_point')
            
        metadata = {
            '_quant_dtype': self._quant_dtype,
            '_compute_dtype': self._compute_dtype,
            '_axiswise_dim': self._axiswise_dim,
        }
        return inner_tensors, metadata
    
    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
        outer_size: Tuple[int, ...],
        outer_stride: Tuple[int, ...],
    ) -> "QuantizedTensor":
        """Reconstruct from serialized form.
        
        Args:
            inner_tensors: Dict with '_data', '_scale', optionally '_zero_point'
            metadata: Dict with dtype info
            outer_size: Original tensor size
            outer_stride: Original tensor stride
            
        Returns:
            Reconstructed QuantizedTensor
        """
        return QuantizedTensor(
            data=inner_tensors['_data'],
            scale=inner_tensors['_scale'],
            quant_dtype=metadata['_quant_dtype'],
            compute_dtype=metadata['_compute_dtype'],
            zero_point=inner_tensors.get('_zero_point'),
            axiswise_dim=metadata['_axiswise_dim'],
        )
    
    # PyTorch dispatch - handle operations
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """Dispatch PyTorch operations on QuantizedTensor.
        
        Strategy:
        1. For supported ops (matmul, etc.): use specialized quantized implementation
        2. For unsupported ops: dequantize and run in compute dtype
        """
        if kwargs is None:
            kwargs = {}
            
        # Check if all args are QuantizedTensor or supported types
        def unwrap(t):
            if isinstance(t, QuantizedTensor):
                return t.dequantize()
            return t
        
        # For now, dequantize for all operations (safe fallback)
        # In the future, add specialized implementations for matmul, etc.
        args_dequant = tree_map(unwrap, args)
        kwargs_dequant = tree_map(unwrap, kwargs)
        
        return func(*args_dequant, **kwargs_dequant)
    
    # FSDP2 extension hooks
    def fsdp_pre_all_gather(self, mesh):
        """Prepare tensors for FSDP2 all-gather.

        Returns the FP8 _data tensor (bandwidth-efficient) plus metadata
        needed to reconstruct after gathering.
        """
        all_gather_inputs = (self._data,)
        metadata = (self._scale, self._quant_dtype, self._compute_dtype, self._zero_point, self._axiswise_dim)
        return all_gather_inputs, metadata

    def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
        """Reconstruct dequantized tensor after FSDP2 all-gather.

        Two paths:
        - First call (out=None): return (dequantized_tensor, (data_tensor,))
        - Subsequent calls (out provided): write into out in-place, return None
        """
        (data,) = all_gather_outputs
        scale, quant_dtype, compute_dtype, zero_point, axiswise_dim = metadata

        # Dequantize: mirrors QuantizedTensor.dequantize() but on raw tensors
        if zero_point is not None:
            dequant = data.to(compute_dtype) - zero_point.to(compute_dtype)
        else:
            dequant = data.to(compute_dtype)

        if axiswise_dim is not None:
            s = scale.view(*[1 if i != axiswise_dim else -1 for i in range(data.ndim)])
        else:
            s = scale

        dequant = dequant * s

        # Cast to param_dtype for mixed precision
        if dequant.dtype != param_dtype:
            dequant = dequant.to(param_dtype)

        if out is not None:
            out.copy_(dequant)
            return

        return dequant, (data,)

    # Disable torch_function to ensure dispatch works correctly
    __torch_function__ = torch._C._disabled_torch_function_impl


def is_quantized_tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a QuantizedTensor."""
    return isinstance(tensor, QuantizedTensor)


def maybe_dequantize(tensor: torch.Tensor) -> torch.Tensor:
    """Dequantize if tensor is QuantizedTensor, otherwise return as-is."""
    if isinstance(tensor, QuantizedTensor):
        return tensor.dequantize()
    return tensor
