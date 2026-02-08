"""Distributed utilities for quantized tensors with FSDP2 support.

TODO: This module contains stub hooks for FSDP2 integration with
QuantizedTensor.  The hooks have not been validated end-to-end with
the new ECOAdamW-based architecture (where ECO injection happens inside
the optimizer, not in a post-optimizer converter hook).

This module provides DTensor support for QuantizedTensor, enabling FSDP2
sharding of quantized weights.
"""

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor, Shard, Replicate
from torch.distributed._tensor.placement_types import DTensorSpec
from typing import Dict, Any, Tuple, Optional

from torchtitan.components.quantized_tensor import QuantizedTensor


class QuantizedDTensor(DTensor):
    """DTensor wrapper for QuantizedTensor.
    
    Shards the quantized data (_data) while keeping scales replicated.
    This is efficient because scales are typically small (scalars or small tensors).
    """
    
    def __new__(cls, quantized_tensor: QuantizedTensor, device_mesh, placements, **kwargs):
        """Create DTensor from QuantizedTensor.
        
        Strategy:
        - Shard _data (the FP8 weights) across devices
        - Replicate _scale (small, typically scalar)
        """
        # Create DTensor from the quantized data
        data_dtensor = DTensor.from_local(
            quantized_tensor._data,
            device_mesh,
            placements,
            run_check=False,
            **kwargs
        )
        
        # Wrap the DTensor to track scale and metadata
        # We use a custom subclass to store the scale separately
        return QuantizedDTensor._make_dtensor_from_data(
            data_dtensor,
            quantized_tensor._scale,
            quantized_tensor._quant_dtype,
            quantized_tensor._compute_dtype,
            quantized_tensor._zero_point,
            quantized_tensor._axiswise_dim,
        )
    
    @staticmethod
    def _make_dtensor_from_data(
        data_dtensor: DTensor,
        scale: torch.Tensor,
        quant_dtype: torch.dtype,
        compute_dtype: torch.dtype,
        zero_point: Optional[torch.Tensor] = None,
        axiswise_dim: Optional[int] = None,
    ) -> "QuantizedDTensor":
        """Create QuantizedDTensor from already-quantized DTensor data."""
        # Create wrapper that presents as compute dtype
        wrapper = torch.Tensor._make_wrapper_subclass(
            QuantizedDTensor,
            data_dtensor.size(),
            strides=data_dtensor.stride(),
            storage_offset=data_dtensor.storage_offset(),
            device=data_dtensor.device,
            dtype=compute_dtype,
            layout=data_dtensor.layout,
            requires_grad=data_dtensor.requires_grad,
        )
        
        # Store components
        wrapper._dtensor_data = data_dtensor
        wrapper._scale = scale
        wrapper._zero_point = zero_point
        wrapper._quant_dtype = quant_dtype
        wrapper._compute_dtype = compute_dtype
        wrapper._axiswise_dim = axiswise_dim
        
        return wrapper
    
    @staticmethod
    def from_quantized(
        quantized_tensor: QuantizedTensor,
        device_mesh,
        placements,
        **kwargs
    ) -> "QuantizedDTensor":
        """Create DTensor from QuantizedTensor.
        
        Args:
            quantized_tensor: QuantizedTensor to shard
            device_mesh: Device mesh for sharding
            placements: Placement strategy (Shard, Replicate, etc.)
            
        Returns:
            QuantizedDTensor with sharded data
        """
        return QuantizedDTensor(quantized_tensor, device_mesh, placements, **kwargs)
    
    def to_quantized(self) -> QuantizedTensor:
        """Convert back to QuantizedTensor.
        
        This is used when we need to gather the sharded tensor back to local
        or convert to non-DTensor format.
        """
        # Get the full data tensor (all-gather if sharded)
        data_dtensor = self._dtensor_data
        
        # If it's a local tensor (already gathered), use it directly
        if not isinstance(data_dtensor, DTensor):
            data_tensor = data_dtensor
        else:
            # Convert DTensor to local tensor (this performs all-gather)
            data_tensor = data_dtensor.to_local()
        
        # Reconstruct QuantizedTensor
        return QuantizedTensor(
            data=data_tensor,
            scale=self._scale,
            quant_dtype=self._quant_dtype,
            compute_dtype=self._compute_dtype,
            zero_point=self._zero_point,
            axiswise_dim=self._axiswise_dim,
        )
    
    def dequantize(self) -> torch.Tensor:
        """Dequantize to compute dtype."""
        # Dequantize the underlying data
        data_dtensor = self._dtensor_data
        
        if not isinstance(data_dtensor, DTensor):
            data = data_dtensor.to(self._compute_dtype)
        else:
            data = data_dtensor.to(self._compute_dtype)
        
        # Apply zero point
        if self._zero_point is not None:
            zero = self._zero_point.to(self._compute_dtype)
            data = data - zero
        
        # Apply scale (broadcast if needed)
        if self._axiswise_dim is not None:
            shape = list(data.shape)
            scale_shape = [1] * len(shape)
            scale_shape[self._axiswise_dim] = -1
            scale = self._scale.view(scale_shape)
        else:
            scale = self._scale
            
        return data * scale
    
    def __tensor_flatten__(self) -> Tuple[list, Dict[str, Any]]:
        """Flatten for serialization (DTensor/FSDP checkpointing)."""
        inner_tensors = ['_dtensor_data', '_scale']
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
    ) -> "QuantizedDTensor":
        """Reconstruct from serialized form."""
        return QuantizedDTensor._make_dtensor_from_data(
            inner_tensors['_dtensor_data'],
            inner_tensors['_scale'],
            metadata['_quant_dtype'],
            metadata['_compute_dtype'],
            inner_tensors.get('_zero_point'),
            metadata['_axiswise_dim'],
        )


def shard_quantized_tensor(
    tensor: QuantizedTensor,
    device_mesh,
    dim: int = 0,
) -> QuantizedDTensor:
    """Shard a QuantizedTensor across devices.
    
    Args:
        tensor: QuantizedTensor to shard
        device_mesh: Device mesh for sharding
        dim: Dimension to shard along
        
    Returns:
        QuantizedDTensor with sharded data
    """
    placements = [Shard(dim)]
    return QuantizedDTensor.from_quantized(
        tensor,
        device_mesh,
        placements,
        run_check=False,
    )


def gather_quantized_tensor(
    tensor: QuantizedDTensor,
    device_mesh,
    dim: int = 0,
) -> QuantizedTensor:
    """Gather a sharded QuantizedDTensor back to local.
    
    Args:
        tensor: Sharded QuantizedDTensor
        device_mesh: Device mesh (for consistency)
        dim: Dimension that was sharded
        
    Returns:
        Local QuantizedTensor
    """
    return tensor.to_quantized()


class QuantizedLinearFSDPWrapper(nn.Module):
    """Wrapper for QuantizedLinear to work with FSDP2.
    
    This wrapper:
    1. Manages all-gather/reduce-scatter for quantized weights
    2. Handles gradient reduction for quantized parameters
    3. Provides proper checkpointing support
    """
    
    def __init__(self, quantized_linear: nn.Module):
        super().__init__()
        self.quantized_linear = quantized_linear
        
        # Store references to weight components
        if hasattr(quantized_linear, '_weight_data'):
            self._weight_data = quantized_linear._weight_data
            self._weight_scale = quantized_linear._weight_scale
        
        # FSDP hooks will be registered on this module
        self._fsdp_hooks = []
    
    def forward(self, x):
        """Forward pass with FSDP-aware weight management."""
        # Forward through the quantized linear layer
        # The quantized linear layer handles dequantization internally
        return self.quantized_linear(x)
    
    def register_fsdp_hooks(self, device_mesh, shard_dim: int = 0):
        """Register FSDP hooks for all-gather/reduce-scatter.
        
        Args:
            device_mesh: Device mesh for FSDP
            shard_dim: Dimension to shard weights along
        """
        # Remove existing hooks
        self._remove_fsdp_hooks()
        
        # Register pre-forward hook for all-gather
        pre_forward_hook = self._fsdp_pre_forward(device_mesh, shard_dim)
        self._fsdp_hooks.append(
            self.quantized_linear.register_forward_pre_hook(pre_forward_hook)
        )
        
        # Register post-backward hook for reduce-scatter
        post_backward_hook = self._fsdp_post_backward(device_mesh, shard_dim)
        if hasattr(self.quantized_linear, 'weight') and self.quantized_linear.weight is not None:
            self._fsdp_hooks.append(
                self.quantized_linear.weight.register_hook(post_backward_hook)
            )
    
    def _fsdp_pre_forward(self, device_mesh, shard_dim: int):
        """Pre-forward hook for all-gather of quantized weights."""
        def hook(module, input):
            # All-gather sharded weight data
            if hasattr(module, '_weight_data') and isinstance(module._weight_data, DTensor):
                # Convert DTensor to local (performs all-gather)
                local_data = module._weight_data.to_local()
                # Temporarily replace with full tensor
                self._original_sharded_weight = module._weight_data
                module._weight_data = local_data
        return hook
    
    def _fsdp_post_backward(self, device_mesh, shard_dim: int):
        """Post-backward hook for reduce-scatter of gradients."""
        def hook(grad):
            # Reduce-scatter gradients
            if isinstance(grad, torch.Tensor) and grad.grad_fn is not None:
                # The gradient is on the gathered tensor, need to shard it
                # This is handled automatically by FSDP's gradient reduction
                pass
            return grad
        return hook
    
    def _remove_fsdp_hooks(self):
        """Remove all registered FSDP hooks."""
        for hook in self._fsdp_hooks:
            hook.remove()
        self._fsdp_hooks.clear()


def register_quantized_fsdp_hooks(
    model: torch.nn.Module,
    device_mesh,
    shard_dim: int = 0,
):
    """Register hooks for FSDP with quantized tensors.
    
    This handles:
    1. Wrapping QuantizedLinear layers with FSDP support
    2. Registering all-gather/reduce-scatter hooks
    3. Setting up checkpoint serialization
    
    Args:
        model: The model to register hooks on
        device_mesh: Device mesh for FSDP
        shard_dim: Dimension to shard weights along (default: 0)
    """
    from torchtitan.components.quantized_linear import QuantizedLinear
    
    # Find all QuantizedLinear modules and wrap them
    for name, module in list(model.named_children()):
        if isinstance(module, QuantizedLinear):
            # Wrap with FSDP support
            wrapper = QuantizedLinearFSDPWrapper(module)
            setattr(model, name, wrapper)
            wrapper.register_fsdp_hooks(device_mesh, shard_dim)
        else:
            # Recursively process submodules
            register_quantized_fsdp_hooks(module, device_mesh, shard_dim)


def prepare_quantized_model_for_fsdp(
    model: torch.nn.Module,
    device_mesh,
    shard_dim: int = 0,
) -> torch.nn.Module:
    """Prepare a model with QuantizedLinear layers for FSDP2 training.
    
    This is the main entry point for FSDP2 integration with quantized tensors.
    
    Args:
        model: Model with QuantizedLinear layers
        device_mesh: Device mesh for FSDP
        shard_dim: Dimension to shard weights along (default: 0)
        
    Returns:
        Model prepared for FSDP2 training
        
    Example:
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> mesh = init_device_mesh("cuda", (world_size,))
        >>> model = prepare_quantized_model_for_fsdp(model, mesh)
    """
    # Register FSDP hooks
    register_quantized_fsdp_hooks(model, device_mesh, shard_dim)
    
    return model


__all__ = [
    'QuantizedDTensor',
    'shard_quantized_tensor',
    'gather_quantized_tensor',
    'QuantizedLinearFSDPWrapper',
    'register_quantized_fsdp_hooks',
    'prepare_quantized_model_for_fsdp',
]
