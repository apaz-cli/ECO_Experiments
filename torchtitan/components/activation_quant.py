"""Activation quantization converter for ECO training.

Provides a ModelConverter that wraps nn.Linear layers to quantize activations
to FP8 during the forward pass, saving the compact FP8 representation for
backward (reducing activation memory).

This is independent of weight quantization — ECOAdamW handles weight quant
via its simulated quant path.  The two compose cleanly: this converter only
touches activations, the optimizer only touches weights.

Per the paper, row-wise scaling is used: each row (token) gets its own scale
factor equal to max(|row|) / max_fp8.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConverter, register_model_converter


__all__ = ["ActivationQuantConverter", "ActivationQuantLinear"]


# ---------------------------------------------------------------------------
# Custom autograd function — saves FP8 activations for backward
# ---------------------------------------------------------------------------

class _ActivationQuantLinearFn(torch.autograd.Function):
    """Linear forward with FP8 activation storage for backward.

    Forward:  y = x @ W^T + b           (full-precision matmul)
    Saved:    x_q (FP8), scale, weight   (compact activation storage)
    Backward: dequant x_q, compute grads (slight approximation in dW)
    """

    @staticmethod
    def forward(ctx, input, weight, bias, quant_dtype):
        # --- Row-wise quantize input for compact backward storage ---
        # input shape: (*, in_features).  We scale per-row (last dim kept).
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])           # (N, in)

        fp8_max = torch.finfo(quant_dtype).max
        row_amax = input_2d.abs().amax(dim=1, keepdim=True)    # (N, 1)
        scale = (row_amax / fp8_max).clamp(min=1e-12)          # (N, 1)
        input_q = (input_2d / scale).to(quant_dtype)            # (N, in) fp8

        ctx.save_for_backward(input_q, scale, weight)
        ctx.has_bias = bias is not None
        ctx.input_dtype = input.dtype
        ctx.orig_shape = orig_shape

        # Forward uses full-precision input (no approximation in y).
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input_q, scale, weight = ctx.saved_tensors

        # Dequantize activation for gradient computation.
        input_deq = input_q.to(ctx.input_dtype) * scale        # (N, in)
        input_deq = input_deq.reshape(ctx.orig_shape)

        # grad_input = grad_output @ W                          # (*, in)
        grad_input = grad_output @ weight

        # grad_weight = grad_output^T @ input_deq               # (out, in)
        go_2d = grad_output.reshape(-1, grad_output.shape[-1])  # (N, out)
        in_2d = input_deq.reshape(-1, input_deq.shape[-1])      # (N, in)
        grad_weight = go_2d.T @ in_2d

        grad_bias = go_2d.sum(0) if ctx.has_bias else None

        return grad_input, grad_weight, grad_bias, None


# ---------------------------------------------------------------------------
# Wrapper module — same Parameter objects as the original nn.Linear
# ---------------------------------------------------------------------------

class ActivationQuantLinear(nn.Module):
    """Drop-in wrapper around nn.Linear that quantizes activations.

    The weight and bias are the *same* Parameter objects from the original
    nn.Linear, so FSDP, ECOAdamW, and state-dict all work unchanged.
    """

    def __init__(self, original: nn.Linear, quant_dtype: torch.dtype):
        super().__init__()
        # Steal parameters — same objects, not copies.
        self.weight = original.weight
        self.bias = original.bias
        self.quant_dtype = quant_dtype
        self.in_features = original.in_features
        self.out_features = original.out_features

    def forward(self, x):
        return _ActivationQuantLinearFn.apply(x, self.weight, self.bias, self.quant_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"quant_dtype={self.quant_dtype}, bias={self.bias is not None}"
        )


# ---------------------------------------------------------------------------
# ModelConverter
# ---------------------------------------------------------------------------

_ACTIVATION_DTYPE_MAP = {
    "none": None,
    "fp8": torch.float8_e4m3fn,
}


class ActivationQuantConverter(ModelConverter):
    """Converter that wraps nn.Linear with activation quantization.

    Does NOT change weight storage — only activations are affected.
    Composes with ECOAdamW's simulated weight quantization.

    Reads ``eco.activation_dtype`` from config.  If "none", convert()
    is a no-op.
    """

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        activation_dtype_str = getattr(job_config.eco, "activation_dtype", "none")
        self.quant_dtype = _ACTIVATION_DTYPE_MAP.get(
            activation_dtype_str.lower(), None
        )

    def convert(self, model: nn.Module):
        """Replace nn.Linear layers with ActivationQuantLinear wrappers.

        Per the paper, only linear layers *within transformer blocks* are
        quantized — embedding and output projection layers are excluded.
        We use the fully-qualified name (FQN) to decide: only modules whose
        FQN starts with "layers." are wrapped.
        """
        if self.quant_dtype is None:
            return model

        self._convert_with_fqn(model, model, prefix="")
        return model

    def _convert_with_fqn(self, root: nn.Module, module: nn.Module, prefix: str):
        for name, child in list(module.named_children()):
            fqn = f"{prefix}.{name}" if prefix else name
            # Depth-first: convert children before checking this child.
            self._convert_with_fqn(root, child, fqn)
            if isinstance(child, nn.Linear) and fqn.startswith("layers."):
                wrapped = ActivationQuantLinear(child, self.quant_dtype)
                setattr(module, name, wrapped)

    def post_optimizer_hook(self, model, **kwargs):
        pass  # Nothing needed.


register_model_converter(ActivationQuantConverter, "activation_quant")
