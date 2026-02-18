"""Hardware-accelerated quantized GEMM dispatch for FP8 and future low-precision formats.

Provides:
  - quantize_fp8_rowwise / quantize_fp8_tensorwise  — quantization helpers
  - scaled_gemm                                      — dtype-dispatched GEMM
  - HardwareQuantLinearFn                            — autograd.Function (FP8 forward, BF16 backward)
  - HardwareQuantLinear                              — nn.Module with QuantizedTensor weight storage

Adding a new dtype (nvfp4, mxfp4, …):
  1. Add dtype to _HARDWARE_GEMM_DTYPES
  2. Add elif branch in scaled_gemm()
  3. Add quantize_* helper
  4. Handle storage dtype in HardwareQuantLinear.__init__
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from torchtitan.components.quantized_tensor import QuantizedTensor


__all__ = [
    "_HAS_FP8_GEMM",
    "_HARDWARE_GEMM_DTYPES",
    "quantize_fp8_rowwise",
    "quantize_fp8_tensorwise",
    "scaled_gemm",
    "HardwareQuantLinearFn",
    "HardwareQuantLinear",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware capability detection (done once at import time)
# ---------------------------------------------------------------------------

try:
    _device_cap = torch.cuda.get_device_capability()
    _HAS_FP8_GEMM: bool = _device_cap >= (8, 9)
except Exception:
    _HAS_FP8_GEMM: bool = False

# Set of torch.dtypes that map to HardwareQuantLinear (FP8 storage + hardware GEMM).
# Activation quantization dtype → this set → dispatch to HardwareQuantLinear.
# Add future dtypes here (torch.float4_e2m1fn, etc.) when hardware support lands.
_HARDWARE_GEMM_DTYPES: set = {torch.float8_e4m3fn}

_FP8_E4M3_MAX: float = torch.finfo(torch.float8_e4m3fn).max


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def quantize_fp8_rowwise(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor row-wise (per-token) to FP8 E4M3.

    Each row gets its own scale = amax(|row|) / fp8_max.
    Used for activation quantization in the forward pass.

    Args:
        tensor: Input tensor of shape (*, in_features), any float dtype.

    Returns:
        fp8_2d: (N, in_features) tensor of dtype float8_e4m3fn
        scale:  (N, 1) per-row scale factors, same dtype as input
    """
    orig_shape = tensor.shape
    tensor_2d = tensor.reshape(-1, orig_shape[-1])       # (N, in_features)

    row_amax = tensor_2d.abs().amax(dim=1, keepdim=True)  # (N, 1)
    scale = (row_amax / _FP8_E4M3_MAX).clamp(min=1e-12)  # (N, 1)
    fp8_2d = (tensor_2d / scale).to(torch.float8_e4m3fn)

    return fp8_2d, scale


def quantize_fp8_tensorwise(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor per-tensor to FP8 E4M3.

    scale = tensor.abs().amax() / fp8_max

    This is the same formula as ECOAdamW._requantize, so the optimizer's
    error model e = θ̃ − θ̂ correctly represents the quantization that the
    forward pass applies — no divergence between optimizer and hardware.

    Args:
        tensor: Input tensor, any float dtype.

    Returns:
        fp8_tensor: Same shape as input, dtype float8_e4m3fn
        scale:      Scalar tensor (0-dim), same dtype as input
    """
    amax = tensor.abs().amax()
    scale = torch.where(amax > 0, amax / _FP8_E4M3_MAX, torch.ones_like(amax))
    fp8_tensor = (tensor / scale).to(torch.float8_e4m3fn)
    return fp8_tensor, scale


# ---------------------------------------------------------------------------
# Dispatched hardware GEMM
# ---------------------------------------------------------------------------

def scaled_gemm(
    a_fp8: torch.Tensor,
    a_scale: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scale: torch.Tensor,
    quant_dtype_str: str,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Hardware-accelerated quantized GEMM with explicit scaling.

    Dispatches to torch._scaled_mm for FP8.  Add new dtype branches here
    for future quantization formats (nvfp4, mxfp4, etc.).

    Args:
        a_fp8:           First operand (M, K) in quant dtype.
        a_scale:         Per-row scale for a — shape (M, 1) or scalar.
        b_fp8:           Second operand (K, N) in quant dtype.
                         For weight matrices, pass weight.T (column-contiguous).
        b_scale:         Per-tensor scale for b — scalar.
        quant_dtype_str: "fp8" or future dtype names.
        out_dtype:       Output dtype (e.g. bfloat16).

    Returns:
        Result tensor (M, N) in out_dtype.
    """
    if quant_dtype_str == "fp8":
        return torch._scaled_mm(
            a_fp8,
            b_fp8,
            scale_a=a_scale,
            scale_b=b_scale,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )
    elif quant_dtype_str == "nvfp4":
        raise NotImplementedError(
            "nvfp4 GEMM not yet implemented — add dispatch here"
        )
    else:
        raise ValueError(
            f"scaled_gemm: unsupported quant_dtype_str {quant_dtype_str!r}. "
            "Use 'fp8' or implement a new branch."
        )


# ---------------------------------------------------------------------------
# Autograd function — FP8 forward, BF16 backward
# ---------------------------------------------------------------------------

class HardwareQuantLinearFn(autograd.Function):
    """Hardware FP8 GEMM in forward with BF16 grad computation in backward.

    Forward:  output = input_fp8 @ weight_fp8.T * scales + bias   (FP8 GEMM)
    Saved:    (input_fp8, input_scale, weight_bf16)                (compact)
    Backward: standard BF16 grad computation

    The weight may be:
      - A QuantizedTensor (FP8 storage): use ._data and ._scale directly.
      - A plain BF16 tensor:             quantize on-the-fly (fallback).
    """

    @staticmethod
    def forward(ctx, input, weight, bias, quant_dtype_str):
        # --- Prepare weight (FP8 storage or on-the-fly quantization) ---
        if isinstance(weight, QuantizedTensor):
            weight_fp8 = weight._data                     # (out, in), fp8
            weight_scale = weight._scale                   # scalar
            weight_bf16 = weight.dequantize()              # (out, in), bf16
        else:
            # Plain tensor: quantize once for this forward pass
            weight_fp8, weight_scale = quantize_fp8_tensorwise(
                weight.to(torch.float32)
            )
            weight_bf16 = weight

        # --- Quantize input row-wise (per token) ---
        orig_shape = input.shape
        input_fp8, input_scale = quantize_fp8_rowwise(input.float())
        # input_fp8: (N, in_features), input_scale: (N, 1)

        ctx.save_for_backward(input_fp8, input_scale, weight_bf16)
        ctx.has_bias = bias is not None
        ctx.input_dtype = input.dtype
        ctx.orig_shape = orig_shape
        ctx.quant_dtype_str = quant_dtype_str

        # --- FP8 GEMM ---
        # cuBLASLt requires mat1 to be row-major and mat2 to be column-major.
        # weight_fp8 (out, in) is row-major; .T gives (in, out) in Fortran order
        # (column-major) — exactly what cuBLASLt expects.  Do NOT call
        # .contiguous() here; that would convert to row-major, which is wrong.
        # _scaled_mm RowWise mode requires scale_a=(M,1) and scale_b=(1,N).
        # weight_scale is per-tensor (scalar); broadcast to (1, out_features).
        b_fp8_T = weight_fp8.T   # (in_features, out_features), Fortran-contiguous
        N_out = b_fp8_T.shape[-1]
        b_scale = weight_scale.reshape(1, 1).expand(1, N_out).contiguous()
        out = scaled_gemm(
            input_fp8,
            input_scale,
            b_fp8_T,
            b_scale,
            quant_dtype_str,
            input.dtype,
        )  # (N, out_features)

        out = out.reshape(*orig_shape[:-1], out.shape[-1])
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_fp8, input_scale, weight_bf16 = ctx.saved_tensors

        # Dequantize saved FP8 activation
        input_bf16 = input_fp8.to(ctx.input_dtype) * input_scale  # (N, in)

        go_2d = grad_output.reshape(-1, grad_output.shape[-1])     # (N, out)

        # grad_input = grad_output @ weight                         (*, in)
        grad_input = go_2d @ weight_bf16
        grad_input = grad_input.reshape(ctx.orig_shape)

        # grad_weight = grad_output.T @ input                       (out, in)
        grad_weight = go_2d.T @ input_bf16

        grad_bias = go_2d.sum(0) if ctx.has_bias else None

        return grad_input, grad_weight, grad_bias, None


# ---------------------------------------------------------------------------
# Module — QuantizedTensor weight + hardware GEMM dispatch
# ---------------------------------------------------------------------------

class HardwareQuantLinear(nn.Module):
    """nn.Linear replacement with FP8 weight storage and hardware FP8 GEMM.

    Wraps an existing nn.Linear, converting its weight to QuantizedTensor
    (FP8 ``_data``, BF16 ``compute_dtype``) and using ``torch._scaled_mm``
    in the forward pass on supported hardware.  Falls back to a dequantized
    BF16 ``F.linear`` on CPUs and older GPUs.

    Memory layout (per parameter element):
      - No master weights: FP8 (1 byte) + FP32 momentum + FP32 variance = 9 bytes
      - With master weights: same + FP32 master in optimizer state         = 13 bytes

    Only quant_dtype_str="fp8" is implemented.  To add nvfp4/mxfp4:
      1. Add the dtype to _HARDWARE_GEMM_DTYPES
      2. Add an elif branch in scaled_gemm()
      3. Handle packed storage here in __init__

    Usage::

        original = nn.Linear(in_features, out_features)
        layer = HardwareQuantLinear(original, quant_dtype_str="fp8")
    """

    def __init__(self, original: nn.Linear, quant_dtype_str: str):
        super().__init__()

        if quant_dtype_str != "fp8":
            raise ValueError(
                f"HardwareQuantLinear expects quant_dtype_str='fp8', "
                f"got {quant_dtype_str!r}. "
                "For other storage dtypes, add new handling here."
            )

        self._quant_dtype_str = quant_dtype_str
        self.in_features = original.in_features
        self.out_features = original.out_features

        # Convert weight to QuantizedTensor (FP8 storage, BF16 compute).
        # Direct insertion into _parameters bypasses isinstance(v, Parameter)
        # guard in register_parameter() — this is a known pattern.
        # parameters(), named_parameters(), FSDP2, and ECOAdamW all iterate
        # _parameters directly and work correctly with tensor subclasses.
        qt = QuantizedTensor.quantize(original.weight.data, torch.float8_e4m3fn)
        qt.requires_grad_(True)
        self._parameters["weight"] = qt

        # Steal the bias Parameter from the original module (same object,
        # not a copy).  nn.Module.__setattr__ auto-registers nn.Parameter.
        self.bias = original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_FP8_GEMM and x.is_cuda:
            return HardwareQuantLinearFn.apply(
                x, self.weight, self.bias, self._quant_dtype_str
            )
        else:
            # CPU / old GPU: let __torch_dispatch__ dequantize the QuantizedTensor
            # weight transparently.  This path keeps gradients flowing correctly
            # (autograd accumulates grad_weight from the linear backward).
            return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"quant_dtype={self._quant_dtype_str}, bias={self.bias is not None}"
        )
