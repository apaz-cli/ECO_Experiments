# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ECOAdamW: AdamW optimizer with ECO error-compensating injection.

Implements Algorithm 3 from:
    "ECO: Quantized Training without Full-Precision Master Weights"
    Nikdan et al., 2025

ECO injection happens INSIDE the optimizer step, between the Adam update
and the requantization.  This avoids the old converter design where θ̃ was
destroyed by premature requantization before the hook could see it.

For QuantizedTensor parameters the per-step flow is:
    1. dequant → θ_bf16
    2. standard Adam update → θ̃  (the "ideal" updated weight in BF16)
    3. requant θ̃ → θ_hat  (the FP8-representable weight)
    4. error  e = θ̃ − θ_hat
    5. inject e into momentum:  m += coeff · diag(sqrt(v/bc2)+eps) · e
    6. store θ_hat back into the QuantizedTensor

For regular (non-quantized) parameters, a standard AdamW step is performed.
"""

import math
from typing import Any, Callable, Optional

import torch
from torch.optim.optimizer import Optimizer

from torchtitan.components.quantized_tensor import QuantizedTensor


__all__ = ["ECOAdamW"]


_DTYPE_TO_QUANT_STR = {
    torch.float8_e4m3fn: "fp8",
    torch.bfloat16: "bf16",
}


class ECOAdamW(Optimizer):
    """AdamW optimizer with ECO (Error-Compensating Optimizer) support.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for running averages (default: (0.9, 0.999))
        eps: Numerical stability term (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        optim_state_dtype: Dtype for optimizer states (default: torch.float32)
        optim_compute_dtype: Dtype for computation (default: torch.float32)
        eco_enabled: Whether to perform ECO injection (default: True)
        stochastic_rounding: Whether to use stochastic rounding during
            requantization (default: False)
        heuristic_log_freq: How often (in steps) to compute heuristic
            diagnostics.  0 means never (default: 0)
        quantize_weights: Whether to simulate quantized weight storage via
            quant→dequant round-trip on regular params (default: True)
        quant_dtype: Quantization dtype string for simulated weight storage.
            'fp8' for FP8 E4M3, 'bf16' for bfloat16 baseline (default: 'bf16')
        master_weights_dtype: Dtype for master weights buffer. None means no
            master weights (pure ECO). torch.float32 or torch.bfloat16 creates
            a separate high-precision copy (traditional quantized training).
            (default: None)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        optim_state_dtype: torch.dtype = torch.float32,
        optim_compute_dtype: torch.dtype = torch.float32,
        eco_enabled: bool = True,
        stochastic_rounding: bool = True,
        heuristic_log_freq: int = 0,
        quantize_weights: bool = True,
        quant_dtype: str = "bf16",
        master_weights_dtype: torch.dtype | None = None,
        include_weight_decay_in_injection: bool = True,
        exclude_from_quant: frozenset | None = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            optim_state_dtype=optim_state_dtype,
            optim_compute_dtype=optim_compute_dtype,
        )
        super().__init__(params, defaults)

        self._eco_enabled = eco_enabled
        self._stochastic_rounding = stochastic_rounding
        self._heuristic_log_freq = heuristic_log_freq
        self._quantize_weights = quantize_weights
        self._quant_dtype = quant_dtype
        self._master_weights_dtype = master_weights_dtype
        self._include_weight_decay_in_injection = include_weight_decay_in_injection
        self._exclude_from_quant = exclude_from_quant or frozenset()
        self._eco_step_count = 0
        # Accumulated heuristic metrics (consumed by get_eco_metrics)
        self._eco_metrics: dict[str, float] = {}

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("optim_state_dtype", torch.float32)
            group.setdefault("optim_compute_dtype", torch.float32)

    # ------------------------------------------------------------------
    # Public API for metrics collection
    # ------------------------------------------------------------------
    def get_eco_metrics(self) -> dict[str, Any]:
        """Return and clear accumulated ECO heuristic metrics."""
        metrics = dict(self._eco_metrics)
        self._eco_metrics.clear()
        return metrics

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._eco_step_count += 1
        should_log = (
            self._heuristic_log_freq > 0
            and self._eco_step_count % self._heuristic_log_freq == 0
        )

        # Per-layer metric accumulators
        layer_metrics: dict[str, list[float]] = {
            "eco/heuristic/norm_ratio": [],
            "eco/heuristic/cosine_sim": [],
            "eco/heuristic/rel_diff_norm": [],
        }

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            optim_state_dtype = group["optim_state_dtype"]
            optim_compute_dtype = group["optim_compute_dtype"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("ECOAdamW does not support sparse gradients")

                # Lazy state init
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(param, dtype=optim_state_dtype)
                    state["exp_avg_sq"] = torch.zeros_like(param, dtype=optim_state_dtype)
                    # Initialize master weights if requested
                    if self._master_weights_dtype is not None:
                        if isinstance(param, QuantizedTensor):
                            # Dequantize FP8 to get initial high-precision values.
                            # QuantizedTensor is already in its quantized form; no
                            # pre-quantization of param needed.
                            state["master_weights"] = param.dequantize().clone().to(self._master_weights_dtype)
                        else:
                            # Store original unquantized weights as initial master weights
                            state["master_weights"] = param.detach().clone().to(self._master_weights_dtype)
                            # Quantize param.data so the first forward pass uses quantized weights
                            # (otherwise first forward uses unquantized, all others use quantized)
                            if self._quantize_weights and param.dim() >= 2 and id(param) not in self._exclude_from_quant:
                                param_f = param.data.to(optim_compute_dtype) if param.dtype != optim_compute_dtype else param.data.clone()
                                theta_hat = self._requantize(param_f, self._quant_dtype, self._stochastic_rounding)
                                param.data.copy_(theta_hat.to(param.dtype))

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"].item()

                if isinstance(param, QuantizedTensor):
                    self._step_quantized(
                        param, grad, exp_avg, exp_avg_sq,
                        lr, beta1, beta2, eps, weight_decay, step,
                        optim_compute_dtype, should_log, layer_metrics,
                    )
                elif self._quantize_weights and param.dim() >= 2 and id(param) not in self._exclude_from_quant:
                    self._step_simulated_quant(
                        param, grad, exp_avg, exp_avg_sq,
                        lr, beta1, beta2, eps, weight_decay, step,
                        optim_compute_dtype, should_log, layer_metrics,
                    )
                else:
                    self._step_regular(
                        param, grad, exp_avg, exp_avg_sq,
                        lr, beta1, beta2, eps, weight_decay, step,
                    )

        # Aggregate heuristic metrics if we logged this step
        if should_log:
            self._aggregate_metrics(layer_metrics)

        return loss

    # ------------------------------------------------------------------
    # Standard AdamW (non-quantized params)
    # ------------------------------------------------------------------
    def _step_regular(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        step: int,
    ) -> None:
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = lr / bias_correction1

        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        param.addcdiv_(exp_avg, denom, value=-step_size)

    # ------------------------------------------------------------------
    # Simulated quantization (regular params with quant→dequant round-trip)
    # ------------------------------------------------------------------
    def _step_simulated_quant(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        step: int,
        optim_compute_dtype: torch.dtype,
        should_log: bool,
        layer_metrics: dict[str, list[float]],
    ):
        # Check if we're using master weights
        state = self.state[param]
        has_master_weights = "master_weights" in state

        # Work in optim_compute_dtype (FP32 by default) to avoid cancellation
        if has_master_weights:
            # Start from master weights (high precision)
            param_f = state["master_weights"].to(optim_compute_dtype) if state["master_weights"].dtype != optim_compute_dtype else state["master_weights"].clone()
        else:
            # Start from quantized param (pure ECO mode)
            param_f = param.data.to(optim_compute_dtype) if param.dtype != optim_compute_dtype else param.data.clone()

        grad_compute = grad if grad.dtype == optim_compute_dtype else grad.to(optim_compute_dtype)

        if exp_avg.dtype == optim_compute_dtype:
            exp_avg_c = exp_avg
            exp_avg_sq_c = exp_avg_sq
            copy_back = False
        else:
            exp_avg_c = exp_avg.to(optim_compute_dtype)
            exp_avg_sq_c = exp_avg_sq.to(optim_compute_dtype)
            copy_back = True

        # 1. Standard Adam update
        exp_avg_c.mul_(beta1).add_(grad_compute, alpha=1 - beta1)
        exp_avg_sq_c.mul_(beta2).addcmul_(grad_compute, grad_compute, value=1 - beta2)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = lr / bias_correction1

        denom = (exp_avg_sq_c.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        if weight_decay != 0:
            param_f.mul_(1 - lr * weight_decay)

        param_f.addcdiv_(exp_avg_c, denom, value=-step_size)

        # θ̃ is now in param_f — the ideal updated weight BEFORE requant.

        # 2. Quant→dequant round-trip to simulate FP8 storage
        theta_hat = self._requantize(
            param_f, self._quant_dtype, self._stochastic_rounding
        )

        # 3. ECO injection (Algorithm 3)
        if self._eco_enabled:
            error = param_f - theta_hat  # e = θ̃ − θ_hat

            injection_coeff = (bias_correction1 / lr) * (1.0 - 1.0 / beta1)
            if self._include_weight_decay_in_injection and weight_decay != 0:
                injection_coeff *= (1.0 - lr * weight_decay)
            adaptive_scale = (exp_avg_sq_c / bias_correction2).sqrt().add_(eps)
            error_injected = injection_coeff * adaptive_scale * error

            if exp_avg_c.dtype != error_injected.dtype:
                error_injected = error_injected.to(exp_avg_c.dtype)
            exp_avg_c.add_(error_injected)

            # Heuristic diagnostics
            if should_log:
                prev_error = self.state[param].get("prev_error")
                if prev_error is not None:
                    self._compute_heuristics(
                        layer_metrics, prev_error, error,
                    )
                self.state[param]["prev_error"] = error.detach().clone()

        # Write momentum back if we changed dtype
        if copy_back:
            exp_avg.copy_(exp_avg_c.to(exp_avg.dtype))
            exp_avg_sq.copy_(exp_avg_sq_c.to(exp_avg_sq.dtype))

        # 4. Store results
        if has_master_weights:
            # Store θ̃ (pre-quantization) to master weights
            state["master_weights"].copy_(param_f.to(state["master_weights"].dtype))
            # Store θ̂ (quantized) to param
            param.data.copy_(theta_hat.to(param.dtype))
        else:
            # Pure ECO mode: only store θ̂ to param
            param.data.copy_(theta_hat.to(param.dtype))

    # ------------------------------------------------------------------
    # ECO-aware AdamW (QuantizedTensor params)
    # ------------------------------------------------------------------
    def _step_quantized(
        self,
        param: QuantizedTensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        step: int,
        optim_compute_dtype: torch.dtype,
        should_log: bool,
        layer_metrics: dict[str, list[float]],
    ):
        # 1. Get high-precision starting point for Adam update.
        #    For master-weights mode: start from stored master weights (θ_mw).
        #    For pure ECO mode:       dequantize FP8 param (θ̂ from last step).
        #    The paper requires θ̃ in "high precision" so the error
        #    e = θ̃ − θ_hat is computed without catastrophic cancellation.
        state = self.state[param]
        has_master_weights = "master_weights" in state

        if has_master_weights:
            mw = state["master_weights"]
            param_f = mw.to(optim_compute_dtype) if mw.dtype != optim_compute_dtype else mw.clone()
        else:
            param_dequant = param.dequantize()
            param_f = param_dequant.to(optim_compute_dtype) if param_dequant.dtype != optim_compute_dtype else param_dequant

        grad_compute = grad if grad.dtype == optim_compute_dtype else grad.to(optim_compute_dtype)

        if exp_avg.dtype == optim_compute_dtype:
            exp_avg_c = exp_avg
            exp_avg_sq_c = exp_avg_sq
            copy_back = False
        else:
            exp_avg_c = exp_avg.to(optim_compute_dtype)
            exp_avg_sq_c = exp_avg_sq.to(optim_compute_dtype)
            copy_back = True

        # 2. Standard Adam update
        exp_avg_c.mul_(beta1).add_(grad_compute, alpha=1 - beta1)
        exp_avg_sq_c.mul_(beta2).addcmul_(grad_compute, grad_compute, value=1 - beta2)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = lr / bias_correction1

        denom = (exp_avg_sq_c.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        if weight_decay != 0:
            param_f.mul_(1 - lr * weight_decay)

        param_f.addcdiv_(exp_avg_c, denom, value=-step_size)

        # θ̃ is now in param_f — the ideal updated weight BEFORE requant.

        # 3. Requantize θ̃ → θ_hat (optionally with stochastic rounding)
        quant_str = _DTYPE_TO_QUANT_STR[param._quant_dtype]
        theta_hat = self._requantize(
            param_f, quant_str, self._stochastic_rounding
        )

        # 4. ECO injection (Algorithm 3)
        if self._eco_enabled:
            error = param_f - theta_hat  # e = θ̃ − θ_hat

            injection_coeff = (bias_correction1 / lr) * (1.0 - 1.0 / beta1)
            if self._include_weight_decay_in_injection and weight_decay != 0:
                injection_coeff *= (1.0 - lr * weight_decay)
            adaptive_scale = (exp_avg_sq_c / bias_correction2).sqrt().add_(eps)
            error_injected = injection_coeff * adaptive_scale * error

            if exp_avg_c.dtype != error_injected.dtype:
                error_injected = error_injected.to(exp_avg_c.dtype)
            exp_avg_c.add_(error_injected)

            # 5. Heuristic diagnostics
            if should_log:
                prev_error = state.get("prev_error")
                if prev_error is not None:
                    self._compute_heuristics(
                        layer_metrics, prev_error, error,
                    )
                state["prev_error"] = error.detach().clone()

        # Write momentum back if we changed dtype
        if copy_back:
            exp_avg.copy_(exp_avg_c.to(exp_avg.dtype))
            exp_avg_sq.copy_(exp_avg_sq_c.to(exp_avg_sq.dtype))

        # 6. Store θ_hat back into the QuantizedTensor
        new_q = QuantizedTensor.quantize(
            theta_hat, quant_dtype=param._quant_dtype, axiswise_dim=param._axiswise_dim,
        )
        param._data.copy_(new_q._data)
        param._scale.copy_(new_q._scale)

        # 7. Update master weights with θ̃ (pre-quantization ideal value)
        if has_master_weights:
            state["master_weights"].copy_(param_f.to(state["master_weights"].dtype))

    # ------------------------------------------------------------------
    # Requantization (with optional stochastic rounding)
    # ------------------------------------------------------------------
    @staticmethod
    def _requantize(
        tensor: torch.Tensor,
        quant_dtype: str,
        stochastic_rounding: bool,
    ) -> torch.Tensor:
        """Round-trip through quant_dtype with optional stochastic rounding.

        For fp8: per-tensor scaling (required by limited FP8 dynamic range).
        For bf16: simple cast round-trip (no scaling needed).
        Returns the dequantized value in the original dtype.
        """
        if quant_dtype == "fp8":
            fp8 = torch.float8_e4m3fn
            fp8_max = torch.finfo(fp8).max
            amax = tensor.abs().amax()
            scale = torch.where(amax > 0, amax / fp8_max, torch.ones_like(amax))
            scaled = tensor / scale

            if stochastic_rounding:
                # Exact ULP for FP8 E4M3: in binade [2^e, 2^(e+1)),
                # ULP = 2^(e - 3) where e = floor(log2(|x|)).
                # The old formula |x| * 2^-3 overestimates by up to 2x.
                abs_scaled = scaled.abs()
                min_normal = 2.0 ** -6   # smallest normal for E4M3 (bias=7)
                subnormal_ulp = 2.0 ** -9  # fixed step in subnormal region
                log2_abs = torch.floor(torch.log2(abs_scaled.clamp(min=min_normal)))
                ulp = torch.exp2(log2_abs - 3)
                ulp = torch.where(abs_scaled < min_normal, subnormal_ulp, ulp)
                ulp = torch.where(abs_scaled == 0, subnormal_ulp, ulp)
                noise = (torch.rand_like(scaled) - 0.5) * ulp
                scaled = (scaled + noise).clamp(-fp8_max, fp8_max)

            quantized = scaled.to(fp8)
            return quantized.to(tensor.dtype) * scale
        elif quant_dtype == "bf16":
            return tensor.to(torch.bfloat16).to(tensor.dtype)
        else:
            raise ValueError(f"Unsupported quant_dtype: {quant_dtype!r}. Use 'fp8' or 'bf16'.")

    # ------------------------------------------------------------------
    # Heuristic helpers
    # ------------------------------------------------------------------
    def _compute_heuristics(
        self,
        layer_metrics: dict[str, list[float]],
        e_prev: torch.Tensor,
        e_curr: torch.Tensor,
    ):
        e_prev_f = e_prev.flatten().float()
        e_curr_f = e_curr.flatten().float()
        norm_prev = e_prev_f.norm()
        norm_curr = e_curr_f.norm()
        if norm_prev < 1e-12:
            return

        layer_metrics["eco/heuristic/norm_ratio"].append(
            (norm_curr / norm_prev).item()
        )
        layer_metrics["eco/heuristic/cosine_sim"].append(
            (torch.dot(e_prev_f, e_curr_f) / (norm_prev * norm_curr + 1e-12)).item()
        )
        diff_norm = (e_prev_f - e_curr_f).norm()
        layer_metrics["eco/heuristic/rel_diff_norm"].append(
            (diff_norm / norm_prev).item()
        )

    def _aggregate_metrics(self, layer_metrics: dict[str, list[float]]):
        self._eco_metrics.clear()
        for name, values in layer_metrics.items():
            if values:
                self._eco_metrics[f"{name}/min"] = min(values)
                self._eco_metrics[f"{name}/mean"] = sum(values) / len(values)
                self._eco_metrics[f"{name}/max"] = max(values)
