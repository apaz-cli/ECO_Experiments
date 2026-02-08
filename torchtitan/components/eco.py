# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ECO (Error-Compensating Optimizer) for quantized training without master weights.

Implements Algorithm 3 from:
    "ECO: Quantized Training without Full-Precision Master Weights"
    Nikdan et al., 2025

After each optimizer step, ECO:
  1. Round-trips each quantized-layer weight through FP8 (quantize then dequantize)
  2. Computes the quantization error  e = theta_tilde - theta_hat
  3. Injects a correction into Adam's first-moment buffer (exp_avg)

The injection for Adam (Algorithm 3, line 216 of the paper):
    m_hat = m_tilde + [(1 - beta1^t) / lr] * (1 - 1/beta1)
            * (sqrt(v / (1 - beta2^t)) + eps) * e

Usage:
    Add "eco" to model.converters list (AFTER Float8):

    [model]
    converters = ["quantize.linear.float8", "eco"]

    [eco]
    enabled = true
"""

import math
from typing import Any, Union

import torch
import torch.nn as nn

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import register_model_converter
from torchtitan.tools.logging import logger


class ECOConverter:
    """ECO model converter: quantize weights and inject error into momentum."""

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        eco_config = job_config.eco
        self.enabled = eco_config.enabled
        if not self.enabled:
            return

        self.heuristic_log_freq = eco_config.heuristic_log_freq
        self.stochastic_rounding = eco_config.stochastic_rounding

        # Optimizer hyperparams needed for the injection formula
        self.beta1 = job_config.optimizer.beta1
        self.beta2 = job_config.optimizer.beta2
        self.eps = job_config.optimizer.eps

        # Step counter
        self.step_count = 0

        # Set of parameter ids that ECO applies to (populated in convert())
        self._eco_param_ids: set[int] = set()

        # Previous quantization errors for heuristic validation
        # Maps param id -> error tensor (held for exactly 1 step)
        self._prev_errors: dict[int, torch.Tensor] = {}

        # Accumulated heuristic metrics for the current logging window
        self._pending_metrics: dict[str, float] = {}

    def convert(self, model: nn.Module):
        """Identify ECO-eligible parameters (those in Float8Linear modules).

        Must be called AFTER Float8LinearConverter.convert() so that
        Float8Linear modules already exist in the model.
        """
        if not self.enabled:
            return

        try:
            from torchao.float8.float8_linear import Float8Linear
        except ImportError:
            try:
                from torchao.float8 import Float8Linear
            except ImportError:
                raise ImportError(
                    "ECO requires torchao with Float8 support. "
                    "Install with: pip install torchao"
                )

        for _name, module in model.named_modules():
            if isinstance(module, Float8Linear):
                self._eco_param_ids.add(id(module.weight))

        n_params = len(self._eco_param_ids)
        logger.info(f"ECO: enabled for {n_params} Float8Linear weight parameters")
        if n_params == 0:
            logger.warning(
                "ECO: no Float8Linear modules found. "
                "Ensure 'quantize.linear.float8' is listed before 'eco' in model.converters."
            )

    def _quantize_fp8_roundtrip(self, tensor: torch.Tensor) -> torch.Tensor:
        """Round-trip a tensor through FP8 E4M3: quantize then dequantize.

        Returns the dequantized tensor (same dtype as input) that represents
        the value as if it were stored in FP8.
        """
        fp8_dtype = torch.float8_e4m3fn
        fp8_max = torch.finfo(fp8_dtype).max  # 448.0

        # Compute tensorwise scale
        amax = tensor.abs().amax()
        # Avoid division by zero for all-zero tensors
        scale = torch.where(amax > 0, amax / fp8_max, torch.ones_like(amax))

        scaled = tensor / scale

        if self.stochastic_rounding:
            # Stochastic rounding via noise injection before round-to-nearest.
            # For FP8 E4M3 with 3 mantissa bits, ULP ≈ |x| * 2^(-3).
            # Adding U(-0.5*ulp, 0.5*ulp) before RtN gives stochastic rounding.
            ulp = scaled.abs() * (2**-3)
            noise = (torch.rand_like(scaled) - 0.5) * ulp
            scaled = scaled + noise
            # Clamp to FP8 representable range so the cast doesn't produce NaN
            scaled = scaled.clamp(-fp8_max, fp8_max)

        quantized = scaled.to(fp8_dtype)
        dequantized = quantized.to(tensor.dtype) * scale
        return dequantized

    def post_optimizer_hook(
        self,
        model: Union[nn.Module, list[nn.Module]],
        optimizers: OptimizersContainer | None = None,
    ):
        """Apply ECO after each optimizer step.

        For each eligible parameter:
        1. Round-trip through FP8 to get quantized weights
        2. Compute quantization error
        3. Inject error into Adam's momentum buffer
        4. Replace parameter data with quantized version (simulation mode)
        """
        if not self.enabled or optimizers is None:
            return

        self.step_count += 1

        # Determine if we should stash/compare errors for heuristic validation
        should_stash = (
            self.heuristic_log_freq > 0
            and self.step_count % self.heuristic_log_freq == 0
        )
        should_compare = len(self._prev_errors) > 0

        models = [model] if isinstance(model, nn.Module) else model

        # Collect per-layer heuristic metrics for aggregation
        layer_metrics: dict[str, list[float]] = {
            "eco/heuristic/norm_ratio": [],
            "eco/heuristic/cosine_sim": [],
            "eco/heuristic/rel_diff_norm": [],
            "eco/heuristic/injection_rel_error": [],
        }

        for model_part, optimizer in zip(models, optimizers.optimizers):
            lr = optimizer.param_groups[0]["lr"]

            for name, param in model_part.named_parameters():
                if id(param) not in self._eco_param_ids:
                    continue
                if not param.requires_grad:
                    continue

                state = optimizer.state.get(param)
                if state is None or "exp_avg" not in state:
                    continue

                # --- Step 1: FP8 round-trip ---
                theta_tilde = param.data  # after optimizer step, before quantization
                theta_hat = self._quantize_fp8_roundtrip(theta_tilde)

                # --- Step 2: Quantization error ---
                error = theta_tilde - theta_hat

                # --- Step 3: Inject into momentum (Algorithm 3) ---
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step_t = state["step"]
                # Handle step as tensor (fused Adam) or scalar
                if isinstance(step_t, torch.Tensor):
                    step_val = step_t.item()
                else:
                    step_val = step_t

                bias_correction1 = 1.0 - self.beta1**step_val
                bias_correction2 = 1.0 - self.beta2**step_val

                # injection_coeff = (1 - beta1^t) / lr * (1 - 1/beta1)
                injection_coeff = (bias_correction1 / lr) * (1.0 - 1.0 / self.beta1)

                # adaptive_scale = sqrt(v / (1 - beta2^t)) + eps
                adaptive_scale = (
                    torch.sqrt(exp_avg_sq / bias_correction2) + self.eps
                )

                # m += coeff * adaptive_scale * error
                exp_avg.add_(injection_coeff * adaptive_scale * error)

                # --- Step 4: Heuristic validation ---
                pid = id(param)

                if should_compare and pid in self._prev_errors:
                    e_prev = self._prev_errors.pop(pid)
                    self._compute_heuristic_metrics(
                        layer_metrics, e_prev, error, exp_avg, lr
                    )

                if should_stash:
                    self._prev_errors[pid] = error.detach().clone()

                # --- Step 5: Replace param with quantized version ---
                param.data.copy_(theta_hat)

        # Aggregate and store heuristic metrics
        if should_compare:
            self._aggregate_heuristic_metrics(layer_metrics)
            # Clean up any remaining stashed errors (shouldn't happen but be safe)
            if not should_stash:
                self._prev_errors.clear()

    def _compute_heuristic_metrics(
        self,
        layer_metrics: dict[str, list[float]],
        e_prev: torch.Tensor,
        e_curr: torch.Tensor,
        momentum: torch.Tensor,
        lr: float,
    ):
        """Compute heuristic validation metrics for one layer.

        Metrics (from Section 1.4 of the experiment plan):
        - Relative norm ratio: ||e_{t+1}|| / ||e_t||
        - Cosine similarity: cos(e_t, e_{t+1})
        - Relative diff norm: ||e_t - e_{t+1}|| / ||e_t||
        - Injection-relative error: (1/lr) * ||e_t - e_{t+1}|| / ||m_t||
        """
        with torch.no_grad():
            e_prev_flat = e_prev.flatten().float()
            e_curr_flat = e_curr.flatten().float()

            norm_prev = e_prev_flat.norm()
            norm_curr = e_curr_flat.norm()

            # Skip degenerate cases (e.g. all-zero errors)
            if norm_prev < 1e-12:
                return

            # Relative norm ratio
            norm_ratio = (norm_curr / norm_prev).item()
            layer_metrics["eco/heuristic/norm_ratio"].append(norm_ratio)

            # Cosine similarity
            cos_sim = (
                torch.dot(e_prev_flat, e_curr_flat) / (norm_prev * norm_curr + 1e-12)
            ).item()
            layer_metrics["eco/heuristic/cosine_sim"].append(cos_sim)

            # Relative diff norm
            diff_norm = (e_prev_flat - e_curr_flat).norm()
            rel_diff = (diff_norm / norm_prev).item()
            layer_metrics["eco/heuristic/rel_diff_norm"].append(rel_diff)

            # Injection-relative error
            m_norm = momentum.flatten().float().norm()
            if m_norm > 1e-12:
                inj_rel_error = ((1.0 / lr) * diff_norm / m_norm).item()
                layer_metrics["eco/heuristic/injection_rel_error"].append(
                    inj_rel_error
                )

    def _aggregate_heuristic_metrics(
        self, layer_metrics: dict[str, list[float]]
    ):
        """Aggregate per-layer metrics to model-wide min/mean/max."""
        self._pending_metrics.clear()

        for metric_name, values in layer_metrics.items():
            if not values:
                continue
            self._pending_metrics[f"{metric_name}/min"] = min(values)
            self._pending_metrics[f"{metric_name}/mean"] = sum(values) / len(values)
            self._pending_metrics[f"{metric_name}/max"] = max(values)

    def get_extra_metrics(self) -> dict[str, Any]:
        """Return and clear any pending ECO metrics for logging."""
        metrics = dict(self._pending_metrics)
        self._pending_metrics.clear()
        return metrics


register_model_converter(ECOConverter, "eco")
