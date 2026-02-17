# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ECOMuon: Muon optimizer with ECO error-compensating injection.

Implements ECO for the Muon optimizer, which uses Newton-Schulz iterations
to approximate the polar decomposition of momentum.

Muon update:
    m_{t+1} = β·m_t + g_t  (no (1-β) scaling)
    u_{t+1} = zeropower_via_newtonschulz5(m_{t+1})  (orthogonal polar factor)
    θ_{t+1} = θ_t - η·u_{t+1}

ECO approaches for Muon:
    1. "naive_sgdm": Apply SGDM formula, ignoring Newton-Schulz nonlinearity
       Δm = (1/η)(1 - 1/β) · e

    2. "frobenius": Scale by Frobenius norm (Adam analogy)
       Δm = (‖m‖_F/η)(1 - 1/β) · e

    3. "jacobian": Exact Jacobian-based injection using finite differences
       Solve: η·J·Δm ≈ e where J = ∂NS(m)/∂m

    4. "pre_ns": Inject error before Newton-Schulz transformation
       m ← m + (1/η)(1 - 1/β)·e, then u = NS(m)

For 1D parameters (biases, norms), falls back to regular AdamW.
"""

import math
from typing import Any, Callable, Optional

import torch
from torch.optim.optimizer import Optimizer

from torchtitan.components.quantized_tensor import QuantizedTensor


__all__ = ["ECOMuon"]


_DTYPE_TO_QUANT_STR = {
    torch.float8_e4m3fn: "fp8",
    torch.bfloat16: "bf16",
}


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Compute the polar factor (orthogonal component) of G using Newton-Schulz.

    This is the core of Muon optimizer. NS5 uses 5 iterations of the
    Newton-Schulz algorithm to approximate U where G = U·P (polar decomposition).

    Args:
        G: Input matrix (typically momentum)
        steps: Number of Newton-Schulz iterations (default: 5)

    Returns:
        Orthogonal matrix U ≈ polar(G)
    """
    assert G.ndim == 2, "zeropower_via_newtonschulz5 requires 2D input"

    # Normalize to prevent overflow
    norm_G = G.norm(p='fro')
    if norm_G < 1e-12:
        return torch.zeros_like(G)

    X = G / norm_G

    # Newton-Schulz iteration: X_{k+1} = X_k · (3I - X_k^T X_k) / 2
    # The paper version uses: X_{k+1} = X_k · (aI + bX_k^T X_k + c(X_k^T X_k)^2)
    # Using standard NS5 coefficients optimized for convergence
    for _ in range(steps):
        A = X.T @ X
        # NS iteration: X = X @ (3I - A) / 2
        # Equivalent form: X = 1.5*X - 0.5*X@A
        X = 1.5 * X - 0.5 * X @ A

    return X


class ECOMuon(Optimizer):
    """Muon optimizer with ECO (Error-Compensating Optimizer) support.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 3e-4)
        momentum: Momentum coefficient β (default: 0.95)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        adam_betas: Betas for Adam on 1D params (default: (0.9, 0.999))
        adam_eps: Epsilon for Adam on 1D params (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.0)
        optim_state_dtype: Dtype for optimizer states (default: torch.float32)
        optim_compute_dtype: Dtype for computation (default: torch.float32)
        eco_enabled: Whether to perform ECO injection (default: True)
        eco_approach: ECO injection approach (default: "frobenius")
            Options: "naive_sgdm", "frobenius", "jacobian", "pre_ns"
        stochastic_rounding: Whether to use stochastic rounding (default: False)
        heuristic_log_freq: How often to log diagnostics, 0=never (default: 0)
        quantize_weights: Simulate quantized weight storage (default: True)
        quant_dtype: Quantization dtype string (default: "bf16")
            Options: "fp8", "bf16"
        jacobian_fd_eps: Finite difference epsilon for Jacobian (default: 1e-5)
        master_weights_dtype: Dtype for master weights buffer. None means no
            master weights (pure ECO). torch.float32 or torch.bfloat16 creates
            a separate high-precision copy (traditional quantized training).
            (default: None)
    """

    def __init__(
        self,
        params,
        # Muon hparams (2D weight matrices)
        lr: float = 3e-4,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        # AdamW hparams (1D params: biases, norms, embeddings)
        adam_lr: float = 3e-3,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        adam_weight_decay: float = 0.0,
        # shared ECO / dtype params
        optim_state_dtype: torch.dtype = torch.float32,
        optim_compute_dtype: torch.dtype = torch.float32,
        eco_enabled: bool = True,
        eco_approach: str = "frobenius",
        stochastic_rounding: bool = False,
        heuristic_log_freq: int = 0,
        quantize_weights: bool = True,
        quant_dtype: str = "bf16",
        jacobian_fd_eps: float = 1e-5,
        master_weights_dtype: torch.dtype | None = None,
        include_weight_decay_in_injection: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if adam_lr < 0.0:
            raise ValueError(f"Invalid adam_lr: {adam_lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        if not 0.0 <= adam_betas[0] < 1.0:
            raise ValueError(f"Invalid adam beta1: {adam_betas[0]}")
        if not 0.0 <= adam_betas[1] < 1.0:
            raise ValueError(f"Invalid adam beta2: {adam_betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if eco_approach not in ["naive_sgdm", "frobenius", "jacobian", "pre_ns"]:
            raise ValueError(
                f"Invalid eco_approach: {eco_approach}. "
                f"Must be one of: naive_sgdm, frobenius, jacobian, pre_ns"
            )

        # Split params into two groups: Muon (dim>=2) and AdamW (dim<2)
        params_list = list(params)
        muon_params = [p for p in params_list if p.dim() >= 2]
        adam_params = [p for p in params_list if p.dim() < 2]

        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            weight_decay=weight_decay,
            optim_state_dtype=optim_state_dtype,
            optim_compute_dtype=optim_compute_dtype,
            _is_adam_group=False,
        )
        # Pass both groups as a list of dicts so PyTorch doesn't reject empty params.
        # PyTorch's Optimizer.__init__ treats a list of dicts as pre-formed param groups.
        super().__init__(
            [
                {"params": muon_params, "lr": lr, "weight_decay": weight_decay, "_is_adam_group": False},
                {"params": adam_params, "lr": adam_lr, "weight_decay": adam_weight_decay, "_is_adam_group": True},
            ],
            defaults,
        )

        self._eco_enabled = eco_enabled
        self._eco_approach = eco_approach
        self._stochastic_rounding = stochastic_rounding
        self._heuristic_log_freq = heuristic_log_freq
        self._quantize_weights = quantize_weights
        self._quant_dtype = quant_dtype
        self._jacobian_fd_eps = jacobian_fd_eps
        self._master_weights_dtype = master_weights_dtype
        self._include_weight_decay_in_injection = include_weight_decay_in_injection
        self._eco_step_count = 0
        self._eco_metrics: dict[str, float] = {}

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("optim_state_dtype", torch.float32)
            group.setdefault("optim_compute_dtype", torch.float32)
            group.setdefault("_is_adam_group", False)

    # ------------------------------------------------------------------
    # Public API
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

        layer_metrics: dict[str, list[float]] = {
            "eco/heuristic/norm_ratio": [],
            "eco/heuristic/cosine_sim": [],
            "eco/heuristic/rel_diff_norm": [],
            "eco/muon/frobenius_norm": [],
            "eco/muon/update_norm": [],
            "eco/muon/injection_scale": [],
        }

        for group in self.param_groups:
            is_adam = group["_is_adam_group"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            optim_state_dtype = group["optim_state_dtype"]
            optim_compute_dtype = group["optim_compute_dtype"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            adam_betas = group["adam_betas"]
            adam_eps = group["adam_eps"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("ECOMuon does not support sparse gradients")

                # Lazy state init
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    if is_adam:
                        state["exp_avg"] = torch.zeros_like(param, dtype=optim_state_dtype)
                        state["exp_avg_sq"] = torch.zeros_like(param, dtype=optim_state_dtype)
                    else:
                        state["momentum_buffer"] = torch.zeros_like(param, dtype=optim_state_dtype)
                    # Initialize master weights if requested
                    if self._master_weights_dtype is not None:
                        state["master_weights"] = param.detach().clone().to(self._master_weights_dtype)

                state["step"] += 1
                step = state["step"].item()

                # Dispatch: Adam group uses AdamW, Muon group uses Muon
                if is_adam:
                    self._step_adam(
                        param, grad, state["exp_avg"], state["exp_avg_sq"],
                        lr, adam_betas, adam_eps, weight_decay, step,
                    )
                else:
                    if self._quantize_weights:
                        self._step_muon_simulated_quant(
                            param, grad, state["momentum_buffer"],
                            lr, momentum, ns_steps, weight_decay, step,
                            optim_compute_dtype, should_log, layer_metrics,
                        )
                    elif isinstance(param, QuantizedTensor):
                        self._step_muon_quantized(
                            param, grad, state["momentum_buffer"],
                            lr, momentum, ns_steps, weight_decay, step,
                            optim_compute_dtype, should_log, layer_metrics,
                        )
                    else:
                        self._step_muon_regular(
                            param, grad, state["momentum_buffer"],
                            lr, momentum, ns_steps, weight_decay,
                        )

        # Aggregate metrics
        if should_log:
            self._aggregate_metrics(layer_metrics)

        return loss

    # ------------------------------------------------------------------
    # Standard Muon (non-quantized 2D params)
    # ------------------------------------------------------------------
    def _step_muon_regular(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        momentum_buffer: torch.Tensor,
        lr: float,
        momentum: float,
        ns_steps: int,
        weight_decay: float,
    ) -> None:
        # Muon momentum: m = β·m + g (no (1-β) scaling)
        momentum_buffer.mul_(momentum).add_(grad)

        # Newton-Schulz to get orthogonal update
        update = zeropower_via_newtonschulz5(momentum_buffer, steps=ns_steps)

        # Weight decay (applied to params directly)
        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        # Apply update: θ ← θ - η·u
        param.add_(update, alpha=-lr)

    # ------------------------------------------------------------------
    # Simulated quantization for 2D Muon params
    # ------------------------------------------------------------------
    def _step_muon_simulated_quant(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        momentum_buffer: torch.Tensor,
        lr: float,
        momentum: float,
        ns_steps: int,
        weight_decay: float,
        step: int,
        optim_compute_dtype: torch.dtype,
        should_log: bool,
        layer_metrics: dict[str, list[float]],
    ):
        # Check if we're using master weights
        state = self.state[param]
        has_master_weights = "master_weights" in state

        # Work in high precision
        if has_master_weights:
            # Start from master weights (high precision)
            param_f = state["master_weights"].to(optim_compute_dtype) if state["master_weights"].dtype != optim_compute_dtype else state["master_weights"].clone()
        else:
            # Start from quantized param (pure ECO mode)
            param_f = param.data.to(optim_compute_dtype) if param.dtype != optim_compute_dtype else param.data.clone()

        grad_compute = grad if grad.dtype == optim_compute_dtype else grad.to(optim_compute_dtype)

        if momentum_buffer.dtype == optim_compute_dtype:
            m_c = momentum_buffer
            copy_back = False
        else:
            m_c = momentum_buffer.to(optim_compute_dtype)
            copy_back = True

        # 1. Update momentum (Muon style: no (1-β) scaling)
        m_c.mul_(momentum).add_(grad_compute)

        # 2. Apply update based on ECO approach
        if self._eco_approach == "pre_ns":
            # For pre_ns, we inject error BEFORE Newton-Schulz
            # We'll handle this in a special path below
            pass  # Handle in dedicated section

        # 3. Compute Newton-Schulz update
        m_tilde = m_c.clone()  # Save for error injection
        update = zeropower_via_newtonschulz5(m_c, steps=ns_steps)

        # 4. Apply weight decay and update
        if weight_decay != 0:
            param_f.mul_(1 - lr * weight_decay)

        param_f.add_(update, alpha=-lr)

        # θ̃ is now in param_f — ideal weight before requant

        # 5. Quant→dequant round-trip
        theta_hat = self._requantize(
            param_f, self._quant_dtype, self._stochastic_rounding
        )

        # 6. ECO injection
        if self._eco_enabled:
            error = param_f - theta_hat  # e = θ̃ - θ̂

            if self._eco_approach == "naive_sgdm":
                # Approach 1: Ignore Newton-Schulz nonlinearity
                injection_coeff = (1.0 / lr) * (1.0 - 1.0 / momentum)
                if self._include_weight_decay_in_injection and weight_decay != 0:
                    injection_coeff *= (1.0 - lr * weight_decay)
                delta_m = injection_coeff * error

            elif self._eco_approach == "frobenius":
                # Approach 2: Scale by Frobenius norm
                m_norm = m_tilde.norm(p='fro')
                injection_coeff = (m_norm / lr) * (1.0 - 1.0 / momentum)
                if self._include_weight_decay_in_injection and weight_decay != 0:
                    injection_coeff *= (1.0 - lr * weight_decay)
                delta_m = injection_coeff * error

                if should_log:
                    layer_metrics["eco/muon/frobenius_norm"].append(m_norm.item())
                    layer_metrics["eco/muon/injection_scale"].append(injection_coeff.item())

            elif self._eco_approach == "jacobian":
                # Approach 3: Jacobian-based (finite differences)
                delta_m = self._compute_jacobian_injection(
                    m_tilde, error, lr, momentum, ns_steps
                )

            elif self._eco_approach == "pre_ns":
                # Approach 4: Inject before Newton-Schulz
                # We need to re-do the update with injected momentum
                injection_coeff = (1.0 / lr) * (1.0 - 1.0 / momentum)
                if self._include_weight_decay_in_injection and weight_decay != 0:
                    injection_coeff *= (1.0 - lr * weight_decay)
                delta_m = injection_coeff * error

                # Inject into momentum BEFORE NS
                m_c.add_(delta_m)

                # Recompute update with corrected momentum
                update_corrected = zeropower_via_newtonschulz5(m_c, steps=ns_steps)

                # Recompute param_f
                param_f = param.data.to(optim_compute_dtype) if param.dtype != optim_compute_dtype else param.data.clone()
                if weight_decay != 0:
                    param_f.mul_(1 - lr * weight_decay)
                param_f.add_(update_corrected, alpha=-lr)

                # Requantize again
                theta_hat = self._requantize(
                    param_f, self._quant_dtype, self._stochastic_rounding
                )

                # No additional injection needed (already done)
                delta_m = None

            else:
                raise ValueError(f"Unknown eco_approach: {self._eco_approach}")

            # Apply injection (except for pre_ns which already did it)
            if delta_m is not None:
                if m_c.dtype != delta_m.dtype:
                    delta_m = delta_m.to(m_c.dtype)
                m_c.add_(delta_m)

            # Heuristic diagnostics
            if should_log:
                prev_error = self.state[param].get("prev_error")
                if prev_error is not None:
                    self._compute_heuristics(layer_metrics, prev_error, error)
                self.state[param]["prev_error"] = error.detach().clone()

                layer_metrics["eco/muon/update_norm"].append(update.norm().item())

        # Write momentum back if needed
        if copy_back:
            momentum_buffer.copy_(m_c.to(momentum_buffer.dtype))

        # 7. Store results
        if has_master_weights:
            # Store θ̃ (pre-quantization) to master weights
            state["master_weights"].copy_(param_f.to(state["master_weights"].dtype))
            # Store θ̂ (quantized) to param
            param.data.copy_(theta_hat.to(param.dtype))
        else:
            # Pure ECO mode: only store θ̂ to param
            param.data.copy_(theta_hat.to(param.dtype))

    # ------------------------------------------------------------------
    # Quantized 2D Muon params
    # ------------------------------------------------------------------
    def _step_muon_quantized(
        self,
        param: QuantizedTensor,
        grad: torch.Tensor,
        momentum_buffer: torch.Tensor,
        lr: float,
        momentum: float,
        ns_steps: int,
        weight_decay: float,
        step: int,
        optim_compute_dtype: torch.dtype,
        should_log: bool,
        layer_metrics: dict[str, list[float]],
    ):
        # Check if we're using master weights
        state = self.state[param]
        has_master_weights = "master_weights" in state

        # Dequantize and upcast
        if has_master_weights:
            param_dequant = state["master_weights"].to(optim_compute_dtype) if state["master_weights"].dtype != optim_compute_dtype else state["master_weights"].clone()
        else:
            param_dequant = param.dequantize()
            if param_dequant.dtype != optim_compute_dtype:
                param_dequant = param_dequant.to(optim_compute_dtype)

        grad_compute = grad if grad.dtype == optim_compute_dtype else grad.to(optim_compute_dtype)

        if momentum_buffer.dtype == optim_compute_dtype:
            m_c = momentum_buffer
            copy_back = False
        else:
            m_c = momentum_buffer.to(optim_compute_dtype)
            copy_back = True

        # Update momentum
        m_c.mul_(momentum).add_(grad_compute)

        # Compute NS update
        m_tilde = m_c.clone()
        update = zeropower_via_newtonschulz5(m_c, steps=ns_steps)

        # Apply update
        if weight_decay != 0:
            param_dequant.mul_(1 - lr * weight_decay)

        param_dequant.add_(update, alpha=-lr)

        # Requantize
        quant_str = _DTYPE_TO_QUANT_STR[param._quant_dtype]
        theta_hat = self._requantize(
            param_dequant, quant_str, self._stochastic_rounding
        )

        # ECO injection (same logic as simulated_quant)
        if self._eco_enabled:
            error = param_dequant - theta_hat

            if self._eco_approach == "naive_sgdm":
                injection_coeff = (1.0 / lr) * (1.0 - 1.0 / momentum)
                delta_m = injection_coeff * error

            elif self._eco_approach == "frobenius":
                m_norm = m_tilde.norm(p='fro')
                injection_coeff = (m_norm / lr) * (1.0 - 1.0 / momentum)
                delta_m = injection_coeff * error

                if should_log:
                    layer_metrics["eco/muon/frobenius_norm"].append(m_norm.item())
                    layer_metrics["eco/muon/injection_scale"].append(injection_coeff.item())

            elif self._eco_approach == "jacobian":
                delta_m = self._compute_jacobian_injection(
                    m_tilde, error, lr, momentum, ns_steps
                )

            elif self._eco_approach == "pre_ns":
                injection_coeff = (1.0 / lr) * (1.0 - 1.0 / momentum)
                delta_m = injection_coeff * error

                m_c.add_(delta_m)
                update_corrected = zeropower_via_newtonschulz5(m_c, steps=ns_steps)

                param_dequant = param.dequantize().to(optim_compute_dtype)
                if weight_decay != 0:
                    param_dequant.mul_(1 - lr * weight_decay)
                param_dequant.add_(update_corrected, alpha=-lr)

                theta_hat = self._requantize(
                    param_dequant, quant_str, self._stochastic_rounding
                )
                delta_m = None

            else:
                raise ValueError(f"Unknown eco_approach: {self._eco_approach}")

            if delta_m is not None:
                if m_c.dtype != delta_m.dtype:
                    delta_m = delta_m.to(m_c.dtype)
                m_c.add_(delta_m)

            if should_log:
                prev_error = self.state[param].get("prev_error")
                if prev_error is not None:
                    self._compute_heuristics(layer_metrics, prev_error, error)
                self.state[param]["prev_error"] = error.detach().clone()

                layer_metrics["eco/muon/update_norm"].append(update.norm().item())

        if copy_back:
            momentum_buffer.copy_(m_c.to(momentum_buffer.dtype))

        # Store results
        if has_master_weights:
            # Store θ̃ (pre-quantization) to master weights
            state["master_weights"].copy_(param_dequant.to(state["master_weights"].dtype))

        # Store θ̂ (quantized) to param
        new_q = QuantizedTensor.quantize(
            theta_hat, quant_dtype=param._quant_dtype, axiswise_dim=param._axiswise_dim,
        )
        param._data.copy_(new_q._data)
        param._scale.copy_(new_q._scale)

    # ------------------------------------------------------------------
    # Jacobian-based injection (Approach 3)
    # ------------------------------------------------------------------
    def _compute_jacobian_injection(
        self,
        m: torch.Tensor,
        error: torch.Tensor,
        lr: float,
        momentum: float,
        ns_steps: int,
        weight_decay: float = 0.0,
    ) -> torch.Tensor:
        """Compute Δm such that η·J·Δm ≈ e where J = ∂NS(m)/∂m.

        We use finite differences to approximate J·v for arbitrary v,
        then solve the linear system using conjugate gradient.
        """
        # Target: η·J·Δm = e
        # So we want: J·Δm = e/η
        target = error / lr
        wd_factor = 1.0
        if self._include_weight_decay_in_injection and weight_decay != 0:
            wd_factor = 1.0 - lr * weight_decay
        target *= wd_factor

        # Define Jacobian-vector product using finite differences
        def jvp(v):
            """Jacobian-vector product J·v ≈ [NS(m+εv) - NS(m-εv)] / (2ε)"""
            eps = self._jacobian_fd_eps
            m_plus = m + eps * v
            m_minus = m - eps * v
            ns_plus = zeropower_via_newtonschulz5(m_plus, steps=ns_steps)
            ns_minus = zeropower_via_newtonschulz5(m_minus, steps=ns_steps)
            return (ns_plus - ns_minus) / (2 * eps)

        # Solve J·Δm = target using conjugate gradient
        # Start with naive SGDM injection as initial guess
        delta_m = (1.0 / lr) * (1.0 - 1.0 / momentum) * error
        delta_m *= wd_factor

        # CG iterations
        r = target - jvp(delta_m)  # residual
        p = r.clone()  # search direction
        rs_old = (r * r).sum()

        for _ in range(min(10, m.numel())):  # Max 10 CG steps
            Ap = jvp(p)
            alpha = rs_old / ((p * Ap).sum() + 1e-12)
            delta_m = delta_m + alpha * p
            r = r - alpha * Ap
            rs_new = (r * r).sum()

            if rs_new < 1e-10:  # Converged
                break

            beta = rs_new / (rs_old + 1e-12)
            p = r + beta * p
            rs_old = rs_new

        return delta_m

    # ------------------------------------------------------------------
    # Adam for 1D params (biases, norms, embeddings)
    # ------------------------------------------------------------------
    def _step_adam(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        step: int,
    ) -> None:
        """Standard Adam for 1D parameters."""
        beta1, beta2 = betas

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
    # Requantization (shared with ECOAdamW)
    # ------------------------------------------------------------------
    @staticmethod
    def _requantize(
        tensor: torch.Tensor,
        quant_dtype: str,
        stochastic_rounding: bool,
    ) -> torch.Tensor:
        """Round-trip through quant_dtype with optional stochastic rounding."""
        if quant_dtype == "fp8":
            fp8 = torch.float8_e4m3fn
            fp8_max = torch.finfo(fp8).max
            amax = tensor.abs().amax()
            scale = torch.where(amax > 0, amax / fp8_max, torch.ones_like(amax))
            scaled = tensor / scale

            if stochastic_rounding:
                ulp = scaled.abs() * (2 ** -3)  # 3 mantissa bits for E4M3
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
        """Compute consecutive error similarity metrics."""
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
        """Aggregate per-layer metrics into min/mean/max statistics."""
        self._eco_metrics.clear()
        for name, values in layer_metrics.items():
            if values:
                self._eco_metrics[f"{name}/min"] = min(values)
                self._eco_metrics[f"{name}/mean"] = sum(values) / len(values)
                self._eco_metrics[f"{name}/max"] = max(values)
