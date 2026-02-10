# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ECOAdamW optimizer — dtype handling, convergence, error conditions."""

import pytest
import torch
import torch.nn as nn

from torchtitan.components.eco_optimizer import ECOAdamW
from torchtitan.components.quantized_tensor import QuantizedTensor


class TestECOAdamW:
    """Core optimizer behavior tests."""

    def test_optimizer_state_dtype_decoupling(self):
        param = torch.randn(64, 64, dtype=torch.float32, requires_grad=True)
        optimizer = ECOAdamW([param], lr=1e-3, optim_state_dtype=torch.bfloat16)
        param.grad = torch.randn_like(param)
        optimizer.step()
        state = optimizer.state[param]
        assert state["exp_avg"].dtype == torch.bfloat16
        assert state["exp_avg_sq"].dtype == torch.bfloat16

    def test_multiple_steps(self):
        param = torch.randn(64, 64, requires_grad=True)
        original = param.data.clone()
        optimizer = ECOAdamW([param], lr=1e-3, optim_state_dtype=torch.bfloat16)
        for _ in range(5):
            param.grad = torch.randn_like(param) * 0.01
            optimizer.step()
            optimizer.zero_grad()
        assert not torch.allclose(param.data, original)

    def test_weight_decay(self):
        param = torch.ones(64, 64, requires_grad=True)
        optimizer = ECOAdamW(
            [param], lr=1e-3, weight_decay=0.1,
            optim_state_dtype=torch.float32, optim_compute_dtype=torch.float32,
            quantize_weights=False,
        )
        param.grad = torch.zeros_like(param)
        optimizer.step()
        assert torch.all(param.data < 1.0)

    def test_different_state_and_compute_dtypes(self):
        param = torch.randn(64, 64, requires_grad=True)
        optimizer = ECOAdamW(
            [param], lr=1e-3,
            optim_state_dtype=torch.float32, optim_compute_dtype=torch.bfloat16,
        )
        param.grad = torch.randn_like(param)
        optimizer.step()
        state = optimizer.state[param]
        assert state["exp_avg"].dtype == torch.float32

    def test_state_dict_save_load(self):
        param = torch.randn(64, 64, requires_grad=True)
        optimizer = ECOAdamW([param], lr=1e-3, optim_state_dtype=torch.bfloat16)
        param.grad = torch.randn_like(param)
        optimizer.step()
        state_dict = optimizer.state_dict()

        new_param = torch.randn(64, 64, requires_grad=True)
        new_opt = ECOAdamW([new_param], lr=1e-3, optim_state_dtype=torch.bfloat16)
        new_opt.load_state_dict(state_dict)
        assert len(new_opt.state) > 0

    def test_eco_enabled_and_disabled_flags(self):
        """eco_enabled controls whether injection runs for QuantizedTensor."""
        fp = torch.randn(32, 32)
        qt = QuantizedTensor.quantize(fp, torch.float8_e4m3fn)
        qt.requires_grad_(True)

        opt_on = ECOAdamW([qt], lr=1e-3, eco_enabled=True, heuristic_log_freq=1)
        qt.grad = torch.randn_like(qt.dequantize()) * 0.01
        opt_on.step()
        assert "prev_error" in opt_on.state[qt]

    def test_get_eco_metrics_api(self):
        """get_eco_metrics returns dict and clears."""
        fp = torch.randn(16, 16)
        qt = QuantizedTensor.quantize(fp, torch.float8_e4m3fn)
        qt.requires_grad_(True)

        opt = ECOAdamW([qt], lr=1e-3, eco_enabled=True, heuristic_log_freq=1)
        qt.grad = torch.randn_like(qt.dequantize()) * 0.01
        opt.step()  # step 1 — stores prev_error
        qt.grad = torch.randn_like(qt.dequantize()) * 0.01
        opt.step()  # step 2 — compares and emits metrics

        m = opt.get_eco_metrics()
        assert isinstance(m, dict)
        assert len(m) > 0
        assert opt.get_eco_metrics() == {}


class TestECOAdamWConvergence:
    def test_simple_optimization_converges(self):
        torch.manual_seed(42)
        X = torch.randn(100, 10)
        y_true = torch.randn(100, 1)
        model = nn.Linear(10, 1)
        optimizer = ECOAdamW(model.parameters(), lr=0.01, optim_state_dtype=torch.bfloat16)
        losses = []
        for _ in range(100):
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(X), y_true)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]


class TestECOAdamWErrorConditions:
    def test_sparse_gradient_error(self):
        param = torch.randn(64, 64, requires_grad=True)
        optimizer = ECOAdamW([param])
        param.grad = torch.randn_like(param).to_sparse()
        with pytest.raises(RuntimeError, match="sparse"):
            optimizer.step()

    def test_nan_gradient_no_crash(self):
        param = torch.ones(32, 32, requires_grad=True)
        optimizer = ECOAdamW([param], lr=1e-3, optim_state_dtype=torch.float32, optim_compute_dtype=torch.float32)
        param.grad = torch.full_like(param, float("nan"))
        optimizer.step()
        assert param.dtype == torch.float32

    def test_zero_learning_rate(self):
        param = torch.randn(32, 32, requires_grad=True)
        original = param.data.clone()
        optimizer = ECOAdamW([param], lr=0.0, quantize_weights=False)
        param.grad = torch.randn_like(param)
        optimizer.step()
        torch.testing.assert_close(param.data, original, atol=1e-6, rtol=0)

    def test_invalid_hyperparameters(self):
        p = torch.randn(16, 16, requires_grad=True)
        with pytest.raises(ValueError):
            ECOAdamW([p], lr=-1.0)
        with pytest.raises(ValueError):
            ECOAdamW([p], betas=(-0.1, 0.999))
        with pytest.raises(ValueError):
            ECOAdamW([p], betas=(0.9, 1.1))
        with pytest.raises(ValueError):
            ECOAdamW([p], eps=-1e-8)
        with pytest.raises(ValueError):
            ECOAdamW([p], weight_decay=-0.01)

    def test_mixed_parameter_types(self):
        """Optimizer handles both regular and QuantizedTensor params."""
        param1 = torch.randn(16, 16, requires_grad=True)
        fp2 = torch.randn(16, 16)
        param2 = QuantizedTensor.quantize(fp2, torch.float8_e4m3fn)
        param2.requires_grad_(True)
        param2.grad = torch.randn_like(param2.dequantize())

        optimizer = ECOAdamW([param1, param2], lr=1e-3, optim_state_dtype=torch.bfloat16)
        param1.grad = torch.randn_like(param1)
        optimizer.step()

        assert param1 in optimizer.state
        assert param2 in optimizer.state
        assert optimizer.state[param1]["exp_avg"].dtype == torch.bfloat16
        assert optimizer.state[param2]["exp_avg"].dtype == torch.bfloat16

    def test_empty_parameter_list(self):
        with pytest.raises(ValueError, match="optimizer got an empty parameter list"):
            ECOAdamW([])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
