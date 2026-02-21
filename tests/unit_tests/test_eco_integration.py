# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for ECOAdamW optimizer with build_optimizers."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.eco_adamw import ECOAdamW
from torchtitan.components.quantized_linear import QuantizedLinear
from torchtitan.components.quantized_tensor import QuantizedTensor
from torchtitan.config import ConfigManager
from torchtitan.distributed import ParallelDims


def _make_config(*extra_args):
    config_manager = ConfigManager()
    return config_manager.parse_args(list(extra_args))


def _build_parallel_dims(job_config, world_size=1):
    p = job_config.parallelism
    return ParallelDims(
        dp_shard=p.data_parallel_shard_degree,
        dp_replicate=p.data_parallel_replicate_degree,
        cp=p.context_parallel_degree,
        tp=p.tensor_parallel_degree,
        pp=p.pipeline_parallel_degree,
        ep=p.expert_parallel_degree,
        etp=p.expert_tensor_parallel_degree,
        world_size=world_size,
    )


class TestBuildOptimizers:
    def test_eco_adamw_built_with_config(self):
        config = _make_config(
            "--optimizer.name", "ECOAdamW",
            "--eco.enabled",
            "--eco.optim_state_dtype", "bf16",
            "--eco.optim_compute_dtype", "bf16",
        )
        parallel_dims = _build_parallel_dims(config)
        model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 10))

        optimizers = build_optimizers(
            [model], config.optimizer, parallel_dims, eco_config=config.eco,
        )

        assert len(optimizers.optimizers) == 1
        opt = optimizers.optimizers[0]
        assert isinstance(opt, ECOAdamW)
        assert opt.defaults["optim_state_dtype"] == torch.bfloat16
        assert opt._eco_enabled is True

    def test_eco_disabled_still_creates_ecoadamw(self):
        config = _make_config("--optimizer.name", "ECOAdamW")
        parallel_dims = _build_parallel_dims(config)
        model = nn.Linear(64, 10)

        eco_config = config.eco if config.eco.enabled else None
        optimizers = build_optimizers(
            [model], config.optimizer, parallel_dims, eco_config=eco_config,
        )
        assert isinstance(optimizers.optimizers[0], ECOAdamW)


class TestDtypeOptions:
    @pytest.mark.parametrize("optim_state_dtype", ["fp32", "bf16", "fp16"])
    @pytest.mark.parametrize("optim_compute_dtype", ["fp32", "bf16", "fp16"])
    def test_various_dtype_combinations(self, optim_state_dtype, optim_compute_dtype):
        config = _make_config(
            "--optimizer.name", "ECOAdamW",
            "--eco.enabled",
            "--eco.optim_state_dtype", optim_state_dtype,
            "--eco.optim_compute_dtype", optim_compute_dtype,
        )
        parallel_dims = _build_parallel_dims(config)
        model = nn.Linear(64, 10)

        optimizers = build_optimizers(
            [model], config.optimizer, parallel_dims, eco_config=config.eco,
        )
        dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
        opt = optimizers.optimizers[0]
        assert opt.defaults["optim_state_dtype"] == dtype_map[optim_state_dtype]
        assert opt.defaults["optim_compute_dtype"] == dtype_map[optim_compute_dtype]


class TestFullTrainingCycle:
    def test_eco_adamw_training_converges(self):
        """ECOAdamW converges on a simple regression with nn.Linear layers."""
        torch.manual_seed(42)
        X = torch.randn(100, 32)
        y = torch.randn(100, 8)

        model = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 8),
        )

        config = _make_config(
            "--optimizer.name", "ECOAdamW",
            "--eco.enabled",
            "--optimizer.adamw.lr", "0.01",
        )
        parallel_dims = _build_parallel_dims(config)
        optimizers = build_optimizers(
            [model], config.optimizer, parallel_dims, eco_config=config.eco,
        )
        optimizer = optimizers.optimizers[0]

        losses = []
        for epoch in range(20):
            optimizer.zero_grad()
            pred = model(X)
            loss = F.mse_loss(pred, y)
            if epoch == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < initial_loss * 0.9, (
            f"ECO training should converge: {initial_loss:.4f} -> {losses[-1]:.4f}"
        )

    def test_get_eco_metrics_from_optimizer(self):
        """Metrics flow: optimizer.get_eco_metrics() → train.py."""
        fp = torch.randn(16, 16)
        qt = QuantizedTensor.quantize(fp, torch.float8_e4m3fn)
        qt.requires_grad_(True)

        opt = ECOAdamW([qt], lr=1e-3, eco_enabled=True, heuristic_log_freq=1)
        qt.grad = torch.randn_like(qt.dequantize()) * 0.01
        opt.step()
        qt.grad = torch.randn_like(qt.dequantize()) * 0.01
        opt.step()

        m = opt.get_eco_metrics()
        assert "eco/heuristic/norm_ratio/mean" in m


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
