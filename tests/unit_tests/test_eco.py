# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ECOConverter (torchtitan/components/eco.py)."""

import pytest
import torch
import torch.nn as nn

from torchtitan.components.eco import ECOConverter
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ConfigManager
from torchtitan.distributed import ParallelDims


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(*extra_args):
    """Create a real JobConfig via ConfigManager."""
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


def _make_eco(*extra_args):
    """Create a real ECOConverter from ConfigManager args."""
    config = _make_config("--eco.enabled", *extra_args)
    parallel_dims = _build_parallel_dims(config)
    return ECOConverter(config, parallel_dims), config


def _make_model(in_features=32, out_features=16):
    """Create a tiny nn.Linear model for testing."""
    model = nn.Linear(in_features, out_features, bias=False)
    model.weight.data.normal_(0, 0.1)
    return model


def _make_optimizer_container(models, lr=1e-3, beta1=0.9, beta2=0.98, eps=1e-9):
    """Create a real OptimizersContainer wrapping real AdamW optimizers."""
    if isinstance(models, nn.Module):
        models = [models]
    return OptimizersContainer(
        model_parts=models,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs=dict(lr=lr, betas=(beta1, beta2), eps=eps),
    )


def _do_fake_backward_and_step(models, container):
    """Set random gradients and do an optimizer step (populates optimizer state)."""
    if isinstance(models, nn.Module):
        models = [models]
    for m in models:
        for p in m.parameters():
            p.grad = torch.randn_like(p) * 0.01
    container.step()
    container.zero_grad()


# ---------------------------------------------------------------------------
# Tests: FP8 round-trip quantization
# ---------------------------------------------------------------------------


class TestFP8RoundTrip:
    @pytest.fixture
    def eco(self):
        e, _ = _make_eco()
        return e

    @pytest.fixture
    def eco_sr(self):
        e, _ = _make_eco("--eco.stochastic_rounding")
        return e

    def test_roundtrip_error_is_small(self, eco):
        """Quantization error should be small relative to tensor magnitude."""
        t = torch.randn(128, 128)
        result = eco._quantize_fp8_roundtrip(t)
        error = (t - result).abs()
        # FP8 E4M3 has 3 mantissa bits -> relative error bounded by ~2^-3 = 0.125
        rel_error = error / (t.abs() + 1e-12)
        assert rel_error.mean() < 0.1

    def test_roundtrip_is_idempotent(self, eco):
        """Applying round-trip twice should give the same result as once."""
        t = torch.randn(64, 64)
        once = eco._quantize_fp8_roundtrip(t)
        twice = eco._quantize_fp8_roundtrip(once)
        assert torch.equal(once, twice), "FP8 round-trip should be idempotent"

    def test_stochastic_rounding_changes_output(self, eco_sr):
        """With stochastic rounding, different calls may give different results."""
        t = torch.randn(128, 128)
        results = [eco_sr._quantize_fp8_roundtrip(t) for _ in range(10)]
        any_differ = any(
            not torch.equal(results[i], results[j])
            for i in range(len(results))
            for j in range(i + 1, len(results))
        )
        assert any_differ, "Stochastic rounding should produce varying outputs"

    def test_stochastic_rounding_is_unbiased(self, eco_sr):
        """Stochastic rounding should be unbiased: E[round(x)] ~ x."""
        t = torch.randn(256, 256)
        N = 50
        results = torch.stack([eco_sr._quantize_fp8_roundtrip(t) for _ in range(N)])
        mean_result = results.float().mean(dim=0)
        error = (mean_result - t.float()).abs().mean()
        assert error < 0.01, f"Stochastic rounding bias too large: {error:.6f}"


# ---------------------------------------------------------------------------
# Tests: Momentum injection (Algorithm 3)
# ---------------------------------------------------------------------------


class TestMomentumInjection:
    def test_injection_formula_is_correct(self):
        """Verify the injection matches Algorithm 3 formula exactly."""
        beta1, beta2, eps, lr = 0.9, 0.98, 1e-9, 1e-3
        eco, _ = _make_eco(
            "--optimizer.beta1", str(beta1),
            "--optimizer.beta2", str(beta2),
            "--optimizer.eps", str(eps),
        )
        model = _make_model()
        eco._eco_param_ids.add(id(model.weight))
        container = _make_optimizer_container(model, lr=lr, beta1=beta1,
                                              beta2=beta2, eps=eps)
        _do_fake_backward_and_step(model, container)

        param = model.weight
        optimizer = container.optimizers[0]
        state = optimizer.state[param]
        m_before = state["exp_avg"].clone()
        v = state["exp_avg_sq"].clone()
        step_val = state["step"].item() if isinstance(state["step"], torch.Tensor) else state["step"]
        w_before = param.data.clone()

        eco.post_optimizer_hook(model, optimizers=container)

        # Manually compute expected injection
        theta_hat = eco._quantize_fp8_roundtrip(w_before)
        error = w_before - theta_hat

        bias_correction1 = 1.0 - beta1 ** step_val
        bias_correction2 = 1.0 - beta2 ** step_val
        injection_coeff = (bias_correction1 / lr) * (1.0 - 1.0 / beta1)
        adaptive_scale = torch.sqrt(v / bias_correction2) + eps

        expected_m = m_before + injection_coeff * adaptive_scale * error
        actual_m = state["exp_avg"]

        torch.testing.assert_close(actual_m, expected_m, atol=1e-6, rtol=1e-5)

        # Weight should now be FP8-representable
        assert torch.equal(param.data, theta_hat)

    def test_non_eco_params_untouched(self):
        """Parameters NOT in _eco_param_ids should not be modified."""
        eco, _ = _make_eco()
        model = _make_model()
        # Do NOT add to eco_param_ids
        container = _make_optimizer_container(model)
        _do_fake_backward_and_step(model, container)

        param = model.weight
        w_before = param.data.clone()
        state = container.optimizers[0].state[param]
        m_before = state["exp_avg"].clone()

        eco.post_optimizer_hook(model, optimizers=container)

        assert torch.equal(param.data, w_before), \
            "Non-ECO params should not be quantized"
        assert torch.equal(state["exp_avg"], m_before), \
            "Non-ECO params momentum should not change"

    def test_multi_model_parts(self):
        """ECO should work with a list of model parts (multi-GPU simulation)."""
        eco, _ = _make_eco()
        m1 = _make_model(32, 16)
        m2 = _make_model(16, 8)
        eco._eco_param_ids.add(id(m1.weight))
        eco._eco_param_ids.add(id(m2.weight))

        container = _make_optimizer_container([m1, m2])
        _do_fake_backward_and_step([m1, m2], container)

        eco.post_optimizer_hook([m1, m2], optimizers=container)

        # Both should be FP8-representable now
        assert torch.equal(m1.weight.data, eco._quantize_fp8_roundtrip(m1.weight.data))
        assert torch.equal(m2.weight.data, eco._quantize_fp8_roundtrip(m2.weight.data))


# ---------------------------------------------------------------------------
# Tests: Heuristic metrics
# ---------------------------------------------------------------------------


class TestHeuristicMetrics:
    def test_metrics_emitted_at_correct_frequency(self):
        """Metrics appear when prev errors exist from a prior stash step.

        With heuristic_log_freq=2:
        - Step 1: step_count=1, 1%2!=0 -> no stash, no compare
        - Step 2: step_count=2, 2%2==0 -> stash errors (first time, nothing to compare)
        - Step 3: step_count=3, prev_errors non-empty -> compare & emit metrics
        - Step 4: step_count=4, 4%2==0 -> stash again (prev_errors was cleared)
        - Step 5: step_count=5, prev_errors non-empty -> compare & emit metrics
        """
        eco, _ = _make_eco("--eco.heuristic_log_freq", "2")
        model = _make_model()
        eco._eco_param_ids.add(id(model.weight))
        container = _make_optimizer_container(model)

        # Step 1: no stash, no compare
        _do_fake_backward_and_step(model, container)
        eco.post_optimizer_hook(model, optimizers=container)
        assert eco.get_extra_metrics() == {}

        # Step 2: stash errors (first time, nothing to compare)
        _do_fake_backward_and_step(model, container)
        eco.post_optimizer_hook(model, optimizers=container)
        assert eco.get_extra_metrics() == {}

        # Step 3: prev_errors exist -> compare with current, emit metrics
        _do_fake_backward_and_step(model, container)
        eco.post_optimizer_hook(model, optimizers=container)
        metrics = eco.get_extra_metrics()
        assert len(metrics) > 0, "Should have heuristic metrics on step after stash"
        assert "eco/heuristic/norm_ratio/mean" in metrics
        assert "eco/heuristic/cosine_sim/mean" in metrics

        # Step 4: stash again (prev_errors cleared on step 3)
        _do_fake_backward_and_step(model, container)
        eco.post_optimizer_hook(model, optimizers=container)
        assert eco.get_extra_metrics() == {}

        # Step 5: compare again
        _do_fake_backward_and_step(model, container)
        eco.post_optimizer_hook(model, optimizers=container)
        metrics = eco.get_extra_metrics()
        assert len(metrics) > 0, "Should have heuristic metrics on second cycle"

    def test_get_extra_metrics_clears_after_read(self):
        """get_extra_metrics should clear pending metrics."""
        eco, _ = _make_eco("--eco.heuristic_log_freq", "1")
        model = _make_model()
        eco._eco_param_ids.add(id(model.weight))
        container = _make_optimizer_container(model)

        # Two consecutive log steps to produce metrics
        _do_fake_backward_and_step(model, container)
        eco.post_optimizer_hook(model, optimizers=container)
        _do_fake_backward_and_step(model, container)
        eco.post_optimizer_hook(model, optimizers=container)

        m1 = eco.get_extra_metrics()
        m2 = eco.get_extra_metrics()
        assert len(m1) > 0
        assert m2 == {}, "Second call should return empty dict"

    def test_heuristic_metric_values_are_reasonable(self):
        """Spot-check that metric values are in expected ranges."""
        eco, _ = _make_eco("--eco.heuristic_log_freq", "1")
        model = _make_model(64, 64)
        eco._eco_param_ids.add(id(model.weight))
        container = _make_optimizer_container(model)

        # Run two log-freq steps
        _do_fake_backward_and_step(model, container)
        eco.post_optimizer_hook(model, optimizers=container)
        _do_fake_backward_and_step(model, container)
        eco.post_optimizer_hook(model, optimizers=container)

        metrics = eco.get_extra_metrics()

        assert 0.0 < metrics["eco/heuristic/norm_ratio/mean"] < 100.0
        assert -1.0 <= metrics["eco/heuristic/cosine_sim/mean"] <= 1.0
        assert metrics["eco/heuristic/rel_diff_norm/mean"] >= 0.0

        for base in ["eco/heuristic/norm_ratio", "eco/heuristic/cosine_sim",
                      "eco/heuristic/rel_diff_norm"]:
            assert f"{base}/min" in metrics
            assert f"{base}/mean" in metrics
            assert f"{base}/max" in metrics


# ---------------------------------------------------------------------------
# Tests: convert() with Float8Linear (requires torchao)
# ---------------------------------------------------------------------------


class TestConvertWithFloat8:
    @pytest.fixture(autouse=True)
    def _skip_without_torchao(self):
        pytest.importorskip("torchao")

    def test_convert_finds_float8_modules(self):
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        eco, _ = _make_eco()
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        config = Float8LinearConfig(emulate=True)
        convert_to_float8_training(model, config=config)

        eco.convert(model)
        assert len(eco._eco_param_ids) == 2

    def test_convert_without_float8_warns(self):
        """If no Float8Linear modules exist, ECO should log a warning."""
        eco, _ = _make_eco()
        model = nn.Linear(32, 16)

        eco.convert(model)
        assert len(eco._eco_param_ids) == 0

    def test_end_to_end_with_float8(self):
        """Full pipeline: Float8 convert -> optimizer step -> ECO hook."""
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        eco, _ = _make_eco()
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        config = Float8LinearConfig(emulate=True)
        convert_to_float8_training(model, config=config)

        eco.convert(model)
        assert len(eco._eco_param_ids) == 2

        container = _make_optimizer_container(model)

        # Real forward + backward
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        container.step()
        container.zero_grad()

        eco.post_optimizer_hook(model, optimizers=container)
        assert eco.step_count == 1


# ---------------------------------------------------------------------------
# Tests: Actual training
# ---------------------------------------------------------------------------


def _train_loop(model, x, y, optimizer, steps, post_step_hook=None):
    """Train a model on a fixed batch, return per-step losses."""
    losses = []
    for _ in range(steps):
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if post_step_hook is not None:
            post_step_hook()
    return losses


def _naive_fp8_hook(model):
    """Post-step hook that quantizes weights through FP8 without compensation."""
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    with torch.no_grad():
        for p in model.parameters():
            amax = p.data.abs().amax()
            scale = torch.where(amax > 0, amax / fp8_max, torch.ones_like(amax))
            quantized = (p.data / scale).to(fp8_dtype)
            p.data.copy_(quantized.to(p.data.dtype) * scale)


class TestTraining:
    """Train a tiny model and verify ECO actually helps."""

    STEPS = 200
    LR = 1e-3
    BETA1 = 0.9
    BETA2 = 0.98
    EPS = 1e-9
    SEED = 42

    def _make_problem(self):
        """Fixed synthetic regression problem: overfit a single batch."""
        torch.manual_seed(self.SEED)
        x = torch.randn(64, 16)
        # Target is a nonlinear function so the MLP has something to learn
        y = torch.sin(x[:, :1] * 3) + 0.5 * x[:, 1:2] ** 2
        return x, y

    def _make_mlp(self):
        torch.manual_seed(self.SEED)
        return nn.Sequential(
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def _train_bf16_baseline(self, x, y):
        """Plain BF16 training, no quantization."""
        model = self._make_mlp()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.LR,
            betas=(self.BETA1, self.BETA2), eps=self.EPS,
        )
        return _train_loop(model, x, y, optimizer, self.STEPS)

    def _train_naive_fp8(self, x, y):
        """FP8 round-trip every step, no error compensation."""
        model = self._make_mlp()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.LR,
            betas=(self.BETA1, self.BETA2), eps=self.EPS,
        )
        hook = lambda: _naive_fp8_hook(model)
        return _train_loop(model, x, y, optimizer, self.STEPS, post_step_hook=hook)

    def _train_eco(self, x, y):
        """FP8 round-trip every step with ECO momentum injection."""
        model = self._make_mlp()
        eco, _ = _make_eco(
            "--optimizer.beta1", str(self.BETA1),
            "--optimizer.beta2", str(self.BETA2),
            "--optimizer.eps", str(self.EPS),
        )
        # Register all params as ECO-eligible (simulating Float8Linear convert)
        for p in model.parameters():
            eco._eco_param_ids.add(id(p))

        container = _make_optimizer_container(
            model, lr=self.LR, beta1=self.BETA1,
            beta2=self.BETA2, eps=self.EPS,
        )
        hook = lambda: eco.post_optimizer_hook(model, optimizers=container)
        losses = _train_loop(model, x, y, container.optimizers[0], self.STEPS,
                             post_step_hook=hook)
        return losses

    def test_bf16_and_eco_converge(self):
        x, y = self._make_problem()
        bf16_losses = self._train_bf16_baseline(x, y)
        eco_losses = self._train_eco(x, y)

        # BF16 baseline should converge easily (sanity check)
        assert bf16_losses[-1] < bf16_losses[0] * 0.01, (
            f"BF16 didn't converge: {bf16_losses[0]:.4f} -> {bf16_losses[-1]:.4f}"
        )

        # ECO should also converge meaningfully (> 50% loss reduction)
        assert eco_losses[-1] < eco_losses[0] * 0.5, (
            f"ECO didn't converge: {eco_losses[0]:.4f} -> {eco_losses[-1]:.4f}"
        )

    def test_naive_fp8_barely_learns(self):
        """Naive FP8 quantization every step cripples training on a tiny model.

        This demonstrates the problem ECO is designed to solve.
        """
        x, y = self._make_problem()
        naive_losses = self._train_naive_fp8(x, y)

        # Naive FP8 should barely reduce loss -- quantization noise
        # destroys optimizer state each step
        reduction = 1 - naive_losses[-1] / naive_losses[0]
        assert reduction < 0.5, (
            f"Naive FP8 reduced loss by {reduction:.0%}, expected it to struggle"
        )

    def test_eco_beats_naive_fp8(self):
        x, y = self._make_problem()
        naive_losses = self._train_naive_fp8(x, y)
        eco_losses = self._train_eco(x, y)

        # ECO should achieve dramatically lower final loss than naive FP8.
        # On this task: ECO ~0.025 vs naive ~1.25 (50x better).
        assert eco_losses[-1] < naive_losses[-1] * 0.5, (
            f"ECO ({eco_losses[-1]:.6f}) should substantially beat "
            f"naive FP8 ({naive_losses[-1]:.6f})"
        )
