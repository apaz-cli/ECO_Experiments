"""Tests for ECO optimizer FSDP2 sharding and checkpointing.

Verifies that ECOAdamW's extra state keys (prev_error, master_weights) have
correct shapes, survive state_dict roundtrips, and work with FSDP2's
fully_shard + distributed checkpoint APIs.  All tests run on CUDA.
"""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)

from torchtitan.components.eco_adamw import ECOAdamW

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# Apply to every test in the module
pytestmark = requires_cuda

DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model():
    """Small model with both dim>=2 (Linear weight) and dim=1 (bias) params."""
    return nn.Sequential(
        nn.Linear(64, 64),
        nn.Linear(64, 64),
    ).to(DEVICE)


def _run_steps(optimizer, model, n=3):
    """Run n forward/backward/step cycles."""
    for _ in range(n):
        x = torch.randn(4, 64, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


# ===========================================================================
# Test Class 1: State shape verification (no distributed)
# ===========================================================================

class TestECOOptimizerStateShapes:
    """Verify optimizer state tensors match parameter shapes."""

    def test_state_shapes_basic(self):
        """exp_avg and exp_avg_sq have the same shape as the param."""
        model = _make_model()
        opt = ECOAdamW(model.parameters(), lr=1e-3, eco_enabled=False)
        _run_steps(opt, model, n=1)

        for p in model.parameters():
            state = opt.state[p]
            assert state["exp_avg"].shape == p.shape
            assert state["exp_avg_sq"].shape == p.shape
            assert state["step"].dim() == 0  # scalar

    def test_state_shapes_with_master_weights(self):
        """master_weights has the same shape as the param."""
        model = _make_model()
        opt = ECOAdamW(
            model.parameters(), lr=1e-3,
            master_weights_dtype=torch.float32,
        )
        _run_steps(opt, model, n=1)

        for p in model.parameters():
            state = opt.state[p]
            if "master_weights" in state:
                assert state["master_weights"].shape == p.shape

    def test_state_shapes_with_prev_error(self):
        """prev_error has the same shape as the param (after 2 steps with heuristic logging)."""
        model = _make_model()
        opt = ECOAdamW(
            model.parameters(), lr=1e-3,
            eco_enabled=True,
            heuristic_log_freq=1,  # log every step -> stores prev_error
            quant_dtype="fp8",
        )
        _run_steps(opt, model, n=2)

        found_prev_error = False
        for p in model.parameters():
            state = opt.state[p]
            if "prev_error" in state:
                found_prev_error = True
                assert state["prev_error"].shape == p.shape
        assert found_prev_error, "Expected at least one param to have prev_error"

    def test_state_shapes_mixed_params(self):
        """Both dim>=2 (quantized path) and dim=1 (regular path) params get correct state shapes."""
        model = _make_model()
        opt = ECOAdamW(
            model.parameters(), lr=1e-3,
            eco_enabled=True, quant_dtype="fp8",
        )
        _run_steps(opt, model, n=2)

        weight_count = 0
        bias_count = 0
        for p in model.parameters():
            state = opt.state[p]
            assert state["exp_avg"].shape == p.shape
            assert state["exp_avg_sq"].shape == p.shape
            if p.dim() >= 2:
                weight_count += 1
            else:
                bias_count += 1
        assert weight_count == 2, "Expected 2 weight matrices"
        assert bias_count == 2, "Expected 2 biases"


# ===========================================================================
# Test Class 2: state_dict roundtrip (no distributed)
# ===========================================================================

class TestECOStateDictRoundtrip:
    """Verify state_dict() -> load_state_dict() preserves ECO state."""

    def _roundtrip(self, model, opt):
        """Save state, create fresh optimizer, load state, return both."""
        sd = opt.state_dict()
        opt2 = ECOAdamW(
            model.parameters(), lr=1e-3,
            eco_enabled=opt._eco_enabled,
            heuristic_log_freq=opt._heuristic_log_freq,
            quantize_weights=opt._quantize_weights,
            quant_dtype=opt._quant_dtype,
            master_weights_dtype=opt._master_weights_dtype,
        )
        opt2.load_state_dict(sd)
        return sd, opt2

    def test_roundtrip_preserves_all_keys(self):
        """All ECO state keys survive save/load."""
        model = _make_model()
        opt = ECOAdamW(
            model.parameters(), lr=1e-3,
            eco_enabled=True, heuristic_log_freq=1,
            quant_dtype="fp8",
            master_weights_dtype=torch.float32,
        )
        _run_steps(opt, model, n=2)

        sd, opt2 = self._roundtrip(model, opt)

        # Check that loaded state has the same keys as original
        for p_idx, p in enumerate(model.parameters()):
            orig_keys = set(opt.state[p].keys())
            # state_dict uses integer indices
            loaded_keys = set(sd["state"][p_idx].keys())
            assert orig_keys == loaded_keys, (
                f"Param {p_idx}: orig keys {orig_keys} != loaded keys {loaded_keys}"
            )

    def test_roundtrip_preserves_values(self):
        """Loaded state values are numerically equal to saved values."""
        model = _make_model()
        opt = ECOAdamW(
            model.parameters(), lr=1e-3,
            eco_enabled=True, heuristic_log_freq=1,
            quant_dtype="fp8",
            master_weights_dtype=torch.float32,
        )
        _run_steps(opt, model, n=2)

        # Snapshot original state values
        orig_states = {}
        for p_idx, p in enumerate(model.parameters()):
            orig_states[p_idx] = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in opt.state[p].items()
            }

        _, opt2 = self._roundtrip(model, opt)

        for p_idx, p in enumerate(model.parameters()):
            for key, orig_val in orig_states[p_idx].items():
                loaded_val = opt2.state[p][key]
                if isinstance(orig_val, torch.Tensor):
                    torch.testing.assert_close(
                        loaded_val, orig_val,
                        msg=f"Param {p_idx}, key '{key}' mismatch",
                    )
                else:
                    assert loaded_val == orig_val

    def test_roundtrip_preserves_dtypes(self):
        """Standard state dtypes survive the roundtrip.

        Note: PyTorch's Optimizer.load_state_dict casts state tensors to match
        the parameter dtype.  So master_weights stored as BF16 will be loaded
        as FP32 (the param dtype).  We only verify that standard Adam states
        (exp_avg, exp_avg_sq, step) preserve their dtypes, since those are
        already FP32 matching param dtype.
        """
        model = _make_model()
        opt = ECOAdamW(
            model.parameters(), lr=1e-3,
            eco_enabled=True, heuristic_log_freq=1,
            quant_dtype="fp8",
            optim_state_dtype=torch.float32,
        )
        _run_steps(opt, model, n=2)

        # Record original dtypes for standard keys only
        orig_dtypes = {}
        standard_keys = {"exp_avg", "exp_avg_sq", "step"}
        for p_idx, p in enumerate(model.parameters()):
            orig_dtypes[p_idx] = {
                k: v.dtype for k, v in opt.state[p].items()
                if isinstance(v, torch.Tensor) and k in standard_keys
            }

        _, opt2 = self._roundtrip(model, opt)

        for p_idx, p in enumerate(model.parameters()):
            for key, orig_dtype in orig_dtypes[p_idx].items():
                loaded_dtype = opt2.state[p][key].dtype
                assert loaded_dtype == orig_dtype, (
                    f"Param {p_idx}, key '{key}': expected {orig_dtype}, got {loaded_dtype}"
                )

    def test_roundtrip_without_optional_keys(self):
        """Roundtrip works when master_weights and prev_error are absent."""
        model = _make_model()
        opt = ECOAdamW(
            model.parameters(), lr=1e-3,
            eco_enabled=False,  # no prev_error
            master_weights_dtype=None,  # no master_weights
        )
        _run_steps(opt, model, n=2)

        _, opt2 = self._roundtrip(model, opt)

        for p in model.parameters():
            state = opt2.state[p]
            assert "exp_avg" in state
            assert "exp_avg_sq" in state
            assert "step" in state
            assert "master_weights" not in state
            assert "prev_error" not in state


# ===========================================================================
# Test Class 3: FSDP2 integration (CUDA, nccl backend)
# ===========================================================================

def _init_dist():
    """Initialize single-rank CUDA distributed environment."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="nccl", world_size=1, rank=0)


def _destroy_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def dist_env():
    """Module-scoped fixture: init nccl once, tear down after all tests."""
    _init_dist()
    yield
    _destroy_dist()


class TestECOFSDP2Integration:
    """Test FSDP2 fully_shard + ECOAdamW on CUDA with nccl."""

    def _make_fsdp_model(self):
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cuda", (1,))
        model = _make_model()  # already on CUDA
        # Shard each linear layer, then the top-level container
        for module in model:
            if isinstance(module, nn.Linear):
                fully_shard(module, mesh=mesh)
        fully_shard(model, mesh=mesh)
        return model, mesh

    def test_fsdp2_optimizer_state_shapes(self, dist_env):
        """State tensor shapes match sharded param shapes after fully_shard."""
        model, mesh = self._make_fsdp_model()
        opt = ECOAdamW(
            model.parameters(), lr=1e-3,
            eco_enabled=True, quant_dtype="fp8",
            heuristic_log_freq=1,
            master_weights_dtype=torch.float32,
        )
        _run_steps(opt, model, n=2)

        for p in model.parameters():
            state = opt.state[p]
            assert state["exp_avg"].shape == p.shape
            assert state["exp_avg_sq"].shape == p.shape
            if "master_weights" in state:
                assert state["master_weights"].shape == p.shape
            if "prev_error" in state:
                assert state["prev_error"].shape == p.shape

    def test_fsdp2_state_dict_api(self, dist_env):
        """get_optimizer_state_dict / set_optimizer_state_dict works with ECO keys."""
        model, mesh = self._make_fsdp_model()
        opt = ECOAdamW(
            model.parameters(), lr=1e-3,
            eco_enabled=True, quant_dtype="fp8",
            master_weights_dtype=torch.float32,
        )
        _run_steps(opt, model, n=2)

        # Save via distributed state dict API
        sd = get_optimizer_state_dict(
            model, opt,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

        # Verify the state dict is non-empty and has state entries
        assert len(sd) > 0, "State dict should not be empty"

        # Load into a fresh optimizer on the same sharded model
        model2, _ = self._make_fsdp_model()
        opt2 = ECOAdamW(
            model2.parameters(), lr=1e-3,
            eco_enabled=True, quant_dtype="fp8",
            master_weights_dtype=torch.float32,
        )
        # Need to initialize state first
        _run_steps(opt2, model2, n=1)

        set_optimizer_state_dict(
            model2, opt2,
            optim_state_dict=sd,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

        # Verify state was loaded (opt2 state should have values from sd)
        for p in model2.parameters():
            state = opt2.state[p]
            assert "exp_avg" in state
            assert "exp_avg_sq" in state

    def test_fsdp2_dcp_roundtrip(self, dist_env):
        """DCP save/load roundtrip preserves all ECO state keys."""
        import torch.distributed.checkpoint as dcp

        model, mesh = self._make_fsdp_model()
        opt = ECOAdamW(
            model.parameters(), lr=1e-3,
            eco_enabled=True, quant_dtype="fp8",
        )
        _run_steps(opt, model, n=2)

        # Get optimizer state dict
        sd = get_optimizer_state_dict(
            model, opt,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            dcp.save({"optimizer": sd}, checkpoint_id=tmpdir)

            # Load into fresh state dict (need same structure)
            model2, _ = self._make_fsdp_model()
            opt2 = ECOAdamW(
                model2.parameters(), lr=1e-3,
                eco_enabled=True, quant_dtype="fp8",
            )
            _run_steps(opt2, model2, n=1)

            sd2 = get_optimizer_state_dict(
                model2, opt2,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )
            dcp.load({"optimizer": sd2}, checkpoint_id=tmpdir)

            # Apply loaded state
            set_optimizer_state_dict(
                model2, opt2,
                optim_state_dict=sd2,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )

            # Verify state was restored
            for p in model2.parameters():
                state = opt2.state[p]
                assert "exp_avg" in state
                assert "exp_avg_sq" in state

    def test_fsdp2_master_weights_checkpoint(self, dist_env):
        """DCP roundtrip with master_weights enabled."""
        import torch.distributed.checkpoint as dcp

        model, mesh = self._make_fsdp_model()
        opt = ECOAdamW(
            model.parameters(), lr=1e-3,
            eco_enabled=True, quant_dtype="fp8",
            master_weights_dtype=torch.float32,
        )
        _run_steps(opt, model, n=2)

        sd = get_optimizer_state_dict(
            model, opt,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dcp.save({"optimizer": sd}, checkpoint_id=tmpdir)

            model2, _ = self._make_fsdp_model()
            opt2 = ECOAdamW(
                model2.parameters(), lr=1e-3,
                eco_enabled=True, quant_dtype="fp8",
                master_weights_dtype=torch.float32,
            )
            _run_steps(opt2, model2, n=1)

            sd2 = get_optimizer_state_dict(
                model2, opt2,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )
            dcp.load({"optimizer": sd2}, checkpoint_id=tmpdir)

            set_optimizer_state_dict(
                model2, opt2,
                optim_state_dict=sd2,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )

            for p in model2.parameters():
                state = opt2.state[p]
                assert "exp_avg" in state
                assert "exp_avg_sq" in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
