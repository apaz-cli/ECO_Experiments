"""Tests for ActivationQuantConverter and ActivationQuantLinear.

Tests:
  - Converter correctly wraps nn.Linear layers inside transformer blocks
  - Skips output projection (FQN doesn't start with "layers.")
  - Parameters are the *same* objects (not copies)
  - Forward gives correct output
  - Backward saves FP8 activations (not full-precision)
  - Row-wise scaling matches paper spec
  - Gradients are correct (within quant tolerance)
  - Composes with ECOAdamW (simulated weight quant + activation quant)
  - Convergence on a toy task
  - No-op when activation_dtype="none"
  - Edge cases: no bias, different input shapes
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.components.activation_quant import (
    ActivationQuantConverter,
    ActivationQuantLinear,
    _ActivationQuantLinearFn,
)
from torchtitan.components.quant_gemm import HardwareQuantLinear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockECO:
    def __init__(self, activation_dtype="fp8"):
        self.activation_dtype = activation_dtype

class _MockJobConfig:
    def __init__(self, activation_dtype="fp8"):
        self.eco = _MockECO(activation_dtype)
        class _model:
            converters = []
        self.model = _model


def _make_converter(activation_dtype="fp8"):
    """Create a converter without needing real ParallelDims."""
    return ActivationQuantConverter(_MockJobConfig(activation_dtype), None)


class _FakeTransformerBlock(nn.Module):
    """Mimics a transformer block with attention + FFN linears."""
    def __init__(self, dim):
        super().__init__()
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.ffn = nn.Linear(dim, dim)

    def forward(self, x):
        return self.ffn(F.relu(self.wq(x) + self.wk(x)))


class _FakeTransformer(nn.Module):
    """Mimics the Llama3 Transformer structure for FQN testing."""
    def __init__(self, dim=32, n_layers=2, vocab_size=100):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleDict({
            str(i): _FakeTransformerBlock(dim) for i in range(n_layers)
        })
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens):
        h = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            h = layer(h)
        return self.output(h)


# ---------------------------------------------------------------------------
# Converter tests
# ---------------------------------------------------------------------------

class TestActivationQuantConverter:
    def test_wraps_layers_inside_transformer_blocks(self):
        model = _FakeTransformer()
        _make_converter("fp8").convert(model)

        # Linears inside layers.* should be wrapped with HardwareQuantLinear
        # (FP8 dtype dispatches to hardware GEMM path, not ActivationQuantLinear)
        assert isinstance(model.layers["0"].wq, HardwareQuantLinear)
        assert isinstance(model.layers["0"].wk, HardwareQuantLinear)
        assert isinstance(model.layers["0"].ffn, HardwareQuantLinear)
        assert isinstance(model.layers["1"].wq, HardwareQuantLinear)

    def test_skips_output_projection(self):
        """Paper: 'excluding the embedding and output layers'."""
        model = _FakeTransformer()
        _make_converter("fp8").convert(model)

        # output is NOT inside layers.* — should remain nn.Linear
        assert isinstance(model.output, nn.Linear)
        assert not isinstance(model.output, ActivationQuantLinear)

    def test_skips_embedding(self):
        """Embedding is nn.Embedding, not nn.Linear — naturally skipped."""
        model = _FakeTransformer()
        _make_converter("fp8").convert(model)
        assert isinstance(model.tok_embeddings, nn.Embedding)

    def test_parameters_are_same_objects(self):
        """Weight/bias Parameters must be the same objects, not copies."""
        original = nn.Linear(16, 8)
        orig_weight = original.weight
        orig_bias = original.bias

        wrapped = ActivationQuantLinear(original, torch.float8_e4m3fn)

        assert wrapped.weight is orig_weight
        assert wrapped.bias is orig_bias

    def test_noop_when_none(self):
        model = _FakeTransformer()
        _make_converter("none").convert(model)

        # Everything should remain unchanged
        assert isinstance(model.layers["0"].wq, nn.Linear)
        assert not isinstance(model.layers["0"].wq, ActivationQuantLinear)

    def test_no_bias(self):
        """Linears without bias should work."""
        model = nn.Module()
        model.layers = nn.ModuleDict({"0": nn.Linear(8, 4, bias=False)})
        _make_converter("fp8").convert(model)

        wrapped = model.layers["0"]
        # FP8 dtype → HardwareQuantLinear
        assert isinstance(wrapped, HardwareQuantLinear)
        assert wrapped.bias is None

    def test_state_dict_keys_unchanged(self):
        """state_dict keys must not change after wrapping."""
        model = _FakeTransformer()
        keys_before = set(model.state_dict().keys())

        _make_converter("fp8").convert(model)

        keys_after = set(model.state_dict().keys())
        assert keys_before == keys_after

    def test_forward_still_works(self):
        """Full model forward pass after conversion."""
        torch.manual_seed(42)
        model = _FakeTransformer(dim=32, n_layers=2, vocab_size=100)
        _make_converter("fp8").convert(model)

        tokens = torch.randint(0, 100, (2, 8))
        out = model(tokens)
        assert out.shape == (2, 8, 100)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Forward / backward correctness
# ---------------------------------------------------------------------------

class TestForwardBackward:
    def test_forward_matches_nn_linear(self):
        """Forward output should be identical to nn.Linear (quant only affects backward)."""
        torch.manual_seed(42)
        linear = nn.Linear(32, 16)
        wrapped = ActivationQuantLinear(linear, torch.float8_e4m3fn)

        x = torch.randn(4, 32)
        y_ref = linear(x)
        y_act = wrapped(x)

        torch.testing.assert_close(y_ref, y_act)

    def test_backward_produces_gradients(self):
        torch.manual_seed(42)
        linear = nn.Linear(16, 8)
        wrapped = ActivationQuantLinear(linear, torch.float8_e4m3fn)

        x = torch.randn(4, 16, requires_grad=True)
        y = wrapped(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert wrapped.weight.grad is not None
        assert wrapped.bias.grad is not None

    def test_backward_grad_input_exact(self):
        """grad_input should be exact (not affected by activation quant)."""
        torch.manual_seed(42)
        linear = nn.Linear(32, 16)
        wrapped = ActivationQuantLinear(linear, torch.float8_e4m3fn)

        x1 = torch.randn(4, 32, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        linear(x1).sum().backward()
        wrapped(x2).sum().backward()

        # grad_input = grad_output @ W — no quant involved
        torch.testing.assert_close(x1.grad, x2.grad)

    def test_backward_grad_weight_approximate(self):
        """grad_weight uses dequantized activations — close but not exact."""
        torch.manual_seed(42)
        original = nn.Linear(32, 16)
        wrapped = ActivationQuantLinear(original, torch.float8_e4m3fn)

        x = torch.randn(4, 32)

        # Reference: exact grad_weight
        out_ref = F.linear(x, original.weight, original.bias)
        out_ref.sum().backward()
        grad_w_ref = original.weight.grad.clone()

        original.weight.grad = None
        original.bias.grad = None

        # Wrapped: approximate grad_weight
        out_wrapped = wrapped(x)
        out_wrapped.sum().backward()
        grad_w_approx = wrapped.weight.grad.clone()

        assert torch.allclose(grad_w_ref, grad_w_approx, atol=0.05, rtol=0.05), (
            f"grad_weight too far off: max diff = {(grad_w_ref - grad_w_approx).abs().max()}"
        )

    def test_3d_input(self):
        """Should handle (batch, seq, hidden) inputs."""
        torch.manual_seed(42)
        linear = nn.Linear(64, 32)
        wrapped = ActivationQuantLinear(linear, torch.float8_e4m3fn)

        x = torch.randn(2, 8, 64)
        y = wrapped(x)
        assert y.shape == (2, 8, 32)

        loss = y.sum()
        loss.backward()
        assert wrapped.weight.grad is not None


# ---------------------------------------------------------------------------
# FP8 activation storage verification
# ---------------------------------------------------------------------------

class TestFP8Storage:
    def test_saved_tensors_are_fp8(self):
        """Verify that the saved activations are actually FP8."""
        packed_tensors = []

        def pack_hook(tensor):
            packed_tensors.append(tensor)
            return tensor

        torch.manual_seed(42)
        linear = nn.Linear(16, 8)
        wrapped = ActivationQuantLinear(linear, torch.float8_e4m3fn)

        x = torch.randn(4, 16, requires_grad=True)

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda t: t):
            y = wrapped(x)

        y.sum().backward()

        fp8_tensors = [t for t in packed_tensors if t.dtype == torch.float8_e4m3fn]
        assert len(fp8_tensors) >= 1, (
            f"Expected at least one FP8 saved tensor, got dtypes: "
            f"{[t.dtype for t in packed_tensors]}"
        )
        assert fp8_tensors[0].shape == (4, 16)

    def test_row_wise_scaling(self):
        """Each row should have its own scale = max(|row|) / fp8_max."""
        torch.manual_seed(42)
        x = torch.randn(3, 8)

        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        row_amax = x.abs().amax(dim=1, keepdim=True)
        expected_scale = (row_amax / fp8_max).clamp(min=1e-12)
        expected_q = (x / expected_scale).to(torch.float8_e4m3fn)

        packed_tensors = []

        def pack_hook(tensor):
            packed_tensors.append(tensor)
            return tensor

        linear = nn.Linear(8, 4)
        wrapped = ActivationQuantLinear(linear, torch.float8_e4m3fn)

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda t: t):
            y = wrapped(x.requires_grad_(True))

        y.sum().backward()

        fp8_tensors = [t for t in packed_tensors if t.dtype == torch.float8_e4m3fn]
        scale_tensors = [
            t for t in packed_tensors
            if t.dtype == torch.float32 and t.shape == (3, 1)
        ]

        assert len(fp8_tensors) == 1
        assert len(scale_tensors) == 1

        torch.testing.assert_close(scale_tensors[0], expected_scale)
        assert torch.equal(fp8_tensors[0], expected_q)


# ---------------------------------------------------------------------------
# Composition with ECOAdamW
# ---------------------------------------------------------------------------

class TestCompositionWithECO:
    def test_eco_plus_activation_quant_convergence(self):
        """ECOAdamW (simulated weight quant) + activation quant should converge."""
        from torchtitan.components.eco_optimizer import ECOAdamW

        torch.manual_seed(42)

        # Use a structure where linears are under "layers." FQN
        model = nn.Module()
        model.layers = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        model.forward = lambda x: model.layers(x)

        _make_converter("fp8").convert(model)

        optimizer = ECOAdamW(
            model.parameters(),
            lr=1e-3,
            eco_enabled=True,
            quantize_weights=True,
            quant_dtype="fp8",
        )

        x = torch.randn(64, 16)
        y = x[:, :4].sum(dim=1, keepdim=True)

        losses = []
        for _ in range(200):
            optimizer.zero_grad()
            pred = model.forward(x)
            loss = F.mse_loss(pred, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        assert losses[-1] < losses[0] * 0.5, (
            f"Should converge with ECO + activation quant: "
            f"initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        )

    def test_eco_without_activation_quant_convergence(self):
        """ECOAdamW without activation quant should also converge (baseline)."""
        from torchtitan.components.eco_optimizer import ECOAdamW

        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        optimizer = ECOAdamW(
            model.parameters(),
            lr=1e-3,
            eco_enabled=True,
            quantize_weights=True,
            quant_dtype="fp8",
        )

        x = torch.randn(64, 16)
        y = x[:, :4].sum(dim=1, keepdim=True)

        losses = []
        for _ in range(200):
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        assert losses[-1] < losses[0] * 0.5

    def test_parameter_identity_after_optimizer_step(self):
        """After optimizer.step(), the wrapped module's params should still be
        the same objects that the optimizer holds."""
        from torchtitan.components.eco_optimizer import ECOAdamW

        model = nn.Module()
        model.layers = nn.Sequential(nn.Linear(8, 4))
        model.forward = lambda x: model.layers(x)
        _make_converter("fp8").convert(model)

        param_ids_before = {id(p) for p in model.parameters()}

        optimizer = ECOAdamW(model.parameters(), lr=1e-3)

        x = torch.randn(2, 8)
        model.forward(x).sum().backward()
        optimizer.step()

        param_ids_after = {id(p) for p in model.parameters()}
        assert param_ids_before == param_ids_after


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_activation_quant_does_not_prevent_convergence(self):
        """A model with activation quant should converge on a simple task.

        Note: HardwareQuantLinear uses QuantizedTensor weights, which require
        ECOAdamW (not standard AdamW) for correct in-place parameter updates.
        """
        from torchtitan.components.eco_optimizer import ECOAdamW

        torch.manual_seed(42)

        model = nn.Module()
        model.layers = nn.Sequential(
            nn.Linear(16, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        model.forward = lambda x: model.layers(x)

        _make_converter("fp8").convert(model)

        # ECOAdamW correctly handles QuantizedTensor weights (HardwareQuantLinear)
        # via _step_quantized.  Standard AdamW in-place ops don't update _data.
        optimizer = ECOAdamW(model.parameters(), lr=1e-3, eco_enabled=True)

        x = torch.randn(128, 16)
        y = torch.sin(x[:, 0:1]) + 0.5 * x[:, 1:2]

        losses = []
        for _ in range(300):
            optimizer.zero_grad()
            pred = model.forward(x)
            loss = F.mse_loss(pred, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        assert losses[-1] < losses[0] * 0.3, (
            f"Should converge: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_input(self):
        linear = nn.Linear(8, 4)
        wrapped = ActivationQuantLinear(linear, torch.float8_e4m3fn)

        x = torch.zeros(2, 8, requires_grad=True)
        y = wrapped(x)
        y.sum().backward()
        assert torch.isfinite(y).all()
        assert torch.isfinite(x.grad).all()

    def test_large_values(self):
        """FP8 E4M3 max is 448 — large inputs should be clamped by scaling."""
        linear = nn.Linear(8, 4)
        wrapped = ActivationQuantLinear(linear, torch.float8_e4m3fn)

        x = torch.randn(2, 8) * 1000
        x.requires_grad_(True)
        y = wrapped(x)
        y.sum().backward()
        assert torch.isfinite(y).all()
        assert torch.isfinite(x.grad).all()

    def test_single_element_rows(self):
        """Input with in_features=1 should work."""
        linear = nn.Linear(1, 4)
        wrapped = ActivationQuantLinear(linear, torch.float8_e4m3fn)

        x = torch.randn(3, 1, requires_grad=True)
        y = wrapped(x)
        y.sum().backward()
        assert y.shape == (3, 4)

    def test_full_model_backward(self):
        """Full FakeTransformer model backward pass after conversion."""
        torch.manual_seed(42)
        model = _FakeTransformer(dim=32, n_layers=2, vocab_size=100)
        _make_converter("fp8").convert(model)

        tokens = torch.randint(0, 100, (2, 8))
        out = model(tokens)
        loss = out.sum()
        loss.backward()

        # All parameters should have gradients
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
