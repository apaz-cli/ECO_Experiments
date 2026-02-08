# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ECO-compatible Llama model with quantized layers.

This module provides an ECO-compatible version of Llama3 that uses QuantizedLinear
for weights, enabling FP8 storage with BF16 compute.
"""

import math
import torch
import torch.nn as nn
from typing import Optional

# Import the original model components
from torchtitan.models.llama3.model.model import (
    Transformer,
    TransformerBlock,
    Attention,
    FeedForward,
    precompute_freqs_cis,
    reshape_for_broadcast,
    apply_rotary_emb,
)
from torchtitan.models.llama3.model.args import TransformerModelArgs
from torchtitan.components.quantized_linear import QuantizedLinear


def _compute_hidden_dim(dim: int, multiple_of: int = 256, ffn_dim_multiplier: Optional[float] = None) -> int:
    """Compute hidden dim for feedforward layer (same as original FeedForward)."""
    hidden_dim = 4 * dim
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class ECOAttention(Attention):
    """Attention module with quantized linear layers."""
    
    def __init__(self, model_args: TransformerModelArgs, layer_idx: int):
        # Initialize without calling parent __init__
        nn.Module.__init__(self)
        
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads
        self.enable_gqa = self.n_heads > self.n_kv_heads
        
        # Use QuantizedLinear for weights
        self.wq = QuantizedLinear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = QuantizedLinear(
            model_args.dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.wv = QuantizedLinear(
            model_args.dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.wo = QuantizedLinear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )
        
        # Import and set up attention mechanism
        from torchtitan.models.attention import (
            FlexAttentionWrapper,
            VarlenAttentionWrapper,
            ScaledDotProductAttentionWrapper,
        )
        
        self.attn_type = model_args.attn_type
        match self.attn_type:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "varlen":
                self.inner_attention = VarlenAttentionWrapper()
            case "sdpa":
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")
    
    def init_weights(self, init_std: float):
        """Initialize weights for quantized attention."""
        for linear in (self.wq, self.wk, self.wv):
            # Reset and re-init with proper std
            linear.reset_parameters()
        # For wo, use the provided init_std
        nn.init.trunc_normal_(self.wo.get_weight().dequantize(), mean=0.0, std=init_std)


class ECOFeedForward(FeedForward):
    """FeedForward module with quantized linear layers."""
    
    def __init__(self, dim: int, hidden_dim: int, layer_idx: int):
        nn.Module.__init__(self)
        
        self.w1 = QuantizedLinear(dim, hidden_dim, bias=False)
        self.w2 = QuantizedLinear(hidden_dim, dim, bias=False)
        self.w3 = QuantizedLinear(dim, hidden_dim, bias=False)
    
    def init_weights(self, init_std: float):
        """Initialize weights for quantized layers."""
        # Re-initialize with specific std
        nn.init.trunc_normal_(self.w1.get_weight().dequantize(), mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.get_weight().dequantize(), mean=0.0, std=init_std)


class ECOTransformerBlock(TransformerBlock):
    """Transformer block with quantized attention and feedforward."""
    
    def __init__(self, layer_id: int, model_args: TransformerModelArgs):
        nn.Module.__init__(self)
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.head_dim = model_args.dim // model_args.n_heads
        self.attention = ECOAttention(model_args, layer_id)
        
        # Compute hidden dim same as original
        hidden_dim = _compute_hidden_dim(
            model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        
        self.feed_forward = ECOFeedForward(
            dim=model_args.dim,
            hidden_dim=hidden_dim,
            layer_idx=layer_id,
        )
        self.layer_id = layer_id
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)


class ECOTransformer(Transformer):
    """ECO-compatible Transformer with quantized weights."""
    
    def __init__(self, model_args: TransformerModelArgs):
        nn.Module.__init__(self)
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        
        # Use ModuleDict like the original Transformer
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = ECOTransformerBlock(layer_id, model_args)
        
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        
        # Output layer can be quantized too
        self.output = QuantizedLinear(model_args.dim, model_args.vocab_size, bias=False)
        
        # Initialize freqs_cis like the original Transformer
        self.register_buffer(
            "freqs_cis",
            self._precompute_freqs_cis(),
            persistent=False
        )
        self._register_load_state_dict_pre_hook(self._eco_load_hook)
    
    def _precompute_freqs_cis(self) -> torch.Tensor:
        """Precompute RoPE frequencies."""
        from .model import precompute_freqs_cis
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
            self.model_args.rope_scaling_args,
        )
    
    @staticmethod
    def _eco_load_hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Hook to handle loading state dict with quantized weights."""
        # Convert regular weights to quantized format if needed
        pass
    
    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks=None,
        positions: torch.Tensor | None = None,
    ):
        """Forward pass for ECOTransformer."""
        h = self.tok_embeddings(tokens)
        
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis, attention_masks=attention_masks, positions=positions)
        
        h = self.norm(h)
        output = self.output(h)
        return output
    
    def init_weights(self, buffer_device: Optional[torch.device] = None):
        """Initialize weights with appropriate init for quantized layers."""
        for layer in self.layers.values():
            # Compute init std per layer (same as original)
            if hasattr(layer, 'weight_init_std'):
                init_std = layer.weight_init_std
            else:
                init_std = 0.02 / (2 * (layer.layer_id + 1)) ** 0.5
            
            layer.attention.init_weights(init_std)
            layer.feed_forward.init_weights(init_std)
        
        # Init embeddings
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)


__all__ = [
    "ECOTransformer",
    "ECOTransformerBlock",
    "ECOAttention",
    "ECOFeedForward",
]
