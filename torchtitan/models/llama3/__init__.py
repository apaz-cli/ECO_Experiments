# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_llama
from .model.args import TransformerModelArgs
from .model.model import Transformer
from .model.state_dict_adapter import Llama3StateDictAdapter

__all__ = [
    "parallelize_llama",
    "TransformerModelArgs",
    "Transformer",
    "llama3_args",
]


llama3_args = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=6, n_heads=16, vocab_size=2048, rope_theta=500000
    ),
    # --- Scaling law models (ECO experiments, matching QUEST paper configs) ---
    # Architecture: Llama-style (SwiGLU FFN, RoPE, no bias).
    # FFN hidden dim = int(8/3 * dim) rounded to multiple_of=256 (QUEST default).
    # All have head_dim=128. Use with T5 tokenizer (vocab_size=32100).
    # QUEST peak LRs: 30M=1.2e-3, 50M=1.2e-3, 100M=6e-4, 200M=3e-4,
    #                  430M=1.5e-4, 800M=7.5e-5 (set in TOML, not here).
    "30M": TransformerModelArgs(
        dim=640, n_layers=6, n_heads=5, vocab_size=32100, rope_theta=500000
    ),
    "50M": TransformerModelArgs(
        dim=768, n_layers=7, n_heads=6, vocab_size=32100, rope_theta=500000
    ),
    "100M": TransformerModelArgs(
        dim=1024, n_layers=8, n_heads=8, vocab_size=32100, rope_theta=500000
    ),
    "200M": TransformerModelArgs(
        dim=1280, n_layers=10, n_heads=10, vocab_size=32100, rope_theta=500000
    ),
    "430M": TransformerModelArgs(
        dim=1664, n_layers=13, n_heads=13, vocab_size=32100, rope_theta=500000
    ),
    "800M": TransformerModelArgs(
        dim=2048, n_layers=16, n_heads=16, vocab_size=32100, rope_theta=500000
    ),
    "debugmodel_flex_attn": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        vocab_size=2048,
        rope_theta=500000,
        attn_type="flex",
        attn_mask_type="block_causal",
    ),
    "debugmodel_varlen_attn": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        vocab_size=2048,
        rope_theta=500000,
        attn_type="varlen",
        attn_mask_type="block_causal",
    ),
    "8B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "8B_flex": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        attn_type="flex",
        attn_mask_type="block_causal",
    ),
    "8B_varlen": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        attn_type="varlen",
        attn_mask_type="block_causal",
    ),
    "70B": TransformerModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": TransformerModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Transformer,
        model_args=llama3_args,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3StateDictAdapter,
    )
