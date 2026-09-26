from .batch_invariant_ops import (
    set_batch_invariant_mode,
    is_batch_invariant_mode_enabled,
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
    matmul_persistent,
    bmm_persistent,
    conv1d_batch_invariant,
    conv2d_batch_invariant,
    conv3d_batch_invariant,
    scaled_dot_product_attention_batch_invariant,
    softmax,
    log_softmax,
    mean_dim,
    get_batch_invariant_attention_block_size,
    AttentionBlockSize,
)

__version__ = "0.1.0"

__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
    "matmul_persistent",
    "bmm_persistent",
    "conv1d_batch_invariant",
    "conv2d_batch_invariant",
    "conv3d_batch_invariant",
    "scaled_dot_product_attention_batch_invariant",
    "softmax",
    "log_softmax",
    "mean_dim",
    "get_batch_invariant_attention_block_size",
    "AttentionBlockSize",
]
