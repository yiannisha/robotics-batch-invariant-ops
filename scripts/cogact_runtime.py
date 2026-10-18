"""Runtime helpers for memory-safe exact CogACT-Small construction."""

from __future__ import annotations

from typing import Any

import torch


def load_model_with_bf16_llama_construction(adapter: Any) -> Any:
    """Build the empty Llama directly in its documented deployment dtype.

    The released CogACT loader constructs a full FP32 Llama-2 7B and then,
    after loading the checkpoint, converts ``model.vlm`` to BF16. Loading FP32
    checkpoint tensors into a BF16 parameter has the same rounding result as
    the immediate post-load ``.to(torch.bfloat16)`` conversion, but avoids the
    transient extra ~14 GB FP32 allocation that exceeds this container's
    cgroup limit while the other evaluators are resident.
    """

    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    from transformers.modeling_utils import no_init_weights

    had_local_descriptor = "_from_config" in LlamaForCausalLM.__dict__
    local_descriptor = LlamaForCausalLM.__dict__.get("_from_config")
    original = LlamaForCausalLM._from_config

    @classmethod
    def bf16_from_config(cls, config, **kwargs):
        kwargs["torch_dtype"] = torch.bfloat16
        # Every learned tensor is supplied by the strict CogACT checkpoint
        # load immediately after construction. Avoid spending CPU time filling
        # those soon-to-be-overwritten tensors with random values.
        with no_init_weights():
            return original(config, **kwargs)

    LlamaForCausalLM._from_config = bf16_from_config
    try:
        return adapter.load_model()
    finally:
        if had_local_descriptor:
            LlamaForCausalLM._from_config = local_descriptor
        else:
            delattr(LlamaForCausalLM, "_from_config")
