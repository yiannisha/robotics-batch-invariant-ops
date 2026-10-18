"""Load GR00T N1.7 from its complete finetuned checkpoint without gated base I/O."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Iterator


BASE_MODEL = "nvidia/Cosmos-Reason2-2B"
LOCAL_METADATA = Path("/workspace/assets/groot_n17/processor")


@contextlib.contextmanager
def local_complete_checkpoint_construction() -> Iterator[None]:
    """Redirect nested base construction to pinned local metadata only.

    The released LIBERO checkpoint contains every backbone and action-head
    tensor.  GR00T nevertheless calls the gated base model's ``from_pretrained``
    while constructing the outer model, before the complete finetuned state is
    loaded.  This context constructs that nested module from its exact local
    config without base weights; Transformers' outer ``from_pretrained`` then
    fills every tensor from the complete LIBERO shards.
    """
    from transformers import (
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
    )
    from transformers.modeling_utils import no_init_weights

    original_model = Qwen3VLForConditionalGeneration.from_pretrained.__func__
    original_processor = Qwen3VLProcessor.from_pretrained.__func__

    def model_from_pretrained(
        cls: type, pretrained_model_name_or_path: str | Path, *args: Any, **kwargs: Any,
    ) -> Any:
        if str(pretrained_model_name_or_path) != BASE_MODEL:
            return original_model(cls, pretrained_model_name_or_path, *args, **kwargs)
        config = Qwen3VLConfig.from_pretrained(LOCAL_METADATA, local_files_only=True)
        attention = kwargs.get("attn_implementation")
        if attention is not None:
            config._attn_implementation = attention
        dtype = kwargs.get("torch_dtype") or kwargs.get("dtype")
        if dtype is not None:
            config.dtype = dtype
        with no_init_weights():
            return cls(config)

    def processor_from_pretrained(
        cls: type, pretrained_model_name_or_path: str | Path, *args: Any, **kwargs: Any,
    ) -> Any:
        redirected = str(pretrained_model_name_or_path) == BASE_MODEL
        if redirected:
            pretrained_model_name_or_path = LOCAL_METADATA
            kwargs = dict(kwargs)
            kwargs["local_files_only"] = True
        processor = original_processor(cls, pretrained_model_name_or_path, *args, **kwargs)
        if redirected and processor.chat_template is None:
            # The exact template is embedded in the pinned tokenizer_config.json.
            # Recent Transformers releases no longer copy it onto the multimodal
            # processor automatically when chat_template.json is absent.
            processor.chat_template = processor.tokenizer.chat_template
        if redirected and not processor.chat_template:
            raise RuntimeError("pinned GR00T processor metadata lacks a chat template")
        return processor

    Qwen3VLForConditionalGeneration.from_pretrained = classmethod(model_from_pretrained)
    Qwen3VLProcessor.from_pretrained = classmethod(processor_from_pretrained)
    try:
        yield
    finally:
        Qwen3VLForConditionalGeneration.from_pretrained = classmethod(original_model)
        Qwen3VLProcessor.from_pretrained = classmethod(original_processor)


def load_official_libero_policy(checkpoint: Path, device: str = "cuda") -> Any:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    with local_complete_checkpoint_construction():
        return Gr00tPolicy(
            embodiment_tag=EmbodimentTag.LIBERO_PANDA,
            model_path=str(checkpoint),
            device=device,
            strict=True,
        )
