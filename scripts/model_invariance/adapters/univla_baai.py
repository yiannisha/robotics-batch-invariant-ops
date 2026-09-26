"""Official BAAI UniVLA PyTorch LIBERO image-policy adapter.

Required environment variables:

``UNIVLA_CHECKPOINT``
    Local ``Yuqi1997/UniVLA`` ``UNIVLA_LIBERO_IMG_BS192_8K`` directory.
``UNIVLA_VISION_TOKENIZER``
    Local ``BAAI/Emu3-VisionTokenizer`` snapshot.
``UNIVLA_SOURCE``
    Official ``baaivision/UniVLA`` checkout.
``UNIVLA_SAMPLE_DIR``
    Real LIBERO fixture containing the two camera frame directories.

This follows the released PyTorch LIBERO wrapper: both camera images are
resized to 200 x 200, encoded by Emu3-VisionTokenizer, formatted in ``VLA``
mode, greedily decoded under the FAST action-vocabulary constraint, and
unnormalized with the released LIBERO constants.  Image VQ codes are produced
one observation at a time, matching online policy use and keeping preprocessing
independent of the inference request batch.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

TASK = "put the white mug on the left plate and put the yellow and white mug " "on the right plate"
EOA_TOKEN_ID = 151845
ACTION_HIGH = np.array(
    [
        0.93712500009996,
        0.86775000009256,
        0.93712500009996,
        0.13175314309916836,
        0.19275000005139997,
        0.3353504997073735,
        0.9996000000999599,
    ]
)
ACTION_LOW = np.array(
    [
        -0.7046250000751599,
        -0.80100000008544,
        -0.9375000001,
        -0.11467779149968735,
        -0.16395000004372,
        -0.2240490058320433,
        -1.0000000001,
    ]
)


@dataclass
class _Policy:
    model: torch.nn.Module
    tokenizer: Any
    processor: Any
    action_tokenizer: Any
    generation_config: Any
    logits_processor: Any
    last_action_token_id: int

    def eval(self):
        self.model.eval()
        return self


class UniVLAAdapter:
    name = "univla-baai-libero-image-official-pytorch"

    def __init__(self) -> None:
        self.checkpoint = Path(os.environ.get("UNIVLA_CHECKPOINT", ""))
        self.vision_checkpoint = Path(os.environ.get("UNIVLA_VISION_TOKENIZER", ""))
        self.source = Path(os.environ.get("UNIVLA_SOURCE", ""))
        self.sample_dir = Path(os.environ.get("UNIVLA_SAMPLE_DIR", ""))
        self.device = torch.device("cuda")
        self.image_processor = None
        self.image_tokenizer = None
        self.trace_callback = None

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    def _validate_paths(self) -> None:
        for filename in (
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
            "emu3.tiktoken",
        ):
            if not (self.checkpoint / filename).is_file():
                raise FileNotFoundError(
                    f"set UNIVLA_CHECKPOINT to the released checkpoint containing {filename}"
                )
        for filename in (
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "modeling_emu3visionvq.py",
        ):
            if not (self.vision_checkpoint / filename).is_file():
                raise FileNotFoundError(
                    "set UNIVLA_VISION_TOKENIZER to the official snapshot " f"containing {filename}"
                )
        for filename in (
            "reference/Emu3/emu3/mllm/modeling_emu3.py",
            "pretrain/fast/processing_action_tokenizer.py",
            "pretrain/fast/tokenizer.json",
        ):
            if not (self.source / filename).is_file():
                raise FileNotFoundError(
                    f"set UNIVLA_SOURCE to the official checkout containing {filename}"
                )
        for filename in (
            "frames/image_001.png",
            "frames/image_064.png",
            "frames/image2_001.png",
            "frames/image2_064.png",
        ):
            if not (self.sample_dir / filename).is_file():
                raise FileNotFoundError(
                    f"set UNIVLA_SAMPLE_DIR to the real LIBERO fixture containing {filename}"
                )

    def load_model(self):
        self._validate_paths()
        emu_source = str(self.source / "reference" / "Emu3")
        source_root = str(self.source)
        for path in (source_root, emu_source):
            if path not in sys.path:
                sys.path.insert(0, path)

        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoProcessor,
            GenerationConfig,
            LogitsProcessor,
        )

        from emu3.mllm import Emu3MoE, Emu3Processor, Emu3Tokenizer

        class ActionIDConstraintLogitsProcessor(LogitsProcessor):
            def __init__(self, allowed_token_ids):
                self.allowed_token_ids = allowed_token_ids

            def __call__(self, input_ids, scores):
                mask = torch.zeros_like(scores, dtype=torch.bool)
                mask[:, self.allowed_token_ids] = True
                scores[~mask] = -float("inf")
                return scores

        model = (
            Emu3MoE.from_pretrained(
                self.checkpoint,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=True,
            )
            .to(self.device)
            .eval()
        )
        tokenizer = Emu3Tokenizer.from_pretrained(
            self.checkpoint,
            model_max_length=model.config.max_position_embeddings,
            padding_side="right",
            use_fast=False,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.vision_checkpoint, trust_remote_code=True
        )
        # Required by the released inference wrapper.  Without this assignment
        # the processor expands a 200px input to 512px and exceeds the policy's
        # 1,600-token context after encoding the two camera streams.
        self.image_processor.min_pixels = 80 * 80
        self.image_tokenizer = (
            AutoModel.from_pretrained(self.vision_checkpoint, trust_remote_code=True)
            .to(self.device)
            .eval()
        )
        processor = Emu3Processor(self.image_processor, self.image_tokenizer, tokenizer)
        action_tokenizer = AutoProcessor.from_pretrained(
            self.source / "pretrain" / "fast", trust_remote_code=True
        )
        last_action_token_id = tokenizer.pad_token_id - 1
        allowed_token_ids = list(
            range(
                last_action_token_id - action_tokenizer.vocab_size,
                last_action_token_id + 1,
            )
        ) + [EOA_TOKEN_ID]
        generation_config = GenerationConfig(
            pad_token_id=model.config.pad_token_id,
            bos_token_id=model.config.bos_token_id,
            eos_token_id=EOA_TOKEN_ID,
            do_sample=False,
        )
        return _Policy(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            action_tokenizer=action_tokenizer,
            generation_config=generation_config,
            logits_processor=ActionIDConstraintLogitsProcessor(allowed_token_ids),
            last_action_token_id=last_action_token_id,
        ).eval()

    def _encode_image(self, camera: str, frame_number: int) -> torch.Tensor:
        image = Image.open(self.sample_dir / "frames" / f"{camera}_{frame_number:03d}.png").convert(
            "RGB"
        )
        image = image.resize((200, 200))
        pixels = self.image_processor(image, return_tensors="pt")["pixel_values"].to(self.device)
        return self.image_tokenizer.encode(pixels).cpu()

    def load_example_inputs(self, count: int):
        if self.image_processor is None or self.image_tokenizer is None:
            raise RuntimeError("load_model must be called before load_example_inputs")
        if count > 64:
            raise ValueError("the reproducible LIBERO fixture contains 64 frames")
        target_frame = int(os.environ.get("UNIVLA_TARGET_FRAME", "0"))
        if not 0 <= target_frame < 64:
            raise ValueError("UNIVLA_TARGET_FRAME must be between 0 and 63")
        frame_indices = [target_frame]
        frame_indices.extend(index for index in range(64) if index != target_frame)

        examples = []
        self.image_tokenizer.to(self.device)
        with torch.inference_mode():
            for frame_index in frame_indices[:count]:
                frame_number = frame_index + 1
                examples.append(
                    {
                        "video_tokens": self._encode_image("image", frame_number),
                        "gripper_tokens": self._encode_image("image2", frame_number),
                    }
                )
        # VQ encoding is preprocessing, not part of the policy request batch.
        # Offload it after the fixed fixture is encoded to leave more memory for
        # large inference batches.
        self.image_tokenizer.cpu()
        torch.cuda.empty_cache()
        return examples

    @staticmethod
    def compose_batch(samples):
        return {key: torch.cat([sample[key] for sample in samples], dim=0) for key in samples[0]}

    def prepare_batch(self, policy: _Policy, batch):
        batch_size = batch["video_tokens"].shape[0]
        video_tokens = batch["video_tokens"].unsqueeze(1)
        gripper_tokens = batch["gripper_tokens"].unsqueeze(1)
        self._trace("input.video_tokens", video_tokens)
        self._trace("input.gripper_tokens", gripper_tokens)

        inputs = policy.processor.video_process(
            text=[TASK] * batch_size,
            video_tokens=video_tokens,
            gripper_tokens=gripper_tokens,
            context_frames=1,
            frames=1,
            mode="VLA",
            padding="longest",
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        self._trace("input.input_ids", input_ids)
        return input_ids, attention_mask

    def run_model(self, policy: _Policy, batch):
        batch_size = batch["video_tokens"].shape[0]
        input_ids, attention_mask = self.prepare_batch(policy, batch)

        outputs = policy.model.generate(
            input_ids,
            policy.generation_config,
            max_new_tokens=80,
            logits_processor=[policy.logits_processor],
            attention_mask=attention_mask,
        )
        generated = outputs[:, input_ids.shape[1] :]
        self._trace("generation.tokens", generated)

        actions = []
        for batch_index in range(batch_size):
            row = generated[batch_index]
            eoa_locations = torch.nonzero(row == EOA_TOKEN_ID, as_tuple=False)
            if eoa_locations.numel():
                row = row[: int(eoa_locations[0].item())]
            processed = policy.last_action_token_id - row
            decoded = policy.action_tokenizer.decode(
                processed.unsqueeze(0).cpu(), time_horizon=10, action_dim=7
            )[0]
            action = 0.5 * (decoded + 1) * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW
            action[..., -1] = np.where(action[..., -1] > 0, 1, -1)
            actions.append(torch.from_numpy(action.copy()))

        result = torch.stack(actions)
        self._trace("actions.unnormalized", result)
        return result

    @staticmethod
    def extract_robot_output(output: Any):
        return output


adapter = UniVLAAdapter()
