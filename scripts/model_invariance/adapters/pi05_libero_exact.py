"""Fine-tuned LeRobot PI05 LIBERO adapter over the audited native-sim corpus."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.verify_pi05_libero_checkpoint import load_verified_policy


class PI05LiberoExactAdapter:
    name = "pi05-libero-finetuned-v044-official-sampler"

    def __init__(self):
        self.checkpoint = Path(os.environ.get("PI05_LIBERO_CHECKPOINT", ""))
        self.tokenizer = Path(os.environ.get("PI05_TEXT_TOKENIZER", ""))
        self.replay_manifest_path = Path(os.environ.get("PI05_LIBERO_REPLAY_MANIFEST", ""))
        self.manifest = None
        self.noise = None
        self.trace_callback = None
        self.postprocessor = None
        self.config = None
        self.device = torch.device("cuda")

    def _trace(self, label, tensor):
        if self.trace_callback is not None:
            self.trace_callback(label, tensor)

    def load_model(self):
        from lerobot.policies.factory import make_pre_post_processors

        if not self.replay_manifest_path.is_file():
            raise FileNotFoundError(self.replay_manifest_path)
        self.manifest = json.loads(self.replay_manifest_path.read_text())
        if self.manifest.get("model_id") != "pi05_libero" or len(self.manifest["observations"]) != 1000:
            raise ValueError("expected complete fine-tuned PI05 LIBERO corpus")
        audit = Path(self.manifest["corpus_audit_path"])
        if hashlib.sha256(audit.read_bytes()).hexdigest() != self.manifest["corpus_audit_sha256"]:
            raise ValueError("corpus audit changed")
        if json.loads(audit.read_text()).get("status") != "complete_corpus_independently_verified":
            raise ValueError("corpus has not passed its independent audit")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        policy, self.loading_report = load_verified_policy(self.checkpoint)
        self.config = policy.config
        _, self.postprocessor = make_pre_post_processors(
            self.config, pretrained_path=self.checkpoint,
            preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": str(self.tokenizer)},
                                    "device_processor": {"device": "cpu"}})
        original = policy.model.denoise_step
        self.flow_index = 0

        def traced_step(*args, **kwargs):
            result = original(*args, **kwargs)
            if self.trace_callback is not None:
                if "x_t" in kwargs:
                    self._trace(f"flow.step_{self.flow_index:02d}.state", kwargs["x_t"])
                self._trace(f"flow.step_{self.flow_index:02d}.velocity", result)
            self.flow_index += 1
            return result

        policy.model.denoise_step = traced_step
        self.noise = np.load(self.manifest["noise_file"], mmap_mode="r")
        if self.noise.shape != (1000, 3, 50, 32):
            raise ValueError("noise geometry differs")
        return policy.to(self.device).eval()

    @lru_cache(maxsize=64)
    def _inputs(self, index):
        row = self.manifest["observations"][index]
        path = Path(self.manifest["fixture_root"]) / row["input_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["input_file_sha256"]:
            raise ValueError(f"input bytes changed: {path}")
        return torch.load(path, map_location="cpu", weights_only=True)

    def load_replay_input(self, replay_index, noise_id=0):
        if self.manifest is None:
            raise RuntimeError("load_model must precede loading requests")
        if not 0 <= replay_index < 1000 or noise_id not in (0, 1, 2):
            raise IndexError((replay_index, noise_id))
        value = np.array(self.noise[replay_index, noise_id], copy=True)
        row = self.manifest["observations"][replay_index]
        if hashlib.sha256(value.tobytes()).hexdigest() != row["noise"][noise_id]["sha256"]:
            raise ValueError("request-local noise changed")
        return {"inputs": self._inputs(replay_index), "noise": torch.from_numpy(value).unsqueeze(0)}

    def load_example_inputs(self, count):
        rows = [row for row in self.manifest["observations"] if row["split"] == "diagnostic"]
        # Cover unrelated tasks/episodes before taking later frames. No held-out
        # input is used for operator selection, capacity or timing qualification.
        rows.sort(key=lambda row: (row["observation_index"], row["episode_id"], row["task"]))
        if count > len(rows):
            raise ValueError("too many requested diagnostic examples")
        return [self.load_replay_input(row["replay_index"], 0) for row in rows[:count]]

    @staticmethod
    def compose_batch(samples):
        return {"inputs": {key: torch.cat([sample["inputs"][key] for sample in samples])
                           for key in samples[0]["inputs"]},
                "noise": torch.cat([sample["noise"] for sample in samples])}

    compose_fixed_batch = compose_batch

    @staticmethod
    def reset_model(policy):
        policy.reset()

    def run_model(self, policy, batch):
        self.flow_index = 0
        self._trace("input.noise", batch["noise"].to(self.device))
        normalized = policy.predict_action_chunk(
            {key: value.to(self.device) for key, value in batch["inputs"].items()},
            noise=batch["noise"].to(self.device).clone())
        self._trace("actions.normalized", normalized)
        actions = self.postprocessor(normalized.detach().cpu())
        self._trace("actions.physical", actions)
        return {"normalized": normalized, "physical": actions.to(self.device)}

    @staticmethod
    def extract_robot_output(output):
        return output

    def execution_configuration(self):
        return {"checkpoint_revision": "7152f13e34c6c115d808aea0e5d08aca74eac14b",
                "suite": "libero_10", "language_padding": "max_length_right",
                "language_tokens": self.config.tokenizer_max_length,
                "sampler": "unchanged_official_predict_action_chunk",
                "flow_steps": self.config.num_inference_steps,
                "chunk_size": self.config.chunk_size, "max_action_dim": self.config.max_action_dim,
                "robot_action_dim": 7, "normalization": "checkpoint_mean_std",
                "precision": self.config.dtype, "tf32": False, "execution": "eager"}

    def fixed_batch_shape_metadata(self, batch):
        return {"language_padding": "max_length_right", "language_tokens": self.config.tokenizer_max_length,
                "canonical_length": self.config.tokenizer_max_length}


adapter = PI05LiberoExactAdapter()
