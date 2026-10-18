"""Full UVA joint video/action adapter with deterministic video-noise replay."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.model_invariance.adapters.uva_libero10_full import UVAFullActionAdapter


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class UVAFullJointAdapter(UVAFullActionAdapter):
    name = "uva-libero10-official-pytorch-joint-full-numerical"

    def __init__(self) -> None:
        super().__init__()
        self.replay_manifest_path = Path(os.environ.get(
            "UVA_JOINT_REPLAY_MANIFEST",
            str(Path(__file__).resolve().parents[3]
                / "results/cross_model/uva_joint/full/replay_manifest.json"),
        ))
        self.task_mode = "full_dynamic_model"
        self.include_video_output = True
        self.action_noise = None

    def load_replay_input(self, replay_index: int, noise_id: int = 0):
        if self.config is None or self.num_diffusion_steps is None:
            raise RuntimeError("load_model must be called before load_replay_input")
        if self.replay_manifest is None:
            self.replay_manifest = json.loads(self.replay_manifest_path.read_text())
            contract = self.replay_manifest["noise_regeneration_contract"]
            if contract["torch_version"] != torch.__version__:
                raise RuntimeError("UVA joint video-noise generator uses another torch version")
            self.action_noise = np.load(
                self.replay_manifest["stored_action_noise_file"], mmap_mode="r"
            )
        rows = self.replay_manifest["observations"]
        if not 0 <= replay_index < len(rows) or not 0 <= noise_id < 3:
            raise IndexError((replay_index, noise_id))
        row = rows[replay_index]
        record = row["noise"][noise_id]
        if row["replay_index"] != replay_index or record["noise_id"] != noise_id:
            raise RuntimeError("UVA joint replay request identity differs from frozen index")
        flat = self.action_noise[replay_index, noise_id]
        result = {
            "images": self.load_real_context(replay_index).unsqueeze(0),
            **self._tokenize(row["task"]),
            "context_index": torch.tensor([[replay_index]], dtype=torch.int64),
        }
        for name in ("vae_noise", "action_initial_noise", "action_step_noise"):
            specification = self.replay_manifest["noise_layout"][name]
            shape = tuple(specification["shape"])
            start = specification["offset"]
            width = int(np.prod(shape))
            result[name] = torch.from_numpy(
                np.array(flat[start : start + width], copy=True).reshape((1, *shape))
            )
        contract_hash = self.replay_manifest["noise_regeneration_contract_sha256"]
        video_contract = {
            "contract_sha256": contract_hash,
            "replay_index": replay_index, "noise_id": noise_id,
            "seed": record["video_seed"],
        }
        if canonical_sha256(video_contract) != record["video_contract_sha256"]:
            raise RuntimeError("UVA joint video-noise contract hash differs")
        generator = torch.Generator(device="cpu").manual_seed(record["video_seed"])
        result["video_initial_noise"] = torch.randn(
            1, 1024, 16, dtype=torch.float32, generator=generator,
        )
        result["video_step_noise"] = torch.randn(
            1, self.num_video_diffusion_steps, 1024, 16,
            dtype=torch.float32, generator=generator,
        )
        return result

    def _explicit_video_sample(self, policy, z: torch.Tensor, batch: dict[str, torch.Tensor]):
        diffusion_head = policy.model.diffloss
        batch_size = batch["images"].shape[0]
        sequence_length = z.shape[0] // batch_size
        if sequence_length != 1024:
            raise ValueError(f"expected 1024 UVA video tokens, got {sequence_length}")
        sample = (
            batch["video_initial_noise"].to(self.device)
            .reshape(batch_size * sequence_length, diffusion_head.in_channels)
        )
        step_noises = batch["video_step_noise"].to(self.device)
        diffusion = diffusion_head.gen_diffusion
        temperature = float(self.config.model.policy.autoregressive_model_params.temperature)
        for step_index, timestep in enumerate(range(diffusion.num_timesteps - 1, -1, -1)):
            timesteps = torch.full(
                (sample.shape[0],), timestep, dtype=torch.long, device=self.device
            )
            result = diffusion.p_mean_variance(
                diffusion_head.net.forward, sample, timesteps, clip_denoised=False,
                model_kwargs={"c": z},
            )
            noise = step_noises[:, step_index].reshape_as(sample)
            nonzero = (timesteps != 0).float().view(-1, 1)
            sample = result["mean"] + (
                nonzero * torch.exp(0.5 * result["log_variance"]) * noise * temperature
            )
            self._trace(
                f"video.diffusion.step_{step_index:03d}",
                sample.reshape(batch_size, sequence_length, -1),
            )
        return sample


adapter = UVAFullJointAdapter()
