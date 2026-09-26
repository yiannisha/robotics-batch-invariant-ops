"""Official OpenDW DW05 joint world-video/action PyTorch adapter."""

from __future__ import annotations

import numpy as np
import torch

from scripts.model_invariance.adapters.opendw_dw05 import OpenDWActionAdapter


class OpenDWJointAdapter(OpenDWActionAdapter):
    name = "opendw-dw05-robotwin-official-pytorch-joint-video-action"
    keep_vae = True
    num_video_frames = 9

    def load_example_inputs(self, count: int):
        examples = super().load_example_inputs(count)
        latent_t = (self.num_video_frames - 1) // 4 + 1
        for example in examples:
            generator = torch.Generator().manual_seed(int(example["noise_seed"]))
            example["video_noise"] = torch.randn(
                48, latent_t, 24, 20, dtype=torch.float32, generator=generator
            )
        return examples

    def _decode_video(self, model, latents: torch.Tensor) -> torch.Tensor:
        video = model.vae.decode(latents, device=self.device, tiled=False)
        video = video.detach().float().clamp(-1, 1)
        return ((video + 1.0) * 127.5).to(torch.uint8).cpu()

    def run_model(self, model, batch):
        first_frame = batch["first_frame"].to(self.device)
        context = batch["context"].to(self.device)
        context_mask = batch["context_mask"].to(self.device)
        latents_video = batch["video_noise"].to(self.device, dtype=torch.bfloat16)
        latents_action = batch["action_noise"].to(self.device, dtype=torch.bfloat16)
        batch_size = latents_action.shape[0]
        latents_video[:, :, 0:1] = first_frame

        self._trace("preprocess.first_frame", batch["first_frame"])
        self._trace("preprocess.context", batch["context"])
        self._trace("input.video_noise", batch["video_noise"])
        self._trace("input.action_noise", batch["action_noise"])

        video_steps, video_deltas = model.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=5,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=None,
        )
        action_steps, action_deltas = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=5,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=None,
        )
        fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
        for step, (video_t, video_delta, action_t, action_delta) in enumerate(
            zip(video_steps, video_deltas, action_steps, action_deltas)
        ):
            timestep_video = video_t.expand(batch_size).to(
                dtype=latents_video.dtype, device=self.device
            )
            timestep_action = action_t.expand(batch_size).to(
                dtype=latents_action.dtype, device=self.device
            )
            pred_video, pred_action = model._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=None,
            )
            self._trace(f"joint.step_{step:02d}.video_velocity", pred_video)
            self._trace(f"joint.step_{step:02d}.action_velocity", pred_action)
            latents_video = model.infer_video_scheduler.step(pred_video, video_delta, latents_video)
            latents_action = model.infer_action_scheduler.step(
                pred_action, action_delta, latents_action
            )
            latents_video[:, :, 0:1] = first_frame
            self._trace(f"joint.step_{step:02d}.video_state", latents_video)
            self._trace(f"joint.step_{step:02d}.action_state", latents_action)

        video = self._decode_video(model, latents_video)
        action = self._denormalize(latents_action)
        self._trace("video.uint8", video)
        self._trace("actions.unnormalized", action)
        return {"video": video, "action": action}

    def run_official_model(self, model, batch):
        if batch["raw_image"].shape[0] != 1:
            raise ValueError("released DW05 infer_joint only supports B=1")
        output = model.infer_joint(
            prompt=self.prompt,
            input_image=batch["raw_image"].to(self.device),
            num_video_frames=self.num_video_frames,
            action_horizon=32,
            proprio=batch["proprio"].to(self.device),
            num_inference_steps=5,
            seed=int(batch["noise_seed"][0]),
            rand_device="cpu",
            tiled=False,
            test_action_with_infer_action=False,
        )
        frames = [
            torch.from_numpy(np.asarray(frame).copy()).permute(2, 0, 1) for frame in output["video"]
        ]
        video = torch.stack(frames, dim=1).unsqueeze(0)
        action = self._denormalize(output["action"].unsqueeze(0))
        return {"video": video, "action": action}


adapter = OpenDWJointAdapter()
