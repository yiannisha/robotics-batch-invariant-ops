"""Full numerical-campaign UVA action adapter.

The completed closed-loop campaign remains tied to ``uva_libero10.py``.  This
subclass preserves that inference implementation while exposing every one of
the 100 action-diffusion states required by the full numerical audit.
"""

from __future__ import annotations

import torch
from einops import rearrange

from scripts.model_invariance.adapters.uva_libero10 import UVALibero10Adapter


class UVAFullActionAdapter(UVALibero10Adapter):
    name = "uva-libero10-official-pytorch-full-numerical"
    causal_ablation_inapplicable_operations = {
        "explicit_sdpa": {
            "reason": (
                "The pinned Transformers 4.28 CLIP implementation expresses attention "
                "as explicit torch.bmm calls and never dispatches fused SDPA."
            ),
            "evidence_path": (
                "/workspace/envs/uva_overlay/transformers/models/clip/modeling_clip.py"
            ),
        },
    }

    def _explicit_action_sample(self, policy, z: torch.Tensor, batch: dict[str, torch.Tensor]):
        diffusion_head = policy.model.diffactloss
        if diffusion_head.act_model_type != "conv_ori":
            raise ValueError(f"unsupported UVA action head: {diffusion_head.act_model_type}")

        features = rearrange(z, "b (t s) c -> b t s c", t=diffusion_head.n_frames)
        features = rearrange(features, "b t (w h) c -> b c t w h", w=diffusion_head.w)
        self._trace("action.conv_transpose3d.input", features)
        features = diffusion_head.conv_transpose3d(features)
        self._trace("action.conv_transpose3d.output", features)
        features = diffusion_head.avg_pool(features)
        features = rearrange(features, "b c t w h -> b (t w h) c")
        self._trace("action.conditioning", features)
        batch_size, sequence_length, _ = features.shape
        conditioning = features.reshape(batch_size * sequence_length, -1)

        sample = (
            batch["action_initial_noise"]
            .to(self.device)
            .reshape(batch_size * sequence_length, diffusion_head.in_channels)
        )
        step_noises = batch["action_step_noise"].to(self.device)
        self._trace("action.diffusion.initial", sample.reshape(batch_size, sequence_length, -1))
        diffusion = diffusion_head.gen_diffusion
        temperature = float(self.config.model.policy.autoregressive_model_params.temperature)
        for step_index, timestep in enumerate(range(diffusion.num_timesteps - 1, -1, -1)):
            timesteps = torch.full(
                (sample.shape[0],), timestep, dtype=torch.long, device=self.device
            )
            result = diffusion.p_mean_variance(
                diffusion_head.net.forward,
                sample,
                timesteps,
                clip_denoised=True,
                model_kwargs={"c": conditioning},
            )
            noise = step_noises[:, step_index].reshape_as(sample)
            nonzero = (timesteps != 0).float().view(-1, 1)
            sample = result["mean"] + (
                nonzero * torch.exp(0.5 * result["log_variance"]) * noise * temperature
            )
            self._trace(
                f"action.diffusion.step_{step_index:03d}",
                sample.reshape(batch_size, sequence_length, -1),
            )
        return sample.reshape(batch_size, sequence_length, -1)


adapter = UVAFullActionAdapter()
