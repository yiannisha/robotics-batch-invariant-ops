"""OpenWAM Alpha RoboTwin adapter exposing released decoded video pixels."""

from __future__ import annotations

from scripts.model_invariance.adapters.openwam_alpha_robotwin import (
    OpenWAMAlphaRobotwinAdapter,
)


class OpenWAMAlphaRobotwinVideoAdapter(OpenWAMAlphaRobotwinAdapter):
    name = "openwam-alpha-robotwin-official-pytorch-decoded-video-action"

    def run_model(self, model, batch):
        output = super().run_model(model, batch)
        return {
            "actions": output["actions"],
            "video": self.decode_video_latents(model, output["video_latent"]),
        }


adapter = OpenWAMAlphaRobotwinVideoAdapter()
