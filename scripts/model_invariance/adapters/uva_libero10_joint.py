"""Official UVA joint video-latent/action inference adapter."""

import os
from pathlib import Path

from scripts.model_invariance.adapters.uva_libero10 import UVALibero10Adapter


class UVALibero10JointAdapter(UVALibero10Adapter):
    name = "uva-libero10-official-pytorch-joint-video-action"

    def __init__(self) -> None:
        super().__init__()
        self.replay_manifest_path = Path(os.environ.get(
            "UVA_JOINT_REPLAY_MANIFEST",
            str(Path(__file__).resolve().parents[3]
                / "results/cross_model/uva_joint/full/replay_manifest.json"),
        ))
        self.task_mode = "full_dynamic_model"
        self.include_video_output = True


adapter = UVALibero10JointAdapter()
