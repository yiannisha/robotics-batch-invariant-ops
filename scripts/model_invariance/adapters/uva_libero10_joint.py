"""Official UVA joint video-latent/action inference adapter."""

from scripts.model_invariance.adapters.uva_libero10 import UVALibero10Adapter


class UVALibero10JointAdapter(UVALibero10Adapter):
    name = "uva-libero10-official-pytorch-joint-video-action"

    def __init__(self) -> None:
        super().__init__()
        self.task_mode = "full_dynamic_model"
        self.include_video_output = True


adapter = UVALibero10JointAdapter()
