"""LingBot exact adapter with precision enforced after official construction.

The released constructor selects matmul precision 'high', which re-enables
TF32. Preserve the original adapter and its rejected diagnostic provenance.
"""
import torch
from scripts.model_invariance.adapters.lingbot_robotwin_exact import LingBotRobotwinExactAdapter


class LingBotRobotwinExactAdapterV2(LingBotRobotwinExactAdapter):
    def load_model(self):
        model = super().load_model()
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self._require_precision()
        return model

    @staticmethod
    def _require_precision():
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise RuntimeError('LingBot exact inference requires TF32 disabled')

    def run_model(self, model, batch):
        self._require_precision()
        result = super().run_model(model, batch)
        self._require_precision()
        return result


adapter = LingBotRobotwinExactAdapterV2()
