"""Reject constructor/per-call precision drift without loading checkpoint weights."""
import unittest
from unittest.mock import patch

import torch

from scripts.collect_lingbot_robotwin_campaign_replay_v3 import CorrectedCorpusAdapter
from scripts.model_invariance.adapters.lingbot_vla2 import LingBotVLA2Adapter


class CorpusPrecision(unittest.TestCase):
    def setUp(self):
        self.previous = (torch.get_float32_matmul_precision(),
                         torch.backends.cuda.matmul.allow_tf32,
                         torch.backends.cudnn.allow_tf32,
                         torch.backends.cudnn.benchmark)

    def tearDown(self):
        precision, matmul, cudnn, benchmark = self.previous
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.backends.cudnn.benchmark = benchmark

    @staticmethod
    def disable_relaxed_modes():
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    def test_official_constructor_cannot_reenable_tf32(self):
        model = object()

        def constructor(_adapter):
            torch.set_float32_matmul_precision('high')
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            return model

        with patch.object(LingBotVLA2Adapter, 'load_model', constructor):
            adapter = CorrectedCorpusAdapter()
            self.assertIs(adapter.load_model(), model)
            adapter.require_precision()
            self.assertEqual(torch.get_float32_matmul_precision(), 'highest')

    def test_wrong_precision_is_rejected_before_inference(self):
        for backend, name in [(torch.backends.cuda.matmul, 'allow_tf32'),
                              (torch.backends.cudnn, 'allow_tf32'),
                              (torch.backends.cudnn, 'benchmark')]:
            with self.subTest(backend=backend, name=name):
                self.disable_relaxed_modes()
                setattr(backend, name, True)
                with patch.object(LingBotVLA2Adapter, 'run_model') as infer:
                    with self.assertRaisesRegex(RuntimeError, 'TF32'):
                        CorrectedCorpusAdapter().run_model(None, {})
                    infer.assert_not_called()

    def test_mid_call_precision_change_is_rejected(self):
        self.disable_relaxed_modes()

        def inference(*_args):
            torch.backends.cuda.matmul.allow_tf32 = True
            return object()

        with patch.object(LingBotVLA2Adapter, 'run_model', inference):
            with self.assertRaisesRegex(RuntimeError, 'TF32'):
                CorrectedCorpusAdapter().run_model(None, {})


if __name__ == '__main__':
    unittest.main()
