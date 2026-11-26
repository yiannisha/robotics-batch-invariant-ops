"""Regression for CogACT's non-step noise label and indexed DDIM samples."""
import unittest
from unittest.mock import patch

import torch

from scripts.model_invariance.adapters.cogact_simpler_exact import CogACTSimplerExactAdapter,implementation


class CogACTTraceLabelsTest(unittest.TestCase):
    def test_noise_is_preserved_and_ten_steps_are_aliased(self):
        adapter=CogACTSimplerExactAdapter();adapter.device=torch.device('cpu')
        labels=[];traces={}
        def callback(name,value):
            labels.append(name);traces[name]=value
        adapter.trace_callback=callback
        normalized=torch.zeros(1,16,7)
        def released_trace(inner,policy,batch):
            inner._trace('diffusion.noise',normalized)
            for step in range(10):inner._trace(f'diffusion.{step}.sample',torch.cat([normalized,normalized+1]))
            inner._trace('actions.normalized',normalized)
            return normalized.double()
        with patch.object(implementation.CogACTSmallAdapter,'run_model',autospec=True,side_effect=released_trace):
            output=adapter.run_model(None,None)
        self.assertIn('diffusion.noise',labels)
        self.assertEqual([name for name in labels if name.startswith('flow.step_')],[f'flow.step_{i:02d}.sample' for i in range(10)])
        self.assertIs(adapter.trace_callback,callback)
        for i in range(10):
            value=traces[f'flow.step_{i:02d}.sample']
            self.assertEqual(list(value.shape),[1,2,16,7])
            self.assertTrue(torch.equal(value[:,0],normalized))
            self.assertTrue(torch.equal(value[:,1],normalized+1))
        self.assertEqual(output['physical'].dtype,torch.float64)
        self.assertTrue(torch.equal(output['normalized'],normalized))


if __name__=='__main__':unittest.main()
