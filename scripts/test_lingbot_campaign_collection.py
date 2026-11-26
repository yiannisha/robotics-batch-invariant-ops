import gzip
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.collect_lingbot_robotwin_campaign_replay import content_digest,frame_indices,retain
from scripts.qualify_simpler_fixed_action_replay import digest


class LingBotCollectionTests(unittest.TestCase):
    def test_selection_is_distinct_and_keeps_short_episodes(self):
        self.assertEqual(frame_indices(400,'diagnostic'),[0,399])
        selected=frame_indices(400,'heldout');self.assertEqual(len(selected),8)
        self.assertEqual(len(set(selected)),8);self.assertEqual(selected[0],1)
        self.assertEqual(frame_indices(3,'heldout'),[1,2])
        self.assertEqual(frame_indices(1,'heldout'),[0])
        with self.assertRaises(ValueError):frame_indices(0,'heldout')

    def test_compressed_raw_retention_preserves_original_dtypes_and_bits(self):
        value=dict(image=np.arange(24,dtype=np.uint8).reshape(2,4,3),
            state=torch.tensor([0.,-0.,1.],dtype=torch.float32))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);relative=retain(root,'frames',value)
            with gzip.open(root/relative,'rb') as stream:saved=torch.load(stream,map_location='cpu',weights_only=False)
            self.assertEqual(content_digest(saved),content_digest(value))
            self.assertEqual(Path(relative).name.split('.')[0],content_digest(value))
            self.assertEqual(saved['state'].view(torch.uint8).tolist(),value['state'].view(torch.uint8).tolist())
            self.assertEqual(retain(root,'frames',value),relative)

    def test_numpy_worker_frame_hash_remains_compatible(self):
        value=dict(observation={'rgb':np.arange(12,dtype=np.uint8)},rng={'cpu':np.arange(8,dtype=np.uint8)})
        self.assertEqual(content_digest(value),digest(value))
        self.assertNotEqual(content_digest(torch.tensor([1.],dtype=torch.float32)),content_digest(torch.tensor([1.],dtype=torch.float64)))


if __name__=='__main__':unittest.main()
