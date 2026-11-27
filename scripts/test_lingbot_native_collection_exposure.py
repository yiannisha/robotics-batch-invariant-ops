"""Freshness must include real source-policy calls, excluding request noise."""
import gzip
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import torch
from scripts.audit_lingbot_heldout_observation_disjointness_v2 import KEYS,content_hash,native_collection_exposure


class NativeCollectionExposure(unittest.TestCase):
    def fixture(self,root):
        bank=root/'requests';bank.mkdir();files={}
        for noise in [0.,1.]:
            value={k:torch.tensor([2.]) for k in KEYS};value['noise']=torch.tensor([noise])
            path=bank/(content_hash(value)+'.pt.gz');stream=io.BytesIO();torch.save(value,stream)
            path.write_bytes(gzip.compress(stream.getvalue(),mtime=0));files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        audit=dict(status='native_corpus_completed_episode_inputs_and_actions_independently_rebuilt',
                   completed_native_episodes=200,all_prescribed_native_episodes_rebuilt=True,raw_files_sha256=files)
        p=root/'audit.json';p.write_text(json.dumps(audit))
        return dict(fixture_root=str(root),corpus_audit_path=str(p),corpus_audit_sha256=hashlib.sha256(p.read_bytes()).hexdigest())

    def test_source_policy_calls_count_as_one_input_for_two_noises(self):
        with tempfile.TemporaryDirectory() as temp:
            exposed,files=native_collection_exposure(self.fixture(Path(temp)))
            self.assertEqual((len(exposed),len(files)),(1,2))

    def test_unrecorded_source_request_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);m=self.fixture(root);(root/'requests/extra.pt.gz').write_bytes(b'extra')
            with self.assertRaises(ValueError):native_collection_exposure(m)

    def test_changed_source_request_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);m=self.fixture(root);next((root/'requests').iterdir()).write_bytes(b'changed')
            with self.assertRaises(ValueError):native_collection_exposure(m)

    def test_partial_source_audit_cannot_seal_freshness(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);m=self.fixture(root);p=Path(m['corpus_audit_path']);a=json.loads(p.read_text())
            a.update(completed_native_episodes=64,all_prescribed_native_episodes_rebuilt=False);p.write_text(json.dumps(a));m['corpus_audit_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):native_collection_exposure(m)


if __name__=='__main__':unittest.main()
