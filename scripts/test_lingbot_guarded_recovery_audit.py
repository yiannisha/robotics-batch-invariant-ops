"""Independent recovery checks must reject a forged forward authority."""
import gzip
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
import torch
from scripts.audit_lingbot_guarded_recovery import verify_record,KEYS,content_hash


class RecoveryAudit(unittest.TestCase):
    def test_known_request_and_noise_independent_identity_are_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);value={k:torch.tensor([1.]) for k in KEYS};value['noise']=torch.tensor([2.])
            path=root/(content_hash(value)+'.pt.gz');buffer=io.BytesIO();torch.save(value,buffer);path.write_bytes(gzip.compress(buffer.getvalue(),mtime=0))
            h=hashlib.sha256(path.read_bytes()).hexdigest();original='/old/requests/'+path.name
            authorities={'authority':dict(raw_files_sha256={original:h})}
            row=dict(path=str(path),sha256=h,historical_authorities=[dict(original_path=original,original_sha256=h,authority_sha256='authority')],inference_attempt_authorized=True,prepared_input_already_archived=True)
            record=verify_record(root,row,authorities);self.assertEqual(record['sha256'],h)
            row['historical_authorities'][0]['original_sha256']='forged'
            with self.assertRaises(ValueError):verify_record(root,row,authorities)
            with self.assertRaises(ValueError):verify_record(root,row,{})


if __name__=='__main__':unittest.main()
