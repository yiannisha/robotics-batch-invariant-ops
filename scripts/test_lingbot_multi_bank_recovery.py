"""Historical recovery must deduplicate matching bytes and reject mismatches."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from scripts.recover_lingbot_historical_native_request_matches_v2 import retain_matches


class MultiBankRecovery(unittest.TestCase):
    def test_duplicate_sources_retain_one_original_container(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);sources=[]
            for name in ['one','two']:
                bank=root/name;bank.mkdir();p=bank/'same.pt.gz';p.write_bytes(b'original');sources.append(p)
            expected=hashlib.sha256(b'original').hexdigest()
            rows=retain_matches({'same.pt.gz':('/original/requests/same.pt.gz',expected)},sources,root/'retained')
            self.assertEqual(len(rows),1);self.assertEqual(len(rows[0]['source_paths']),2)
            self.assertEqual((root/'retained/same.pt.gz').read_bytes(),b'original')

    def test_matching_name_with_wrong_original_hash_is_not_recovered(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=root/'same.pt.gz';p.write_bytes(b'changed')
            rows=retain_matches({'same.pt.gz':('/original/requests/same.pt.gz',hashlib.sha256(b'original').hexdigest())},[p],root/'retained')
            self.assertEqual(rows,[]);self.assertFalse((root/'retained/same.pt.gz').exists())


if __name__=='__main__':unittest.main()
