"""Recovery must not execute a forward on any unarchived input bytes."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from scripts.lingbot_historical_forward_guard import authorize,record_before_forward,load_authorities,UnarchivedRecoveryInput


class HistoricalForwardGuard(unittest.TestCase):
    def test_exact_archived_container_is_authorized_and_durably_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'known.pt.gz';path.write_bytes(b'archived container')
            h=hashlib.sha256(path.read_bytes()).hexdigest();known={path.name:[dict(original_path='original',original_sha256=h,authority_sha256='authority')]}
            matches=authorize(path,known);self.assertEqual(matches,known[path.name])
            ledger=root/'ledger.jsonl';record_before_forward(ledger,path,known,'request-0')
            record=json.loads(ledger.read_text());self.assertTrue(record['prepared_input_already_archived']);self.assertEqual(record['sha256'],h)

    def test_wrong_name_or_changed_bytes_never_create_authorization_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'known.pt.gz';path.write_bytes(b'changed')
            known={path.name:[dict(original_sha256=hashlib.sha256(b'original').hexdigest())]}
            with self.assertRaises(UnarchivedRecoveryInput):record_before_forward(root/'ledger.jsonl',path,known,'request')
            self.assertFalse((root/'ledger.jsonl').exists())
            with self.assertRaises(UnarchivedRecoveryInput):authorize(path,{})

    def test_unknown_authority_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'authority.json';path.write_text('{}')
            with self.assertRaises(ValueError):load_authorities([path])

    def test_actual_retained_authorities_bind_original_request_rosters(self):
        root=Path(__file__).resolve().parents[1]/'results/four_model/current_evidence'
        paths=[root/'lingbot_full200_native_corpus_independent_audit.json',root/'lingbot_native_corpus_partial_audit.json']
        known,bindings=load_authorities(paths)
        self.assertEqual(len(bindings),2);self.assertGreaterEqual(len(known),1077)
        self.assertEqual(sum(len(rows) for rows in known.values()),1077+82)


if __name__=='__main__':unittest.main()
