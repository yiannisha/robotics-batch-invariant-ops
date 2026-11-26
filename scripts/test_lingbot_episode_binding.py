"""Episode-byte provenance controls; no model construction/inference."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_lingbot_paired_study_bound import read_bound_episode, verify_episode_bindings


class EpisodeBinding(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'episode.json.gz'
        self.payload = gzip.compress(json.dumps({'calls': [], 'success': False}).encode(), mtime=0)
        self.path.write_bytes(self.payload)

    def test_binds_exact_compressed_bytes_read(self):
        hashes = {}
        self.assertFalse(read_bound_episode(self.path, hashes)['success'])
        self.assertEqual(hashes[str(self.path.resolve())], hashlib.sha256(self.payload).hexdigest())
        verify_episode_bindings(hashes)

    def test_changed_episode_rejected(self):
        hashes = {}
        read_bound_episode(self.path, hashes)
        self.path.write_bytes(gzip.compress(json.dumps({'calls': [], 'success': True}).encode(), mtime=0))
        with self.assertRaisesRegex(ValueError, 'Episode bytes changed'):
            verify_episode_bindings(hashes)

    def test_recompression_with_equal_json_is_still_detected(self):
        hashes = {}
        read_bound_episode(self.path, hashes)
        self.path.write_bytes(gzip.compress(json.dumps({'calls': [], 'success': False}).encode(), mtime=1))
        with self.assertRaisesRegex(ValueError, 'Episode bytes changed'):
            verify_episode_bindings(hashes)

    def test_duplicate_binding_rejected(self):
        hashes = {}
        read_bound_episode(self.path, hashes)
        with self.assertRaisesRegex(ValueError, 'Duplicate episode'):
            read_bound_episode(self.path, hashes)

    def test_empty_binding_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing episode'):
            verify_episode_bindings({})


if __name__ == '__main__':
    unittest.main()
