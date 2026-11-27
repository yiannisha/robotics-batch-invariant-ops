"""Released expert failures may skip seeds; capture/reset failures must stop."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from scripts.qualify_robotwin_benchmark_replay_v2 import released_expert_rejection
from scripts.audit_robotwin_benchmark_replay_v6 import verify_expert_rejection


class ExpertRejections(unittest.TestCase):
    def test_expert_play_once_failure_is_released_rejection(self):
        self.assertTrue(released_expert_rejection({'op':'expert'},{'ok':False,'error_type':'IndexError'},'info = env.play_once()'))

    def test_instrumentation_and_reset_failures_stop(self):
        for op,fragment in [('expert','canonical=capture()'),('reset','info = env.play_once()')]:
            self.assertFalse(released_expert_rejection({'op':op},{'ok':False,'error_type':'IndexError'},fragment))

    def test_independent_rejection_audit_checks_raw_bytes_and_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);raw=b'prefix\ninfo = env.play_once()\nIndexError: list index out of range\n'
            (d/'worker-stderr.log').write_bytes(raw)
            fragment=raw[7:];response=dict(ok=False,error_type='IndexError',released_expert_seed_rejection=True,
                expert_exception_stderr_offset=7,expert_exception_stderr_length=len(fragment),
                expert_exception_stderr_sha256=hashlib.sha256(fragment).hexdigest())
            verify_expert_rejection(d,{'op':'expert'},response)
            with self.assertRaises(ValueError):verify_expert_rejection(d,{'op':'reset'},response)
            (d/'worker-stderr.log').write_bytes(raw.replace(b'IndexError',b'OtherError'))
            with self.assertRaises(ValueError):verify_expert_rejection(d,{'op':'expert'},response)

    def test_capture_exception_cannot_be_independently_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);raw=b'canonical=capture()\nValueError\n';(d/'worker-stderr.log').write_bytes(raw)
            response=dict(ok=False,error_type='ValueError',released_expert_seed_rejection=True,
                expert_exception_stderr_offset=0,expert_exception_stderr_length=len(raw),
                expert_exception_stderr_sha256=hashlib.sha256(raw).hexdigest())
            with self.assertRaises(ValueError):verify_expert_rejection(d,{'op':'expert'},response)


if __name__=='__main__':unittest.main()
