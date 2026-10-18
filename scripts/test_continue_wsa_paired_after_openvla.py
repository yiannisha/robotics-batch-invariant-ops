from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from continue_wsa_paired_after_openvla import PAIRED, condition_worker_pids


def fake_process(root: Path, pid: int, arguments: list[str], state: str = "R") -> None:
    process = root / str(pid)
    process.mkdir()
    (process / "cmdline").write_bytes(b"\0".join(item.encode() for item in arguments) + b"\0")
    (process / "stat").write_text(f"{pid} (python) {state} 0 0 0\n")


class WSAConditionWorkerTest(unittest.TestCase):
    def test_exact_condition_and_output_matching(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            output = str((PAIRED / "native_singleton").resolve())
            common = ["--condition", "native_singleton"]
            fake_process(
                proc, 101,
                ["python", "scripts/run_wsa_paired_client.py", *common, "--output", output],
            )
            fake_process(
                proc, 102,
                ["python", "scripts/run_wsa_paired_server.py", *common],
            )
            fake_process(
                proc, 103,
                ["python", "scripts/run_wsa_paired_client.py", *common, "--output", output + "-other"],
            )
            fake_process(
                proc, 104,
                ["python", "scripts/run_wsa_paired_server.py", *common], "Z",
            )
            self.assertEqual(
                condition_worker_pids("native_singleton", "client", proc), [101]
            )
            self.assertEqual(
                condition_worker_pids("native_singleton", "server", proc), [102]
            )


if __name__ == "__main__":
    unittest.main()
