from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from continue_internvla_paired_after_probes import ROOT, condition_worker_pids


def fake_process(root: Path, pid: int, arguments: list[str], state: str = "R") -> None:
    process = root / str(pid)
    process.mkdir()
    (process / "cmdline").write_bytes(b"\0".join(item.encode() for item in arguments) + b"\0")
    (process / "stat").write_text(f"{pid} (python) {state} 0 0 0\n")


class InternVLAConditionWorkerTest(unittest.TestCase):
    def test_exact_client_and_server_matching(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            output = str(
                ROOT / "results/cross_model/internvla_a15_optimized/full/closed_loop/paired/native_singleton"
            )
            common = ["--backend", "optimized", "--condition", "native_singleton"]
            fake_process(proc, 101, ["python", "scripts/run_internvla_paired_client.py", *common, "--output", output])
            fake_process(proc, 102, ["python", "scripts/run_internvla_paired_server.py", *common])
            fake_process(proc, 103, ["python", "scripts/run_internvla_paired_client.py", *common, "--output", output + "-other"])
            fake_process(proc, 104, ["python", "scripts/run_internvla_paired_server.py", *common], "Z")
            self.assertEqual(
                condition_worker_pids("optimized", "native_singleton", "client", proc), [101]
            )
            self.assertEqual(
                condition_worker_pids("optimized", "native_singleton", "server", proc), [102]
            )


if __name__ == "__main__":
    unittest.main()
