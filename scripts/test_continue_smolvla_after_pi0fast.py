from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from continue_smolvla_after_pi0fast import PAIRED, condition_worker_pids


def fake_process(root: Path, pid: int, arguments: list[str], state: str = "R") -> None:
    process = root / str(pid)
    process.mkdir()
    (process / "cmdline").write_bytes(b"\0".join(item.encode() for item in arguments) + b"\0")
    (process / "stat").write_text(f"{pid} (python) {state} 0 0 0\n")


class ConditionWorkerPidsTest(unittest.TestCase):
    def test_matches_only_live_exact_condition_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            script = "scripts/run_lerobot_smolvla_paired_libero.py"
            output = str(PAIRED / "patched_singleton")
            fake_process(proc, 101, ["python", script, "--condition", "patched_singleton", "--output", output])
            fake_process(proc, 102, ["python", script, "--condition", "native_singleton", "--output", output])
            fake_process(proc, 103, ["python", script, "--condition", "patched_singleton", "--output", output], "Z")
            fake_process(proc, 104, ["python", script, "--condition", "patched_singleton", "--output", output + "-other"])
            fake_process(proc, 105, ["python", "unrelated.py", "--condition", "patched_singleton", "--output", output])

            self.assertEqual(condition_worker_pids("patched_singleton", proc), [101])


if __name__ == "__main__":
    unittest.main()
