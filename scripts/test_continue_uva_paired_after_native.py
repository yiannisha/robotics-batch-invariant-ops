from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from continue_uva_paired_after_native import condition_root, condition_worker_pids


def fake_process(root: Path, pid: int, arguments: list[str], state: str = "R") -> None:
    process = root / str(pid)
    process.mkdir()
    (process / "cmdline").write_bytes(b"\0".join(item.encode() for item in arguments) + b"\0")
    (process / "stat").write_text(f"{pid} (python) {state} 0 0 0\n")


class UVAConditionWorkerTest(unittest.TestCase):
    def test_exact_path_condition_and_output_matching(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            output = str(condition_root("joint", "native_singleton"))
            base = ["python", "scripts/run_uva_paired_libero.py"]
            fake_process(proc, 101, [*base, "--path", "joint", "--condition", "native_singleton", "--output", output])
            fake_process(proc, 102, [*base, "--path", "action", "--condition", "native_singleton", "--output", output])
            fake_process(proc, 103, [*base, "--path", "joint", "--condition", "native_singleton", "--output", output + "-other"])
            fake_process(proc, 104, [*base, "--path", "joint", "--condition", "native_singleton", "--output", output], "Z")
            self.assertEqual(
                condition_worker_pids("joint", "native_singleton", proc), [101]
            )


if __name__ == "__main__":
    unittest.main()
