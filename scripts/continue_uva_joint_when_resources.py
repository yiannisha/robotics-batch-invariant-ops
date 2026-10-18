"""Run the UVA joint paired campaign once enough GPU memory is available."""

from __future__ import annotations

import argparse
import fcntl
import subprocess
import time
from pathlib import Path

from continue_uva_paired_after_native import (
    CONDITIONS,
    condition_complete,
    finalize_path,
    join_external_condition,
    path_lock,
    run_condition_resumable,
)


CGROUP_MEMORY = Path("/sys/fs/cgroup/memory.current")
INTERNVLA_OPTIMIZED_MANIFEST = Path(
    "results/cross_model/internvla_a15_optimized/full/closed_loop/paired/"
    "native_singleton/condition_manifest.json"
)
WSA_SUPERVISOR_LOCK = Path(
    "results/cross_model/wsa_base/full/closed_loop/paired/.supervisor.lock"
)
INTERNVLA_OPTIMIZED_LOCK = Path(
    "results/cross_model/internvla_a15_optimized/full/closed_loop/paired/.supervisor.lock"
)
HIGHER_PRIORITY_LOCKS = (
    ("WSA", WSA_SUPERVISOR_LOCK),
    ("InternVLA optimized", INTERNVLA_OPTIMIZED_LOCK),
)


def gpu_free_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def lock_active(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-gpu-free-mib", type=int, default=8000)
    parser.add_argument("--maximum-cgroup-bytes", type=int, default=150_000_000_000)
    args = parser.parse_args()
    with path_lock("joint"):
        while not INTERNVLA_OPTIMIZED_MANIFEST.is_file():
            print(
                "UVA joint waiting for higher-priority InternVLA optimized launch",
                flush=True,
            )
            time.sleep(60)
        while True:
            active = next(
                (name for name, lock in HIGHER_PRIORITY_LOCKS if lock_active(lock)),
                None,
            )
            if active is None:
                break
            print(f"UVA joint waiting for higher-priority {active} campaign", flush=True)
            time.sleep(60)
        while True:
            free = gpu_free_mib()
            memory = int(CGROUP_MEMORY.read_text())
            if free >= args.minimum_gpu_free_mib and memory <= args.maximum_cgroup_bytes:
                break
            print(
                f"UVA joint waiting: gpu_free_mib={free}/{args.minimum_gpu_free_mib} "
                f"cgroup_memory_bytes={memory}/{args.maximum_cgroup_bytes}",
                flush=True,
            )
            time.sleep(60)
        for condition in CONDITIONS:
            if condition_complete("joint", condition):
                continue
            join_external_condition("joint", condition)
            if not condition_complete("joint", condition):
                run_condition_resumable("joint", condition)
        finalize_path("joint")


if __name__ == "__main__":
    main()
