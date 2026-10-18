"""Resume the WSA paired campaign only when GPU and cgroup memory are safe."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from continue_wsa_paired_after_openvla import (
    CONDITIONS,
    finalize,
    join_external_condition,
    make_task_manifest,
    run_condition_resumable,
    supervisor_lock,
    validate_condition,
)


CGROUP_MEMORY = Path("/sys/fs/cgroup/memory.current")


def gpu_free_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def wait_for_resources(minimum_gpu_free_mib: int, maximum_cgroup_bytes: int) -> None:
    while True:
        free = gpu_free_mib()
        memory = int(CGROUP_MEMORY.read_text())
        if free >= minimum_gpu_free_mib and memory <= maximum_cgroup_bytes:
            print(
                f"WSA resource gate passed: gpu_free_mib={free} "
                f"cgroup_memory_bytes={memory}",
                flush=True,
            )
            return
        print(
            f"WSA waiting: gpu_free_mib={free}/{minimum_gpu_free_mib} "
            f"cgroup_memory_bytes={memory}/{maximum_cgroup_bytes}",
            flush=True,
        )
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-gpu-free-mib", type=int, default=16_000)
    parser.add_argument("--maximum-cgroup-bytes", type=int, default=105_000_000_000)
    args = parser.parse_args()
    task_manifest = make_task_manifest()
    gate = lambda: wait_for_resources(
        args.minimum_gpu_free_mib, args.maximum_cgroup_bytes,
    )
    with supervisor_lock():
        for condition in CONDITIONS:
            if validate_condition(condition):
                continue
            join_external_condition(condition)
            if not validate_condition(condition):
                run_condition_resumable(condition, task_manifest, before_attempt=gate)
        finalize(task_manifest)


if __name__ == "__main__":
    main()
