"""Run optimized InternVLA paired coverage when GPU capacity becomes available."""

from __future__ import annotations

import argparse
import fcntl
import json
import time
from pathlib import Path

from continue_internvla_paired_after_probes import (
    CONDITIONS,
    backend_lock,
    finalize_backend,
    join_external_condition,
    run_condition_resumable,
    validate_condition,
)


CGROUP_MEMORY = Path("/sys/fs/cgroup/memory.current")
WSA_SUPERVISOR_LOCK = Path(
    "results/cross_model/wsa_base/full/closed_loop/paired/.supervisor.lock"
)
PI0FAST_SUPERVISOR_LOCK = Path(
    "results/cross_model/pi_fast/full/closed_loop/paired/.supervisor.lock"
)
SMOLVLA_SUPERVISOR_LOCK = Path(
    "results/cross_model/smolvla/full/closed_loop/paired/.supervisor.lock"
)
HIGHER_PRIORITY_LOCKS = (
    ("WSA", WSA_SUPERVISOR_LOCK),
    ("PI0-FAST", PI0FAST_SUPERVISOR_LOCK),
    ("SmolVLA", SMOLVLA_SUPERVISOR_LOCK),
)


def gpu_free_mib() -> int:
    import subprocess

    result = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
        ],
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


def wait_for_higher_priority_campaigns() -> None:
    while True:
        active = next(
            (name for name, path in HIGHER_PRIORITY_LOCKS if lock_active(path)),
            None,
        )
        if active is None:
            return
        print(f"InternVLA optimized waiting for higher-priority {active} campaign", flush=True)
        time.sleep(60)


def wait_for_resources(minimum_gpu_free_mib: int, maximum_cgroup_bytes: int) -> None:
    while True:
        free = gpu_free_mib()
        memory = int(CGROUP_MEMORY.read_text())
        if free >= minimum_gpu_free_mib and memory <= maximum_cgroup_bytes:
            print(
                json.dumps({"gpu_free_mib": free, "cgroup_memory_bytes": memory}),
                flush=True,
            )
            return
        print(
            f"InternVLA optimized waiting: gpu_free_mib={free}/{minimum_gpu_free_mib} "
            f"cgroup_memory_bytes={memory}/{maximum_cgroup_bytes}",
            flush=True,
        )
        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-gpu-free-mib", type=int, default=15000)
    parser.add_argument("--maximum-cgroup-bytes", type=int, default=105_000_000_000)
    args = parser.parse_args()
    if args.minimum_gpu_free_mib <= 0 or args.maximum_cgroup_bytes <= 0:
        raise ValueError("resource thresholds must be positive")

    with backend_lock("optimized"):
        wait_for_higher_priority_campaigns()
        wait_for_resources(args.minimum_gpu_free_mib, args.maximum_cgroup_bytes)
        for condition in CONDITIONS:
            if validate_condition("optimized", condition):
                continue
            join_external_condition("optimized", condition)
            if not validate_condition("optimized", condition):
                run_condition_resumable("optimized", condition)
        finalize_backend("optimized")


if __name__ == "__main__":
    main()
