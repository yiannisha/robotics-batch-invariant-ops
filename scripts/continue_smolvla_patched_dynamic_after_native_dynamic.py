"""Finish SmolVLA dynamic conditions without overlapping their GPU residency."""

from __future__ import annotations

import argparse

from continue_smolvla_after_pi0fast import (
    join_external_condition,
    run_condition_resumable,
    validate_condition,
    wait_for_active_condition,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dynamic-pid", type=int, required=True)
    args = parser.parse_args()
    if args.native_dynamic_pid <= 0:
        raise ValueError("invalid native_dynamic worker PID")

    wait_for_active_condition(args.native_dynamic_pid)
    if not validate_condition("native_dynamic"):
        run_condition_resumable("native_dynamic")

    if validate_condition("patched_dynamic"):
        print("Already verified SmolVLA patched_dynamic; preserving evidence", flush=True)
        return
    join_external_condition("patched_dynamic")
    if not validate_condition("patched_dynamic"):
        run_condition_resumable("patched_dynamic")
    print("Verified both SmolVLA dynamic conditions", flush=True)


if __name__ == "__main__":
    main()
