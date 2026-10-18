"""Compare raw LIBERO resets across singleton and dynamic vector batches.

This probe intentionally stops before policy/environment preprocessing.  It is
used to distinguish a physical reset mismatch from a batch-dependent policy
input transform when a closed-loop pairing audit reports different processed
sensor hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def array_evidence(value: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        "minimum": float(contiguous.min()),
        "maximum": float(contiguous.max()),
    }


def flatten_arrays(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            rows.update(flatten_arrays(value[key], f"{prefix}/{key}"))
    elif isinstance(value, np.ndarray):
        rows[prefix] = value
    return rows


def slice_sample(value: Any, index: int) -> Any:
    if isinstance(value, dict):
        return {key: slice_sample(item, index) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value[index]
    return value


def set_explicit_init_state_ids(env: Any, identifiers: list[int]) -> None:
    setter = getattr(env, "set_attr", None)
    if setter is None:
        ensure = getattr(env, "_ensure", None)
        if not callable(ensure):
            raise RuntimeError("vector env exposes neither set_attr nor _ensure")
        ensure()
        setter = getattr(getattr(env, "_env", None), "set_attr", None)
    if not callable(setter):
        raise RuntimeError("vector env does not expose a callable set_attr")
    setter("init_state_id", identifiers)
    observed = list(env.get_attr("init_state_id"))
    if observed != identifiers:
        raise RuntimeError(f"init-state assignment failed: {observed} != {identifiers}")


def capture_many(
    *,
    task: int,
    initialization_ids: list[int],
    batch_size: int,
    base_seed: int,
) -> dict[int, dict[str, Any]]:
    from lerobot.envs.configs import LiberoEnv as LiberoConfig
    from lerobot.utils.random_utils import set_seed

    set_seed(base_seed)
    config = LiberoConfig(
        task="libero_10",
        task_ids=[task],
        control_mode="absolute",
        episode_length=800,
        max_parallel_tasks=1,
    )
    env = config.create_envs(n_envs=batch_size, use_async_envs=True)["libero_10"][task]
    try:
        if batch_size == 1:
            groups = [[initialization_id] for initialization_id in initialization_ids]
        else:
            group_starts = sorted({
                (initialization_id // batch_size) * batch_size
                for initialization_id in initialization_ids
            })
            groups = [list(range(start, start + batch_size)) for start in group_starts]
        requested = set(initialization_ids)
        captured: dict[int, dict[str, Any]] = {}
        for identifiers in groups:
            seeds = [base_seed + identifier for identifier in identifiers]
            set_explicit_init_state_ids(env, identifiers)
            observation, _ = env.reset(seed=seeds)
            for target_index, initialization_id in enumerate(identifiers):
                if initialization_id not in requested:
                    continue
                sample = slice_sample(observation, target_index)
                arrays = flatten_arrays(sample)
                if not arrays:
                    raise RuntimeError("raw LIBERO reset exposed no array-valued observation fields")
                captured[initialization_id] = {
                    "batch_size": batch_size,
                    "identifiers": identifiers,
                    "seeds": seeds,
                    "target_index": target_index,
                    "fields": {key: array_evidence(value) for key, value in arrays.items()},
                }
        if set(captured) != requested:
            raise RuntimeError(
                f"captured reset IDs differ from request: {sorted(captured)} != {sorted(requested)}"
            )
        return captured
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, nargs="+", required=True)
    parser.add_argument("--initialization-ids", type=int, nargs="+", required=True)
    parser.add_argument("--base-seed", type=int, default=20261005)
    parser.add_argument("--source", type=Path, default=Path("/workspace/models/lerobot"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not os.environ.get("LIBERO_CONFIG_PATH") or os.environ.get("MUJOCO_GL") != "egl":
        raise EnvironmentError("set pinned LIBERO_CONFIG_PATH and MUJOCO_GL=egl")
    sys.path.insert(0, str((args.source / "src").resolve()))

    comparisons = []
    for task in args.tasks:
        singleton_rows = capture_many(
            task=task,
            initialization_ids=args.initialization_ids,
            batch_size=1,
            base_seed=args.base_seed,
        )
        dynamic_rows = capture_many(
            task=task,
            initialization_ids=args.initialization_ids,
            batch_size=4,
            base_seed=args.base_seed,
        )
        for initialization_id in args.initialization_ids:
            singleton = singleton_rows[initialization_id]
            dynamic = dynamic_rows[initialization_id]
            singleton_fields = singleton["fields"]
            dynamic_fields = dynamic["fields"]
            keys_match = set(singleton_fields) == set(dynamic_fields)
            field_matches = {
                key: singleton_fields[key]["sha256"] == dynamic_fields[key]["sha256"]
                for key in sorted(set(singleton_fields) & set(dynamic_fields))
            }
            comparisons.append(
                {
                    "task": task,
                    "initialization_id": initialization_id,
                    "keys_match": keys_match,
                    "all_fields_match": keys_match and all(field_matches.values()),
                    "field_matches": field_matches,
                    "singleton": singleton,
                    "dynamic": dynamic,
                }
            )

    report = {
        "schema_version": 1,
        "scope": "raw_libero_reset_before_preprocessing",
        "base_seed": args.base_seed,
        "comparison_count": len(comparisons),
        "all_comparisons_match": all(row["all_fields_match"] for row in comparisons),
        "comparisons": comparisons,
    }
    if args.output:
        write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "comparison_count": report["comparison_count"],
                "all_comparisons_match": report["all_comparisons_match"],
                "mismatches": [
                    {
                        "task": row["task"],
                        "initialization_id": row["initialization_id"],
                        "field_matches": row["field_matches"],
                    }
                    for row in comparisons
                    if not row["all_fields_match"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
