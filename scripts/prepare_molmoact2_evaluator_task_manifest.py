"""Freeze MolmoAct2's exact checkpoint-matched LIBERO-10 evaluator tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from libero.libero import benchmark, get_libero_path

from scripts.model_invariance.adapters.molmoact2 import CHECKPOINT_REVISION, CHECKPOINT_SHARDS
from scripts.run_molmoact2_paired_libero import (
    CHECKPOINT,
    LEROBOT_REVISION,
    OFFICIAL_MODEL_REVISION,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    tasks = []
    for task_id in range(10):
        task = suite.get_task(task_id)
        init_path = (
            Path(get_libero_path("init_states"))
            / task.problem_folder / Path(task.init_states_file).name
        )
        bddl_path = (
            Path(get_libero_path("bddl_files"))
            / task.problem_folder / task.bddl_file
        )
        init_count = len(suite.get_task_init_states(task_id))
        if init_count != 50 or not init_path.is_file() or not bddl_path.is_file():
            raise ValueError(f"LIBERO task {task_id} does not expose exactly 50 initializations")
        tasks.append({
            "task_id": task_id,
            "task_name": task.name,
            "task_description": task.language,
            "initializations": init_count,
            "init_states_path": str(init_path.resolve()),
            "init_states_sha256": sha256(init_path),
            "bddl_path": str(bddl_path.resolve()),
            "bddl_sha256": sha256(bddl_path),
        })
    index = CHECKPOINT / "model.safetensors.index.json"
    report = {
        "schema_version": 1,
        "model_id": "molmoact2",
        "task_suite": "libero_10",
        "task_ids": list(range(10)),
        "task_descriptions": [row["task_description"] for row in tasks],
        "initializations_per_task": 50,
        "tasks": tasks,
        "checkpoint_repository": "allenai/MolmoAct2-LIBERO",
        "checkpoint_repository_revision": CHECKPOINT_REVISION,
        "checkpoint_index_path": str(index.resolve()),
        "checkpoint_index_sha256": sha256(index),
        "checkpoint_shards": [
            {"name": name, "bytes": size, "sha256": digest}
            for name, size, digest in CHECKPOINT_SHARDS
        ],
        "official_model_repository_revision": OFFICIAL_MODEL_REVISION,
        "official_lerobot_revision": LEROBOT_REVISION,
    }
    content = json.dumps(report, sort_keys=True, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.is_file() and args.output.read_text() != content:
        raise FileExistsError("existing MolmoAct2 evaluator task manifest differs")
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(content)
    os.replace(temporary, args.output)
    print(json.dumps({"tasks": len(tasks), "output": str(args.output), "sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
