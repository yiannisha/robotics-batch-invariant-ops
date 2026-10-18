"""Independently audit raw four-condition SpatialVLA SimplerEnv pairing."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CONDITIONS = (
    "native_singleton", "native_dynamic",
    "patched_singleton", "patched_dynamic",
)
CHECKPOINT = Path("/workspace/assets/spatialvla/checkpoint")
CHECKPOINT_HASHES = {
    "model-00001-of-00002.safetensors": "713ee1bd116c8f77e2f56903d15e6c3f3dff6ea6b12fc71bfcc3bd4a53a2cc2b",
    "model-00002-of-00002.safetensors": "b8c87abc1b91598ad237c31ee2286c5c783bfb9f4142696d9b1c36e62f634a34",
    "config.json": "0c5d94be86f9e15b708650c4b829033ae99243f85507012b1dafa0263f799909",
    "model.safetensors.index.json": "6c351156136b2452db86d3a426ff30adc9d11363b43f46fa69fe79d76b6d5c79",
    "processor_config.json": "8712b6d5171f74e90f951ec7b148ab90509f28434e36b2da959527fa0ddb57ea",
    "processing_spatialvla.py": "c20b15460633f65809788c633936347fada2f2bbcfffba898d2f0b4b99807b33",
    "modeling_spatialvla.py": "a04008cf364e094079493e5982fa2b32fca12874b0a42b645e0aecab024e7236",
    "action_tokenizer.py": "b6b18eee0320e77156881aea4a92ea5f54dc278d43c1ea72c3a0deaa9d505a31",
    "configuration_spatialvla.py": "63e100324daf81d2cd7b6c5e6fafcace67d660586ef7b509bb38cc8fba72c1e3",
    "tokenizer.json": "2523a63c898ebf0a32c7282a2e459ef2c950a846c5f3172305089e4149b6b6c3",
    "tokenizer_config.json": "33f38858ac90afd5ac29942110dd3b98f40e1e194fca254714bab5d88f57cc65",
    "preprocessor_config.json": "bdb79dc3255a75c1c4be8ac1ddd4e2aa08b9a29eee0431755b7b0faa3b54e413",
    "generation_config.json": "79c657e6d0660218a91c360516cae5b8fa0cf1dd447476c8dcdb815c8ff994b1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(data)


def finite_vector(value: Any, length: int) -> bool:
    return (isinstance(value, list) and len(value) == length
            and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value))


def finite_matrix(value: Any, rows: int, columns: int) -> bool:
    return isinstance(value, list) and len(value) == rows and all(
        finite_vector(row, columns) for row in value
    )


def expected_groups(tasks: list[dict], batch_size: int) -> set[tuple[int, tuple[int, ...]]]:
    groups = set()
    for task in tasks:
        count = task["initializations"]
        for start in range(0, count, batch_size):
            groups.add((task["task_id"], tuple(range(start, min(start + batch_size, count)))))
    return groups


def audit(root: Path, task_manifest_path: Path) -> dict[str, Any]:
    issues: list[str] = []
    task_manifest = json.loads(task_manifest_path.read_text())
    tasks = task_manifest.get("tasks")
    if (task_manifest.get("schema_version") != 1
            or task_manifest.get("model_id") != "spatialvla"
            or task_manifest.get("task_count") != 12
            or not isinstance(tasks, list) or len(tasks) != 12
            or sum(task.get("initializations", 0) for task in tasks) != 496):
        issues.append("task manifest does not define the pinned 12-task/496-initialization suite")
        tasks = tasks if isinstance(tasks, list) else []
    expected_pairs = {
        (task["task_id"], initialization)
        for task in tasks for initialization in range(task["initializations"])
    }
    client_source = Path(__file__).with_name("run_spatialvla_paired_client.py")
    server_source = Path(__file__).with_name("run_spatialvla_paired_server.py")
    client_hash = sha256(client_source) if client_source.is_file() else None
    server_hash = sha256(server_source) if server_source.is_file() else None
    task_hash = sha256(task_manifest_path)
    expected_artifacts = [
        {"path": str((CHECKPOINT / name).resolve()), "sha256": digest}
        for name, digest in CHECKPOINT_HASHES.items()
    ]
    for item in expected_artifacts:
        artifact = Path(item["path"])
        if not artifact.is_file() or sha256(artifact) != item["sha256"]:
            issues.append(f"exact SpatialVLA checkpoint artifact differs: {artifact}")
    conditions: dict[str, dict[str, Any]] = {}
    cross_condition: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}

    for condition in CONDITIONS:
        folder = root / condition
        manifest_path = folder / "condition_manifest.json"
        eval_path = folder / "official/eval_info.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        batch_size = 1 if condition.endswith("singleton") else 4
        if (manifest.get("schema_version") != 1 or manifest.get("model") != "spatialvla"
                or manifest.get("condition") != condition or manifest.get("batch_size") != batch_size
                or manifest.get("invariant_ops") != condition.startswith("patched")
                or manifest.get("task_suite") != task_manifest.get("task_suite")
                or manifest.get("task_manifest_sha256") != task_hash
                or manifest.get("task_count") != 12
                or manifest.get("planned_initializations") != 496
                or manifest.get("deterministic_policy") is not True
                or manifest.get("explicit_physical_configuration") is not True
                or manifest.get("client_source_sha256") != client_hash
                or manifest.get("server_source_sha256") != server_hash
                or manifest.get("server_metadata_path") != str((folder / "server_metadata.json").resolve())):
            issues.append(f"{condition}: condition manifest differs from pinned execution contract")
        metadata_path = folder / "server_metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        if (manifest.get("server_metadata_sha256") != (
                sha256(metadata_path) if metadata_path.is_file() else None)
                or metadata.get("protocol_version") != "2.1"
                or metadata.get("policy_type") != "spatialvla"
                or metadata.get("checkpoint") != str(CHECKPOINT.resolve())
                or metadata.get("task_suite") != task_manifest.get("task_suite")
                or metadata.get("paired_condition") != condition
                or metadata.get("genuine_batched_policy_call") is not True
                or metadata.get("maximum_request_batch") != batch_size
                or metadata.get("batch_invariant_ops") != condition.startswith("patched")
                or metadata.get("deterministic_policy") is not True
                or metadata.get("unnorm_keys") != {
                    "google_robot": "fractal20220817_data/0.1.0",
                    "widowx_bridge": "bridge_orig/1.0.0",
                }
                or metadata.get("image_size") != [224, 224]
                or metadata.get("action_chunk_size") != 4
                or metadata.get("action_dim") != 7
                or metadata.get("action_ensemble_temperature") != -0.8
                or metadata.get("sticky_gripper_repeat") != 10
                or metadata.get("artifact_files") != expected_artifacts):
            issues.append(f"{condition}: server handshake/checkpoint evidence differs")
        observed_groups: set[tuple[int, tuple[int, ...]]] = set()
        rows: dict[tuple[int, int], dict[str, Any]] = {}
        files = sorted((folder / "rollouts").glob("*.json")) + sorted(
            (folder / "rollouts").glob("*.json.gz")
        )
        for path in files:
            try:
                value = read_json(path)
            except Exception as exc:
                issues.append(f"{condition}: unreadable rollout {path}: {exc}")
                continue
            task = value.get("task_id"); ids = value.get("explicit_initialization_ids")
            if not isinstance(task, int) or not isinstance(ids, list) or not ids:
                issues.append(f"{condition}: malformed physical group {path}")
                continue
            group = (task, tuple(ids))
            if group in observed_groups:
                issues.append(f"{condition}: duplicate physical group {group}")
            observed_groups.add(group)
            if value.get("seeds") != [manifest.get("seed", 0) + item for item in ids]:
                issues.append(f"{condition}: reset seeds differ for physical group {group}")
            episodes = value.get("episodes")
            if not isinstance(episodes, list) or len(episodes) != len(ids):
                issues.append(f"{condition}: episode count differs for physical group {group}")
                continue
            for episode in episodes:
                initialization = episode.get("initialization_id"); key = (task, initialization)
                sizes = episode.get("policy_batch_sizes")
                actions = episode.get("action_sequence"); raw = episode.get("raw_action_sequence")
                chunks = episode.get("raw_action_chunks"); trajectory = episode.get("trajectory")
                rewards = episode.get("rewards"); terminated = episode.get("terminated")
                truncated = episode.get("truncated")
                length = len(actions) if isinstance(actions, list) else -1
                valid_sequences = (
                    length > 0 and all(finite_vector(row, 7) for row in actions)
                    and isinstance(raw, list) and len(raw) == length
                    and all(finite_vector(row, 7) for row in raw)
                    and isinstance(chunks, list) and len(chunks) == length
                    and all(finite_matrix(row, 4, 7) for row in chunks)
                    and isinstance(trajectory, list) and len(trajectory) == length + 1
                    and isinstance(rewards, list) and len(rewards) == length
                    and isinstance(terminated, list) and len(terminated) == length
                    and isinstance(truncated, list) and len(truncated) == length
                    and isinstance(sizes, list) and len(sizes) == length
                    and sizes and sizes[0] == len(ids)
                    and all(isinstance(size, int) and 1 <= size <= len(ids) for size in sizes)
                )
                if (episode.get("task") != task or initialization not in ids
                        or episode.get("reset_seed") != manifest.get("seed", 0) + initialization
                        or episode.get("deterministic_policy") is not True
                        or not isinstance(episode.get("initial_observation_sha256"), str)
                        or len(episode["initial_observation_sha256"]) != 64
                        or episode.get("success") != any(terminated or [])
                        or not valid_sequences or key in rows):
                    issues.append(f"{condition}: malformed or duplicate episode {key} in {path}")
                    continue
                rows[key] = episode
                cross_condition.setdefault(key, {})[condition] = episode
        required_groups = expected_groups(tasks, batch_size)
        if observed_groups != required_groups:
            issues.append(
                f"{condition}: physical group schedule differs "
                f"(observed={len(observed_groups)}, expected={len(required_groups)})"
            )
        if set(rows) != expected_pairs:
            issues.append(
                f"{condition}: reset coverage differs "
                f"(observed={len(rows)}, expected={len(expected_pairs)})"
            )
        info = json.loads(eval_path.read_text()) if eval_path.is_file() else {}
        overall = info.get("overall", {})
        successes = sum(bool(row["success"]) for row in rows.values())
        if (overall.get("n_episodes") != 496 or overall.get("n_success") != successes
                or info.get("condition") != condition
                or info.get("task_suite") != task_manifest.get("task_suite")):
            issues.append(f"{condition}: official aggregate differs from raw episodes")
        conditions[condition] = {
            "physical_groups": len(observed_groups), "episodes": len(rows),
            "successes": successes, "expected_physical_groups": len(required_groups),
        }

    for key in expected_pairs:
        group = cross_condition.get(key, {})
        if set(group) != set(CONDITIONS):
            issues.append(f"paired reset {key} lacks all four conditions")
            continue
        hashes = {row["initial_observation_sha256"] for row in group.values()}
        seeds = {row["reset_seed"] for row in group.values()}
        if len(hashes) != 1 or len(seeds) != 1:
            issues.append(f"paired reset {key} differs physically across conditions")

    report = {
        "schema_version": 1,
        "model_id": "spatialvla",
        "evidence_tier": "independent_raw_pairing_audit",
        "task_manifest_path": str(task_manifest_path.resolve()),
        "task_manifest_sha256": task_hash,
        "expected_resets_per_condition": len(expected_pairs),
        "conditions": conditions,
        "fully_paired": not issues,
        "issues": issues,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, default=Path(
        "results/cross_model/spatialvla/full/closed_loop/paired/evaluator_task_manifest.json"
    ))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.root.resolve(), args.task_manifest.resolve())
    output = args.output or args.root / "reset_pairing_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "issues"}, indent=2))
    if report["issues"]:
        print(json.dumps({"issues": report["issues"][:20]}, indent=2))
    raise SystemExit(0 if report["fully_paired"] else 1)


if __name__ == "__main__":
    main()
