"""Freeze exact model/adapter/kernel bytes only after complete diagnostics.

This content-addressed freeze is used when a shared dirty worktree makes a
single Git SHA insufficient to identify the executed numerical implementation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from audit_full_cross_model_campaign import EXPECTED_EXECUTION_CONFIG, capacity_audit, required_arrangements


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(command: list[str]) -> str:
    return subprocess.run(["git", *command], check=True, capture_output=True, text=True).stdout.strip()


def diagnostic_keys(path: Path, manifest: dict, expected_requests: set[str], manifest_hash: str, feasible: dict, expected_configuration: dict | None):
    opener = gzip.open if path.suffix == ".gz" else open
    groups = defaultdict(set)
    observations = manifest["observations"]
    request_source = {
        request_id: (row, noise_id)
        for row in observations if row["split"] == "diagnostic"
        for noise_id, request_id in enumerate(row["request_ids"])
    }
    with opener(path, "rt") as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("replay_manifest_sha256") != manifest_hash or item.get("numerical_freeze_sha256") is not None:
                raise ValueError("diagnostic record is not bound to the selected unfrozen replay manifest")
            if expected_configuration is not None and item.get("adapter_execution_configuration") != expected_configuration:
                raise ValueError("diagnostic record uses another adapter execution configuration")
            request_id = item.get("request_id")
            mode = item.get("implementation")
            if request_id not in expected_requests or mode not in ("native", "batch_invariant"):
                raise ValueError(f"unexpected diagnostic request/mode: {request_id}, {mode}")
            row, noise_id = request_source[request_id]
            logical_size = item.get("logical_batch_size")
            target_index = item.get("target_batch_index")
            ordering = item.get("request_ordering")
            if logical_size not in (1, 2, 4, 8, 16, 32) or not isinstance(target_index, int) or not isinstance(ordering, list) or len(ordering) != logical_size:
                raise ValueError(f"malformed diagnostic batching schedule: {request_id}")
            companions = [observations[index]["request_ids"][noise_id] for index in row.get("companion_replay_indices", [])[: logical_size - 1]]
            if len(companions) != logical_size - 1:
                raise ValueError(f"frozen diagnostic companions missing: {request_id}")
            transformation = item.get("transformation")
            if item.get("companion_type") == "duplicate":
                companions = [request_id] * (logical_size - 1)
            if transformation == "singleton":
                expected_order = [request_id]
            elif transformation == "batch_position":
                expected_order = list(companions)
                expected_order.insert(target_index, request_id)
            elif transformation == "permutation":
                expected_order = list(reversed([request_id, *companions]))
            else:
                expected_order = [request_id, *companions]
            if ordering != expected_order or item.get("companion_ids") != [value for index, value in enumerate(ordering) if index != target_index]:
                raise ValueError(f"diagnostic schedule differs from frozen companions: {request_id}")
            key = (
                item.get("transformation"), item.get("logical_batch_size"),
                item.get("target_batch_position"), item.get("companion_type"),
            )
            if key in groups[(request_id, mode)]:
                raise ValueError(f"duplicate diagnostic arrangement: {request_id}, {mode}, {key}")
            groups[(request_id, mode)].add(key)
            if mode == "batch_invariant" and not item.get("output", {}).get("exact"):
                raise ValueError(f"full invariant diagnostic failed: {request_id}, {key}")
    expected_groups = {(request_id, mode) for request_id in expected_requests for mode in ("native", "batch_invariant")}
    required_by_mode = {
        mode: required_arrangements(tuple([1, 2, 4, 8] + [
            size for size in (16, 32) if feasible.get(f"{mode}|B{size}")
        ]))
        for mode in ("native", "batch_invariant")
    }
    if set(groups) != expected_groups or any(required_by_mode[mode] != keys for (_, mode), keys in groups.items()):
        raise ValueError("diagnostic campaign is incomplete for the probed feasible batch-size matrix")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    diagnostic_observations = [item for item in manifest["observations"] if item["split"] == "diagnostic"]
    if len(diagnostic_observations) != 200:
        raise ValueError("numerical freeze requires 200 diagnostic observations")
    stochastic = all(len(item.get("noise", [])) == 3 and len(item.get("request_ids", [])) == 3 for item in diagnostic_observations)
    contract = manifest.get("deterministic_decoding_contract") or {}
    deterministic = (
        all(len(item.get("request_ids", [])) == 1 for item in diagnostic_observations)
        and isinstance(contract, dict)
        and contract.get("reason")
        and contract.get("evidence_path")
        and Path(contract["evidence_path"]).is_file()
        and sha256(Path(contract["evidence_path"])) == contract.get("evidence_sha256")
    )
    if not stochastic and not deterministic:
        raise ValueError("freeze needs three stored noises or a pinned deterministic contract")
    expected_requests = {request for item in diagnostic_observations for request in item["request_ids"]}
    expected_count = 600 if stochastic else 200
    if len(expected_requests) != expected_count:
        raise ValueError(f"numerical freeze requires {expected_count} distinct diagnostic requests")
    capacity_issues = []
    capacity = capacity_audit(args.capacity, args.manifest.parent, capacity_issues)
    if not capacity.get("complete"):
        raise ValueError(f"large-batch capacity probe is incomplete: {capacity_issues[:5]}")
    expected_configuration = EXPECTED_EXECUTION_CONFIG.get(args.model_id)
    groups = diagnostic_keys(args.diagnostic, manifest, expected_requests, sha256(args.manifest), capacity["feasible"], expected_configuration)
    repository = Path(__file__).resolve().parents[1]
    mandatory_sources = [
        repository / "scripts" / "run_model_replay_invariance.py",
        repository / "scripts" / "model_invariance" / "harness.py",
        repository / "batch_invariant_ops" / "batch_invariant_ops.py",
    ]
    if args.model_id in ("uva", "uva_joint"):
        mandatory_sources.append(Path(
            "/workspace/envs/uva_overlay/transformers/models/clip/modeling_clip.py"
        ))
    if manifest.get("source_replay_manifest_path"):
        mandatory_sources.append(Path(manifest["source_replay_manifest_path"]))
    paths = list(dict.fromkeys([*mandatory_sources, *args.source, *args.checkpoint, args.manifest, args.capacity, args.diagnostic]))
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("source/checkpoint/manifest/diagnostic file missing")
    hashes = {str(path.resolve()): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths}
    report = {
        "schema_version": 1,
        "model_id": args.model_id,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git(["rev-parse", "HEAD"]),
        "git_status": git(["status", "--short"]),
        "content_addressed_freeze": hashes,
        "diagnostic_request_pairs": len(expected_requests),
        "diagnostic_implementation_groups": len(groups),
        "diagnostic_arrangement_records": sum(map(len, groups.values())),
        "large_batch_feasible": capacity["feasible"],
        "adapter_execution_configuration": expected_configuration,
        "invariant_diagnostic_exact": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
