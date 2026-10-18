"""Fail-closed prefix audit for queued UniVLA and MolmoAct2 LIBERO runs.

The structural audit is shared with the already exercised LeRobot campaigns;
this layer independently pins every model-specific source and readiness record
before accepting any exact-checkpoint rollout prefix.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_lerobot_paired_prefix import audit as audit_structure


ROOT = Path(__file__).resolve().parents[1]
SHARED_WRAPPER_SHA256 = "4757b4b63634283c7e78528eede888e50c615f7a430f0678a779dee7bdc8c207"
UNIVLA_NOISE_CONTRACT = "944a2294c3a2924b64298a77aaa45babe4e98bf47652b771b409265c716028d8"
UNIVLA_SHARD_HASHES = {
    "b7e6e472d974d1c572138839008e308486b526ffe1c9065306bd2107513b9e80",
    "c700932f380da894537a191b1b60f335a617efa01dc44ae6d8216a4f4e97f28d",
    "eeef8be71a34052ff2a590a2120f5d4067114f4f2f72b3c9b78d771968b207c3",
    "0932b91cf0b322993213a8fc5f0eb591171a130addd93d2bff701a3dc5e4259f",
    "89536431c69b08b10b449ec309f52dcea22f14b7647317f30f5715273392bbf1",
}
MOLMO_SHARD_HASHES = {
    "5aa56113a3c1c95f0c03b90bada5b7ac2babf59c752a3d6201b8362a530c6809",
    "1fd95aec9ad144d3f4eac42db88a000321e016beeae606ecfcb84fcbdb13ecf4",
    "e6c455434915e74a8d750bedd0fc4c23b5a8ce37954aaff9c3e3f22085a26890",
    "4e5bcf5bad81573153d6deaa3d25c3e53dc697658488b1cdbab4c8a3dffba1b6",
    "76140ad00c225f95b80eb6b6a5fea2f985e4fdde3356a3c67c93d87d083d4ee2",
}

MODELS: dict[str, dict[str, Any]] = {
    "univla_baai": {
        "manifest": {
            "checkpoint_sha256": "fdadf800d3f743375d1db8f4c44527e9622c602b9341c74b88bbbf65488ce922",
            "official_evaluator_sha256": "ab99413e0daeeda070e6bae66a67b9553acd987d832d7e871b4be54ef2774990",
            "official_policy_sha256": "fafe0a15c0f4e5f7bfc909b4c81e4a2d41c3f768994852e398c194d843d68ec1",
            "paired_wrapper_sha256": "32cde04238120c844e6af2a85aee85b744cf6146afb83fecbd36b80dc816d6a4",
            "adapter_sha256": "1eb23af6fea544556c92368d43e90d2d36ba7f589fe59a41fbebe468c9d89ddb",
            "generation_compatibility": "transformers_post_4_50_explicit_generation_mixin_and_legacy_dynamic_cache_api",
            "shared_rollout_wrapper_sha256": SHARED_WRAPPER_SHA256,
            "task_manifest_sha256": "e629fc969eb364d999f5e2a6ffc8e67bac3be08b03fc5acbf7a6625fa6742d28",
            "official_univla_revision": "a91ee5f269d7ad46d46635ee6481d2eb655b4af9",
            "official_lerobot_revision": "e595b7902714ba51f91e47523f66f89c5181b649",
            "policy_checkpoint_revision": "df6c94ff872f4fc056693e507c033ca390694118",
            "vision_checkpoint_revision": "c81f916ad371289e205310a7539255e8a9396488",
            "deterministic_contract_sha256": UNIVLA_NOISE_CONTRACT,
            "checkpoint_loading": "five_hash_verified_sealed_memfd_artifacts",
        },
        "files": {
            ROOT / "scripts/run_univla_paired_libero.py": "32cde04238120c844e6af2a85aee85b744cf6146afb83fecbd36b80dc816d6a4",
            ROOT / "scripts/run_lerobot_paired_libero.py": SHARED_WRAPPER_SHA256,
            ROOT / "scripts/model_invariance/adapters/univla_baai.py": "1eb23af6fea544556c92368d43e90d2d36ba7f589fe59a41fbebe468c9d89ddb",
            Path("/workspace/models/univla/reference/RoboVLMs/eval/libero/evaluate_libero_emu.py"): "ab99413e0daeeda070e6bae66a67b9553acd987d832d7e871b4be54ef2774990",
            Path("/workspace/models/univla/reference/RoboVLMs/eval/libero/model_wrapper_emu.py"): "fafe0a15c0f4e5f7bfc909b4c81e4a2d41c3f768994852e398c194d843d68ec1",
            ROOT / "results/cross_model/univla_baai/full/runtime_checkpoint/policy/UNIVLA_LIBERO_IMG_BS192_8K/model.safetensors.index.json": "fdadf800d3f743375d1db8f4c44527e9622c602b9341c74b88bbbf65488ce922",
            ROOT / "results/cross_model/univla_baai/full/closed_loop/paired/evaluator_task_manifest.json": "e629fc969eb364d999f5e2a6ffc8e67bac3be08b03fc5acbf7a6625fa6742d28",
            ROOT / "results/cross_model/univla_baai/full/closed_loop/paired_preflight/exact_memfd_v6/preflight.json": "03ab08956d7d3115007dc7a706d26d4f9b5380d156d51bbad8d8b2caea03eea0",
        },
        "receipt": ROOT / "results/cross_model/univla_baai/full/checkpoint_cache_receipt.json",
        "daemon_command": "serve_univla_checkpoint_memfd.py",
        "shard_hashes": UNIVLA_SHARD_HASHES,
    },
    "molmoact2": {
        "manifest": {
            "checkpoint_sha256": "273799bed0cbdc2c4a7a524c294652a0538c482e5e1148cf403951e836722c06",
            "official_evaluator_sha256": "2a5962b162e225c1621e5b4058f0d9f9c082eea0db42981dfebd40bf9c3e1c50",
            "official_policy_sha256": "3e0090b8cc045d59911c0a579815ebd17ce0423f9fdd03be887412df5c7b8b49",
            "paired_wrapper_sha256": "81ca5bccd88ad7803254664854562a9677dc7ac1d4c28b668026e4daa29f0674",
            "shared_rollout_wrapper_sha256": SHARED_WRAPPER_SHA256,
            "task_manifest_sha256": "c55177a1e40912b5e9c2e1cf39ff78be8c532d3f225f48027afb221803290e7c",
            "paired_preflight_sha256": "4b0686136499eb0365b6c67021bbef92722f700f82e99c6eb5d324b87855e3c6",
            "checkpoint_revision": "0d24a92bd1faf321ef497c3bbd5681af97c65aa2",
            "official_model_repository_revision": "66b87e64efd99dfd103241418113955cf64dfa9c",
            "official_lerobot_revision": "e595b7902714ba51f91e47523f66f89c5181b649",
            "official_environment_sha256": "97c984f12331527626812ec19967ef399e545535b3571becf044db2417ae9d71",
            "checkpoint_loading": "five_hash_verified_shards_streamed_from_RAM_to_meta_allocated_GPU_model",
        },
        "files": {
            ROOT / "scripts/run_molmoact2_paired_libero.py": "81ca5bccd88ad7803254664854562a9677dc7ac1d4c28b668026e4daa29f0674",
            ROOT / "scripts/run_lerobot_paired_libero.py": SHARED_WRAPPER_SHA256,
            ROOT / "scripts/model_invariance/adapters/molmoact2.py": "957e493b01887ad98c0b87f6d0f137d7f665aa0e8fd45e5c13b6d64fa5541c59",
            Path("/workspace/models/lerobot/src/lerobot/scripts/lerobot_eval.py"): "2a5962b162e225c1621e5b4058f0d9f9c082eea0db42981dfebd40bf9c3e1c50",
            Path("/workspace/models/lerobot/src/lerobot/envs/libero.py"): "97c984f12331527626812ec19967ef399e545535b3571becf044db2417ae9d71",
            Path("/workspace/assets/molmoact2/checkpoint/modeling_molmoact2.py"): "3e0090b8cc045d59911c0a579815ebd17ce0423f9fdd03be887412df5c7b8b49",
            Path("/workspace/assets/molmoact2/checkpoint/model.safetensors.index.json"): "273799bed0cbdc2c4a7a524c294652a0538c482e5e1148cf403951e836722c06",
            ROOT / "results/cross_model/molmoact2/full/closed_loop/paired/evaluator_task_manifest.json": "c55177a1e40912b5e9c2e1cf39ff78be8c532d3f225f48027afb221803290e7c",
            ROOT / "results/cross_model/molmoact2/full/closed_loop/paired_preflight/exact_ram_stream_v5/preflight.json": "4b0686136499eb0365b6c67021bbef92722f700f82e99c6eb5d324b87855e3c6",
        },
        "receipt": ROOT / "results/cross_model/molmoact2/full/checkpoint_memfd_prefetch_receipt.json",
        "daemon_command": "serve_molmoact2_checkpoint_memfd.py",
        "shard_hashes": MOLMO_SHARD_HASHES,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sealed_receipt(
    path: Path, command_fragment: str, expected_hashes: set[str]
) -> None:
    receipt = json.loads(path.read_text())
    rows = receipt.get("files") or {}
    pid = receipt.get("daemon_pid")
    command_path = Path(f"/proc/{pid}/cmdline") if isinstance(pid, int) else Path("/missing")
    if (
        receipt.get("all_pinned_files_verified") is not True
        or receipt.get("storage") != "sealed_memfd"
        or {row.get("sha256") for row in rows.values()} != expected_hashes
        or not command_path.is_file()
        or command_fragment.encode() not in command_path.read_bytes()
    ):
        raise ValueError("exact-checkpoint sealed-memory receipt is no longer live and pinned")
    required = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
    for row in rows.values():
        source = Path(row.get("path") or row.get("memfd_path") or "")
        if (
            not source.is_file()
            or source.stat().st_size != row.get("bytes")
            or row.get("verification") != "full_stream_sha256_then_memfd_sealed_immutable"
        ):
            raise ValueError(f"sealed checkpoint member differs: {source}")
        with source.open("rb") as stream:
            if fcntl.fcntl(stream.fileno(), fcntl.F_GET_SEALS) & required != required:
                raise ValueError(f"checkpoint member is not immutable: {source}")


def audit_readiness(model: str) -> dict[str, Any]:
    if model not in MODELS:
        raise ValueError("exact LeRobot readiness supports only UniVLA and MolmoAct2")
    expected = MODELS[model]
    for path, digest in expected["files"].items():
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"{model} pinned provenance file differs: {path}")
    validate_sealed_receipt(
        expected["receipt"], expected["daemon_command"], expected["shard_hashes"]
    )
    return {
        "schema_version": 1,
        "model_id": model,
        "exact_checkpoint_ready": True,
        "pinned_provenance_files_verified": len(expected["files"]),
        "sealed_checkpoint_members_verified": len(expected["shard_hashes"]),
    }


def audit(root: Path) -> dict[str, Any]:
    manifest_path = root / "condition_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model = manifest.get("model")
    if model not in MODELS:
        raise ValueError("exact LeRobot prefix auditor supports only UniVLA and MolmoAct2")
    expected = MODELS[model]
    readiness = audit_readiness(model)
    for key, value in expected["manifest"].items():
        if manifest.get(key) != value:
            raise ValueError(f"{model} condition provenance differs at {key}")
    if model == "univla_baai" and manifest.get("checkpoint_receipt_sha256") != file_sha256(
        expected["receipt"]
    ):
        raise ValueError("UniVLA condition does not pin the live checkpoint receipt")
    if model == "molmoact2" and set(manifest.get("checkpoint_shard_sha256") or []) != MOLMO_SHARD_HASHES:
        raise ValueError("MolmoAct2 condition shard set differs")
    report = audit_structure(root)
    report.update({
        "provenance_integrity_passed": True,
        "exact_checkpoint_ready": readiness["exact_checkpoint_ready"],
        "paired_wrapper_sha256": manifest["paired_wrapper_sha256"],
        "shared_rollout_wrapper_sha256": manifest["shared_rollout_wrapper_sha256"],
        "condition_manifest_sha256": file_sha256(manifest_path),
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--condition-root", type=Path)
    target.add_argument("--readiness-model", choices=tuple(MODELS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = (
        audit(args.condition_root)
        if args.condition_root is not None
        else audit_readiness(args.readiness_model)
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
