"""Run every non-success-rate stage required for full DexVLA coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from audit_full_cross_model_campaign import capacity_audit


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "dexvla"
MODEL_ROOT = ROOT / "results/cross_model" / MODEL_ID
FULL = MODEL_ROOT / "full"
MANIFEST = FULL / "replay_manifest.json"
ADAPTER = "scripts.model_invariance.adapters.dexvla"
PYTHON = Path("/workspace/envs/dexvla/bin/python")
CHECKPOINT = Path("/workspace/assets/dexvla/repo/checkpoint-60000")
SOURCE = Path("/workspace/assets/dexvla/sample/episode_47.hdf5")
OFFICIAL = Path("/workspace/models/dexvla")
PILOT = MODEL_ROOT / "pilots/non_success_pre_full_coverage"
START_MARKER = FULL / "non_success_full_campaign_started.json"
SOURCE_SHA256 = "abd1a0681df1a14e1fc09a799a7b5965e4a73bc7f233e2bbd5441303adb4203a"
CHECKPOINT_HASHES = {
    "model-00001-of-00002.safetensors": "3b096461b657171a3b37585e44e2e32171d045aa3bddea0295f4d2c3971baf7a",
    "model-00002-of-00002.safetensors": "c5b2b1e07e55e37b0cbaa1dab222872d4ea79ddd063062b2e65c0f8a2073aa87",
    "model.safetensors.index.json": "c118abc7c960406ead47e938276d47ec44aecb8076daaa9af1c0199c3d8b7c64",
    "config2.json": "9186f1dad06f01223abdc863d6af6a63289d3af9496202fba668a68534e50c88",
    "dataset_stats.pkl": "7bee355ea318a692b2b263a027792bd0e56b10429f6bb3f0e3938ad109ffe533",
    "preprocessor_config.json": "b5eaad0c2815f07631535dcc58f3c462b0d73693638ad21d19f3c50820eae1cc",
    "tokenizer.json": "88a3a6fcb80132f76da8aa40cdc3fccd7e5d8468ef15421f5b0c2715e85217d2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(arguments: list[str]) -> None:
    print("RUN", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, env=os.environ.copy(), check=True)


def wait_for_predecessor(pid: int) -> None:
    while True:
        proc = Path(f"/proc/{pid}")
        if not proc.exists():
            return
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
        except FileNotFoundError:
            return
        if state == "Z":
            return
        if "run_cogact_full_campaign.py" not in command:
            raise RuntimeError(f"PID {pid} no longer identifies the CogACT supervisor")
        print(f"waiting for CogACT full-campaign supervisor PID {pid}", flush=True)
        time.sleep(30)


def verify_assets() -> None:
    if not PYTHON.is_file() or not MANIFEST.is_file() or not SOURCE.is_file():
        raise FileNotFoundError("DexVLA environment, replay manifest, or source is missing")
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError("DexVLA retained public HDF5 source changed")
    manifest = json.loads(MANIFEST.read_text())
    if (len(manifest.get("observations", [])) != 396
            or manifest.get("split_counts") != {"diagnostic": 200, "heldout": 196}
            or manifest.get("noise_shape") != [396, 3, 50, 14]):
        raise ValueError("DexVLA replay does not contain the full source-limited campaign")
    for name, expected in CHECKPOINT_HASHES.items():
        path = CHECKPOINT / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"DexVLA checkpoint differs: {name}")
    revision = subprocess.run(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != "fc21a822f4c774e242eb6f1ab4a235788de7aba9":
        raise ValueError("DexVLA official code revision differs")


def archive_pilots() -> None:
    expected = {
        "adapter_sha256": sha256(ROOT / "scripts/model_invariance/adapters/dexvla.py"),
        "invariant_ops_sha256": sha256(ROOT / "batch_invariant_ops/batch_invariant_ops.py"),
        "replay_manifest_sha256": sha256(MANIFEST),
        "source_sha256": SOURCE_SHA256,
        "checkpoint_shard_sha256": [
            CHECKPOINT_HASHES["model-00001-of-00002.safetensors"],
            CHECKPOINT_HASHES["model-00002-of-00002.safetensors"],
        ],
    }
    if START_MARKER.is_file():
        marker = json.loads(START_MARKER.read_text())
        if any(marker.get(key) != value for key, value in expected.items()):
            raise RuntimeError("DexVLA numerical sources changed after campaign start")
        return
    if PILOT.exists():
        raise RuntimeError(f"pilot archive exists without a start marker: {PILOT}")
    PILOT.mkdir(parents=True)
    archived = []
    for name in (
        "baseline.json", "batch_transformations.json", "first_divergence.json",
        "fixed.json", "multiple_inputs.json", "official_equivalence.json",
        "performance", "provenance.json", "REPORT.md",
    ):
        source = MODEL_ROOT / name
        if source.exists():
            shutil.move(str(source), str(PILOT / name))
            archived.append(str(source.relative_to(ROOT)))
    status = FULL / "status.json"
    if status.exists():
        shutil.move(str(status), str(PILOT / "full_status.json"))
        archived.append(str(status.relative_to(ROOT)))
    START_MARKER.write_text(json.dumps({
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Prior DexVLA outputs were pilots over at most three source frames.",
        "pilot_archive": str(PILOT.resolve()),
        "archived": archived,
        **expected,
    }, indent=2) + "\n")


def replay_indices(count: int) -> str:
    rows = [
        row for row in json.loads(MANIFEST.read_text())["observations"]
        if row.get("split") == "diagnostic"
    ]
    if len(rows) < count:
        raise ValueError(f"only {len(rows)} DexVLA diagnostic inputs for B{count}")
    return ",".join(str(row["replay_index"]) for row in rows[:count])


def audited_capacity() -> dict:
    issues: list[str] = []
    result = capacity_audit(FULL / "diagnostic/large_batch_capacity.jsonl", FULL, issues)
    if not result.get("complete"):
        raise RuntimeError(f"DexVLA capacity probe failed independent audit: {issues}")
    return result


def run_replay(split: str, capacity: dict, freeze: Path | None = None) -> None:
    common = [
        str(PYTHON), "-u", "scripts/run_model_replay_invariance.py",
        "--adapter", ADAPTER, "--manifest", str(MANIFEST),
        "--output", str(FULL / f"{split}/invariance.jsonl.gz"),
        "--fidelity-output", str(FULL / f"{split}/singleton_fidelity.jsonl.gz"),
        "--split", split,
    ]
    if freeze is not None:
        common.extend(("--freeze", str(freeze)))
    run([*common, "--batch-sizes", "1", "2", "4", "--resume"])
    for mode in ("native", "batch_invariant"):
        optional = [
            size for size in (8, 16, 32)
            if capacity["feasible"].get(f"{mode}|B{size}")
        ]
        if optional:
            run([
                *common, "--modes", mode, "--batch-sizes", "1", "2", "4",
                *map(str, optional), "--resume",
            ])


def summarize(raw: Path, output: Path, script: str) -> None:
    run([str(PYTHON), script, "--input", str(raw), "--output", str(output)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-pid", type=int)
    args = parser.parse_args()
    if args.wait_for_pid is not None:
        if args.wait_for_pid <= 0:
            raise ValueError("wait PID must be positive")
        wait_for_predecessor(args.wait_for_pid)
    os.environ.update({
        "PYTHONPATH": f"{OFFICIAL}:{ROOT}",
        "DEXVLA_CHECKPOINT": str(CHECKPOINT),
        "DEXVLA_SAMPLE": str(SOURCE),
        "DEXVLA_REPLAY_MANIFEST": str(MANIFEST),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    verify_assets()
    archive_pilots()

    capacity_path = FULL / "diagnostic/large_batch_capacity.jsonl"
    if not capacity_path.is_file():
        run([
            str(PYTHON), "-u", "scripts/probe_model_large_batches.py",
            "--adapter", ADAPTER, "--manifest", str(MANIFEST),
            "--batch-sizes", "8", "16", "32", "--output", str(capacity_path),
        ])
    capacity = audited_capacity()
    run_replay("diagnostic", capacity)

    traces = ((1, "native"), (8, "batch_invariant"))
    for size, implementation in traces:
        trace = FULL / f"operator/aten_trace_B{size}.json"
        if not trace.is_file():
            run([
                str(PYTHON), "-u", "scripts/trace_model_operator_shapes.py",
                "--adapter", ADAPTER, "--manifest", str(MANIFEST),
                "--batch-size", str(size), "--implementation", implementation,
                "--replay-indices", replay_indices(size), "--output", str(trace),
            ])
    operator_config = FULL / "operator/configurations.json"
    if not operator_config.is_file():
        run([
            str(PYTHON), "scripts/build_dexvla_operator_config.py",
            "--trace-b1", str(FULL / "operator/aten_trace_B1.json"),
            "--trace-b8", str(FULL / "operator/aten_trace_B8.json"),
            "--checkpoint", str(CHECKPOINT), "--output", str(operator_config),
        ])
    run([
        str(PYTHON), "-u", "scripts/run_model_causal_ablation.py",
        "--adapter", ADAPTER, "--manifest", str(MANIFEST),
        "--output", str(FULL / "diagnostic/causal_ablation.jsonl"), "--resume",
    ])

    freeze = FULL / "numerical_freeze.json"
    if not freeze.is_file():
        sources = [
            ROOT / "scripts/model_invariance/adapters/dexvla.py",
            *sorted((OFFICIAL / "qwen2_vla").rglob("*.py")),
            *sorted((OFFICIAL / "policy_heads").rglob("*.py")),
        ]
        checkpoint_files = [CHECKPOINT / name for name in json.loads(MANIFEST.read_text())["checkpoint_files"]]
        checkpoint_files.extend((SOURCE, FULL / "request_noise.npy"))
        run([
            str(PYTHON), "scripts/freeze_model_numerics.py", "--model-id", MODEL_ID,
            "--manifest", str(MANIFEST),
            "--diagnostic", str(FULL / "diagnostic/invariance.jsonl.gz"),
            "--capacity", str(capacity_path), "--source", *map(str, sources),
            "--checkpoint", *map(str, checkpoint_files), "--output", str(freeze),
        ])

    run_replay("heldout", capacity, freeze)
    run([
        str(PYTHON), "scripts/summarize_model_replay_invariance.py",
        "--input", str(FULL / "heldout/invariance.jsonl.gz"),
        "--fidelity", str(FULL / "heldout/singleton_fidelity.jsonl.gz"),
        "--output", str(FULL / "heldout/summary.json"),
        "--flow-output", str(FULL / "heldout/flow_summary.json"),
        "--flow-steps", "10", "--flow-label-template", "flow.step_{step:02d}.state",
    ])
    run([
        str(PYTHON), "-u", "scripts/qualify_real_shape_matrix_ops.py",
        "--config", str(operator_config), "--seeds", "100",
        "--output", str(FULL / "operator/raw.jsonl"), "--resume",
    ])

    base_indices = replay_indices(4)
    policy = FULL / "performance/policy_timings.jsonl"
    run([
        str(PYTHON), "-u", "scripts/benchmark_model_policy.py", "--adapter", ADAPTER,
        "--batch-sizes", "1,2,4", "--sessions", "5", "--warmup", "20",
        "--repetitions", "200", "--replay-manifest", str(MANIFEST),
        "--replay-indices", base_indices, "--output", str(policy), "--resume",
        "--full-campaign",
    ])
    summarize(policy, FULL / "performance/policy_summary.json", "scripts/summarize_model_policy_benchmark.py")
    for size in (8, 16, 32):
        for mode in ("native", "batch_invariant"):
            if not capacity["feasible"].get(f"{mode}|B{size}"):
                continue
            raw = FULL / f"performance/policy_{mode}_B{size}_timings.jsonl"
            run([
                str(PYTHON), "-u", "scripts/benchmark_model_policy.py", "--adapter", ADAPTER,
                "--batch-sizes", str(size), "--implementations", mode,
                "--sessions", "5", "--warmup", "20", "--repetitions", "200",
                "--replay-manifest", str(MANIFEST), "--replay-indices", replay_indices(size),
                "--output", str(raw), "--resume", "--full-campaign",
            ])
            summarize(raw, FULL / f"performance/policy_{mode}_B{size}_summary.json", "scripts/summarize_model_policy_benchmark.py")

    serving = FULL / "performance/serving_timings.jsonl"
    run([
        str(PYTHON), "-u", "scripts/benchmark_model_serving.py", "--adapter", ADAPTER,
        "--logical-batch-size", "2", "--fixed-batch-size", "4", "--sessions", "5",
        "--warmup", "20", "--repetitions", "200", "--replay-manifest", str(MANIFEST),
        "--replay-indices", base_indices, "--output", str(serving), "--resume",
        "--full-campaign",
    ])
    summarize(serving, FULL / "performance/serving_summary.json", "scripts/summarize_model_serving_benchmark.py")
    large_serving = []
    feasible = capacity["feasible"]
    shapes = []
    if feasible.get("native|B16"):
        shapes.append((8, 16))
    if all(feasible.get(key) for key in (
        "native|B16", "batch_invariant|B16", "native|B32"
    )):
        shapes.append((16, 32))
    for logical, fixed in shapes:
        indices = replay_indices(fixed)
        raw = FULL / f"performance/serving_fixed_B{fixed}_timings.jsonl"
        run([
            str(PYTHON), "-u", "scripts/benchmark_model_serving.py",
            "--adapter", ADAPTER, "--logical-batch-size", str(logical),
            "--fixed-batch-size", str(fixed), "--sessions", "5",
            "--warmup", "20", "--repetitions", "200",
            "--replay-manifest", str(MANIFEST), "--replay-indices", indices,
            "--output", str(raw), "--resume", "--full-campaign",
        ])
        summarize(
            raw,
            FULL / f"performance/serving_fixed_B{fixed}_summary.json",
            "scripts/summarize_model_serving_benchmark.py",
        )
        large_serving.append((logical, fixed, indices, raw))
    launches = FULL / "performance/kernel_launches.json"
    if not launches.is_file():
        run([
            str(PYTHON), "-u", "scripts/profile_model_kernel_launches.py", "--adapter", ADAPTER,
            "--batch-sizes", "1", "2", "4", "--logical-batch-size", "2",
            "--fixed-batch-size", "4", "--replay-manifest", str(MANIFEST),
            "--replay-indices", base_indices, "--output", str(launches),
        ])
    queue = FULL / "serving/requests.jsonl"
    run([
        str(PYTHON), "-u", "scripts/benchmark_model_request_queue.py", "--adapter", ADAPTER,
        "--serving-timings", str(serving), "--logical-batch-size", "2",
        "--fixed-batch-size", "4", "--requests-per-group", "250",
        "--replay-manifest", str(MANIFEST), "--replay-indices", base_indices,
        "--output", str(queue), "--resume",
    ])
    run([
        str(PYTHON), "scripts/summarize_model_request_queue.py", "--input", str(queue),
        "--output", str(FULL / "serving/summary.json"),
    ])
    for logical, fixed, indices, timing in large_serving:
        queue = FULL / f"serving/requests_fixed_B{fixed}.jsonl"
        run([
            str(PYTHON), "-u", "scripts/benchmark_model_request_queue.py",
            "--adapter", ADAPTER, "--serving-timings", str(timing),
            "--logical-batch-size", str(logical), "--fixed-batch-size", str(fixed),
            "--requests-per-group", "250", "--replay-manifest", str(MANIFEST),
            "--replay-indices", indices, "--output", str(queue), "--resume",
        ])
        run([
            str(PYTHON), "scripts/summarize_model_request_queue.py",
            "--input", str(queue),
            "--output", str(FULL / f"serving/summary_fixed_B{fixed}.json"),
        ])
    run([str(PYTHON), "scripts/generate_full_path_report.py", "--model-id", MODEL_ID,
         "--results", "results/cross_model"])
    audit_output = ROOT / "results/cross_model/full_coverage_audit.json"
    command = [str(PYTHON), "scripts/audit_full_cross_model_campaign.py", "--results",
               "results/cross_model", "--output", str(audit_output)]
    result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=False)
    if result.returncode not in (0, 1):
        raise subprocess.CalledProcessError(result.returncode, command)
    record = next((row for row in json.loads(audit_output.read_text())["configurations"]
                   if row["id"] == MODEL_ID), None)
    if not record or record.get("complete") is not True:
        raise RuntimeError(f"DexVLA still fails independent audit: {record}")


if __name__ == "__main__":
    main()
