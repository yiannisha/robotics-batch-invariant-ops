"""Run every non-success-rate stage required for full RDT-1B coverage.

The completed four-condition ManiSkill success campaign is preserved under
``full/closed_loop``.  This supervisor starts only after SpatialVLA releases
the GPU and runs RDT's exact replay, diagnostics, held-out sweep, ablations,
operator qualification, timing, and genuine request-queue experiments.
"""

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
MODEL_ROOT = ROOT / "results/cross_model/rdt_1b"
FULL = MODEL_ROOT / "full"
MANIFEST = FULL / "replay_manifest.json"
ADAPTER = "scripts.model_invariance.adapters.rdt_maniskill"
PYTHON = Path("/workspace/envs/rdt/bin/python")
CHECKPOINT = Path("/workspace/assets/rdt/checkpoint")
CHECKPOINT_WEIGHT = CHECKPOINT / "rdt/mp_rank_00_model_states.pt"
SIGLIP = Path("/workspace/assets/rdt/siglip")
SOURCE = Path("/workspace/models/rdt")
PILOT = MODEL_ROOT / "pilots/non_success_pre_full_coverage"
START_MARKER = FULL / "non_success_full_campaign_started.json"


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
        if "run_spatialvla_full_campaign.py" not in command:
            raise RuntimeError(f"PID {pid} no longer identifies the SpatialVLA supervisor")
        print(f"waiting for SpatialVLA full-campaign supervisor PID {pid}", flush=True)
        time.sleep(30)


def archive_pilots() -> None:
    adapter = ROOT / "scripts/model_invariance/adapters/rdt_maniskill.py"
    invariant_ops = ROOT / "batch_invariant_ops/batch_invariant_ops.py"
    if START_MARKER.is_file():
        marker = json.loads(START_MARKER.read_text())
        expected = {
            "adapter_sha256": sha256(adapter),
            "invariant_ops_sha256": sha256(invariant_ops),
            "replay_manifest_sha256": sha256(MANIFEST),
        }
        if any(marker.get(key) != value for key, value in expected.items()):
            raise RuntimeError("RDT numerical sources changed after campaign start")
        return
    if PILOT.exists():
        raise RuntimeError(f"pilot archive exists without a start marker: {PILOT}")
    PILOT.mkdir(parents=True)
    archived = []
    for name in ("diagnostic", "heldout", "operator", "performance", "serving"):
        source = FULL / name
        if source.exists():
            shutil.move(str(source), str(PILOT / f"full_{name}"))
            archived.append(f"full/{name}")
    for name in ("numerical_freeze.json", "report.json", "REPORT.md"):
        source = FULL / name
        if source.exists():
            shutil.move(str(source), str(PILOT / f"full_{name}"))
            archived.append(f"full/{name}")
    for name in (
        "baseline.json", "batch_transformations.json", "first_divergence.json",
        "fixed.json", "multiple_inputs.json", "performance", "provenance.json", "REPORT.md",
    ):
        source = MODEL_ROOT / name
        if source.exists():
            shutil.move(str(source), str(PILOT / name))
            archived.append(str(source.relative_to(ROOT)))
    START_MARKER.write_text(json.dumps({
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Prior RDT numerical outputs were off-domain pilots and predated the exact replay freeze.",
        "pilot_archive": str(PILOT.resolve()),
        "archived": archived,
        "adapter_sha256": sha256(adapter),
        "invariant_ops_sha256": sha256(invariant_ops),
        "replay_manifest_sha256": sha256(MANIFEST),
    }, indent=2) + "\n")


def replay_indices(count: int) -> str:
    rows = [
        row for row in json.loads(MANIFEST.read_text())["observations"]
        if row.get("split") == "diagnostic"
    ]
    selected, seen_tasks, seen_episodes = [], set(), set()
    for row in rows:
        if row["task_index"] not in seen_tasks:
            selected.append(row["replay_index"])
            seen_tasks.add(row["task_index"])
            seen_episodes.add(row["episode_id"])
    for row in rows:
        if row["episode_id"] not in seen_episodes:
            selected.append(row["replay_index"])
            seen_episodes.add(row["episode_id"])
    chosen = set(selected)
    selected.extend(row["replay_index"] for row in rows if row["replay_index"] not in chosen)
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} RDT diagnostic inputs for B{count}")
    return ",".join(map(str, selected[:count]))


def audited_capacity() -> dict:
    issues: list[str] = []
    result = capacity_audit(FULL / "diagnostic/large_batch_capacity.jsonl", FULL, issues)
    if not result.get("complete"):
        raise RuntimeError(f"RDT capacity probe failed independent audit: {issues}")
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
    run([*common, "--batch-sizes", "1", "2", "4", "8", "--resume"])
    for mode in ("native", "batch_invariant"):
        large = [size for size in (16, 32) if capacity["feasible"].get(f"{mode}|B{size}")]
        if large:
            run([
                *common, "--modes", mode,
                "--batch-sizes", "1", "2", "4", "8", *map(str, large), "--resume",
            ])


def summarize_timing(raw: Path, output: Path, script: str) -> None:
    run([str(PYTHON), script, "--input", str(raw), "--output", str(output)])


def run_global_audit() -> None:
    output = ROOT / "results/cross_model/full_coverage_audit.json"
    command = [
        str(PYTHON), "scripts/audit_full_cross_model_campaign.py",
        "--results", "results/cross_model", "--output", str(output),
    ]
    print("RUN", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=False)
    if result.returncode not in (0, 1):
        raise subprocess.CalledProcessError(result.returncode, command)
    report = json.loads(output.read_text())
    record = next(
        (row for row in report.get("configurations", []) if row.get("id") == "rdt_1b"), None
    )
    if not record or record.get("complete") is not True:
        raise RuntimeError(f"RDT still fails the independent audit: {record}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-pid", type=int)
    args = parser.parse_args()
    if args.wait_for_pid is not None:
        if args.wait_for_pid <= 0:
            raise ValueError("wait PID must be positive")
        wait_for_predecessor(args.wait_for_pid)
    os.environ.update({
        "PYTHONPATH": f"{SOURCE}:{ROOT}",
        "RDT_CHECKPOINT": str(CHECKPOINT),
        "RDT_SIGLIP": str(SIGLIP),
        "RDT_SOURCE": str(SOURCE),
        "RDT_SAMPLE_DIR": "/workspace/assets/lerobot_libero/fixture",
        "RDT_REPLAY_MANIFEST": str(MANIFEST),
        "HF_HOME": "/workspace/.cache/huggingface",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    if not MANIFEST.is_file():
        raise FileNotFoundError("RDT exact replay reconstruction has not completed")
    archive_pilots()

    capacity_path = FULL / "diagnostic/large_batch_capacity.jsonl"
    if not capacity_path.is_file():
        run([
            str(PYTHON), "-u", "scripts/probe_model_large_batches.py",
            "--adapter", ADAPTER, "--manifest", str(MANIFEST),
            "--batch-sizes", "16", "32", "--output", str(capacity_path),
        ])
    capacity = audited_capacity()
    run_replay("diagnostic", capacity)

    for size in (1, 8):
        trace = FULL / f"operator/aten_trace_B{size}.json"
        if not trace.is_file():
            run([
                str(PYTHON), "-u", "scripts/trace_model_operator_shapes.py",
                "--adapter", ADAPTER, "--manifest", str(MANIFEST),
                "--batch-size", str(size), "--replay-indices", replay_indices(size),
                "--output", str(trace),
            ])
    operator_config = FULL / "operator/configurations.json"
    if not operator_config.is_file():
        run([
            str(PYTHON), "scripts/build_rdt_operator_config.py",
            "--trace-b1", str(FULL / "operator/aten_trace_B1.json"),
            "--trace-b8", str(FULL / "operator/aten_trace_B8.json"),
            "--checkpoint", str(CHECKPOINT_WEIGHT), "--output", str(operator_config),
        ])
    run([
        str(PYTHON), "-u", "scripts/run_model_causal_ablation.py",
        "--adapter", ADAPTER, "--manifest", str(MANIFEST),
        "--output", str(FULL / "diagnostic/causal_ablation.jsonl"), "--resume",
    ])

    freeze = FULL / "numerical_freeze.json"
    if not freeze.is_file():
        source_files = [
            ROOT / "scripts/model_invariance/adapters/rdt_maniskill.py",
            ROOT / "batch_invariant_ops/batch_invariant_ops.py",
            *sorted(SOURCE.rglob("*.py")),
            *sorted(SOURCE.rglob("*.yaml")),
        ]
        checkpoint_files = [
            CHECKPOINT_WEIGHT,
            *sorted((CHECKPOINT / "lang_embeds").glob("*.pt")),
            SIGLIP / "model.safetensors", SIGLIP / "config.json",
            SIGLIP / "preprocessor_config.json",
        ]
        run([
            str(PYTHON), "scripts/freeze_model_numerics.py",
            "--model-id", "rdt_1b", "--manifest", str(MANIFEST),
            "--diagnostic", str(FULL / "diagnostic/invariance.jsonl.gz"),
            "--capacity", str(capacity_path), "--source", *map(str, source_files),
            "--checkpoint", *map(str, checkpoint_files), "--output", str(freeze),
        ])

    run_replay("heldout", capacity, freeze)
    run([
        str(PYTHON), "scripts/summarize_model_replay_invariance.py",
        "--input", str(FULL / "heldout/invariance.jsonl.gz"),
        "--fidelity", str(FULL / "heldout/singleton_fidelity.jsonl.gz"),
        "--output", str(FULL / "heldout/summary.json"),
        "--flow-output", str(FULL / "heldout/flow_summary.json"),
        "--flow-steps", "5",
        "--flow-series", "prediction=diffusion.{step}.prediction",
        "--flow-series", "sample=diffusion.{step}.sample",
    ])
    run([
        str(PYTHON), "-u", "scripts/qualify_real_shape_matrix_ops.py",
        "--config", str(operator_config), "--seeds", "100",
        "--output", str(FULL / "operator/raw.jsonl"), "--resume",
    ])

    base_indices = replay_indices(8)
    policy = FULL / "performance/policy_timings.jsonl"
    run([
        str(PYTHON), "-u", "scripts/benchmark_model_policy.py",
        "--adapter", ADAPTER, "--batch-sizes", "1,2,4,8",
        "--sessions", "5", "--warmup", "20", "--repetitions", "200",
        "--replay-manifest", str(MANIFEST), "--replay-indices", base_indices,
        "--output", str(policy), "--resume", "--full-campaign",
    ])
    summarize_timing(policy, FULL / "performance/policy_summary.json",
                     "scripts/summarize_model_policy_benchmark.py")
    for size in (16, 32):
        indices = replay_indices(size)
        for mode in ("native", "batch_invariant"):
            if not capacity["feasible"].get(f"{mode}|B{size}"):
                continue
            raw = FULL / f"performance/policy_{mode}_B{size}_timings.jsonl"
            run([
                str(PYTHON), "-u", "scripts/benchmark_model_policy.py",
                "--adapter", ADAPTER, "--batch-sizes", str(size),
                "--implementations", mode, "--sessions", "5", "--warmup", "20",
                "--repetitions", "200", "--replay-manifest", str(MANIFEST),
                "--replay-indices", indices, "--output", str(raw), "--resume",
                "--full-campaign",
            ])
            summarize_timing(raw, FULL / f"performance/policy_{mode}_B{size}_summary.json",
                             "scripts/summarize_model_policy_benchmark.py")

    serving = FULL / "performance/serving_timings.jsonl"
    run([
        str(PYTHON), "-u", "scripts/benchmark_model_serving.py",
        "--adapter", ADAPTER, "--logical-batch-size", "4", "--fixed-batch-size", "8",
        "--sessions", "5", "--warmup", "20", "--repetitions", "200",
        "--replay-manifest", str(MANIFEST), "--replay-indices", base_indices,
        "--output", str(serving), "--resume", "--full-campaign",
    ])
    summarize_timing(serving, FULL / "performance/serving_summary.json",
                     "scripts/summarize_model_serving_benchmark.py")
    large_serving = []
    feasible = capacity["feasible"]
    shapes = []
    if feasible.get("native|B16"):
        shapes.append((8, 16))
    if all(feasible.get(key) for key in ("native|B16", "batch_invariant|B16", "native|B32")):
        shapes.append((16, 32))
    for logical, fixed in shapes:
        indices = replay_indices(fixed)
        raw = FULL / f"performance/serving_fixed_B{fixed}_timings.jsonl"
        run([
            str(PYTHON), "-u", "scripts/benchmark_model_serving.py",
            "--adapter", ADAPTER, "--logical-batch-size", str(logical),
            "--fixed-batch-size", str(fixed), "--sessions", "5", "--warmup", "20",
            "--repetitions", "200", "--replay-manifest", str(MANIFEST),
            "--replay-indices", indices, "--output", str(raw), "--resume",
            "--full-campaign",
        ])
        summarize_timing(raw, FULL / f"performance/serving_fixed_B{fixed}_summary.json",
                         "scripts/summarize_model_serving_benchmark.py")
        large_serving.append((logical, fixed, indices, raw))

    launches = FULL / "performance/kernel_launches.json"
    if not launches.is_file():
        run([
            str(PYTHON), "-u", "scripts/profile_model_kernel_launches.py",
            "--adapter", ADAPTER, "--batch-sizes", "1", "2", "4", "8",
            "--logical-batch-size", "4", "--fixed-batch-size", "8",
            "--replay-manifest", str(MANIFEST), "--replay-indices", base_indices,
            "--output", str(launches),
        ])

    queue = FULL / "serving/requests.jsonl"
    run([
        str(PYTHON), "-u", "scripts/benchmark_model_request_queue.py",
        "--adapter", ADAPTER, "--serving-timings", str(serving),
        "--logical-batch-size", "4", "--fixed-batch-size", "8",
        "--requests-per-group", "250", "--replay-manifest", str(MANIFEST),
        "--replay-indices", base_indices, "--output", str(queue), "--resume",
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
            str(PYTHON), "scripts/summarize_model_request_queue.py", "--input", str(queue),
            "--output", str(FULL / f"serving/summary_fixed_B{fixed}.json"),
        ])

    run([
        str(PYTHON), "scripts/generate_full_path_report.py",
        "--model-id", "rdt_1b", "--results", "results/cross_model",
    ])
    run_global_audit()


if __name__ == "__main__":
    main()
