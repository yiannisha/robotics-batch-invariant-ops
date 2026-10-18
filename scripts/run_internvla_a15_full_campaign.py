"""Run full non-success coverage for InternVLA-A1.5 standard and optimized."""

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
RESULTS = ROOT / "results/cross_model"
ADAPTER = "scripts.model_invariance.adapters.internvla_a15"
PYTHON = Path("/workspace/envs/internvla_a15/bin/python")
CHECKPOINT = Path("/workspace/assets/internvla_a15/checkpoint")
QWEN = Path("/workspace/assets/internvla_a15/qwen")
MODEL_ROOT = Path("/workspace/models/internvla-a-series")
POLICY_SOURCES = MODEL_ROOT / "src/lerobot/policies/internvla_a1_5"
PATHS = (
    ("internvla_a15_standard", "standard"),
    ("internvla_a15_optimized", "optimized"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
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
        if "run_smolvla_full_campaign.py" not in command:
            raise RuntimeError(f"PID {pid} no longer identifies the SmolVLA supervisor")
        print(f"waiting for SmolVLA full-campaign supervisor PID {pid}", flush=True)
        time.sleep(30)


def full(model_id: str) -> Path:
    return RESULTS / model_id / "full"


def manifest(model_id: str) -> Path:
    return full(model_id) / "replay_manifest.json"


def execution_configuration(backend: str) -> dict | None:
    return (
        {"backend": "optimized", "optimized_graph_cache_policy": "recapture_per_policy_call"}
        if backend == "optimized" else None
    )


def archive_pilots(model_id: str, backend: str) -> None:
    directory = full(model_id)
    pilot = RESULTS / model_id / "pilots/non_success_pre_full_coverage"
    marker = directory / "non_success_full_campaign_started.json"
    adapter = ROOT / "scripts/model_invariance/adapters/internvla_a15.py"
    if marker.is_file():
        record = json.loads(marker.read_text())
        if (record.get("adapter_sha256") != sha256(adapter)
                or record.get("backend") != backend
                or record.get("adapter_execution_configuration") != execution_configuration(backend)):
            raise RuntimeError(f"{model_id} numerical source/configuration changed after campaign start")
        return
    if pilot.exists():
        raise RuntimeError(f"pilot archive exists without a start marker: {pilot}")
    pilot.mkdir(parents=True)
    archived = []
    for name in ("diagnostic", "heldout", "operator", "performance", "serving"):
        source = directory / name
        if source.exists():
            shutil.move(str(source), str(pilot / name))
            archived.append(name)
    for name in ("numerical_freeze.json", "report.json", "REPORT.md"):
        source = directory / name
        if source.exists():
            shutil.move(str(source), str(pilot / name))
            archived.append(name)
    marker.write_text(json.dumps({
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Prior cross-model numerical outputs were pilots and predated the full freeze.",
        "pilot_archive": str(pilot.resolve()), "archived": archived,
        "backend": backend,
        "adapter_execution_configuration": execution_configuration(backend),
        "adapter_sha256": sha256(adapter),
        "replay_manifest_sha256": sha256(manifest(model_id)),
    }, indent=2) + "\n")


def replay_indices(model_id: str, count: int) -> str:
    observations = [
        row for row in json.loads(manifest(model_id).read_text())["observations"]
        if row.get("split") == "diagnostic"
    ]
    selected = []
    seen_tasks = set()
    for row in observations:
        task = row.get("task_index", row.get("task"))
        if task not in seen_tasks:
            selected.append(row["replay_index"])
            seen_tasks.add(task)
    chosen = set(selected)
    seen_episodes = {
        (row.get("task_index", row.get("task")), row.get("episode_id"))
        for row in observations if row["replay_index"] in chosen
    }
    for row in observations:
        episode = (row.get("task_index", row.get("task")), row.get("episode_id"))
        if episode not in seen_episodes:
            selected.append(row["replay_index"])
            seen_episodes.add(episode)
    chosen = set(selected)
    selected.extend(row["replay_index"] for row in observations if row["replay_index"] not in chosen)
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} diagnostic inputs for B{count}")
    return ",".join(map(str, selected[:count]))


def audited_capacity(model_id: str) -> dict:
    directory = full(model_id)
    issues: list[str] = []
    result = capacity_audit(directory / "diagnostic/large_batch_capacity.jsonl", directory, issues)
    if not result.get("complete"):
        raise RuntimeError(f"{model_id} capacity probe failed independent audit: {issues}")
    return result


def run_replay(model_id: str, split: str, capacity: dict, freeze: Path | None = None) -> None:
    directory = full(model_id)
    common = [
        str(PYTHON), "-u", "scripts/run_model_replay_invariance.py",
        "--adapter", ADAPTER, "--manifest", str(manifest(model_id)),
        "--output", str(directory / f"{split}/invariance.jsonl.gz"),
        "--fidelity-output", str(directory / f"{split}/singleton_fidelity.jsonl.gz"),
        "--split", split,
    ]
    if freeze is not None:
        common.extend(("--freeze", str(freeze)))
    run([*common, "--batch-sizes", "1", "2", "4", "8", "--resume"])
    for mode in ("native", "batch_invariant"):
        large = [size for size in (16, 32) if capacity["feasible"].get(f"{mode}|B{size}")]
        if large:
            run([*common, "--modes", mode, "--batch-sizes", "1", "2", "4", "8",
                 *map(str, large), "--resume"])


def summarize_timing(raw: Path, output: Path, script: str) -> None:
    run([str(PYTHON), script, "--input", str(raw), "--output", str(output)])


def run_backend(model_id: str, backend: str) -> None:
    directory = full(model_id)
    replay = manifest(model_id)
    os.environ.update({
        "PYTHONPATH": f"{MODEL_ROOT / 'src'}:{ROOT}",
        "INTERNVLA_A15_CHECKPOINT": str(CHECKPOINT),
        "INTERNVLA_A15_QWEN": str(QWEN),
        "INTERNVLA_A15_SAMPLE_DIR": "/workspace/assets/smolvla/full_replay",
        "INTERNVLA_A15_REPLAY_MANIFEST": str(replay),
        "INTERNVLA_A15_BACKEND": backend,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    archive_pilots(model_id, backend)
    capacity_path = directory / "diagnostic/large_batch_capacity.jsonl"
    if not capacity_path.is_file():
        run([
            str(PYTHON), "-u", "scripts/probe_model_large_batches.py",
            "--adapter", ADAPTER, "--manifest", str(replay),
            "--batch-sizes", "16", "32", "--output", str(capacity_path),
        ])
    capacity = audited_capacity(model_id)
    run_replay(model_id, "diagnostic", capacity)

    for batch_size in (1, 8):
        trace = directory / f"operator/aten_trace_B{batch_size}.json"
        if not trace.is_file():
            run([
                str(PYTHON), "-u", "scripts/trace_model_operator_shapes.py",
                "--adapter", ADAPTER, "--manifest", str(replay),
                "--batch-size", str(batch_size), "--output", str(trace),
            ])
    operator_config = directory / "operator/configurations.json"
    if not operator_config.is_file():
        run([
            str(PYTHON), "scripts/build_internvla_a15_operator_config.py",
            "--model-id", model_id,
            "--trace-b1", str(directory / "operator/aten_trace_B1.json"),
            "--trace-b8", str(directory / "operator/aten_trace_B8.json"),
            "--checkpoint", str(CHECKPOINT / "model.safetensors"),
            "--output", str(operator_config),
        ])
    run([
        str(PYTHON), "-u", "scripts/run_model_causal_ablation.py",
        "--adapter", ADAPTER, "--manifest", str(replay),
        "--output", str(directory / "diagnostic/causal_ablation.jsonl"), "--resume",
    ])

    freeze = directory / "numerical_freeze.json"
    if not freeze.is_file():
        sources = [
            ROOT / "scripts/model_invariance/adapters/internvla_a15.py",
            *sorted(POLICY_SOURCES.rglob("*.py")),
        ]
        checkpoint_files = sorted(path for base in (CHECKPOINT, QWEN)
                                  for path in base.iterdir() if path.is_file())
        run([
            str(PYTHON), "scripts/freeze_model_numerics.py",
            "--model-id", model_id, "--manifest", str(replay),
            "--diagnostic", str(directory / "diagnostic/invariance.jsonl.gz"),
            "--capacity", str(capacity_path), "--source", *map(str, sources),
            "--checkpoint", *map(str, checkpoint_files), "--output", str(freeze),
        ])

    run_replay(model_id, "heldout", capacity, freeze)
    run([
        str(PYTHON), "scripts/summarize_model_replay_invariance.py",
        "--input", str(directory / "heldout/invariance.jsonl.gz"),
        "--fidelity", str(directory / "heldout/singleton_fidelity.jsonl.gz"),
        "--output", str(directory / "heldout/summary.json"),
        "--flow-output", str(directory / "heldout/flow_summary.json"),
        "--flow-steps", "10", "--flow-label-template", "flow.step_{step:02d}.state",
    ])
    run([
        str(PYTHON), "-u", "scripts/qualify_real_shape_matrix_ops.py",
        "--config", str(operator_config), "--seeds", "100",
        "--output", str(directory / "operator/raw.jsonl"), "--resume",
    ])

    base_indices = replay_indices(model_id, 8)
    policy = directory / "performance/policy_timings.jsonl"
    run([
        str(PYTHON), "-u", "scripts/benchmark_model_policy.py",
        "--adapter", ADAPTER, "--batch-sizes", "1,2,4,8",
        "--sessions", "5", "--warmup", "20", "--repetitions", "200",
        "--replay-manifest", str(replay), "--replay-indices", base_indices,
        "--output", str(policy), "--resume", "--full-campaign",
    ])
    summarize_timing(policy, directory / "performance/policy_summary.json",
                     "scripts/summarize_model_policy_benchmark.py")
    for size in (16, 32):
        indices = replay_indices(model_id, size)
        for mode in ("native", "batch_invariant"):
            if not capacity["feasible"].get(f"{mode}|B{size}"):
                continue
            raw = directory / f"performance/policy_{mode}_B{size}_timings.jsonl"
            run([
                str(PYTHON), "-u", "scripts/benchmark_model_policy.py",
                "--adapter", ADAPTER, "--batch-sizes", str(size),
                "--implementations", mode, "--sessions", "5", "--warmup", "20",
                "--repetitions", "200", "--replay-manifest", str(replay),
                "--replay-indices", indices, "--output", str(raw), "--resume", "--full-campaign",
            ])
            summarize_timing(raw, directory / f"performance/policy_{mode}_B{size}_summary.json",
                             "scripts/summarize_model_policy_benchmark.py")

    serving = directory / "performance/serving_timings.jsonl"
    run([
        str(PYTHON), "-u", "scripts/benchmark_model_serving.py",
        "--adapter", ADAPTER, "--logical-batch-size", "4", "--fixed-batch-size", "8",
        "--sessions", "5", "--warmup", "20", "--repetitions", "200",
        "--replay-manifest", str(replay), "--replay-indices", base_indices,
        "--output", str(serving), "--resume", "--full-campaign",
    ])
    summarize_timing(serving, directory / "performance/serving_summary.json",
                     "scripts/summarize_model_serving_benchmark.py")
    feasible = capacity["feasible"]
    large_shapes = []
    if feasible.get("native|B16"):
        large_shapes.append((8, 16))
    if all(feasible.get(key) for key in ("native|B16", "batch_invariant|B16", "native|B32")):
        large_shapes.append((16, 32))
    large_serving = []
    for logical, fixed_size in large_shapes:
        indices = replay_indices(model_id, fixed_size)
        raw = directory / f"performance/serving_fixed_B{fixed_size}_timings.jsonl"
        run([
            str(PYTHON), "-u", "scripts/benchmark_model_serving.py",
            "--adapter", ADAPTER, "--logical-batch-size", str(logical),
            "--fixed-batch-size", str(fixed_size), "--sessions", "5", "--warmup", "20",
            "--repetitions", "200", "--replay-manifest", str(replay),
            "--replay-indices", indices, "--output", str(raw), "--resume", "--full-campaign",
        ])
        summarize_timing(raw, directory / f"performance/serving_fixed_B{fixed_size}_summary.json",
                         "scripts/summarize_model_serving_benchmark.py")
        large_serving.append((logical, fixed_size, indices, raw))

    launches = directory / "performance/kernel_launches.json"
    if not launches.is_file():
        run([
            str(PYTHON), "-u", "scripts/profile_model_kernel_launches.py",
            "--adapter", ADAPTER, "--batch-sizes", "1", "2", "4", "8",
            "--logical-batch-size", "4", "--fixed-batch-size", "8",
            "--replay-manifest", str(replay), "--replay-indices", base_indices,
            "--output", str(launches),
        ])
    queue = directory / "serving/requests.jsonl"
    run([
        str(PYTHON), "-u", "scripts/benchmark_model_request_queue.py",
        "--adapter", ADAPTER, "--serving-timings", str(serving),
        "--logical-batch-size", "4", "--fixed-batch-size", "8",
        "--requests-per-group", "250", "--replay-manifest", str(replay),
        "--replay-indices", base_indices, "--output", str(queue), "--resume",
    ])
    run([str(PYTHON), "scripts/summarize_model_request_queue.py", "--input", str(queue),
         "--output", str(directory / "serving/summary.json")])
    for logical, fixed_size, indices, timing in large_serving:
        queue = directory / f"serving/requests_fixed_B{fixed_size}.jsonl"
        run([
            str(PYTHON), "-u", "scripts/benchmark_model_request_queue.py",
            "--adapter", ADAPTER, "--serving-timings", str(timing),
            "--logical-batch-size", str(logical), "--fixed-batch-size", str(fixed_size),
            "--requests-per-group", "250", "--replay-manifest", str(replay),
            "--replay-indices", indices, "--output", str(queue), "--resume",
        ])
        run([str(PYTHON), "scripts/summarize_model_request_queue.py", "--input", str(queue),
             "--output", str(directory / f"serving/summary_fixed_B{fixed_size}.json")])
    run([str(PYTHON), "scripts/generate_full_path_report.py",
         "--model-id", model_id, "--results", "results/cross_model"])


def run_global_audit() -> None:
    output = RESULTS / "full_coverage_audit.json"
    command = [str(PYTHON), "scripts/audit_full_cross_model_campaign.py",
               "--results", "results/cross_model", "--output", str(output)]
    print("RUN", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=False)
    if result.returncode not in (0, 1):
        raise subprocess.CalledProcessError(result.returncode, command)
    report = json.loads(output.read_text())
    by_id = {row.get("id"): row for row in report.get("configurations", [])}
    failures = {model_id: by_id.get(model_id) for model_id, _ in PATHS
                if not by_id.get(model_id, {}).get("complete")}
    if failures:
        raise RuntimeError(f"InternVLA paths still fail independent audit: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-pid", type=int)
    args = parser.parse_args()
    if args.wait_for_pid is not None:
        if args.wait_for_pid <= 0:
            raise ValueError("wait PID must be positive")
        wait_for_predecessor(args.wait_for_pid)
    for model_id, backend in PATHS:
        run_backend(model_id, backend)
    run_global_audit()


if __name__ == "__main__":
    main()
