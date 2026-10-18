"""Resume every non-success-rate stage required for full PI0-FAST coverage."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from audit_full_cross_model_campaign import capacity_audit


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "results/cross_model/pi_fast/full"
MANIFEST = FULL / "replay_manifest.json"
ADAPTER = "scripts.model_invariance.adapters.lerobot_pi0fast"
PYTHON = Path("/workspace/envs/smolvla/bin/python")
CHECKPOINT = Path("/workspace/assets/pi_fast/checkpoint")
TEXT_TOKENIZER = Path("/workspace/assets/pi_fast/text_tokenizer")
ACTION_TOKENIZER = Path("/workspace/assets/pi_fast/action_tokenizer")
MODEL_SOURCES = Path("/workspace/models/lerobot/src/lerobot/policies")


def run(arguments: list[str]) -> None:
    print("RUN", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, env=os.environ.copy(), check=True)


def wait_for_process(pid: int) -> None:
    expected = ("run_model_replay_invariance.py", "--adapter", ADAPTER, "--split", "diagnostic")
    while True:
        proc = Path(f"/proc/{pid}")
        if not proc.exists():
            return
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
        except FileNotFoundError:
            return
        if state == "Z":
            return
        if any(fragment not in cmdline for fragment in expected):
            raise RuntimeError(f"PID {pid} no longer identifies the PI0-FAST diagnostic")
        print(f"waiting for active PI0-FAST diagnostic PID {pid}", flush=True)
        time.sleep(20)


def replay_indices(count: int) -> str:
    observations = [
        row for row in json.loads(MANIFEST.read_text())["observations"]
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
    selected.extend(
        row["replay_index"] for row in observations
        if row["replay_index"] not in chosen
    )
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} diagnostic inputs for B{count}")
    return ",".join(map(str, selected[:count]))


def trace(batch_size: int) -> None:
    output = FULL / f"operator/aten_trace_B{batch_size}.json"
    if output.is_file():
        return
    run([
        str(PYTHON), "-u", "scripts/trace_model_operator_shapes.py",
        "--adapter", ADAPTER, "--manifest", str(MANIFEST),
        "--batch-size", str(batch_size), "--output", str(output),
    ])


def timing_summary(raw: Path, output: Path, script: str) -> None:
    run([str(PYTHON), script, "--input", str(raw), "--output", str(output)])


def run_global_audit_and_require_pi_fast() -> None:
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
    pi_fast = next(
        (record for record in report.get("configurations", []) if record.get("id") == "pi_fast"),
        None,
    )
    if not pi_fast or pi_fast.get("complete") is not True:
        raise RuntimeError(f"PI0-FAST still fails the independent audit: {pi_fast}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-diagnostic-pid", type=int)
    args = parser.parse_args()
    if args.wait_for_diagnostic_pid is not None:
        if args.wait_for_diagnostic_pid <= 0:
            raise ValueError("wait PID must be positive")
        wait_for_process(args.wait_for_diagnostic_pid)

    os.environ.update({
        "PYTHONPATH": "/workspace/models/lerobot/src:/workspace/robotics-batch-invariant-ops",
        "LEROBOT_PI0FAST_CHECKPOINT": str(CHECKPOINT),
        "LEROBOT_PI0FAST_SAMPLE_DIR": "/workspace/assets/smolvla/full_replay",
        "PI0FAST_REPLAY_MANIFEST": str(MANIFEST),
        "PI0FAST_TEXT_TOKENIZER": str(TEXT_TOKENIZER),
        "PI0FAST_ACTION_TOKENIZER": str(ACTION_TOKENIZER),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })

    capacity_path = FULL / "diagnostic/large_batch_capacity.jsonl"
    if not capacity_path.is_file():
        run([
            str(PYTHON), "-u", "scripts/probe_model_large_batches.py",
            "--adapter", ADAPTER, "--manifest", str(MANIFEST),
            "--batch-sizes", "16", "32", "--output", str(capacity_path),
        ])
    capacity_issues: list[str] = []
    capacity = capacity_audit(capacity_path, FULL, capacity_issues)
    if not capacity.get("complete"):
        raise RuntimeError(f"PI0-FAST capacity probe failed independent audit: {capacity_issues}")

    diagnostic = FULL / "diagnostic/invariance.jsonl.gz"
    diagnostic_fidelity = FULL / "diagnostic/singleton_fidelity.jsonl.gz"
    run([
        str(PYTHON), "-u", "scripts/run_model_replay_invariance.py",
        "--adapter", ADAPTER, "--manifest", str(MANIFEST),
        "--output", str(diagnostic), "--fidelity-output", str(diagnostic_fidelity),
        "--split", "diagnostic", "--batch-sizes", "1", "2", "4", "8", "16", "32",
        "--resume",
    ])

    # Traces precede the ablation so an unsupported selective override cannot
    # prevent independent operator qualification evidence from being created.
    trace(1)
    trace(8)
    operator_config = FULL / "operator/configurations.json"
    if not operator_config.is_file():
        run([
            str(PYTHON), "scripts/build_pi0fast_operator_config.py",
            "--trace-b1", str(FULL / "operator/aten_trace_B1.json"),
            "--trace-b8", str(FULL / "operator/aten_trace_B8.json"),
            "--checkpoint", str(CHECKPOINT / "model.safetensors"),
            "--output", str(operator_config),
        ])

    run([
        str(PYTHON), "-u", "scripts/run_model_causal_ablation.py",
        "--adapter", ADAPTER, "--manifest", str(MANIFEST),
        "--output", str(FULL / "diagnostic/causal_ablation.jsonl"), "--resume",
    ])

    freeze = FULL / "numerical_freeze.json"
    if not freeze.is_file():
        source_files = [
            ROOT / "scripts/model_invariance/adapters/lerobot_pi0fast.py",
            MODEL_SOURCES / "pi0_fast/configuration_pi0_fast.py",
            MODEL_SOURCES / "pi0_fast/modeling_pi0_fast.py",
            MODEL_SOURCES / "pi0_fast/processor_pi0_fast.py",
            MODEL_SOURCES / "pi_gemma.py",
            MODEL_SOURCES / "common/vla_utils.py",
        ]
        checkpoint_files = [
            CHECKPOINT / name for name in (
                "model.safetensors", "config.json", "policy_preprocessor.json",
                "policy_postprocessor.json",
                "policy_preprocessor_step_2_normalizer_processor.safetensors",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            )
        ]
        checkpoint_files.extend(sorted(path for path in TEXT_TOKENIZER.iterdir() if path.is_file()))
        checkpoint_files.extend(sorted(path for path in ACTION_TOKENIZER.iterdir() if path.is_file()))
        run([
            str(PYTHON), "scripts/freeze_model_numerics.py",
            "--model-id", "pi_fast", "--manifest", str(MANIFEST),
            "--diagnostic", str(diagnostic),
            "--capacity", str(FULL / "diagnostic/large_batch_capacity.jsonl"),
            "--source", *map(str, source_files),
            "--checkpoint", *map(str, checkpoint_files),
            "--output", str(freeze),
        ])

    heldout = FULL / "heldout/invariance.jsonl.gz"
    heldout_fidelity = FULL / "heldout/singleton_fidelity.jsonl.gz"
    run([
        str(PYTHON), "-u", "scripts/run_model_replay_invariance.py",
        "--adapter", ADAPTER, "--manifest", str(MANIFEST), "--freeze", str(freeze),
        "--output", str(heldout), "--fidelity-output", str(heldout_fidelity),
        "--split", "heldout", "--batch-sizes", "1", "2", "4", "8", "16", "32",
        "--resume",
    ])
    run([
        str(PYTHON), "scripts/summarize_model_replay_invariance.py",
        "--input", str(heldout), "--fidelity", str(heldout_fidelity),
        "--output", str(FULL / "heldout/summary.json"),
        "--flow-output", str(FULL / "heldout/flow_summary.json"),
        "--flow-steps", "256", "--flow-label-template", "flow.step_{step:03d}.prediction",
        "--variable-flow-steps",
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
    timing_summary(policy, FULL / "performance/policy_summary.json",
                   "scripts/summarize_model_policy_benchmark.py")

    for size in (16, 32):
        indices = replay_indices(size)
        for implementation in ("native", "batch_invariant"):
            raw = FULL / f"performance/policy_{implementation}_B{size}_timings.jsonl"
            run([
                str(PYTHON), "-u", "scripts/benchmark_model_policy.py",
                "--adapter", ADAPTER, "--batch-sizes", str(size),
                "--implementations", implementation,
                "--sessions", "5", "--warmup", "20", "--repetitions", "200",
                "--replay-manifest", str(MANIFEST), "--replay-indices", indices,
                "--output", str(raw), "--resume", "--full-campaign",
            ])
            timing_summary(
                raw, FULL / f"performance/policy_{implementation}_B{size}_summary.json",
                "scripts/summarize_model_policy_benchmark.py",
            )

    serving = FULL / "performance/serving_timings.jsonl"
    run([
        str(PYTHON), "-u", "scripts/benchmark_model_serving.py",
        "--adapter", ADAPTER, "--logical-batch-size", "4", "--fixed-batch-size", "8",
        "--sessions", "5", "--warmup", "20", "--repetitions", "200",
        "--replay-manifest", str(MANIFEST), "--replay-indices", base_indices,
        "--output", str(serving), "--resume", "--full-campaign",
    ])
    timing_summary(serving, FULL / "performance/serving_summary.json",
                   "scripts/summarize_model_serving_benchmark.py")

    large_serving = []
    for logical, fixed in ((8, 16), (16, 32)):
        indices = replay_indices(fixed)
        raw = FULL / f"performance/serving_fixed_B{fixed}_timings.jsonl"
        run([
            str(PYTHON), "-u", "scripts/benchmark_model_serving.py",
            "--adapter", ADAPTER, "--logical-batch-size", str(logical),
            "--fixed-batch-size", str(fixed), "--sessions", "5", "--warmup", "20",
            "--repetitions", "200", "--replay-manifest", str(MANIFEST),
            "--replay-indices", indices, "--output", str(raw), "--resume", "--full-campaign",
        ])
        timing_summary(
            raw, FULL / f"performance/serving_fixed_B{fixed}_summary.json",
            "scripts/summarize_model_serving_benchmark.py",
        )
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
        str(PYTHON), "scripts/summarize_model_request_queue.py",
        "--input", str(queue), "--output", str(FULL / "serving/summary.json"),
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

    run([
        str(PYTHON), "scripts/generate_full_path_report.py",
        "--model-id", "pi_fast", "--results", "results/cross_model",
    ])
    run_global_audit_and_require_pi_fast()


if __name__ == "__main__":
    main()
