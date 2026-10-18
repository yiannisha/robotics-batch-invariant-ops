"""Resume every non-closed-loop stage required for full X-VLA coverage."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "results/cross_model/xvla/full"
MANIFEST = FULL / "replay_manifest.json"
ADAPTER = "scripts.model_invariance.adapters.lerobot_xvla"
PYTHON = Path("/workspace/envs/smolvla/bin/python")
CHECKPOINT = Path("/workspace/assets/xvla/checkpoint")
MODEL_SOURCES = Path("/workspace/models/lerobot/src/lerobot/policies/xvla")


def run(arguments: list[str]) -> None:
    print("RUN", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, env=os.environ.copy(), check=True)


def replay_indices(count: int) -> str:
    observations = [row for row in json.loads(MANIFEST.read_text())["observations"]
                    if row.get("split") == "diagnostic"]
    # Cover every available task first, then every available source episode,
    # before filling from additional observations.  Timing/serving inputs are
    # required to maximize both task and episode diversity; merely choosing a
    # different frame from an already selected episode does not satisfy that
    # contract.
    selected = []
    seen_tasks = set()
    for row in observations:
        if row["task_index"] not in seen_tasks:
            selected.append(row["replay_index"])
            seen_tasks.add(row["task_index"])
    seen_episodes = {
        (row["task_index"], row["episode_id"])
        for row in observations if row["replay_index"] in set(selected)
    }
    for row in observations:
        episode = (row["task_index"], row["episode_id"])
        if episode not in seen_episodes:
            selected.append(row["replay_index"])
            seen_episodes.add(episode)
    selected.extend(row["replay_index"] for row in observations
                    if row["replay_index"] not in set(selected))
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} diagnostic inputs for B{count}")
    return ",".join(map(str, selected[:count]))


def main() -> None:
    os.environ.update({
        "PYTHONPATH": "/workspace/models/lerobot/src:/workspace/robotics-batch-invariant-ops",
        "LEROBOT_XVLA_CHECKPOINT": str(CHECKPOINT),
        "LEROBOT_XVLA_SAMPLE_DIR": "/workspace/assets/smolvla/full_replay",
        "XVLA_REPLAY_MANIFEST": str(MANIFEST),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    diagnostic = FULL / "diagnostic/invariance.jsonl.gz"
    diagnostic_fidelity = FULL / "diagnostic/singleton_fidelity.jsonl.gz"
    run([
        str(PYTHON), "-u", "scripts/run_model_replay_invariance.py",
        "--adapter", ADAPTER, "--manifest", str(MANIFEST),
        "--output", str(diagnostic), "--fidelity-output", str(diagnostic_fidelity),
        "--split", "diagnostic", "--batch-sizes", "1", "2", "4", "8", "16", "32",
        "--resume",
    ])

    run([
        str(PYTHON), "-u", "scripts/run_model_causal_ablation.py",
        "--adapter", ADAPTER, "--manifest", str(MANIFEST),
        "--output", str(FULL / "diagnostic/causal_ablation.jsonl"), "--resume",
    ])

    freeze = FULL / "numerical_freeze.json"
    if not freeze.is_file():
        source_files = [
            ROOT / "scripts/model_invariance/adapters/lerobot_xvla.py",
            *(MODEL_SOURCES / name for name in (
                "action_hub.py", "configuration_xvla.py", "modeling_xvla.py",
                "processor_xvla.py", "soft_transformer.py", "utils.py",
            )),
        ]
        checkpoint_files = [
            CHECKPOINT / name for name in (
                "model.safetensors", "config.json", "policy_preprocessor.json",
                "policy_postprocessor.json",
            )
        ]
        run([
            str(PYTHON), "scripts/freeze_model_numerics.py",
            "--model-id", "xvla", "--manifest", str(MANIFEST),
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
        "--flow-steps", "10", "--flow-label-template", "flow.step_{step:02d}.prediction",
    ])

    run([
        str(PYTHON), "-u", "scripts/qualify_real_shape_matrix_ops.py",
        "--config", str(FULL / "operator/configurations.json"),
        "--seeds", "100", "--output", str(FULL / "operator/raw.jsonl"), "--resume",
    ])

    for logical, fixed in ((8, 16), (16, 32)):
        suffix = f"B{fixed}"
        indices = replay_indices(fixed)
        timings = FULL / f"performance/serving_fixed_{suffix}_timings.jsonl"
        run([
            str(PYTHON), "-u", "scripts/benchmark_model_serving.py",
            "--adapter", ADAPTER, "--logical-batch-size", str(logical),
            "--fixed-batch-size", str(fixed), "--sessions", "5", "--warmup", "20",
            "--repetitions", "200", "--replay-manifest", str(MANIFEST),
            "--replay-indices", indices, "--output", str(timings),
            "--resume", "--full-campaign",
        ])
        run([
            str(PYTHON), "scripts/summarize_model_serving_benchmark.py",
            "--input", str(timings),
            "--output", str(FULL / f"performance/serving_fixed_{suffix}_summary.json"),
        ])
        requests = FULL / f"serving/requests_fixed_{suffix}.jsonl"
        run([
            str(PYTHON), "-u", "scripts/benchmark_model_request_queue.py",
            "--adapter", ADAPTER, "--serving-timings", str(timings),
            "--logical-batch-size", str(logical), "--fixed-batch-size", str(fixed),
            "--requests-per-group", "250", "--replay-manifest", str(MANIFEST),
            "--replay-indices", indices, "--output", str(requests), "--resume",
        ])
        run([
            str(PYTHON), "scripts/summarize_model_request_queue.py",
            "--input", str(requests),
            "--output", str(FULL / f"serving/summary_fixed_{suffix}.json"),
        ])

    run([
        str(PYTHON), "scripts/generate_full_path_report.py",
        "--model-id", "xvla", "--results", "results/cross_model",
    ])


if __name__ == "__main__":
    main()
