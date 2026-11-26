"""Run the verified fine-tuned PI05 policy on real LIBERO observations.

This is a numerical integration probe, not a held-out or success-rate campaign.
The unchanged official sampler receives explicit per-request noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from verify_pi05_libero_checkpoint import load_verified_policy
from qualify_simpler_fixed_action_replay import digest


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    from batch_invariant_ops import set_batch_invariant_mode
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep
    from lerobot.policies.factory import make_pre_post_processors

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    policy, loading = load_verified_policy(args.checkpoint)
    policy.to("cuda").eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": str(args.tokenizer)},
                                "device_processor": {"device": "cpu"}})
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    requests = []
    source_observations = []

    def add_batch(value):
        if isinstance(value, np.ndarray):
            return value[None]
        if isinstance(value, dict):
            return {key: add_batch(item) for key, item in value.items()}
        return value

    for task in range(3):
        seed = 20261108 + task
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = LiberoEnv(suite, task, "libero_10", obs_type="pixels_agent_pos")
        try:
            env.init_state_id = 0
            observation, _ = env.reset(seed=seed)
            batch = preprocess_observation(add_batch(observation))
            batch = LiberoProcessorStep().observation(batch)
            batch["task"] = [env.task_description]
            inputs = preprocess(batch)
            inputs = {key: value for key, value in inputs.items() if isinstance(value, torch.Tensor)}
            noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim,
                                generator=torch.Generator().manual_seed(seed))
            requests.append({"inputs": inputs, "noise": noise})
            source_observations.append({"task": task, "initialization": 0,
                                        "instruction": env.task_description,
                                        "observation_sha256": digest(observation)})
        finally:
            env.close()
    torch.save(requests, args.output / "requests.pt")
    runtime = {"scope": "real_input_numerical_integration_probe", "loading": loading,
               "source_observations": source_observations,
               "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
               "batch_sizes": args.batch_sizes, "sampler": "official_predict_action_chunk",
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "requests_sha256": hashlib.sha256((args.output / "requests.pt").read_bytes()).hexdigest()}
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2, default=str) + "\n")
    with set_batch_invariant_mode(True):
        pass
    records = []

    def run(indices, mode):
        batch = {key: torch.cat([requests[i]["inputs"][key] for i in indices]).to("cuda")
                 for key in requests[0]["inputs"]}
        noise = torch.cat([requests[i]["noise"] for i in indices]).to("cuda")
        policy.reset()
        with torch.inference_mode(), set_batch_invariant_mode(mode):
            normalized = policy.predict_action_chunk(batch, noise=noise.clone())
            actions = postprocess(normalized)
        return normalized.detach().cpu(), actions.detach().cpu()

    for invariant in (False, True):
        for target in range(len(requests)):
            reference, reference_actions = run([target], invariant)
            repeat, repeated_actions = run([target], invariant)
            if tensor_digest(reference) != tensor_digest(repeat) or tensor_digest(reference_actions) != tensor_digest(repeated_actions):
                raise RuntimeError(f"same-condition B1 repeat mismatch: invariant={invariant}, target={target}")
            for size in args.batch_sizes:
                for composition in ("duplicate", "unrelated"):
                    for position in sorted({0, size // 2, size - 1}):
                        indices = [target if composition == "duplicate" else (target + j + 1) % 3
                                   for j in range(size - 1)]
                        indices.insert(position, target)
                        normalized, actions = run(indices, invariant)
                        selected = normalized[position:position+1]
                        selected_actions = actions[position:position+1]
                        record = {"invariant": invariant, "target": target, "batch_size": size,
                                  "position": position, "composition": composition, "ordering": indices,
                                  "normalized_exact": tensor_digest(selected) == tensor_digest(reference),
                                  "actions_exact": tensor_digest(selected_actions) == tensor_digest(reference_actions),
                                  "max_normalized_error": float((selected-reference).abs().max()),
                                  "max_action_error": float((selected_actions-reference_actions).abs().max()),
                                  "normalized_actions": selected.tolist(), "actions": selected_actions.tolist()}
                        with (args.output / "comparisons.jsonl").open("a") as stream:
                            stream.write(json.dumps(record) + "\n")
                        records.append(record)
                        print(f"invariant={invariant} target={target} B={size} pos={position} {composition}: exact={record['actions_exact']}", flush=True)
    summary = {"scope": runtime["scope"], "comparisons": len(records),
               "native_failures": sum(not r["actions_exact"] for r in records if not r["invariant"]),
               "invariant_failures": sum(not r["actions_exact"] or not r["normalized_exact"] for r in records if r["invariant"]),
               "same_condition_b1_repeats_exact": True, "success_rate_campaign_complete": False}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["invariant_failures"]:
        raise RuntimeError("invariant real-input probe failed; preserve evidence and investigate")


if __name__ == "__main__":
    main()
