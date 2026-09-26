"""Compare deterministic LingBot MoE inference with the released atomic kernel."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import deterministic_token_choice_moe
from scripts.model_invariance.harness import (
    _compose_batch,
    _run_once,
    compare_outputs,
    seed_everything,
)


def _comparison(reference, candidate):
    tensors = compare_outputs(reference, candidate)
    return {
        "exact": all(item.exact for item in tensors),
        "tensors": [asdict(item) for item in tensors],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.lingbot_vla2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = importlib.import_module(args.adapter).adapter
    from lingbotvla.models.vla.lingbot_vla import qwen2_action_expert

    released_kernel = qwen2_action_expert.robby_moe_forward
    seed_everything(args.seed)
    policy = adapter.load_model().eval()
    target = adapter.load_example_inputs(1)[0]
    batch = _compose_batch(adapter, [target])

    qwen2_action_expert.robby_moe_forward = deterministic_token_choice_moe
    deterministic = _run_once(adapter, policy, batch, args.seed)
    qwen2_action_expert.robby_moe_forward = released_kernel
    released_first = _run_once(adapter, policy, batch, args.seed)
    released_second = _run_once(adapter, policy, batch, args.seed)
    qwen2_action_expert.robby_moe_forward = deterministic_token_choice_moe

    report = {
        "model": adapter.name,
        "seed": args.seed,
        "released_repeatability": _comparison(released_first, released_second),
        "deterministic_vs_released_first": _comparison(deterministic, released_first),
        "deterministic_vs_released_second": _comparison(deterministic, released_second),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
