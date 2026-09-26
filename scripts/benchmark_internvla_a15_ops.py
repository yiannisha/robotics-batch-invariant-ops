"""Benchmark the operator shapes exercised by InternVLA-A1.5."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch
import triton

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import (
    bmm_persistent,
    conv1d_batch_invariant,
    conv3d_batch_invariant,
)
from scripts.benchmark_batch_invariant_ops import run_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,8")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dtype = torch.bfloat16
    cases = []
    for batch_size in (int(value) for value in args.batch_sizes.split(",")):
        generator = torch.Generator(device="cuda").manual_seed(21_000 + batch_size)

        probabilities = torch.randn(
            batch_size * 8, 58, 708, device="cuda", dtype=dtype, generator=generator
        )
        values = torch.randn(
            batch_size * 8, 708, 256, device="cuda", dtype=dtype, generator=generator
        )
        cases.append(
            run_case(
                "internvla_action_attention_value_bmm",
                {
                    "left": list(probabilities.shape),
                    "right": list(values.shape),
                    "dtype": str(dtype),
                },
                lambda: torch.bmm(probabilities, values),
                lambda: bmm_persistent(probabilities, values),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )
        del probabilities, values

        patches = torch.randn(
            batch_size * 512,
            3,
            2,
            16,
            16,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        patch_weight = torch.randn(
            1024, 3, 2, 16, 16, device="cuda", dtype=dtype, generator=generator
        )
        patch_bias = torch.randn(1024, device="cuda", dtype=dtype, generator=generator)
        cases.append(
            run_case(
                "qwen35_vision_patch_conv3d",
                {
                    "input": list(patches.shape),
                    "weight": list(patch_weight.shape),
                    "stride": [2, 16, 16],
                    "dtype": str(dtype),
                },
                lambda: torch.nn.functional.conv3d(
                    patches, patch_weight, patch_bias, stride=(2, 16, 16)
                ),
                lambda: conv3d_batch_invariant(
                    patches,
                    patch_weight,
                    patch_bias,
                    (2, 16, 16),
                    (0, 0, 0),
                    (1, 1, 1),
                    False,
                    (0, 0, 0),
                    1,
                ),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )
        del patches, patch_weight, patch_bias

        linear_states = torch.randn(
            batch_size,
            6144,
            650,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        causal_weight = torch.randn(6144, 1, 4, device="cuda", dtype=dtype, generator=generator)
        causal_bias = torch.randn(6144, device="cuda", dtype=dtype, generator=generator)
        cases.append(
            run_case(
                "qwen35_depthwise_causal_conv1d",
                {
                    "input": list(linear_states.shape),
                    "weight": list(causal_weight.shape),
                    "padding": [3],
                    "groups": 6144,
                    "dtype": str(dtype),
                },
                lambda: torch.nn.functional.conv1d(
                    linear_states, causal_weight, causal_bias, padding=3, groups=6144
                ),
                lambda: conv1d_batch_invariant(
                    linear_states,
                    causal_weight,
                    causal_bias,
                    (1,),
                    (3,),
                    (1,),
                    False,
                    (0,),
                    6144,
                ),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )
        del linear_states, causal_weight, causal_bias

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    report = {
        "environment": {
            "gpu": properties.name,
            "gpu_memory_bytes": properties.total_memory,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "triton": triton.__version__,
        },
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "cases": cases,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
