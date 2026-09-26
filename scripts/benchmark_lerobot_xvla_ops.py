"""Benchmark X-VLA's divergent Florence-2 Conv2d and MM shapes."""

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

from batch_invariant_ops import conv2d_batch_invariant, matmul_persistent
from scripts.benchmark_batch_invariant_ops import run_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,8")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dtype = torch.float32
    generator = torch.Generator(device="cuda").manual_seed(91_500)
    cases = []
    for batch_size in (int(value) for value in args.batch_sizes.split(",")):
        # X-VLA flattens its two valid camera views before Florence-2.
        images = torch.randn(
            2 * batch_size,
            512,
            28,
            28,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        conv_weight = torch.randn(1_024, 512, 3, 3, device="cuda", dtype=dtype, generator=generator)
        conv_bias = torch.randn(1_024, device="cuda", dtype=dtype, generator=generator)
        cases.append(
            run_case(
                "xvla_florence_stage3_patch_conv2d",
                {
                    "input": list(images.shape),
                    "weight": list(conv_weight.shape),
                    "bias": list(conv_bias.shape),
                    "stride": [2, 2],
                    "padding": [1, 1],
                    "dtype": str(dtype),
                },
                lambda: torch.nn.functional.conv2d(
                    images, conv_weight, conv_bias, stride=(2, 2), padding=(1, 1)
                ),
                lambda: conv2d_batch_invariant(
                    images,
                    conv_weight,
                    conv_bias,
                    (2, 2),
                    (1, 1),
                    (1, 1),
                    False,
                    (0, 0),
                    1,
                ),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )
        del images, conv_weight, conv_bias

        projector_input = torch.randn(
            2 * batch_size * 50,
            2_048,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        projector_weight = torch.randn(
            2_048, 1_024, device="cuda", dtype=dtype, generator=generator
        )
        cases.append(
            run_case(
                "xvla_florence_image_projection_mm",
                {
                    "left": list(projector_input.shape),
                    "right": list(projector_weight.shape),
                    "logical_input": [2 * batch_size, 50, 2_048],
                    "dtype": str(dtype),
                },
                lambda: torch.mm(projector_input, projector_weight),
                lambda: matmul_persistent(projector_input, projector_weight),
                batch_size=batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        )
        del projector_input, projector_weight

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
