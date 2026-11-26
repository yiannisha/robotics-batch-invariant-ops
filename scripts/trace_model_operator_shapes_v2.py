"""Trace exercised ATen operator shapes/strides on a real policy request.

This is provenance for selecting 100-seed qualification cases, not itself an
operator qualification or a policy-level invariance result.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.model_invariance.harness import _compose_batch, _run_once
from batch_invariant_ops import set_batch_invariant_mode


SELECTED = {
    "aten::mm", "aten::addmm", "aten::bmm", "aten::convolution",
    "aten::_softmax", "aten::mean", "aten::scaled_dot_product_attention",
    "aten::linear", "aten::matmul", "aten::conv2d", "aten::softmax",
}


def describe(value):
    if isinstance(value, torch.Tensor):
        result = {
            "shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "device": value.device.type,
        }
        if value.dtype == torch.bool:
            compact = value
            if value.numel() > 4096:
                # Record broadcast masks without repeating a zero-stride
                # dimension thousands of times. Keep original shape/strides.
                selection = tuple(slice(0, 1) if stride == 0 else slice(None)
                                  for stride in value.stride())
                compact = value[selection]
            if compact.numel() <= 4096:
                result["values"] = compact.detach().cpu().tolist()
                if compact.shape != value.shape:
                    result["values_shape"] = list(compact.shape)
        return result
    if isinstance(value, (list, tuple)):
        return [describe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class ShapeRecorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.records = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name
        if name in SELECTED:
            record = {
                "aten_operation": name,
                "arguments": describe(args),
                "keyword_arguments": describe(kwargs or {}),
            }
            self.records[json.dumps(record, sort_keys=True)] += 1
        return func(*args, **(kwargs or {}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--replay-indices",
        help="comma-separated diagnostic replay indices; must match --batch-size",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--implementation", choices=("native", "batch_invariant"), default="native"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    adapter = importlib.import_module(args.adapter).adapter
    model = adapter.load_model()
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError("Operator trace requires TF32 disabled after model construction")
    if hasattr(model, "eval"):
        model.eval()
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        provider = getattr(adapter, "replay_manifest_path", None)
        if provider is None and hasattr(adapter, "sample_dir"):
            provider = adapter.sample_dir / "replay_manifest.json"
        if provider is None or hashlib.sha256(Path(provider).read_bytes()).digest() != hashlib.sha256(args.manifest.read_bytes()).digest():
            raise ValueError("adapter and selected real-input manifests differ")
        diagnostic = [row for row in manifest["observations"] if row["split"] == "diagnostic"]
        if args.replay_indices:
            indexes = [int(value) for value in args.replay_indices.split(",")]
            by_index = {row["replay_index"]: row for row in diagnostic}
            if len(indexes) != args.batch_size or len(set(indexes)) != args.batch_size:
                raise ValueError("replay-indices must contain batch-size distinct entries")
            if any(index not in by_index for index in indexes):
                raise ValueError("operator trace replay indices must belong to the diagnostic split")
            selected = [by_index[index] for index in indexes]
        else:
            selected = diagnostic[: args.batch_size]
        examples = [adapter.load_replay_input(row["replay_index"], 0) for row in selected]
        request_ids = [row["request_ids"][0] for row in selected]
    else:
        examples = list(adapter.load_example_inputs(args.batch_size))
        request_ids = None
    batch = _compose_batch(adapter, examples)
    recorder = ShapeRecorder()
    mode = (
        set_batch_invariant_mode()
        if args.implementation == "batch_invariant" else contextlib.nullcontext()
    )
    with mode, recorder:
        _run_once(adapter, model, batch, args.seed)
        torch.cuda.synchronize()
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError("Operator trace precision changed during inference")
    if not recorder.records:
        raise RuntimeError("no selected ATen operations were captured")
    report = {
        "schema_version": 2,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "adapter": args.adapter,
        "adapter_execution_configuration": (
            adapter.execution_configuration()
            if hasattr(adapter, "execution_configuration") else None
        ),
        "implementation": args.implementation,
        "batch_size": args.batch_size,
        "request_ids": request_ids,
        "replay_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest() if args.manifest else None,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "operations": [dict(json.loads(key), count=count) for key, count in sorted(recorder.records.items())],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    Path(str(args.output) + '.driver.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"operations": len(report["operations"]), "calls": sum(recorder.records.values())}))


if __name__ == "__main__":
    main()
