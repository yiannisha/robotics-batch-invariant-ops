"""Trace Cosmos 3 Nano's first attention divergence and flow propagation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts.model_invariance.harness import (
    _compose_batch,
    _first_sample,
    compare_outputs,
    seed_everything,
)


def _hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().reshape(-1).contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    candidate_full_shape = list(candidate.shape)
    if reference.ndim == candidate.ndim:
        candidate = candidate[tuple(slice(0, size) for size in reference.shape)]
    exact = reference.shape == candidate.shape and torch.equal(reference, candidate)
    report: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_full_shape": candidate_full_shape,
        "dtype": str(reference.dtype),
        "exact": exact,
        "reference_hash": _hash(reference),
        "candidate_hash": _hash(candidate),
    }
    if reference.shape == candidate.shape:
        different = reference != candidate
        locations = torch.nonzero(different, as_tuple=False)
        report["differing_elements"] = int(different.sum().item())
        report["first_differing_index"] = (
            [int(index) for index in locations[0].tolist()] if locations.numel() else None
        )
        if reference.is_floating_point():
            absolute = (reference.float() - candidate.float()).abs()
            report["max_abs_difference"] = float(absolute.max().item())
            report["mean_abs_difference"] = float(absolute.mean().item())
    return report


def _profile_summary(result) -> dict[str, list[str]]:
    return {
        "aten_operators": sorted(
            {
                event.name
                for event in result.events()
                if event.device_type == torch.autograd.DeviceType.CPU
                and event.name.startswith("aten::")
            }
        ),
        "cuda_kernels": sorted(
            {
                event.name
                for event in result.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
            }
        ),
    }


def _run(
    adapter,
    service,
    examples,
    batch_size: int,
    *,
    invariant: bool,
    disable_attention_repair: bool,
    seed: int,
):
    from cosmos_framework.model.generator.mot import attention as attention_module
    from cosmos_framework.model.generator.mot import (
        inference_text_kv_memory as text_kv_module,
    )

    captures: dict[str, list[torch.Tensor]] = {
        "flow.input": [],
        "flow.timestep": [],
        "flow.velocity": [],
    }
    attention_capture: dict[str, Any] = {}
    attention_metadata: list[dict[str, Any]] = []
    profiler_report: dict[str, Any] = {}
    original_get_velocity = service.model._get_velocity
    original_attention = attention_module.attention
    original_text_attention = text_kv_module.attention

    def traced_get_velocity(_self, *args, **kwargs):
        noise = kwargs["noise_x"]
        timestep = kwargs["timestep"]
        captures["flow.input"].append(noise[0].detach().contiguous().cpu())
        captures["flow.timestep"].append(timestep[:1].detach().contiguous().cpu())
        result = original_get_velocity(*args, **kwargs)
        captures["flow.velocity"].append(result[0].detach().contiguous().cpu())
        return result

    def traced_attention(original, source):
        def call(query, key, value, **kwargs):
            q_offsets = kwargs.get("cumulative_seqlen_Q")
            kv_offsets = kwargs.get("cumulative_seqlen_KV")
            q_length = int(q_offsets[1].item()) if q_offsets is not None else query.shape[1]
            kv_length = int(kv_offsets[1].item()) if kv_offsets is not None else key.shape[1]
            if len(attention_metadata) < 4:
                attention_metadata.append(
                    {
                        "source": source,
                        "uses_varlen": q_offsets is not None,
                        "is_causal": bool(kwargs.get("is_causal", False)),
                        "packed_query_shape": list(query.shape),
                        "packed_key_shape": list(key.shape),
                        "target_query_length": q_length,
                        "target_key_length": kv_length,
                    }
                )
            first = not attention_capture
            if first:
                attention_capture.update(
                    {
                        "source": source,
                        "uses_varlen": q_offsets is not None,
                        "packed_query_shape": list(query.shape),
                        "packed_key_shape": list(key.shape),
                        "query": query[:, :q_length].detach().contiguous().cpu(),
                        "key": key[:, :kv_length].detach().contiguous().cpu(),
                        "value": value[:, :kv_length].detach().contiguous().cpu(),
                    }
                )
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as result:
                    output = original(query, key, value, **kwargs)
                    torch.cuda.synchronize()
                profiler_report.update(_profile_summary(result))
                attention_capture["output"] = output[:, :q_length].detach().contiguous().cpu()
                return output
            return original(query, key, value, **kwargs)

        return call

    service.model._get_velocity = types.MethodType(traced_get_velocity, service.model)
    attention_module.attention = traced_attention(original_attention, "two_way_attention")
    text_kv_module.attention = traced_attention(original_text_attention, "cached_text_kv_attention")
    batch = _compose_batch(adapter, examples[:batch_size])
    seed_everything(seed)
    previous_disable = os.environ.get("_COSMOS3_DISABLE_ATTENTION_REPAIR")
    if disable_attention_repair:
        os.environ["_COSMOS3_DISABLE_ATTENTION_REPAIR"] = "1"
    else:
        os.environ.pop("_COSMOS3_DISABLE_ATTENTION_REPAIR", None)
    try:
        with set_batch_invariant_mode(invariant), torch.inference_mode():
            output = adapter.extract_robot_output(adapter.run_model(service, batch))
    finally:
        service.model._get_velocity = original_get_velocity
        attention_module.attention = original_attention
        text_kv_module.attention = original_text_attention
        if previous_disable is None:
            os.environ.pop("_COSMOS3_DISABLE_ATTENTION_REPAIR", None)
        else:
            os.environ["_COSMOS3_DISABLE_ATTENTION_REPAIR"] = previous_disable
    return captures, attention_capture, attention_metadata, output, profiler_report


def _capture_comparisons(reference, candidate):
    records = []
    for label in ("flow.input", "flow.timestep", "flow.velocity"):
        for call, (left, right) in enumerate(zip(reference[label], candidate[label])):
            records.append({"label": label, "call": call, **_comparison(left, right)})
    return records


def _attention_comparisons(reference, candidate):
    records = {}
    for label in ("query", "key", "value", "output"):
        left = reference[label]
        right = candidate[label]
        # The stock B=1 dense shortcut includes one trailing alignment token;
        # varlen metadata correctly excludes it. Compare the shared real-token
        # prefix while retaining both full shapes as dispatch evidence.
        sequence_length = min(left.shape[1], right.shape[1])
        record = _comparison(left[:, :sequence_length], right[:, :sequence_length])
        record["reference_full_shape"] = list(left.shape)
        record["candidate_full_shape"] = list(right.shape)
        record["compared_sequence_length"] = sequence_length
        records[label] = record
    return records


def _output_comparisons(reference, candidate):
    return [asdict(item) for item in compare_outputs(reference, _first_sample(candidate))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter", default="scripts.model_invariance.adapters.cosmos3_nano_policy"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = importlib.import_module(args.adapter).adapter
    service = adapter.load_model()
    if hasattr(service, "eval"):
        service.eval()
    examples = list(adapter.load_example_inputs(2))
    runs = {}
    for mode, invariant, disable_attention_repair in (
        ("stock", False, False),
        ("generic_operators_only", True, True),
        ("fixed", True, False),
    ):
        b1 = _run(
            adapter,
            service,
            examples,
            1,
            invariant=invariant,
            disable_attention_repair=disable_attention_repair,
            seed=args.seed,
        )
        b2 = _run(
            adapter,
            service,
            examples,
            2,
            invariant=invariant,
            disable_attention_repair=disable_attention_repair,
            seed=args.seed,
        )
        runs[mode] = {
            "attention_dispatch": {
                "batch_1": {
                    key: value
                    for key, value in b1[1].items()
                    if not isinstance(value, torch.Tensor)
                },
                "batch_2": {
                    key: value
                    for key, value in b2[1].items()
                    if not isinstance(value, torch.Tensor)
                },
                "first_four_calls_batch_1": b1[2],
                "first_four_calls_batch_2": b2[2],
            },
            "first_attention": _attention_comparisons(b1[1], b2[1]),
            "flow_iterations": _capture_comparisons(b1[0], b2[0]),
            "final_outputs": _output_comparisons(b1[3], b2[3]),
            "first_attention_profile": {"batch_1": b1[4], "batch_2": b2[4]},
        }

    report = {
        "model": adapter.name,
        "comparison": "B=1 target versus first sample at unrelated B=2",
        "root_cause": (
            "After generic MM/BMM projection repairs, Cosmos still selects dense "
            "cuDNN attention at B=1 and packed variable-length CUTLASS attention "
            "at B>1; cached text-KV steps repeat the switch"
        ),
        "runs": runs,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
