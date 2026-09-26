"""Locate the first batch-dependent module in the Cosmos 3 Wan video decoder."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts.model_invariance.harness import _compose_batch, seed_everything


def _fingerprint(tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().contiguous().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def _capture_decode(decoder, decode, latent: torch.Tensor, reference=None):
    captures: dict[str, list[dict[str, object]]] = defaultdict(list)
    handles = []

    for name, module in decoder.named_modules():
        if name and not any(module.children()):

            def hook(_module, _inputs, output, *, name=name):
                if not isinstance(output, torch.Tensor):
                    return
                value = output
                call = len(captures[name])
                if reference is not None and call < len(reference.get(name, [])):
                    reference_shape = reference[name][call]["shape"]
                    if (
                        value.ndim > 0
                        and len(reference_shape) == value.ndim
                        and list(value.shape[1:]) == reference_shape[1:]
                        and value.shape[0] >= reference_shape[0]
                    ):
                        value = value[: reference_shape[0]]
                captures[name].append(_fingerprint(value))

            handles.append(module.register_forward_hook(hook))

    try:
        output = decode(latent)
    finally:
        for handle in handles:
            handle.remove()
    return dict(captures), _fingerprint(output[:1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter", default="scripts.model_invariance.adapters.cosmos3_edge_policy"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = importlib.import_module(args.adapter).adapter
    seed_everything(args.seed)
    service = adapter.load_model()
    example = adapter.load_example_inputs(1)[0]

    with set_batch_invariant_mode():
        generated = adapter.run_model(service, _compose_batch(adapter, [example]))
        latent = generated["vision_latent"]
        tokenizer = service.model.tokenizer_vision_gen
        decoder = tokenizer.model.model
        reference, reference_output = _capture_decode(
            decoder, service.model.decode, latent
        )
        candidate, candidate_output = _capture_decode(
            decoder, service.model.decode, latent.repeat(2, 1, 1, 1, 1), reference
        )

    comparisons = []
    for name, reference_calls in reference.items():
        candidate_calls = candidate.get(name, [])
        for call, reference_record in enumerate(reference_calls):
            candidate_record = candidate_calls[call] if call < len(candidate_calls) else None
            comparisons.append(
                {
                    "module": name,
                    "module_type": type(decoder.get_submodule(name)).__name__,
                    "call": call,
                    "exact": candidate_record == reference_record,
                    "reference": reference_record,
                    "candidate": candidate_record,
                }
            )

    first_divergence = next((item for item in comparisons if not item["exact"]), None)
    first_divergence_index = next(
        (index for index, item in enumerate(comparisons) if not item["exact"]), None
    )
    report = {
        "model": adapter.name,
        "comparison": "identical latent at B=1 versus first sample at duplicate B=2",
        "latent": _fingerprint(latent),
        "reference_output": reference_output,
        "candidate_output_first_sample": candidate_output,
        "first_divergence": first_divergence,
        "total_comparisons": len(comparisons),
        "exact_comparisons_before_first_divergence": first_divergence_index,
        "differing_comparisons": sum(not item["exact"] for item in comparisons),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
