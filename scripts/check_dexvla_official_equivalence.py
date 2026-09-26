"""Check that the generalized DexVLA adapter preserves official B=1 inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.model_invariance.harness import _compose_batch, seed_everything


def _hash(value: torch.Tensor) -> str:
    raw = value.detach().cpu().reshape(-1).contiguous().view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def _comparison(reference: torch.Tensor, candidate: torch.Tensor):
    reference = reference.detach().cpu()
    candidate = candidate.detach().cpu()
    difference = reference != candidate
    absolute = (reference.float() - candidate.float()).abs()
    return {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "exact": torch.equal(reference, candidate),
        "differing_elements": int(difference.sum().item()),
        "max_abs_difference": float(absolute.max().item()),
        "reference_hash": _hash(reference),
        "candidate_hash": _hash(candidate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="scripts.model_invariance.adapters.dexvla")
    parser.add_argument("--noise-seed", type=int, default=96_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = importlib.import_module(args.adapter).adapter
    policy = adapter.load_model().eval()
    example = adapter.load_example_inputs(1)[0]
    batch = _compose_batch(adapter, [example])
    model = policy.model
    device_batch = {key: value.to(adapter.device) for key, value in batch.items() if key != "noise"}

    explicit_noise = batch["noise"]
    original_randn = torch.randn

    def controlled_randn(*shape, **kwargs):
        requested = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        if tuple(requested) == tuple(explicit_noise.shape):
            return explicit_noise.clone()
        return original_randn(*shape, **kwargs)

    seed_everything(args.noise_seed)
    with patch("torch.randn", controlled_randn), torch.inference_mode():
        official_actions, official_text = model.evaluate(
            input_ids=device_batch["input_ids"],
            actions=None,
            states=device_batch["states"],
            is_pad=None,
            tokenizer=policy.processor.tokenizer,
            is_eval=True,
            pixel_values=device_batch["pixel_values"],
            attention_mask=device_batch["attention_mask"],
            image_grid_thw=device_batch["image_grid_thw"],
        )

    captures = {}
    adapter.trace_callback = lambda label, value: captures.__setitem__(
        label, value.detach().clone()
    )
    seed_everything(args.noise_seed)
    try:
        with torch.inference_mode():
            adapter.run_model(policy, batch)
    finally:
        adapter.trace_callback = None

    comparison = _comparison(official_actions, captures["actions.normalized"])
    generated = captures["reasoning.tokens"][:, batch["input_ids"].shape[1] :]
    adapter_text = policy.processor.tokenizer.batch_decode(generated, skip_special_tokens=False)[
        0
    ].strip()
    report = {
        "official_method": "Qwen2VLForConditionalGenerationForVLA.evaluate",
        "noise_seed": args.noise_seed,
        "noise_injection": "official torch.randn replaced with adapter explicit noise",
        "normalized_actions": comparison,
        "official_text": official_text,
        "adapter_text": adapter_text,
        "text_exact": official_text == adapter_text,
        "exact": comparison["exact"] and official_text == adapter_text,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["exact"] else 1)


if __name__ == "__main__":
    main()
