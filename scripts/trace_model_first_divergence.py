"""Locate the first divergent leaf-module output for a campaign adapter."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from scripts.model_invariance.harness import (
    _compose_batch,
    _first_sample,
    _run_once,
    compare_outputs,
    seed_everything,
)


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    for attribute in ("pooler_output", "last_hidden_state", "logits"):
        tensor = getattr(value, attribute, None)
        if isinstance(tensor, torch.Tensor):
            return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _hash(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _target_like(reference: torch.Tensor, candidate: torch.Tensor):
    if candidate.shape == reference.shape:
        return candidate
    if (
        reference.ndim > 0
        and candidate.ndim == reference.ndim
        and candidate.shape[1:] == reference.shape[1:]
        and candidate.shape[0] >= reference.shape[0]
    ):
        return candidate[: reference.shape[0]]
    return None


def _comparison(reference, candidate) -> dict:
    return {"exact": all(item.exact for item in records), "tensors": [asdict(item) for item in records]} if (records := compare_outputs(reference, candidate)) else {"exact": True, "tensors": []}


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return type(value)((key, _to_cpu(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    raise TypeError(f"unsupported output type {type(value).__name__}")


def _named_modules(model):
    if isinstance(model, torch.nn.Module):
        return list(model.named_modules())
    modules = []
    for attribute, value in vars(model).items():
        if isinstance(value, torch.nn.Module):
            for name, module in value.named_modules():
                modules.append((attribute if not name else f"{attribute}.{name}", module))
    if not modules:
        raise TypeError(f"cannot discover modules in {type(model).__name__}")
    return modules


def _trace(
    adapter,
    model,
    batch,
    seed: int,
    references=None,
    module_prefix: str | None = None,
    invariant: bool = False,
):
    records = []
    occurrences = {}
    handles = []

    def register(name, module):
        def hook(_module, _inputs, output):
            tensor = _first_tensor(output)
            if tensor is None:
                return
            occurrence = occurrences.get(name, 0)
            occurrences[name] = occurrence + 1
            record = {
                "name": name,
                "occurrence": occurrence,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
            if references is None:
                record["hash"] = _hash(tensor)
            else:
                index = len(records)
                comparable = index < len(references)
                if comparable:
                    reference = references[index]
                    comparable = reference["name"] == name and reference["occurrence"] == occurrence
                if comparable:
                    placeholder = torch.empty(tuple(reference["shape"]), dtype=tensor.dtype)
                    target = _target_like(placeholder, tensor)
                    comparable = target is not None
                record["comparable"] = comparable
                record["target_hash"] = _hash(target) if comparable else None
                record["exact"] = comparable and record["target_hash"] == reference["hash"]
            records.append(record)

        try:
            handles.append(module.register_forward_hook(hook))
        except RuntimeError as error:
            # TorchScript modules (for example released video tokenizers) do
            # not support Python forward hooks. Other eager leaves still give
            # a valid ordered first-divergence scan.
            if "register_forward_hook is not supported on ScriptModules" not in str(error):
                raise

    for name, module in _named_modules(model):
        if (
            name
            and (module_prefix is None or name.startswith(module_prefix))
            and not any(module.children())
        ):
            register(name, module)
    try:
        mode = set_batch_invariant_mode() if invariant else contextlib.nullcontext()
        with mode:
            _run_once(adapter, model, batch, seed)
    finally:
        for handle in handles:
            handle.remove()
    return records


def _capture(adapter, model, batch, seed, module, occurrence, invariant):
    captured = []

    def hook(_module, _inputs, output):
        tensor = _first_tensor(output)
        if tensor is not None:
            captured.append(tensor.detach().cpu())

    handle = module.register_forward_hook(hook)
    mode = set_batch_invariant_mode() if invariant else contextlib.nullcontext()
    try:
        with mode:
            output = _run_once(adapter, model, batch, seed)
    finally:
        handle.remove()
    if occurrence >= len(captured):
        raise RuntimeError(f"missing occurrence {occurrence}; observed {len(captured)}")
    return captured[occurrence], _to_cpu(_first_sample(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-batch-size", type=int, default=2)
    parser.add_argument("--replay-manifest", type=Path)
    parser.add_argument("--target-replay-index", type=int, default=0)
    parser.add_argument("--noise-id", type=int, default=0)
    parser.add_argument(
        "--invariant-only",
        action="store_true",
        help="scan and capture only the batch-invariant execution path",
    )
    parser.add_argument(
        "--module-prefix",
        help="limit the scan to this named-module prefix (useful around CUDA graphs)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.candidate_batch_size < 2:
        parser.error("--candidate-batch-size must be at least 2")
    if args.output.exists() and not args.force:
        raise FileExistsError(args.output)

    adapter = importlib.import_module(args.adapter).adapter
    seed_everything(args.seed)
    model = adapter.load_model()
    if hasattr(model, "eval"):
        model.eval()
    manifest_hash = None
    if args.replay_manifest:
        if not hasattr(adapter, "load_replay_input"):
            raise TypeError("adapter does not support frozen real-input replay")
        manifest_bytes = args.replay_manifest.read_bytes()
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        provider = getattr(adapter, "replay_manifest_path", None)
        if provider is None and hasattr(adapter, "sample_dir"):
            provider = adapter.sample_dir / "replay_manifest.json"
        if provider is None or hashlib.sha256(Path(provider).read_bytes()).hexdigest() != manifest_hash:
            raise ValueError("adapter replay manifest differs from the selected frozen manifest")
        manifest = json.loads(manifest_bytes)
        rows = manifest["observations"]
        target = rows[args.target_replay_index]
        if target["replay_index"] != args.target_replay_index:
            raise ValueError("replay observation order is not indexed by replay_index")
        companion_indexes = target["companion_replay_indices"][: args.candidate_batch_size - 1]
        if len(companion_indexes) != args.candidate_batch_size - 1:
            raise ValueError("frozen companion schedule is too short")
        companions = [rows[index] for index in companion_indexes]
        if any(row["episode_id"] == target["episode_id"] for row in companions):
            raise ValueError("trace requires companions from unrelated episodes")
        examples = [adapter.load_replay_input(args.target_replay_index, args.noise_id)]
        examples.extend(adapter.load_replay_input(index, args.noise_id) for index in companion_indexes)
    else:
        examples = list(adapter.load_example_inputs(args.candidate_batch_size))
    singleton = _compose_batch(adapter, examples[:1])
    paired = _compose_batch(adapter, examples)

    reference_trace = _trace(
        adapter,
        model,
        singleton,
        args.seed,
        module_prefix=args.module_prefix,
        invariant=args.invariant_only,
    )
    candidate_trace = _trace(
        adapter,
        model,
        paired,
        args.seed,
        reference_trace,
        module_prefix=args.module_prefix,
        invariant=args.invariant_only,
    )
    first_index = next(
        (
            index
            for index, record in enumerate(candidate_trace)
            if record.get("comparable") and not record.get("exact")
        ),
        None,
    )
    if first_index is None:
        structural_index = next(
            (index for index, record in enumerate(candidate_trace) if not record.get("comparable")),
            None,
        )
        modes = {}
        for label, invariant in (
            (("batch_invariant", True),) if args.invariant_only else
            (("native", False), ("batch_invariant", True))
        ):
            mode = set_batch_invariant_mode() if invariant else contextlib.nullcontext()
            with mode:
                reference_output = _to_cpu(_first_sample(_run_once(adapter, model, singleton, args.seed)))
                candidate_output = _to_cpu(_first_sample(_run_once(adapter, model, paired, args.seed)))
            modes[label] = {"final_output": _comparison(reference_output, candidate_output)}
        report = {
            "model": adapter.name,
            "adapter": args.adapter,
            "seed": args.seed,
            "batch_size": args.candidate_batch_size,
            "composition": "unrelated" if args.replay_manifest else "fixture_sample",
            "replay_manifest_sha256": manifest_hash,
            "target_replay_index": args.target_replay_index if args.replay_manifest else None,
            "noise_id": args.noise_id if args.replay_manifest else None,
            "companion_replay_indices": companion_indexes if args.replay_manifest else None,
            "module_prefix": args.module_prefix,
            "host_leaf_scan": {
                "comparable_leaf_outputs": sum(bool(item.get("comparable")) for item in candidate_trace),
                "first_divergent_index": None,
                "first_structural_mismatch_index": structural_index,
                "first_structural_mismatch": (
                    {"reference": reference_trace[structural_index], "candidate": candidate_trace[structural_index]}
                    if structural_index is not None else None
                ),
            },
            "native_first_divergence": None,
            "modes": modes,
            "batch_invariant_boundary_exact": None,
            "limitation": "No divergent comparable leaf was found; a variable-shape structural boundary cannot be assigned a numeric first-divergence operator.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    first = reference_trace[first_index]
    previous = next(
        (
            reference_trace[index]
            for index in range(first_index - 1, -1, -1)
            if candidate_trace[index].get("comparable") and candidate_trace[index].get("exact")
        ),
        None,
    )
    module = dict(_named_modules(model))[first["name"]]
    modes = {}
    capture_modes = (
        (("batch_invariant", True),)
        if args.invariant_only
        else (("native", False), ("batch_invariant", True))
    )
    for label, invariant in capture_modes:
        reference_leaf, reference_output = _capture(
            adapter, model, singleton, args.seed, module, first["occurrence"], invariant
        )
        candidate_leaf_full, candidate_output = _capture(
            adapter, model, paired, args.seed, module, first["occurrence"], invariant
        )
        candidate_leaf = _target_like(reference_leaf, candidate_leaf_full)
        if candidate_leaf is None:
            raise RuntimeError("first divergent leaf cannot be sliced to the target request")
        modes[label] = {
            "first_leaf": _comparison(reference_leaf, candidate_leaf),
            "final_output": _comparison(reference_output, candidate_output),
        }

    report = {
        "model": adapter.name,
        "adapter": args.adapter,
        "seed": args.seed,
        "batch_size": args.candidate_batch_size,
        "composition": "unrelated",
        "replay_manifest_sha256": manifest_hash,
        "target_replay_index": args.target_replay_index if args.replay_manifest else None,
        "noise_id": args.noise_id if args.replay_manifest else None,
        "companion_replay_indices": companion_indexes if args.replay_manifest else None,
        "module_prefix": args.module_prefix,
        "host_leaf_scan": {
            "comparable_leaf_outputs": sum(bool(item.get("comparable")) for item in candidate_trace),
            "first_divergent_index": first_index,
            "last_exact_leaf": previous,
            "first_divergent_leaf": {
                **first,
                "module_type": type(module).__name__,
                "native_comparison": (
                    modes["native"]["first_leaf"] if "native" in modes else None
                ),
                "batch_invariant_comparison": modes["batch_invariant"]["first_leaf"],
            },
        },
        "native_first_divergence": (
            f"{first['name']} output occurrence {first['occurrence']}"
            if "native" in modes
            else None
        ),
        "modes": modes,
        "batch_invariant_boundary_exact": modes["batch_invariant"]["first_leaf"]["exact"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
