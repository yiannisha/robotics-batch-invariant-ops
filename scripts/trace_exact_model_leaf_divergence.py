"""Locate the first comparable leaf difference with explicit request projection.

Unresolved shapes are retained, not assigned an operator-level causal claim.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
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


def _artifact_hash(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


PROJECTION_ADAPTER = None
PROJECTION_BATCH = None
from scripts.leaf_request_projection import project, source_blocks


def _target_like(reference: torch.Tensor, candidate: torch.Tensor, *, name=''):
    if PROJECTION_BATCH is None:
        raise RuntimeError('Explicit logical batch size required')
    return project(reference, candidate, batch_size=PROJECTION_BATCH,
                   target_index=0, blocks=source_blocks(PROJECTION_ADAPTER, name))


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
                    target = _target_like(placeholder, tensor, name=name)
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


def _capture(adapter, model, batch, seed, module, occurrence, invariant, *, artifact_path=None, provenance=None):
    captured = []
    if artifact_path is not None:
        artifact_path = Path(artifact_path)
        if artifact_path.exists() or Path(str(artifact_path) + '.unsupported.json').exists():
            raise FileExistsError(artifact_path)

    def hook(_module, _inputs, output):
        tensor = _first_tensor(output)
        if tensor is not None:
            captured.append(tensor.detach().cpu())
            if artifact_path is not None and len(captured) - 1 == occurrence:
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if len(_inputs) != 1 or not isinstance(output, torch.Tensor):
                        raise ValueError('Requires one positional tensor input and a tensor output')
                    from scripts.captured_leaf_operator import capture_operator
                    record = capture_operator(_module, _inputs[0], output, provenance=provenance)
                except ValueError as error:
                    with Path(str(artifact_path) + '.unsupported.json').open('x') as stream:
                        json.dump(dict(status='local_capture_unsupported', reason=str(error), provenance=provenance), stream, indent=2)
                else:
                    with artifact_path.open('xb') as stream:
                        torch.save(record, stream)

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
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument('--capture-supported-operators', action='store_true',
                        help='Retain real Linear/Conv2d operands, weights, outputs and runtime for independent isolated replay')
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
    allowed = {'scripts.model_invariance.adapters.' + name for name in
               ('pi05_libero_exact', 'rdt_maniskill_exact', 'cogact_simpler_exact', 'lingbot_robotwin_exact')}
    if args.adapter not in allowed:
        parser.error('Requires one of the four exact adapters')
    global PROJECTION_ADAPTER, PROJECTION_BATCH
    PROJECTION_ADAPTER, PROJECTION_BATCH = args.adapter, args.candidate_batch_size
    projection_sources = [Path(__file__), Path(__file__).with_name('leaf_request_projection.py'),
                          Path(__file__).with_name('captured_leaf_operator.py'),
                          Path(__file__).with_name('captured_tensor_layout.py')]
    repo = Path(__file__).resolve().parents[1]
    projection_sources += [repo / 'scripts/model_invariance/harness.py',
                           repo / ('scripts/model_invariance/adapters/' + args.adapter.rsplit('.', 1)[1] + '.py'),
                           repo / 'batch_invariant_ops/__init__.py',
                           repo / 'batch_invariant_ops/batch_invariant_ops.py']
    dependencies = {
        'pi05_libero_exact': ['scripts/verify_pi05_libero_checkpoint.py'],
        'rdt_maniskill_exact': ['scripts/model_invariance/adapters/rdt_maniskill.py', 'scripts/run_rdt_paired_server.py'],
        'cogact_simpler_exact': ['scripts/model_invariance/adapters/cogact_small.py'],
        'lingbot_robotwin_exact': ['scripts/model_invariance/adapters/lingbot_vla2.py', 'batch_invariant_ops/moe.py'],
    }
    projection_sources += [repo / p for p in dependencies[args.adapter.rsplit('.', 1)[1]]]
    if args.adapter.endswith('.cogact_simpler_exact'):
        cog_source = Path(os.environ.get('COGACT_SOURCE', 'missing-cogact-source')) / 'action_model/models.py'
        cog_adapter = Path(__file__).parent / 'model_invariance/adapters/cogact_small.py'
        model_text, adapter_text = cog_source.read_text(), cog_adapter.read_text()
        if ('cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)' not in model_text
                or 'conditioning = torch.cat([cognition, unconditional], dim=0)' not in adapter_text):
            raise ValueError('CogACT source block-layout contract changed')
        projection_sources += [cog_source, cog_adapter]
    projection_metadata = dict(target_index=0, batch_size=args.candidate_batch_size,
        source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in projection_sources},
        cfg_action_network_blocks=2 if args.adapter.endswith('.cogact_simpler_exact') else None,
        unresolved_shapes_excluded_from_numeric_attribution=True,
        operator_attribution_complete=False,
        scope='First comparable leaf only; semantic mapping and functional operations outside leaves can remain unresolved')
    scan_output = Path(str(args.output) + '.leaf_scan.json')
    if scan_output.exists():
        raise FileExistsError(scan_output)
    operator_directory = Path(str(args.output) + '.operators') if args.capture_supported_operators else None
    if operator_directory is not None:
        operator_directory.mkdir(parents=True, exist_ok=False)
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
        if target.get('split') != 'diagnostic':
            raise ValueError('Leaf attribution accepts diagnostic inputs only')
        if target["replay_index"] != args.target_replay_index:
            raise ValueError("replay observation order is not indexed by replay_index")
        companion_indexes = target["companion_replay_indices"][: args.candidate_batch_size - 1]
        if len(companion_indexes) != args.candidate_batch_size - 1:
            raise ValueError("frozen companion schedule is too short")
        companions = [rows[index] for index in companion_indexes]
        if any(row.get('split') != 'diagnostic' for row in companions):
            raise ValueError('Held-out companion exposure is forbidden')
        if any((row["task"], row["episode_id"]) == (target["task"], target["episode_id"]) for row in companions):
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
    scan_output.parent.mkdir(parents=True, exist_ok=True)
    scan_output.write_text(json.dumps(dict(reference=reference_trace, candidate=candidate_trace,
                                         projection_contract=projection_metadata), indent=2) + "\n")
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
        report["request_projection_contract"] = projection_metadata
        report["operator_attribution_complete"] = False
        report['captured_operator_directory'] = str(operator_directory.resolve()) if operator_directory else None
        report['captured_operator_artifact_sha256'] = {}
        report['isolated_operator_replay_complete'] = False
        report['leaf_scan_file'] = str(scan_output.resolve())
        report['leaf_scan_sha256'] = _artifact_hash(scan_output)
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
        capture_provenance = dict(adapter=args.adapter, model=adapter.name,
                                  execution_configuration=adapter.execution_configuration(),
                                  mode=label, seed=args.seed, replay_manifest_sha256=manifest_hash,
                                  target_replay_index=args.target_replay_index, noise_id=args.noise_id,
                                  companion_replay_indices=companion_indexes,
                                  module_name=first['name'], module_occurrence=first['occurrence'],
                                  target_request_id=target['request_ids'][args.noise_id],
                                  request_noise_sha256=target['noise'][args.noise_id]['sha256'],
                                  projection_contract=projection_metadata)
        reference_leaf, reference_output = _capture(
            adapter, model, singleton, args.seed, module, first["occurrence"], invariant,
            artifact_path=operator_directory / (label + '_singleton.pt') if operator_directory else None,
            provenance=dict(capture_provenance, logical_batch_size=1),
        )
        candidate_leaf_full, candidate_output = _capture(
            adapter, model, paired, args.seed, module, first["occurrence"], invariant,
            artifact_path=operator_directory / (label + '_batched.pt') if operator_directory else None,
            provenance=dict(capture_provenance, logical_batch_size=args.candidate_batch_size),
        )
        candidate_leaf = _target_like(reference_leaf, candidate_leaf_full, name=first["name"])
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
    report["request_projection_contract"] = projection_metadata
    report['captured_operator_directory'] = str(operator_directory.resolve()) if operator_directory else None
    report['captured_operator_artifact_sha256'] = {
        str(path.resolve()): _artifact_hash(path) for path in sorted(operator_directory.iterdir())
    } if operator_directory else {}
    report['isolated_operator_replay_complete'] = False
    report['leaf_scan_file'] = str(scan_output.resolve())
    report['leaf_scan_sha256'] = _artifact_hash(scan_output)
    report["operator_attribution_complete"] = False
    report["earlier_unresolved_leaf_count"] = sum(not r.get("comparable") for r in candidate_trace[:first_index])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
