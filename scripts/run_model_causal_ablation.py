"""Run the six π0 causal modes on every frozen diagnostic request."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import set_batch_invariant_mode
from batch_invariant_ops import batch_invariant_ops as invariant_ops
from scripts.model_invariance.harness import _compose_batch
from scripts.run_model_replay_invariance import metrics, run_with_trace, sample_at


MODES = (
    "native", "native_deterministic", "existing_invariant_ops",
    "operators_plus_patch_projection", "explicit_per_matrix_reference",
    "full_invariant",
)
MODE_OPERATIONS = {
    "native": (),
    "native_deterministic": (),
    "existing_invariant_ops": (
        "mm", "addmm", "bmm", "softmax", "log_softmax", "mean",
    ),
    "operators_plus_patch_projection": (
        "mm", "addmm", "bmm", "softmax", "log_softmax", "mean", "convolution",
    ),
    "explicit_per_matrix_reference": (
        "mm", "addmm", "explicit_bmm", "softmax", "log_softmax", "mean", "convolution",
        "explicit_sdpa",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def explicit_per_matrix_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Reference attention with a separate fixed-shape reduction per head."""
    if dropout_p != 0.0:
        raise ValueError("causal ablation reference only supports dropout_p=0")
    if enable_gqa:
        repeats = query.shape[-3] // key.shape[-3]
        key = key.repeat_interleave(repeats, dim=-3)
        value = value.repeat_interleave(repeats, dim=-3)
    leading = query.shape[:-2]
    q_length, k_length = query.shape[-2], key.shape[-2]
    flat_q = query.reshape(-1, q_length, query.shape[-1])
    flat_k = key.reshape(-1, k_length, key.shape[-1])
    flat_v = value.reshape(-1, k_length, value.shape[-1])
    masks = None
    if attn_mask is not None:
        masks = torch.broadcast_to(attn_mask, (*leading, q_length, k_length)).reshape(
            -1, q_length, k_length
        )
    causal = torch.ones((q_length, k_length), dtype=torch.bool, device=query.device).tril()
    outputs = []
    factor = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
    for index in range(flat_q.shape[0]):
        scores = torch.mm(flat_q[index], flat_k[index].transpose(0, 1)) * factor
        if is_causal:
            if masks is not None:
                raise ValueError("attn_mask and is_causal cannot both be set")
            scores = scores.masked_fill(~causal, float("-inf"))
        elif masks is not None:
            mask = masks[index]
            scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.mm(probabilities, flat_v[index]))
    return torch.stack(outputs).reshape(*leading, q_length, value.shape[-1])


class SelectiveOperators:
    def __init__(self, operations: tuple[str, ...]):
        self.operations = operations
        self.library = None
        self.counts = Counter()

    def counted(self, name, function):
        def call(*args, **kwargs):
            self.counts[name] += 1
            return function(*args, **kwargs)
        return call

    def __enter__(self):
        if not self.operations:
            return self

        def pair(values):
            values = tuple(values)
            return values * 2 if len(values) == 1 else values

        def convolution(input, weight, bias, stride, padding, dilation,
                        transposed, output_padding, groups):
            return invariant_ops.convolution_batch_invariant(
                input, weight, bias, pair(stride), pair(padding), pair(dilation),
                transposed, pair(output_padding), groups,
            )

        def explicit_bmm(left, right):
            return torch.stack([
                torch.mm(left[index], right[index]) for index in range(left.shape[0])
            ])

        functions = {
            "mm": invariant_ops.mm_batch_invariant,
            "addmm": invariant_ops.addmm_batch_invariant,
            "bmm": invariant_ops.bmm_batch_invariant,
            "explicit_bmm": explicit_bmm,
            "softmax": invariant_ops._softmax_batch_invariant,
            "log_softmax": invariant_ops._log_softmax_batch_invariant,
            "mean": invariant_ops.mean_batch_invariant,
            "convolution": convolution,
            "explicit_sdpa": explicit_per_matrix_sdpa,
        }
        schemas = {
            "mm": "aten::mm", "addmm": "aten::addmm", "bmm": "aten::bmm",
            "explicit_bmm": "aten::bmm",
            "softmax": "aten::_softmax",
            "log_softmax": "aten::_log_softmax", "mean": "aten::mean.dim",
            "convolution": "aten::convolution",
            "explicit_sdpa": "aten::scaled_dot_product_attention",
        }
        self.library = torch.library.Library("aten", "IMPL")
        for operation in self.operations:
            self.library.impl(
                schemas[operation], self.counted(operation, functions[operation]), "CUDA"
            )
        return self

    def __exit__(self, *_):
        if self.library is not None:
            self.library._destroy()
            self.library = None

    def assert_coverage(
        self, mode: str, *, inapplicable_operations: frozenset[str] = frozenset()
    ) -> None:
        if mode in ("operators_plus_patch_projection", "explicit_per_matrix_reference") \
                and "convolution" not in inapplicable_operations \
                and not self.counts["convolution"]:
            raise RuntimeError(f"{mode} silently bypassed convolution")
        if (mode == "explicit_per_matrix_reference"
                and "explicit_sdpa" not in inapplicable_operations
                and not self.counts["explicit_sdpa"]):
            raise RuntimeError("explicit attention reference was not exercised")
        if mode == "existing_invariant_ops" and not (
            self.counts["mm"] or self.counts["addmm"] or self.counts["bmm"]
        ):
            raise RuntimeError("existing invariant matrix operators were not exercised")


@contextlib.contextmanager
def implementation(mode: str):
    deterministic = mode == "native_deterministic"
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    try:
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        if mode == "full_invariant":
            with set_batch_invariant_mode():
                yield None
        else:
            with SelectiveOperators(MODE_OPERATIONS[mode]) as operators:
                yield operators
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.backends.cudnn.benchmark = previous_benchmark


def append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    manifest_hash = sha256(args.manifest)
    observations = manifest["observations"]
    selected = [row for row in observations if row.get("split") == "diagnostic"]
    multiplicity = 3 if all(len(row.get("request_ids", [])) == 3 for row in selected) else 1
    expected_ids = {request_id for row in selected for request_id in row["request_ids"]}
    if len(selected) != 200 or len(expected_ids) != 200 * multiplicity:
        raise ValueError("ablation requires the complete frozen diagnostic request set")
    completed = set()
    if args.output.is_file():
        for line in args.output.read_text().splitlines():
            item = json.loads(line)
            key = (item.get("mode"), item.get("request_id"))
            if key in completed or item.get("replay_manifest_sha256") != manifest_hash:
                raise ValueError("existing ablation output has duplicate or foreign evidence")
            completed.add(key)

    adapter = importlib.import_module(args.adapter).adapter
    provider = Path(getattr(adapter, "replay_manifest_path"))
    if not provider.is_file() or sha256(provider) != manifest_hash:
        raise ValueError("adapter replay manifest differs from selected ablation manifest")
    model = adapter.load_model()
    if hasattr(model, "eval"):
        model.eval()
    execution_configuration = (
        adapter.execution_configuration() if hasattr(adapter, "execution_configuration") else None
    )
    inapplicable_operations = getattr(
        adapter, "causal_ablation_inapplicable_operations", {}
    )
    if not isinstance(inapplicable_operations, dict):
        raise TypeError("adapter causal-ablation inapplicability contract must be a dictionary")
    pinned_inapplicable = {}
    for operation, evidence in inapplicable_operations.items():
        if (operation not in ("explicit_sdpa", "convolution")
                or not isinstance(evidence, dict)
                or not evidence.get("reason") or not evidence.get("evidence_path")):
            raise ValueError("unsupported causal-ablation inapplicability contract")
        evidence_path = Path(evidence["evidence_path"])
        if not evidence_path.is_file():
            raise FileNotFoundError(evidence_path)
        if operation == "explicit_sdpa":
            if evidence_path.suffix == ".json":
                trace = json.loads(evidence_path.read_text())
                operations = trace.get("operations", [])
                if (
                    trace.get("adapter") != args.adapter
                    or trace.get("replay_manifest_sha256") != manifest_hash
                    or any(
                        row.get("aten_operation") == "aten::scaled_dot_product_attention"
                        for row in operations
                    )
                    or not any(
                        row.get("aten_operation") in {"aten::bmm", "aten::mm"}
                        for row in operations
                    )
                ):
                    raise ValueError(
                        "claimed explicit-SDPA inapplicability differs from pinned trace"
                    )
            else:
                source = evidence_path.read_text()
                eager_attention = any(
                    token in source
                    for token in ("torch.bmm", "torch.matmul", "torch.einsum")
                )
                if not eager_attention or "scaled_dot_product_attention" in source:
                    raise ValueError("claimed explicit-SDPA inapplicability differs from source")
        else:
            trace = json.loads(evidence_path.read_text())
            if (trace.get("adapter") != args.adapter
                    or trace.get("replay_manifest_sha256") != manifest_hash
                    or any(row.get("aten_operation") == "aten::convolution"
                           for row in trace.get("operations", []))):
                raise ValueError("claimed convolution inapplicability differs from pinned trace")
        pinned_inapplicable[operation] = {
            **evidence, "evidence_sha256": sha256(evidence_path),
        }
    for row in selected:
        companion_index = row["companion_replay_indices"][0]
        companion_row = observations[companion_index]
        source_limited = (
            (manifest.get("companion_schedule") or {}).get("version")
            == "distinct_source_observations_v1"
        )
        if companion_index == row["replay_index"] or (
            not source_limited and companion_row["episode_id"] == row["episode_id"]
        ):
            raise ValueError("ablation companion is not from an unrelated episode")
        for noise_id in range(multiplicity):
            request_id = row["request_ids"][noise_id]
            target = adapter.load_replay_input(row["replay_index"], noise_id)
            companion = adapter.load_replay_input(companion_index, noise_id)
            for mode in MODES:
                if (mode, request_id) in completed:
                    continue
                adapter_context = (
                    adapter.causal_ablation_context(mode)
                    if hasattr(adapter, "causal_ablation_context") else
                    contextlib.nullcontext()
                )
                with implementation(mode) as operators, adapter_context:
                    # Some complete-invariant adapters canonicalize dynamic
                    # input shapes during composition.  Compose inside the
                    # active mode so the final stage includes that repair,
                    # while the earlier causal stages retain native padding.
                    singleton = _compose_batch(adapter, [target])
                    paired = _compose_batch(adapter, [target, companion])
                    reference, reference_trace = run_with_trace(
                        adapter, model, singleton, args.seed
                    )
                    candidate, candidate_trace = run_with_trace(
                        adapter, model, paired, args.seed
                    )
                    if operators is not None:
                        mode_inapplicable = frozenset(
                            operation for operation in pinned_inapplicable
                            if operation in MODE_OPERATIONS[mode]
                        )
                        operators.assert_coverage(
                            mode, inapplicable_operations=mode_inapplicable,
                        )
                flow_steps = {
                    label: metrics(reference_trace[label], sample_at(value, 0))
                    for label, value in candidate_trace.items() if label in reference_trace
                }
                divergent = next(
                    (label for label, difference in flow_steps.items() if not difference["exact"]),
                    None,
                )
                output = metrics(reference, sample_at(candidate, 0))
                if mode == "full_invariant" and (
                    not output["exact"] or any(not value["exact"] for value in flow_steps.values())
                ):
                    raise RuntimeError(f"full invariant ablation diverged for {request_id}")
                observed_boundary = divergent or ("returned_output" if not output["exact"] else None)
                record = {
                    "schema_version": 1,
                    "experiment_id": str(uuid.uuid4()),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "mode": mode, "request_id": request_id,
                    "task": row["task"], "episode_id": row["episode_id"],
                    "observation_index": row["observation_index"], "noise_id": noise_id,
                    "replay_manifest_sha256": manifest_hash,
                    "source_image_sha256": row.get(
                        "image_sha256", row.get("source_image_rgb_sha256")
                    ),
                    "request_noise_sha256": (
                        row.get("noise", [])[noise_id]["sha256"] if multiplicity == 3 else None
                    ),
                    "adapter": args.adapter,
                    "adapter_execution_configuration": execution_configuration,
                    "batch_size": 2,
                    "companion_replay_index": companion_index,
                    "companion_request_id": companion_row["request_ids"][noise_id],
                    "first_divergence": (
                        {"observed_boundary": observed_boundary,
                         "metrics": flow_steps[divergent] if divergent is not None else output}
                        if observed_boundary is not None else None
                    ),
                    "output": output,
                    "flow_steps": flow_steps,
                    "override_invocations": dict(operators.counts) if operators is not None else {},
                    "inapplicable_operations": {
                        operation: evidence
                        for operation, evidence in pinned_inapplicable.items()
                        if operation in MODE_OPERATIONS[mode]
                    },
                    "deterministic_algorithms": mode == "native_deterministic",
                }
                append(args.output, record)
                completed.add((mode, request_id))
                print(f"completed ablation {mode} {request_id}", flush=True)
    print(args.output)


if __name__ == "__main__":
    main()
