"""Model-agnostic batch-invariance investigation machinery.

An adapter owns model-specific loading, batching, inference, and output
extraction.  This module owns deterministic seeding, batch composition,
bitwise comparison, diagnostics, OOM handling, and result serialization.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from batch_invariant_ops import set_batch_invariant_mode

DEFAULT_BATCH_SIZES = (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64)


@runtime_checkable
class ModelAdapter(Protocol):
    """The minimal integration surface required for a robotics model."""

    name: str

    def load_model(self) -> Any: ...

    def load_example_inputs(self, count: int) -> Sequence[Any]: ...

    def run_model(self, model: Any, batch: Any) -> Any: ...

    def extract_robot_output(self, output: Any) -> Any: ...


@dataclass(frozen=True)
class TensorDifference:
    path: str
    reference_shape: tuple[int, ...]
    candidate_shape: tuple[int, ...]
    dtype: str
    exact: bool
    differing_elements: int | None
    max_abs_difference: float | None
    mean_abs_difference: float | None
    first_differing_index: tuple[int, ...] | None
    reference_hash: str
    candidate_hash: str


def seed_everything(seed: int) -> None:
    """Reset ordinary host and device RNG state before an inference run."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _flatten_tensors(value: Any, path: str = "output") -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return {path: value}
    if isinstance(value, Mapping):
        flattened: dict[str, torch.Tensor] = {}
        for key in sorted(value, key=str):
            flattened.update(_flatten_tensors(value[key], f"{path}.{key}"))
        return flattened
    if isinstance(value, (tuple, list)):
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(_flatten_tensors(item, f"{path}[{index}]"))
        return flattened
    raise TypeError(f"{path} contains unsupported output type {type(value).__name__}")


def _first_sample(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        return value[:1]
    if isinstance(value, Mapping):
        return type(value)((key, _first_sample(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_first_sample(item) for item in value)
    if isinstance(value, list):
        return [_first_sample(item) for item in value]
    raise TypeError(f"unsupported output type {type(value).__name__}")


def _first_differing_index(mask: torch.Tensor) -> tuple[int, ...] | None:
    locations = torch.nonzero(mask, as_tuple=False)
    if locations.numel() == 0:
        return None
    return tuple(int(index) for index in locations[0].tolist())


def compare_outputs(reference: Any, candidate: Any) -> list[TensorDifference]:
    """Return exact and magnitude diagnostics for every output tensor."""

    reference_tensors = _flatten_tensors(reference)
    candidate_tensors = _flatten_tensors(candidate)
    if reference_tensors.keys() != candidate_tensors.keys():
        raise ValueError(
            "output structures differ: "
            f"reference={sorted(reference_tensors)}, candidate={sorted(candidate_tensors)}"
        )

    comparisons = []
    for path, reference_tensor in reference_tensors.items():
        candidate_tensor = candidate_tensors[path]
        same_metadata = (
            reference_tensor.shape == candidate_tensor.shape
            and reference_tensor.dtype == candidate_tensor.dtype
        )
        exact = same_metadata and torch.equal(reference_tensor, candidate_tensor)
        differing_elements = None
        max_abs_difference = None
        mean_abs_difference = None
        first_differing_index = None
        if same_metadata:
            difference_mask = reference_tensor != candidate_tensor
            differing_elements = int(difference_mask.sum().item())
            first_differing_index = _first_differing_index(difference_mask)
            if reference_tensor.is_floating_point() or reference_tensor.is_complex():
                absolute = (reference_tensor - candidate_tensor).abs().float()
                if absolute.numel():
                    max_abs_difference = float(absolute.max().item())
                    mean_abs_difference = float(absolute.mean().item())
                else:
                    max_abs_difference = mean_abs_difference = 0.0
        comparisons.append(
            TensorDifference(
                path=path,
                reference_shape=tuple(reference_tensor.shape),
                candidate_shape=tuple(candidate_tensor.shape),
                dtype=str(reference_tensor.dtype),
                exact=exact,
                differing_elements=differing_elements,
                max_abs_difference=max_abs_difference,
                mean_abs_difference=mean_abs_difference,
                first_differing_index=first_differing_index,
                reference_hash=_tensor_hash(reference_tensor),
                candidate_hash=_tensor_hash(candidate_tensor),
            )
        )
    return comparisons


def _default_compose_batch(samples: Sequence[Any]) -> Any:
    first = samples[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(list(samples))
    if isinstance(first, Mapping):
        return type(first)(
            (key, _default_compose_batch([sample[key] for sample in samples])) for key in first
        )
    if isinstance(first, tuple):
        return tuple(
            _default_compose_batch([sample[index] for sample in samples])
            for index in range(len(first))
        )
    if isinstance(first, list):
        return [
            _default_compose_batch([sample[index] for sample in samples])
            for index in range(len(first))
        ]
    if isinstance(first, (bool, int, float, str)):
        if not all(sample == first for sample in samples):
            raise ValueError("non-tensor metadata differs; provide adapter.compose_batch")
        return first
    raise TypeError(f"cannot batch {type(first).__name__}; provide adapter.compose_batch")


def _compose_batch(adapter: ModelAdapter, samples: Sequence[Any]) -> Any:
    compose: Callable[[Sequence[Any]], Any] = getattr(
        adapter, "compose_batch", _default_compose_batch
    )
    return compose(samples)


def _reset_model(adapter: ModelAdapter, model: Any) -> None:
    reset = getattr(adapter, "reset_model", None)
    if reset is not None:
        reset(model)


def _run_once(adapter: ModelAdapter, model: Any, batch: Any, seed: int) -> Any:
    seed_everything(seed)
    _reset_model(adapter, model)
    with torch.inference_mode():
        output = adapter.run_model(model, batch)
    return adapter.extract_robot_output(output)


def _is_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _comparison_record(comparisons: list[TensorDifference]) -> dict[str, Any]:
    return {
        "exact": all(comparison.exact for comparison in comparisons),
        "tensors": [asdict(comparison) for comparison in comparisons],
    }


def run_investigation(
    adapter: ModelAdapter,
    *,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    invariant_ops: bool = False,
    seed: int = 0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run deterministic, duplicate, and unrelated batch comparisons."""

    batch_sizes = tuple(sorted(set(batch_sizes)))
    if not batch_sizes or batch_sizes[0] != 1 or any(size < 1 for size in batch_sizes):
        raise ValueError("batch_sizes must be positive and include 1")

    seed_everything(seed)
    model = adapter.load_model()
    if hasattr(model, "eval"):
        model.eval()
    examples = list(adapter.load_example_inputs(max(batch_sizes)))
    if not examples:
        raise ValueError("adapter returned no example inputs")
    if max(batch_sizes) > 1 and len(examples) < 2:
        raise ValueError("unrelated composition requires at least two example inputs")
    target = examples[0]
    reference_batch = _compose_batch(adapter, [target])

    report: dict[str, Any] = {
        "model": adapter.name,
        "batch_invariant_ops": invariant_ops,
        "seed": seed,
        "requested_batch_sizes": list(batch_sizes),
        "repeatability": None,
        "composition_results": {"duplicate": [], "unrelated": []},
        "maximum_tested_batch": None,
        "end_to_end_exact": False,
    }

    mode = set_batch_invariant_mode() if invariant_ops else contextlib.nullcontext()
    with mode:
        reference = _run_once(adapter, model, reference_batch, seed)
        repeated = _run_once(adapter, model, reference_batch, seed)
        repeat_comparisons = compare_outputs(reference, repeated)
        report["repeatability"] = _comparison_record(repeat_comparisons)
        if not report["repeatability"]["exact"]:
            report["failure"] = "same-batch inference is not repeatable"
        else:
            maximum_tested = 1
            for composition in ("duplicate", "unrelated"):
                for batch_size in batch_sizes:
                    if composition == "duplicate":
                        samples = [target] * batch_size
                    else:
                        companions = [
                            examples[1 + index % (len(examples) - 1)]
                            for index in range(batch_size - 1)
                        ]
                        samples = [target, *companions]
                    batch = _compose_batch(adapter, samples)
                    try:
                        candidate = _run_once(adapter, model, batch, seed)
                    except RuntimeError as error:
                        if not _is_oom(error):
                            raise
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        report["composition_results"][composition].append(
                            {"batch_size": batch_size, "status": "OOM"}
                        )
                        break
                    comparisons = compare_outputs(reference, _first_sample(candidate))
                    record = {
                        "batch_size": batch_size,
                        "status": "PASS" if all(item.exact for item in comparisons) else "FAIL",
                    }
                    record.update(_comparison_record(comparisons))
                    report["composition_results"][composition].append(record)
                    maximum_tested = max(maximum_tested, batch_size)
            report["maximum_tested_batch"] = maximum_tested
            completed = [
                record
                for results in report["composition_results"].values()
                for record in results
                if record["status"] != "OOM"
            ]
            report["end_to_end_exact"] = bool(completed) and all(
                record["status"] == "PASS" for record in completed
            )

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
    return report
