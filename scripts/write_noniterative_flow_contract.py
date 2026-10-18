"""Prove that a held-out policy path has boundaries but no iterative flow steps."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.input.is_file() or not args.evidence.is_file() or not args.reason.strip():
        raise ValueError("raw held-out input, implementation evidence, and reason are required")
    nonsingleton = 0
    observed_boundaries = None
    implementations = set()
    for item in records(args.input):
        if item.get("batch_size", 0) <= 1:
            continue
        nonsingleton += 1
        implementation = item.get("implementation")
        if implementation not in ("native", "batch_invariant"):
            raise ValueError(f"unexpected implementation {implementation!r}")
        implementations.add(implementation)
        flow = item.get("flow_steps")
        if not isinstance(flow, dict) or not flow:
            raise ValueError("noniterative evidence needs recorded internal/output boundaries")
        labels = set(flow)
        if any(label.startswith("flow.step_") for label in labels):
            raise ValueError("raw records contain iterative flow steps and cannot be marked N/A")
        if observed_boundaries is None:
            observed_boundaries = labels
        elif labels != observed_boundaries:
            raise ValueError("noniterative boundary coverage varies between held-out records")
    if nonsingleton == 0 or implementations != {"native", "batch_invariant"}:
        raise ValueError("held-out archive lacks both implementations or nonsingleton records")
    report = {
        "schema_version": 1,
        "model_id": args.model_id,
        "status": "not_applicable",
        "reason": args.reason.strip(),
        "evidence_path": str(args.evidence.resolve()),
        "evidence_sha256": sha256(args.evidence),
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "nonsingleton_records": nonsingleton,
        "implementations": ["native", "batch_invariant"],
        "observed_boundaries": sorted(observed_boundaries or ()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "model_id": args.model_id,
        "nonsingleton_records": nonsingleton,
        "observed_boundaries": report["observed_boundaries"],
    }))


if __name__ == "__main__":
    main()
