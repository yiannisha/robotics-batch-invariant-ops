"""Freeze unrelated, episode-distinct companion indices into a replay manifest.

The schedule is stored before diagnostic/held-out measurements so an auditor
can verify each arrangement against immutable request IDs and source episodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_model_replay_invariance import unrelated_companion_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mirror", type=Path)
    parser.add_argument("--max-companions", type=int, default=31)
    args = parser.parse_args()
    if args.max_companions < 7:
        raise ValueError("full campaign requires at least seven unrelated companions")
    if args.input != args.output and args.output.exists():
        raise FileExistsError(args.output)
    if args.mirror and args.mirror != args.input and args.mirror.exists() and args.mirror.read_bytes() != args.input.read_bytes():
        raise ValueError("mirror is not an exact copy of the input manifest")
    manifest = json.loads(args.input.read_text())
    rows = manifest["observations"]
    if any(row["replay_index"] != index for index, row in enumerate(rows)):
        raise ValueError("replay indices must be contiguous and ordered")
    for index, row in enumerate(rows):
        companions = unrelated_companion_rows(rows, row, index, args.max_companions)
        row["companion_replay_indices"] = [item["replay_index"] for item in companions]
    manifest["companion_schedule"] = {
        "version": "unrelated_episode_heads_v1",
        "max_companions": args.max_companions,
        "description": "Other tasks first; one observation from each distinct non-target episode.",
    }
    payload = json.dumps(manifest, indent=2) + "\n"
    args.output.write_text(payload)
    if args.mirror:
        args.mirror.write_text(payload)
    print(json.dumps({
        "observations": len(rows), "max_companions": args.max_companions,
        "output": str(args.output), "mirror": str(args.mirror) if args.mirror else None,
    }))


if __name__ == "__main__":
    main()
