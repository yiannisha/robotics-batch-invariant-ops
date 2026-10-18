"""Summarize raw held-out request/arrangement and iterative-step records."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path


def records(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty error series")
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flow-output", type=Path)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--flow-label-template", default="flow.step_{step:02d}.state")
    parser.add_argument(
        "--flow-series", action="append", default=[], metavar="NAME=TEMPLATE",
        help="summarize multiple complete iterative series (for example action and video)",
    )
    parser.add_argument(
        "--variable-flow-steps",
        action="store_true",
        help=(
            "accept a contiguous per-request prefix of --flow-steps labels and "
            "summarize every observed step; intended for autoregressive decoders "
            "whose stop token produces request-dependent lengths"
        ),
    )
    parser.add_argument(
        "--noniterative",
        action="store_true",
        help="write only the numerical summary for a policy with no iterative flow",
    )
    args = parser.parse_args()
    if args.flow_steps < 1:
        raise ValueError("flow-steps must be positive")
    if not args.noniterative and args.flow_output is None:
        raise ValueError("flow-output is required unless --noniterative is selected")
    if args.noniterative and args.variable_flow_steps:
        raise ValueError("noniterative and variable-flow-steps are mutually exclusive")
    if args.noniterative and args.flow_series:
        raise ValueError("noniterative and flow-series are mutually exclusive")
    series_templates = {}
    for specification in args.flow_series:
        if "=" not in specification:
            raise ValueError("flow-series must be NAME=TEMPLATE")
        name, template = specification.split("=", 1)
        if not name or not template or name in series_templates or "{step" not in template:
            raise ValueError(f"invalid or duplicate flow-series: {specification}")
        series_templates[name] = template
    if not series_templates:
        series_templates["default"] = args.flow_label_template
    arrangement_count = defaultdict(int)
    violations = defaultdict(int)
    request_violations = defaultdict(set)
    output_errors = defaultdict(list)
    step_counts = defaultdict(int)
    step_exact = defaultdict(int)
    step_errors = defaultdict(list)
    for item in records(args.input):
        if item["batch_size"] <= 1:
            continue
        mode = item["implementation"]
        arrangement_count[mode] += 1
        difference = item["output"]
        output_errors[mode].append(difference["max_absolute_error"])
        if not difference["exact"]:
            violations[mode] += 1
            request_violations[mode].add(item["request_id"])
        if not args.noniterative:
            flow_steps = item.get("flow_steps", {})
            for series, template in series_templates.items():
                expected_labels = [
                    template.format(step=step) for step in range(args.flow_steps)
                ]
                if args.variable_flow_steps:
                    present = [label in flow_steps for label in expected_labels]
                    observed_steps = next(
                        (step for step, exists in enumerate(present) if not exists),
                        args.flow_steps,
                    )
                    if observed_steps == 0 or any(present[observed_steps:]):
                        raise ValueError(
                            f"noncontiguous/empty {series} flow trace for "
                            f"{item['request_id']} {mode}"
                        )
                else:
                    observed_steps = args.flow_steps
                for step in range(observed_steps):
                    label = expected_labels[step]
                    flow = flow_steps.get(label)
                    if flow is None:
                        raise ValueError(f"missing {label} for {item['request_id']} {mode}")
                    key = (series, mode, step)
                    step_counts[key] += 1
                    step_exact[key] += int(flow["exact"])
                    step_errors[key].append(flow["max_absolute_error"])
    fidelity = list(records(args.fidelity))
    if not fidelity:
        raise ValueError("missing singleton-fidelity records")
    fidelity_errors = [item["output"]["max_absolute_error"] for item in fidelity]
    summary = {
        "schema_version": 1,
        "source": str(args.input),
        "singleton_fidelity_source": str(args.fidelity),
        "fixed_request_noise_pairs": len(fidelity),
        "implementations": {
            mode: {
                "nonsingleton_arrangements": arrangement_count[mode],
                "arrangement_violations": violations[mode],
                "arrangement_violation_rate": violations[mode] / arrangement_count[mode],
                "request_violations": len(request_violations[mode]),
                "request_violation_rate": len(request_violations[mode]) / len(fidelity),
                "median_max_absolute_error": statistics.median(output_errors[mode]),
                "p95_max_absolute_error": percentile(output_errors[mode], 0.95),
                "p99_max_absolute_error": percentile(output_errors[mode], 0.99),
                "maximum_absolute_error": max(output_errors[mode]),
            }
            for mode in sorted(arrangement_count)
        },
        "singleton_fidelity": {
            "exact_count": sum(item["output"]["exact"] for item in fidelity),
            "maximum_absolute_error": max(fidelity_errors),
            "median_max_absolute_error": statistics.median(fidelity_errors),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if not args.noniterative:
        def series_summary(series: str, template: str) -> dict:
            return {
                "flow_label_template": template,
                "implementations": {
                    mode: {
                        str(step): {
                            "comparisons": step_counts[(series, mode, step)],
                            "exact_count": step_exact[(series, mode, step)],
                            "median_max_absolute_error": statistics.median(
                                step_errors[(series, mode, step)]
                            ),
                            "p95_max_absolute_error": percentile(
                                step_errors[(series, mode, step)], 0.95
                            ),
                            "p99_max_absolute_error": percentile(
                                step_errors[(series, mode, step)], 0.99
                            ),
                            "maximum_absolute_error": max(step_errors[(series, mode, step)]),
                        }
                        for step in sorted(
                            observed_step
                            for observed_series, observed_mode, observed_step in step_counts
                            if observed_series == series and observed_mode == mode
                        )
                    }
                    for mode in sorted(arrangement_count)
                },
            }

        flow_summary = {
            "schema_version": 1,
            "source": str(args.input),
            "variable_length": args.variable_flow_steps,
            "maximum_steps": args.flow_steps,
        }
        if args.flow_series:
            flow_summary["series"] = {
                name: series_summary(name, template)
                for name, template in series_templates.items()
            }
        else:
            default = series_summary("default", args.flow_label_template)
            flow_summary["flow_label_template"] = default["flow_label_template"]
            flow_summary["implementations"] = default["implementations"]
        args.flow_output.parent.mkdir(parents=True, exist_ok=True)
        args.flow_output.write_text(json.dumps(flow_summary, indent=2) + "\n")
    print(json.dumps({"requests": len(fidelity), "arrangements": sum(arrangement_count.values())}))


if __name__ == "__main__":
    main()
