"""Build audited aggregate tables, figures, and a concise campaign report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path

from audit_cross_model_campaign import CONFIGURATIONS, inspect


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def max_tensor_metric(value, key: str) -> float:
    found: list[float] = []

    def visit(item) -> None:
        if isinstance(item, dict):
            if key in item and isinstance(item[key], (int, float)):
                found.append(float(item[key]))
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return max(found, default=0.0)


def configuration_record(root: Path, identifier: str, name: str) -> dict:
    directory = root / identifier
    audit = inspect(root, identifier, name)
    baseline = read_json(directory / "baseline.json")
    fixed = read_json(directory / "fixed.json")
    transformations = read_json(directory / "batch_transformations.json")
    policy = read_json(directory / "performance" / "policy_summary.json")
    serving = read_json(directory / "performance" / "serving_summary.json")

    policy_by_batch = {
        int(item["batch_size"]): item for item in policy.get("comparisons", [])
    }
    serving_by_name = {
        item["configuration"]: item for item in serving.get("comparisons", [])
    }
    singleton = transformations.get("singleton_fidelity", {})
    operator_ratios: list[float] = []
    operator_cases = 0
    for path in sorted((directory / "performance").glob("operator_session_*.json")):
        for case in read_json(path).get("cases", []):
            if isinstance(case.get("latency_ratio"), (int, float)):
                operator_ratios.append(float(case["latency_ratio"]))
                operator_cases += 1

    return {
        "id": identifier,
        "name": name,
        "evidence_tier": "pilot",
        "measurement_core_complete": audit["measurement_core_complete"],
        "native_end_to_end_exact": baseline.get("end_to_end_exact"),
        "fixed_end_to_end_exact": fixed.get("end_to_end_exact"),
        "fixed_transformations_exact": transformations["modes"]["batch_invariant"]["all_exact"],
        "maximum_tested_batch": fixed.get("maximum_tested_batch"),
        "singleton_fidelity_exact": singleton.get("exact"),
        "singleton_fidelity_max_abs_difference": max_tensor_metric(singleton, "max_abs_difference"),
        "singleton_fidelity_mean_abs_difference_max": max_tensor_metric(singleton, "mean_abs_difference"),
        "policy_batch_sizes": sorted(policy_by_batch),
        "policy_invariant_over_native_latency": {
            str(batch): policy_by_batch[batch]["invariant_over_native_latency"]
            for batch in sorted(policy_by_batch)
        },
        "policy_invariant_over_native_throughput": {
            str(batch): policy_by_batch[batch]["invariant_over_native_throughput"]
            for batch in sorted(policy_by_batch)
        },
        "policy_peak_memory_delta_bytes": {
            str(batch): policy_by_batch[batch]["invariant_minus_native_peak_incremental_bytes"]
            for batch in sorted(policy_by_batch)
        },
        "serving_invariant_over_native_latency": serving_by_name["batch_invariant_dynamic"]["latency_over_native_dynamic"],
        "serving_invariant_over_native_throughput": serving_by_name["batch_invariant_dynamic"]["throughput_over_native_dynamic"],
        "serving_fixed_padded_over_native_latency": serving_by_name["fixed_size_padded"]["latency_over_native_dynamic"],
        "serving_serial_over_native_latency": serving_by_name["serial_singleton"]["latency_over_native_dynamic"],
        "operator_measurements": operator_cases,
        "operator_latency_ratio_median": statistics.median(operator_ratios),
        "operator_latency_ratio_min": min(operator_ratios),
        "operator_latency_ratio_max": max(operator_ratios),
    }


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def svg_bars(records: list[dict], output: Path, field: str, title: str, subtitle: str) -> None:
    width, left, right, top, row = 1120, 250, 90, 80, 25
    height = top + row * len(records) + 60
    values = [float(record[field]) for record in records]
    cap = max(values)
    plot_width = width - left - right
    ticks = 5
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="30" font-family="sans-serif" font-size="21" font-weight="bold">{html.escape(title)}</text>',
        f'<text x="20" y="53" font-family="sans-serif" font-size="13" fill="#555">{html.escape(subtitle)}</text>',
    ]
    for index in range(ticks + 1):
        value = cap * index / ticks
        x = left + plot_width * index / ticks
        lines += [
            f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{height-38}" stroke="#ddd"/>',
            f'<text x="{x:.1f}" y="{height-17}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.1f}×</text>',
        ]
    one_x = left + plot_width / cap
    lines.append(f'<line x1="{one_x:.1f}" y1="{top-12}" x2="{one_x:.1f}" y2="{height-38}" stroke="#222" stroke-width="1.5"/>')
    for index, record in enumerate(records):
        y = top + index * row
        value = float(record[field])
        bar_width = plot_width * value / cap
        color = "#d95f02" if value > 1 else "#1b9e77"
        lines += [
            f'<text x="{left-8}" y="{y+14}" text-anchor="end" font-family="sans-serif" font-size="12">{html.escape(record["name"])}</text>',
            f'<rect x="{left}" y="{y+2}" width="{bar_width:.1f}" height="16" fill="{color}" opacity="0.84"/>',
            f'<text x="{left+bar_width+5:.1f}" y="{y+14}" font-family="sans-serif" font-size="11">{value:.3f}×</text>',
        ]
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n")


def write_markdown(records: list[dict], aggregates: dict, output: Path) -> None:
    lines = [
        "# Cross-model pilot results",
        "",
        "These are preliminary measurements from the 29-path integration sweep. They do not satisfy the full π0-equivalent campaign. See `full_coverage_audit.json` for the authoritative full-coverage status. The π0 reference is excluded.",
        "",
        "## Outcome",
        "",
        f"- Pilot measurement files present: **{aggregates['measurement_core_complete']}/{aggregates['configuration_count']}** configurations.",
        f"- Batch-invariant end-to-end exact: **{aggregates['fixed_end_to_end_exact']}/{aggregates['configuration_count']}** configurations.",
        f"- All requested batch transformations exact: **{aggregates['fixed_transformations_exact']}/{aggregates['configuration_count']}** configurations.",
        f"- Native end-to-end exact without repaired operators: **{aggregates['native_end_to_end_exact']}/{aggregates['configuration_count']}** configurations.",
        f"- Geometric-mean complete-policy latency ratio at batch 1: **{aggregates['policy_b1_latency_geomean']:.3f}×** invariant/native.",
        f"- Geometric-mean dynamic-serving latency ratio: **{aggregates['serving_latency_geomean']:.3f}×** invariant/native.",
        "",
        "Singleton fidelity compares repaired B=1 execution with the official/native singleton path. It is reported separately from batch invariance: the repaired path is exactly invariant under all tested compositions even where its arithmetic order differs slightly from native B=1.",
        "",
        "## Per-configuration summary",
        "",
        "| Configuration | Native invariant | Repaired invariant | Max batch | Singleton exact | Singleton max abs | Policy B1 ratio | Largest policy-batch ratio | Serving ratio | Operator median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        policy = record["policy_invariant_over_native_latency"]
        b1 = policy.get("1")
        largest_batch = max((int(key) for key in policy), default=0)
        lines.append(
            "| {name} | {native} | {fixed} | {batch} | {singleton} | {error:.3g} | {b1:.3f}× | B{largest}: {largest_ratio:.3f}× | {serving:.3f}× | {operator:.3f}× |".format(
                name=record["name"],
                native="yes" if record["native_end_to_end_exact"] else "no",
                fixed="yes" if record["fixed_end_to_end_exact"] and record["fixed_transformations_exact"] else "no",
                batch=record["maximum_tested_batch"],
                singleton="yes" if record["singleton_fidelity_exact"] else "no",
                error=record["singleton_fidelity_max_abs_difference"],
                b1=b1,
                largest=largest_batch,
                largest_ratio=policy[str(largest_batch)],
                serving=record["serving_invariant_over_native_latency"],
                operator=record["operator_latency_ratio_median"],
            )
        )
    lines += [
        "",
        "## Figures and raw tables",
        "",
        "- `figures/policy_b1_latency_ratio.svg`: complete-policy B=1 invariant/native latency.",
        "- `figures/serving_latency_ratio.svg`: dynamic serving invariant/native latency.",
        "- `aggregate_results.json`: full nested aggregate, including every tested policy batch size, throughput, memory delta, fidelity error, and operator range.",
        "- `aggregate_results.csv`: flat comparison table for analysis tools.",
        "",
        "The environment, provenance, numerical comparisons, first-divergence traces, raw timing samples, and five independent operator sessions remain under each configuration directory.",
    ]
    output.write_text("\n".join(lines) + "\n")


def write_configuration_markdown(record: dict, directory: Path) -> None:
    policy = record["policy_invariant_over_native_latency"]
    throughput = record["policy_invariant_over_native_throughput"]
    memory = record["policy_peak_memory_delta_bytes"]
    lines = [
        f"# {record['name']} pilot report",
        "",
        "This report describes preliminary integration measurements. Full π0-equivalent coverage is tracked by `../full_coverage_audit.json` and has not been established by this file.",
        "",
        "## Numerical result",
        "",
        f"- Pilot measurement files: **{'present' if record['measurement_core_complete'] else 'incomplete'}**.",
        f"- Native end-to-end batch invariance: **{'exact' if record['native_end_to_end_exact'] else 'not exact'}**.",
        f"- Repaired end-to-end batch invariance: **{'exact' if record['fixed_end_to_end_exact'] else 'not exact'}** through batch {record['maximum_tested_batch']}.",
        f"- Repaired target-position, companion, permutation, and partition transformations: **{'exact' if record['fixed_transformations_exact'] else 'not exact'}**.",
        f"- Repaired/native singleton fidelity: **{'exact' if record['singleton_fidelity_exact'] else 'not bit-identical'}**; maximum absolute difference {record['singleton_fidelity_max_abs_difference']:.9g}.",
        "",
        "## Complete-policy performance",
        "",
        "| Batch | Invariant/native latency | Invariant/native throughput | Peak-memory delta |",
        "|---:|---:|---:|---:|",
    ]
    for batch in sorted(map(int, policy)):
        lines.append(
            f"| {batch} | {policy[str(batch)]:.4f}× | {throughput[str(batch)]:.4f}× | {memory[str(batch)] / (1024**2):.1f} MiB |"
        )
    lines += [
        "",
        "## Serving and operators",
        "",
        f"- Dynamic batch-invariant/native serving latency: **{record['serving_invariant_over_native_latency']:.4f}×**.",
        f"- Fixed-size padded/native dynamic serving latency: **{record['serving_fixed_padded_over_native_latency']:.4f}×**.",
        f"- Serial-singleton/native dynamic serving latency: **{record['serving_serial_over_native_latency']:.4f}×**.",
        f"- Real-shape operator observations: **{record['operator_measurements']}** across five sessions; median invariant/native latency **{record['operator_latency_ratio_median']:.4f}×**, range **[{record['operator_latency_ratio_min']:.4f}×, {record['operator_latency_ratio_max']:.4f}×]**.",
        "",
        "## Retained evidence",
        "",
        "`provenance.json`, `baseline.json`, `fixed.json`, `multiple_inputs.json`, `batch_transformations.json`, and `first_divergence.json` retain the numerical evidence. Raw performance samples and five operator sessions are under `performance/`.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/cross_model"))
    args = parser.parse_args()
    root = args.results
    records = [configuration_record(root, identifier, name) for identifier, name in CONFIGURATIONS]
    b1 = [record["policy_invariant_over_native_latency"]["1"] for record in records]
    serving = [record["serving_invariant_over_native_latency"] for record in records]
    aggregates = {
        "configuration_count": len(records),
        "measurement_core_complete": sum(record["measurement_core_complete"] for record in records),
        "native_end_to_end_exact": sum(bool(record["native_end_to_end_exact"]) for record in records),
        "fixed_end_to_end_exact": sum(bool(record["fixed_end_to_end_exact"]) for record in records),
        "fixed_transformations_exact": sum(bool(record["fixed_transformations_exact"]) for record in records),
        "singleton_fidelity_exact": sum(bool(record["singleton_fidelity_exact"]) for record in records),
        "policy_b1_latency_geomean": geometric_mean(b1),
        "policy_b1_latency_median": statistics.median(b1),
        "policy_b1_latency_range": [min(b1), max(b1)],
        "serving_latency_geomean": geometric_mean(serving),
        "serving_latency_median": statistics.median(serving),
        "serving_latency_range": [min(serving), max(serving)],
        "operator_measurement_count": sum(record["operator_measurements"] for record in records),
    }
    report = {"schema_version": 1, "aggregates": aggregates, "configurations": records}
    report["evidence_tier"] = "pilot"
    (root / "aggregate_results.json").write_text(json.dumps(report, indent=2) + "\n")

    fields = [
        "id", "name", "native_end_to_end_exact", "fixed_end_to_end_exact",
        "fixed_transformations_exact", "maximum_tested_batch", "singleton_fidelity_exact",
        "singleton_fidelity_max_abs_difference", "policy_b1_latency_ratio",
        "policy_largest_batch", "policy_largest_batch_latency_ratio",
        "serving_invariant_over_native_latency", "serving_fixed_padded_over_native_latency",
        "serving_serial_over_native_latency", "operator_measurements", "operator_latency_ratio_median",
    ]
    with (root / "aggregate_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            batches = record["policy_invariant_over_native_latency"]
            largest = max(map(int, batches))
            writer.writerow({
                **{field: record.get(field) for field in fields},
                "policy_b1_latency_ratio": batches["1"],
                "policy_largest_batch": largest,
                "policy_largest_batch_latency_ratio": batches[str(largest)],
            })

    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    policy_rows = [{**record, "policy_b1": record["policy_invariant_over_native_latency"]["1"]} for record in records]
    svg_bars(sorted(policy_rows, key=lambda item: item["policy_b1"]), figures / "policy_b1_latency_ratio.svg", "policy_b1", "Complete-policy latency at batch 1", "Ratio of repaired batch-invariant execution to native execution; lower is faster, 1× is parity.")
    svg_bars(sorted(records, key=lambda item: item["serving_invariant_over_native_latency"]), figures / "serving_latency_ratio.svg", "serving_invariant_over_native_latency", "Dynamic-serving latency", "Ratio of repaired batch-invariant dynamic batching to native dynamic batching for each model's representative serving shape.")
    write_markdown(records, aggregates, root / "CAMPAIGN_REPORT.md")
    for record in records:
        write_configuration_markdown(record, root / record["id"])
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
