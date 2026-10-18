"""Render a source-pinned SVG of completed paired success-rate results."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from report_completed_success_results import build_report


def render_svg(report: dict) -> str:
    models = report["models"]
    width = 1120
    row_height = 105
    height = 170 + row_height * len(models)
    flip_x, flip_width = 215, 330
    delta_x, delta_width = 705, 330
    max_flip = max(
        [0.12]
        + [row["native_batch_comparison"]["outcome_disagreement_rate"] for row in models]
    )
    max_abs_delta = max(
        [0.04]
        + [
            abs(value)
            for row in models
            for value in row["native_batch_comparison"]["ci95"]
        ]
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<metadata>",
        html.escape(
            "source_success_rate_audit_sha256="
            + report["source_success_rate_audit_sha256"]
        ),
        "</metadata>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#17202a}'
        '.title{font-size:22px;font-weight:700}.sub{font-size:13px;fill:#566573}'
        '.label{font-size:14px;font-weight:600}.axis{font-size:11px;fill:#566573}'
        '.value{font-size:12px}</style>',
        '<text x="35" y="38" class="title">Batch context changes individual outcomes; net accuracy shifts remain uncertain</text>',
        '<text x="35" y="62" class="sub">Independently audited paired simulator evaluations only — native versus repaired batching</text>',
        f'<text x="{flip_x}" y="96" class="label">Success/failure outcome flips</text>',
        f'<text x="{delta_x}" y="96" class="label">Dynamic − singleton success rate (95% paired CI)</text>',
    ]

    for tick in range(5):
        rate = max_flip * tick / 4
        x = flip_x + flip_width * tick / 4
        parts.extend([
            f'<line x1="{x:.2f}" y1="110" x2="{x:.2f}" y2="{height-35}" stroke="#e5e8e8"/>',
            f'<text x="{x:.2f}" y="126" text-anchor="middle" class="axis">{100*rate:.1f}%</text>',
        ])
    for tick in range(-2, 3):
        value = max_abs_delta * tick / 2
        x = delta_x + delta_width * (value + max_abs_delta) / (2 * max_abs_delta)
        stroke = "#85929e" if tick == 0 else "#e5e8e8"
        parts.extend([
            f'<line x1="{x:.2f}" y1="110" x2="{x:.2f}" y2="{height-35}" stroke="{stroke}"/>',
            f'<text x="{x:.2f}" y="126" text-anchor="middle" class="axis">{100*value:+.1f} pp</text>',
        ])

    for index, model in enumerate(models):
        y = 165 + index * row_height
        native = model["native_batch_comparison"]
        patched = model["patched_batch_comparison"]
        native_width = flip_width * native["outcome_disagreement_rate"] / max_flip
        patched_width = flip_width * patched["outcome_disagreement_rate"] / max_flip
        ci_low, ci_high = native["ci95"]
        difference = native["right_minus_left"]
        scale_delta = lambda value: delta_x + delta_width * (
            value + max_abs_delta
        ) / (2 * max_abs_delta)
        model_name = html.escape(model["model_id"])
        parts.extend([
            f'<text x="35" y="{y+8}" class="label">{model_name}</text>',
            f'<rect x="{flip_x}" y="{y-15}" width="{native_width:.2f}" height="18" fill="#c0392b" rx="2"/>',
            f'<rect x="{flip_x}" y="{y+9}" width="{patched_width:.2f}" height="12" fill="#2471a3" rx="2"/>',
            f'<text x="{flip_x+native_width+7:.2f}" y="{y-1}" class="value">native {100*native["outcome_disagreement_rate"]:.1f}%</text>',
            f'<text x="{flip_x+patched_width+7:.2f}" y="{y+20}" class="value">repaired {100*patched["outcome_disagreement_rate"]:.1f}%</text>',
            f'<line x1="{scale_delta(ci_low):.2f}" y1="{y+2}" x2="{scale_delta(ci_high):.2f}" y2="{y+2}" stroke="#1f618d" stroke-width="3"/>',
            f'<line x1="{scale_delta(ci_low):.2f}" y1="{y-5}" x2="{scale_delta(ci_low):.2f}" y2="{y+9}" stroke="#1f618d"/>',
            f'<line x1="{scale_delta(ci_high):.2f}" y1="{y-5}" x2="{scale_delta(ci_high):.2f}" y2="{y+9}" stroke="#1f618d"/>',
            f'<circle cx="{scale_delta(difference):.2f}" cy="{y+2}" r="5" fill="#1f618d"/>',
            f'<text x="{delta_x+delta_width+12}" y="{y+7}" class="value">{100*difference:+.1f} pp</text>',
        ])

    parts.extend([
        f'<rect x="35" y="{height-26}" width="14" height="10" fill="#c0392b"/>',
        f'<text x="55" y="{height-17}" class="axis">native flip rate</text>',
        f'<rect x="165" y="{height-26}" width="14" height="10" fill="#2471a3"/>',
        f'<text x="185" y="{height-17}" class="axis">repaired flip rate</text>',
        "</svg>",
    ])
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/cross_model"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.results / "closed_loop/completed_success_results.svg"
    report = build_report(args.results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_svg(report))
    print(output)


if __name__ == "__main__":
    main()
