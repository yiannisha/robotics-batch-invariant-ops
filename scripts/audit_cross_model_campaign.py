"""Generate a progress inventory of the preliminary cross-model pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONFIGURATIONS = (
    ("pi05", "π0.5"),
    ("pi_fast", "π0-FAST"),
    ("internvla_a15_standard", "InternVLA-A1.5 standard"),
    ("internvla_a15_optimized", "InternVLA-A1.5 optimized"),
    ("molmoact2", "MolmoAct2"),
    ("groot_n17", "GR00T N1.7"),
    ("cosmos3_nano", "Cosmos 3 Nano policy"),
    ("cosmos3_nano_video", "Cosmos 3 Nano decoded video"),
    ("cosmos3_edge", "Cosmos 3 Edge policy"),
    ("cosmos3_edge_video", "Cosmos 3 Edge decoded video"),
    ("xvla", "X-VLA"),
    ("uva", "UVA action"),
    ("uva_joint", "UVA joint video/action"),
    ("smolvla", "SmolVLA"),
    ("openvla_oft", "OpenVLA-OFT"),
    ("dexvla", "DexVLA"),
    ("lingbot_vla2", "LingBot-VLA 2.0"),
    ("univla_baai", "BAAI UniVLA"),
    ("spatialvla", "SpatialVLA 4B"),
    ("rdt_1b", "RDT-1B"),
    ("cogact_small", "CogACT-Small"),
    ("openhelix", "OpenHelix"),
    ("wsa_base", "WSA Base"),
    ("vitra_vla_3b", "VITRA-VLA-3B"),
    ("opendw_dw05", "OpenDW DW05 action"),
    ("opendw_dw05_joint", "OpenDW DW05 joint video/action"),
    ("openwam_alpha_robotwin", "OpenWAM Alpha policy"),
    ("openwam_alpha_video", "OpenWAM Alpha decoded video"),
    ("xiaomi_robotics_1", "Xiaomi-Robotics-1"),
)


def _read(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def _timing_progress(path: Path, configuration_count: int) -> dict:
    if not path.is_file():
        return {"records": 0, "expected": None, "complete": False}
    count = 0
    environment = None
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("record_type") == "environment":
                if environment is not None:
                    raise ValueError(f"multiple environment records in {path}")
                environment = record
            elif record.get("record_type") == "timing":
                count += 1
    expected = None
    if environment is not None:
        expected = (
            int(environment["sessions"])
            * int(environment["repetitions"])
            * configuration_count
        )
    return {
        "records": count,
        "expected": expected,
        "complete": expected is not None and count == expected,
    }


def inspect(root: Path, identifier: str, name: str) -> dict:
    directory = root / identifier
    baseline = _read(directory / "baseline.json")
    fixed = _read(directory / "fixed.json")
    transformations = _read(directory / "batch_transformations.json")
    invariant_transformations_exact = bool(
        transformations
        and transformations.get("modes", {}).get("batch_invariant", {}).get("all_exact")
    )
    required = {
        "provenance": (directory / "provenance.json").is_file(),
        "baseline": baseline is not None,
        "fixed": fixed is not None,
        "first_divergence": (directory / "first_divergence.json").is_file(),
        "multiple_inputs": (directory / "multiple_inputs.json").is_file(),
        "batch_transformations": transformations is not None,
    }
    policy_path = directory / "performance" / "policy_timings.jsonl"
    policy_environment = None
    if policy_path.is_file():
        with policy_path.open() as stream:
            for line in stream:
                candidate = json.loads(line)
                if candidate.get("record_type") == "environment":
                    policy_environment = candidate
                    break
    policy_configuration_count = (
        2 * len(policy_environment["batch_sizes"]) if policy_environment else 0
    )
    policy_timings = _timing_progress(policy_path, policy_configuration_count)
    serving_timings = _timing_progress(
        directory / "performance" / "serving_timings.jsonl", 4
    )
    operator_sessions = len(list((directory / "performance").glob("operator_session_*.json")))
    evidence_present = (
        any(required.values())
        or policy_timings["records"] > 0
        or serving_timings["records"] > 0
        or operator_sessions > 0
    )
    numerical_core_complete = (
        all(required.values())
        and bool(fixed and fixed.get("end_to_end_exact"))
        and invariant_transformations_exact
    )
    performance_core_complete = (
        policy_timings["complete"]
        and serving_timings["complete"]
        and operator_sessions >= 5
    )
    measurement_core_complete = numerical_core_complete and performance_core_complete
    return {
        "id": identifier,
        "name": name,
        "evidence_tier": "pilot",
        "status": (
            "PILOT MEASUREMENT CORE PRESENT"
            if measurement_core_complete
            else "PILOT NUMERICAL CORE PRESENT"
            if numerical_core_complete
            else "PILOT IN PROGRESS"
            if evidence_present
            else "PILOT NOT RUN"
        ),
        "artifacts": required,
        "native_end_to_end_exact": baseline.get("end_to_end_exact") if baseline else None,
        "fixed_end_to_end_exact": fixed.get("end_to_end_exact") if fixed else None,
        "invariant_transformations_exact": (
            invariant_transformations_exact if transformations else None
        ),
        "maximum_tested_batch": fixed.get("maximum_tested_batch") if fixed else None,
        "policy_timing_records": policy_timings["records"],
        "policy_timing_expected": policy_timings["expected"],
        "policy_timing_complete": policy_timings["complete"],
        "serving_timing_records": serving_timings["records"],
        "serving_timing_expected": serving_timings["expected"],
        "serving_timing_complete": serving_timings["complete"],
        "operator_benchmark_sessions": operator_sessions,
        "operator_benchmark_complete": operator_sessions >= 5,
        "numerical_core_complete": numerical_core_complete,
        "performance_core_complete": performance_core_complete,
        "measurement_core_complete": measurement_core_complete,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/cross_model"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [inspect(args.results, identifier, name) for identifier, name in CONFIGURATIONS]
    report = {
        "evidence_tier": "pilot_only",
        "warning": "This inventory does not audit full π0-equivalent coverage; use full_coverage_audit.json.",
        "configuration_count": len(records),
        "not_run": sum(record["status"] == "PILOT NOT RUN" for record in records),
        "in_progress": sum(record["status"] == "PILOT IN PROGRESS" for record in records),
        "numerical_core_complete": sum(
            record["status"] == "PILOT NUMERICAL CORE PRESENT" for record in records
        ),
        "measurement_core_complete": sum(
            record["status"] == "PILOT MEASUREMENT CORE PRESENT" for record in records
        ),
        "configurations": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
