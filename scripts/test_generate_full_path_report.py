from pathlib import Path

from generate_full_path_report import required_artifact_paths


def names(capacity: dict) -> set[str]:
    root = Path("/campaign")
    return {str(path.relative_to(root)) for path in required_artifact_paths(root, capacity)}


def test_report_omits_large_artifacts_for_captured_ooms() -> None:
    paths = names({"feasible": {
        "native|B16": False, "batch_invariant|B16": False,
        "native|B32": False, "batch_invariant|B32": False,
    }})
    assert not any("_B16_" in path or "_B32_" in path for path in paths)
    assert not any("fixed_B16" in path or "fixed_B32" in path for path in paths)


def test_report_includes_only_capacity_required_large_artifacts() -> None:
    paths = names({"feasible": {
        "native|B16": True, "batch_invariant|B16": True,
        "native|B32": True, "batch_invariant|B32": False,
    }})
    assert "performance/policy_native_B16_timings.jsonl" in paths
    assert "performance/policy_batch_invariant_B16_timings.jsonl" in paths
    assert "performance/policy_native_B32_timings.jsonl" in paths
    assert "performance/policy_batch_invariant_B32_timings.jsonl" not in paths
    assert "performance/serving_fixed_B16_timings.jsonl" in paths
    assert "performance/serving_fixed_B32_timings.jsonl" in paths
    assert "serving/requests_fixed_B16.jsonl" in paths
    assert "serving/requests_fixed_B32.jsonl" in paths
