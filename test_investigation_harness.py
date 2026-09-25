from __future__ import annotations

import json

import torch

from scripts.model_invariance.harness import compare_outputs, run_investigation


class SyntheticAdapter:
    name = "synthetic"

    def __init__(self, batch_dependent: bool = False) -> None:
        self.batch_dependent = batch_dependent

    def load_model(self):
        return torch.nn.Identity()

    def load_example_inputs(self, count: int):
        return [{"x": torch.tensor([float(index), 2.0])} for index in range(count)]

    def run_model(self, model, batch):
        actions = model(batch["x"])
        if self.batch_dependent:
            actions = actions + actions.shape[0]
        return {"actions": actions, "trajectory": (actions[:, None, :],)}

    def extract_robot_output(self, output):
        return output


def test_compare_outputs_reports_first_difference_and_hashes() -> None:
    reference = {"actions": torch.tensor([[1.0, 2.0]])}
    candidate = {"actions": torch.tensor([[1.0, 3.0]])}

    [comparison] = compare_outputs(reference, candidate)

    assert not comparison.exact
    assert comparison.differing_elements == 1
    assert comparison.first_differing_index == (0, 1)
    assert comparison.max_abs_difference == 1.0
    assert comparison.reference_hash != comparison.candidate_hash


def test_investigation_covers_duplicate_and_unrelated_compositions(tmp_path) -> None:
    output = tmp_path / "result.json"

    report = run_investigation(SyntheticAdapter(), batch_sizes=(1, 2, 3, 5), output_path=output)

    assert report["repeatability"]["exact"]
    assert report["end_to_end_exact"]
    assert report["maximum_tested_batch"] == 5
    assert [item["batch_size"] for item in report["composition_results"]["duplicate"]] == [
        1,
        2,
        3,
        5,
    ]
    assert json.loads(output.read_text())["end_to_end_exact"]


def test_investigation_detects_batch_dependent_final_output() -> None:
    report = run_investigation(SyntheticAdapter(batch_dependent=True), batch_sizes=(1, 2, 3))

    assert report["repeatability"]["exact"]
    assert not report["end_to_end_exact"]
    assert report["composition_results"]["duplicate"][0]["status"] == "PASS"
    assert report["composition_results"]["duplicate"][1]["status"] == "FAIL"
    tensor = report["composition_results"]["unrelated"][2]["tensors"][0]
    assert tensor["differing_elements"] == 2
