from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path("/workspace/robotics-batch-invariant-ops")
MANIFEST_PATH = ROOT / "results/cross_model/molmoact2/full/replay_manifest.json"


def test_molmoact2_manifest_has_full_real_replay_and_explicit_noise() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    rows = manifest["observations"]
    assert manifest["model_id"] == "molmoact2"
    assert len(rows) == 1000
    assert len({row["id"] for row in rows}) == 1000
    assert len({(row["episode_id"], row["observation_index"]) for row in rows}) == 1000
    assert sum(row["split"] == "diagnostic" for row in rows) == 200
    assert sum(row["split"] == "heldout" for row in rows) == 800
    noise = np.load(manifest["noise_file"], mmap_mode="r", allow_pickle=False)
    assert noise.shape == (1000, 3, 10, 32)
    assert noise.dtype == np.float32
    for replay_index in (0, 199, 200, 999):
        for noise_id in range(3):
            digest = hashlib.sha256(
                np.asarray(noise[replay_index, noise_id]).tobytes()
            ).hexdigest()
            assert rows[replay_index]["noise"][noise_id]["sha256"] == digest


def test_molmoact2_derivation_preserves_source_membership_pixels_and_companions() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    source_path = Path(manifest["source_replay_manifest_path"])
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == manifest[
        "source_replay_manifest_sha256"
    ]
    source = json.loads(source_path.read_text())
    fields = (
        "id", "replay_index", "task", "episode_id", "observation_index",
        "split", "state", "images", "image_sha256", "companion_replay_indices",
    )
    assert all(
        all(left[field] == right[field] for field in fields)
        for left, right in zip(source["observations"], manifest["observations"])
    )


class _Stats:
    def normalize_state(self, state, _domain):
        return state

    def get_metadata(self, _domain):
        return {"setup_type": "libero", "control_mode": "relative"}


class _Processor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(padding_side="right", pad_token_id=99)

    def __call__(self, *, text, images, return_tensors):
        assert len(images) == 2 and return_tensors == "pt"
        length = 5 + (len(text) % 3)
        return {
            "input_ids": torch.arange(length, dtype=torch.int64)[None],
            "attention_mask": torch.ones((1, length), dtype=torch.int64),
            "token_type_ids": torch.zeros((1, length), dtype=torch.bool),
            "pixel_values": torch.zeros((2, 3, 8, 8), dtype=torch.float32),
        }


def test_molmoact2_adapter_reconstructs_and_pads_frozen_requests(monkeypatch) -> None:
    monkeypatch.setenv("MOLMOACT2_REPLAY_MANIFEST", str(MANIFEST_PATH))
    from scripts.model_invariance.adapters.molmoact2 import MolmoAct2Adapter

    adapter = MolmoAct2Adapter()
    adapter.model = SimpleNamespace(config=SimpleNamespace(
        num_state_tokens=256, add_setup_tokens=True, add_control_tokens=True,
    ))
    adapter.stats = _Stats()
    adapter.metadata = adapter.stats.get_metadata("libero")
    adapter.modeling_module = SimpleNamespace(
        _normalize_question_text=lambda value: value,
        _build_discrete_state_string=lambda state, count: f"state:{len(state)}:{count}",
        _build_robot_text=lambda **values: values["task"] + values["discrete_state_string"],
    )
    adapter.processor = _Processor()
    left = adapter.load_replay_input(0, 0)
    right = adapter.load_replay_input(100, 2)
    assert left["trajectory"].shape == (10, 32)
    assert right["trajectory"].shape == (10, 32)
    batch = adapter.compose_batch([left, right])
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["token_type_ids"].shape == batch["input_ids"].shape
    assert batch["pixel_values"].shape == (4, 3, 8, 8)
    assert batch["trajectory"].shape == (2, 10, 32)
    for row, source in enumerate((left, right)):
        length = source["input_ids"].shape[1]
        assert torch.equal(batch["input_ids"][row, :length], source["input_ids"][0])
        assert torch.all(batch["input_ids"][row, length:] == 99)
    fixed = adapter.compose_fixed_batch([left, right])
    assert fixed["input_ids"].shape == (2, 497)
    assert adapter.fixed_batch_shape_metadata(fixed) == {
        "language_padding": "max_length_right",
        "language_tokens": 497,
        "canonical_length": 497,
    }
    from batch_invariant_ops import set_batch_invariant_mode

    with set_batch_invariant_mode():
        invariant = adapter.compose_batch([left, right])
    assert invariant["input_ids"].shape == (2, 497)

    live = adapter.prepare_runtime_sample(
        image=torch.zeros((3, 16, 16), dtype=torch.float32),
        wrist_image=torch.ones((3, 16, 16), dtype=torch.float32),
        state=torch.zeros(8), task="put the mug on the plate",
        trajectory=torch.zeros((10, 32)),
    )
    assert live["trajectory"].shape == (10, 32)
    assert live["pixel_values"].shape == (2, 3, 8, 8)


def test_molmoact2_campaign_configures_full_shared_driver_contract() -> None:
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_lingbot_vla2_full_campaign as shared
    import run_molmoact2_full_campaign as runner

    runner.configure_shared_driver()
    assert shared.MODEL_ID == "molmoact2"
    assert shared.MANIFEST == MANIFEST_PATH
    assert shared.ADAPTER == "scripts.model_invariance.adapters.molmoact2"
    assert shared.FLOW_STEPS == 10
    assert shared.FLOW_LABEL_TEMPLATE == "flow.step_{step:02d}.state"
    assert shared.CAMPAIGN_ENVIRONMENT["MOLMOACT2_STREAM_CHECKPOINT"] == "1"
    assert shared.CAMPAIGN_ENVIRONMENT["MOLMOACT2_REPLAY_MANIFEST"] == str(MANIFEST_PATH)
    assert shared.FROZEN_INPUT_FILES == [runner.SOURCE]
    assert callable(shared.CHECKPOINT_FREEZE_FILES)
    assert shared.__doc__.startswith("Run every non-success-rate stage required for full MolmoAct2")


def test_molmoact2_paired_noise_is_request_local_for_dynamic_and_serial() -> None:
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from run_molmoact2_paired_libero import MolmoAct2PairedPolicy

    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1, dtype=torch.bfloat16))
            self.model = SimpleNamespace(action_expert=SimpleNamespace(
                action_embed=SimpleNamespace(weight=self.weight)
            ))

    policy = MolmoAct2PairedPolicy.__new__(MolmoAct2PairedPolicy)
    torch.nn.Module.__init__(policy)
    policy.model = _FakeModel()
    seeds = [1101, 2202]
    policy._paired_generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    policy._paired_first_noise_hashes = [None, None]
    policy._paired_active_request_index = None
    values = policy._request_trajectories(2)
    expected = [
        torch.randn((10, 32), generator=torch.Generator().manual_seed(seed), dtype=torch.bfloat16)
        for seed in seeds
    ]
    assert all(torch.equal(left, right) for left, right in zip(values, expected))
    assert all(isinstance(value, str) and len(value) == 64 for value in policy._paired_first_noise_hashes)

    policy._paired_generators = [torch.Generator().manual_seed(seed) for seed in (1, 2, 3, 4)]
    policy._paired_first_noise_hashes = [None] * 4
    policy._paired_active_request_index = 2
    serial = policy._request_trajectories(1)
    expected_serial = torch.randn(
        (10, 32), generator=torch.Generator().manual_seed(3), dtype=torch.bfloat16
    )
    assert torch.equal(serial[0], expected_serial)
    assert policy._paired_first_noise_hashes[:2] == [None, None]
    assert policy._paired_first_noise_hashes[2] is not None
    assert policy._paired_first_noise_hashes[3] is None
