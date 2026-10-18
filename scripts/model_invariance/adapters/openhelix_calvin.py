"""Official OpenHelix PyTorch adapter for the released CALVIN model.

Required environment variables point to the official source checkout, the
``prompt_tuning_aux`` checkpoint directory, metadata-only LLaVA/CLIP
directories, and a directory of public CALVIN ``episode_*.npz`` transitions.
The initial diffusion trajectory is an explicit per-sample input.
"""

from __future__ import annotations

import math
import os
import random
import fcntl
import gc
import hashlib
import json
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

INSTRUCTION = "push the sliding door to the right side"
INTERPOLATION_LENGTH = 20
DIFFUSION_STEPS = 25
CHECKPOINT_REPOSITORY = "OpenHelix/openhelix"
CHECKPOINT_REVISION = "1a6fabbe148af045ed0710d007ed24c36fa83a53"
CHECKPOINT_SHARDS = [
    ("model-00000.safetensors", 2_105_645_528, "7bc815c581781b9d864ff22fbfd87f84d45adda0b283609d42f5716f6b933ac7"),
    ("model-00001.safetensors", 2_114_018_192, "6f015b79cea364ea356c0e11414e9dbe71f4736b2b5b8523119a34a087e0e758"),
    ("model-00002.safetensors", 2_114_034_864, "2ec780a890f2554385765fe3f4beedeb6b953d7eaf6f3887ad1535e56976433f"),
    ("model-00003.safetensors", 2_124_504_248, "6d552bc92746187c7d1614186e924dee8c3b456c06be03868da9d2ac07410128"),
    ("model-00004.safetensors", 2_057_395_512, "247bc9b65065446a596a3c431077bdbe4126681821334c2019369aaa65f49620"),
    ("model-00005.safetensors", 2_114_018_256, "4172759f8b2e57b1e0d3d8daaddea4b5489adfa546f07b4a7ce1b53db55f2e62"),
    ("model-00006.safetensors", 1_504_179_160, "ee2df39c2fc516148d2d82e108f66c189f2cd769fb4e42cc333e6dca23d87810"),
]
POLICY_CHECKPOINT = (
    "policy.pth", 445_004_793,
    "c1d0b70dc11798c23180cde475afcdd14aa981b33a827367a89bc5e8a9c96638",
)


@dataclass
class _Models:
    planner: torch.nn.Module
    policy: torch.nn.Module
    tokenizer: Any
    image_processor: Any

    def eval(self):
        self.planner.eval()
        self.policy.eval()
        return self


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _camera_basis(eye: np.ndarray, target: np.ndarray, up: np.ndarray):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, up)
    side /= np.linalg.norm(side)
    camera_up = np.cross(side, forward)
    return side, camera_up, forward


def _planner_image_uint8(value: Any) -> np.ndarray:
    """Return the official PIL-compatible RGB representation without rescaling."""
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"OpenHelix planner image shape differs: {image.shape}")
    if image.dtype == np.uint8:
        return image.copy()
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError(f"OpenHelix planner image dtype is not integral: {image.dtype}")
    if image.size and (int(image.min()) < 0 or int(image.max()) > 255):
        raise ValueError("OpenHelix planner image integer values exceed uint8 range")
    return image.astype(np.uint8)


def _deproject(
    depth: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    fov_degrees: float,
) -> np.ndarray:
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    focal = height / (2 * np.tan(np.deg2rad(fov_degrees) / 2))
    x = (u - width // 2) * depth / focal
    y = -(v - height // 2) * depth / focal
    side, camera_up, forward = _camera_basis(eye, target, up)
    points = (
        eye[None, None]
        + x[..., None] * side
        + y[..., None] * camera_up
        + depth[..., None] * forward
    )
    return points.astype(np.float32)


def _calvin_observation(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as item:
        rgb_static = item["rgb_static"].copy()
        rgb_gripper = item["rgb_gripper"].copy()
        depth_static = item["depth_static"].copy()
        depth_gripper = item["depth_gripper"].copy()
        robot = item["robot_obs"].copy()

    static_eye = np.asarray([2.871459009488717, -2.166602199425597, 2.555159848480571])
    static_target = np.asarray([-0.026242351159453392, -0.0302329882979393, 0.3920000493526459])
    static_up = np.asarray([0.4041403970338857, 0.22629790978217404, 0.8862616969685161])
    pcd_static = _deproject(depth_static, static_eye, static_target, static_up, 10.0)

    tcp_rotation = _rpy_matrix(robot[3:6])
    hand_position = robot[:3] - tcp_rotation @ np.asarray([0.0, 0.0, 0.14])
    camera_position = hand_position + tcp_rotation @ np.asarray([-0.1, 0.0, 0.0])
    camera_rotation = tcp_rotation @ _rpy_matrix(np.asarray([1.3, 0.0, -1.57]))
    camera_target = camera_position + camera_rotation[:, 1]
    camera_up = -camera_rotation[:, 2]
    pcd_gripper = _deproject(depth_gripper, camera_position, camera_target, camera_up, 75.0)

    gripper_rgb_tensor = torch.from_numpy(rgb_gripper).permute(2, 0, 1)[None].float()
    gripper_rgb_tensor = F.interpolate(
        gripper_rgb_tensor, size=(200, 200), mode="bilinear", align_corners=False
    )
    rgb_gripper = gripper_rgb_tensor[0].permute(1, 2, 0).byte().numpy()
    gripper_pcd_tensor = torch.from_numpy(pcd_gripper).permute(2, 0, 1)[None]
    gripper_pcd_tensor = F.interpolate(gripper_pcd_tensor, size=(200, 200), mode="nearest")
    pcd_gripper = gripper_pcd_tensor[0].permute(1, 2, 0).numpy()

    from utils.pytorch3d_transforms import euler_angles_to_matrix, matrix_to_quaternion

    euler = torch.from_numpy(robot[3:6]).float()
    quaternion = matrix_to_quaternion(euler_angles_to_matrix(euler, "XYZ")).numpy()
    openness = np.asarray([(robot[-1] + 1) / 2], dtype=np.float32)
    proprio = np.concatenate([robot[:3].astype(np.float32), quaternion, openness])
    return {
        "planner_image": rgb_static,
        "rgb": (np.stack([rgb_static, rgb_gripper]) / 255.0)
        .astype(np.float32)
        .transpose(0, 3, 1, 2),
        "pcd": np.stack([pcd_static, pcd_gripper]).transpose(0, 3, 1, 2),
        "proprio": proprio,
        "robot_euler": robot[:6].astype(np.float32),
    }


class OpenHelixCalvinAdapter:
    name = "openhelix-official-pytorch-calvin"
    causal_ablation_inapplicable_operations = {
        "explicit_sdpa": {
            "reason": (
                "The released OpenHelix policy implements attention with explicit "
                "torch.bmm calls rather than aten::scaled_dot_product_attention."
            ),
            "evidence_path": (
                "/workspace/models/openhelix/diffuser_actor/utils/"
                "multihead_custom_attention.py"
            ),
        },
    }

    def __init__(self) -> None:
        self.source = Path(os.environ.get("OPENHELIX_SOURCE", ""))
        self.checkpoint = Path(os.environ.get("OPENHELIX_CHECKPOINT", ""))
        self.base = Path(os.environ.get("OPENHELIX_BASE", ""))
        self.vision = Path(os.environ.get("OPENHELIX_VISION", ""))
        self.sample_dir = Path(os.environ.get("OPENHELIX_SAMPLE_DIR", ""))
        self.replay_manifest_path = Path(os.environ.get(
            "OPENHELIX_REPLAY_MANIFEST",
            "/workspace/robotics-batch-invariant-ops/results/cross_model/openhelix/full/replay_manifest.json",
        ))
        self.stream_checkpoint = os.environ.get("OPENHELIX_STREAM_CHECKPOINT", "0") == "1"
        self.stream_cache = Path(os.environ.get(
            "OPENHELIX_STREAM_CACHE", "/workspace/assets/openhelix/stream_cache"
        ))
        self.checkpoint_receipt = Path(os.environ.get(
            "OPENHELIX_CHECKPOINT_RECEIPT",
            "/workspace/robotics-batch-invariant-ops/results/cross_model/openhelix/full/checkpoint_stream_receipt.json",
        ))
        self.device = torch.device("cuda")
        self.trace_callback = None
        self.replay_manifest = None
        self.replay_noise = None

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    @staticmethod
    def execution_configuration() -> dict[str, Any]:
        return {
            "source": "KyujinL/CALVIN_ABC",
            "planner_image_shape_hwc": [200, 200, 3],
            "policy_cameras": ["static", "gripper"],
            "policy_rgb_shape_bchw": [2, 3, 200, 200],
            "policy_point_cloud_shape_bchw": [2, 3, 200, 200],
            "proprio_dim": 8,
            "planner_precision": "bfloat16",
            "policy_precision": "float32",
            "ddpm_steps": 25,
            "action_chunk_size": 19,
            "robot_output_dim": 7,
            "instruction": INSTRUCTION,
        }

    def _validate(self) -> list[Path]:
        if self.replay_manifest_path.is_file():
            manifest = json.loads(self.replay_manifest_path.read_text())
            fixture_root = Path(manifest["fixture_root"])
            files = sorted(fixture_root.glob("episode_*.npz"))
        else:
            files = sorted(self.sample_dir.glob("episode_*.npz"))
        required = [
            self.source / "planer.py",
            self.checkpoint / "policy.pth",
            self.base / "config.json",
            self.base / "tokenizer.model",
            self.vision / "config.json",
            self.vision / "preprocessor_config.json",
        ]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing OpenHelix artifacts: {missing}")
        shard_root = self.checkpoint / "llava_ckpt_safetensors"
        missing_shards = [name for name, _, _ in CHECKPOINT_SHARDS if not (shard_root / name).is_file()]
        if missing_shards and not self.stream_checkpoint:
            raise FileNotFoundError(f"missing exact OpenHelix planner shards: {missing_shards}")
        if len(files) < 3:
            raise FileNotFoundError("three public CALVIN transition fixtures are required")
        return files

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _record_checkpoint(
        self, name: str, size: int, digest: str, source: str, **extra: Any
    ) -> None:
        if self.checkpoint_receipt.is_file():
            record = json.loads(self.checkpoint_receipt.read_text())
            if (
                record.get("repository") != CHECKPOINT_REPOSITORY
                or record.get("revision") != CHECKPOINT_REVISION
            ):
                raise RuntimeError("OpenHelix checkpoint receipt belongs to another snapshot")
        else:
            record = {
                "schema_version": 1,
                "repository": CHECKPOINT_REPOSITORY,
                "revision": CHECKPOINT_REVISION,
                "files": {},
            }
        expected = {item[0]: item[1:] for item in [*CHECKPOINT_SHARDS, POLICY_CHECKPOINT]}
        if name not in expected or expected[name] != (size, digest):
            raise RuntimeError(f"checkpoint receipt attempted an unpinned artifact: {name}")
        record["files"][name] = {
            "bytes": size,
            "sha256": digest,
            "verification": "full_file_sha256_before_tensor_load",
            "source": source,
        }
        record.update(extra)
        record["all_pinned_files_verified"] = set(record["files"]) == set(expected)
        self.checkpoint_receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_receipt.with_name(
            self.checkpoint_receipt.name + f".tmp.{os.getpid()}"
        )
        temporary.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        os.replace(temporary, self.checkpoint_receipt)

    def _checkpoint_shard(
        self, name: str, size: int, expected_hash: str
    ) -> tuple[Path, int | None]:
        local = self.checkpoint / "llava_ckpt_safetensors" / name
        if local.is_file():
            if local.stat().st_size != size or self._sha256(local) != expected_hash:
                raise RuntimeError(f"OpenHelix checkpoint shard differs from pinned bytes: {name}")
            self._record_checkpoint(name, size, expected_hash, "resident_local_file")
            return local, None
        if not self.stream_checkpoint:
            raise FileNotFoundError(local)
        # The workspace deliberately does not have room for a second checkpoint
        # shard.  A previous filesystem-backed stream also kept an unlinked shard
        # alive through safetensors' mmap, exhausting disk before the next shard.
        # Keep each exact shard in a sealed anonymous file instead.  The
        # /proc/self/fd path remains seekable for safetensors and consumes no
        # workspace blocks.
        if not hasattr(os, "memfd_create"):
            raise OSError("OpenHelix checkpoint streaming requires Linux memfd_create")
        flags = getattr(os, "MFD_CLOEXEC", 0) | getattr(os, "MFD_ALLOW_SEALING", 0)
        descriptor = os.memfd_create(f"openhelix-{name}", flags)
        target = Path(f"/proc/self/fd/{descriptor}")
        url = (
            f"https://huggingface.co/{CHECKPOINT_REPOSITORY}/resolve/{CHECKPOINT_REVISION}/"
            f"prompt_tuning_aux/llava_ckpt_safetensors/{name}?download=true"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "batch-invariance-openhelix/1"})
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                os.fdopen(os.dup(descriptor), "wb") as output,
            ):
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            if os.fstat(descriptor).st_size != size or self._sha256(target) != expected_hash:
                raise RuntimeError(f"streamed OpenHelix checkpoint shard failed hash/size: {name}")
            seals = (
                fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW
                | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
            )
            fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        except BaseException:
            os.close(descriptor)
            raise
        self._record_checkpoint(name, size, expected_hash, "pinned_remote_memfd_stream")
        return target, descriptor

    def _add_source_path(self) -> None:
        source = str(self.source)
        if source in sys.path:
            sys.path.remove(source)
        sys.path.insert(0, source)

    def _load_planner(self):
        import transformers
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
        from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

        from model.llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
        from planer import LISAForCausalLM

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.base,
            model_max_length=512,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.add_tokens("<ACT>")
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        config = transformers.AutoConfig.from_pretrained(self.base)
        config.mm_vision_tower = str(self.vision)
        with torch.device("meta"):
            planner = LISAForCausalLM(
                config,
                out_dim=512,
                vision_tower=str(self.vision),
                use_mm_start_end=True,
            )
            tower = planner.get_model().get_vision_tower()
            tower.vision_tower = CLIPVisionModel(CLIPVisionConfig.from_pretrained(self.vision))
            tower.is_loaded = True
            try:
                # Transformers >=4.57 defaults to covariance-based initialization,
                # which reads meta tensors during this intentionally weightless
                # construction.  Every embedding is replaced by the released
                # checkpoint below, so retain the earlier initialization behavior.
                planner.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            except TypeError:
                planner.resize_token_embeddings(len(tokenizer))

        parameter_names = set(dict(planner.named_parameters()))
        expected = dict(planner.named_parameters()) | dict(planner.named_buffers())
        loaded: set[str] = set()
        for filename, size, expected_hash in CHECKPOINT_SHARDS:
            shard, ephemeral_descriptor = self._checkpoint_shard(
                filename, size, expected_hash
            )
            try:
                with safe_open(shard, framework="pt", device="cpu") as handle:
                    for name in handle.keys():
                        if name not in expected:
                            continue
                        shard_tensor = handle.get_tensor(name)
                        set_module_tensor_to_device(
                            planner,
                            name,
                            self.device,
                            value=shard_tensor,
                            dtype=torch.bfloat16 if name in parameter_names else None,
                        )
                        loaded.add(name)
                        del shard_tensor
            finally:
                # Ensure no safetensors mmap view survives before closing the
                # anonymous backing file.
                gc.collect()
                if ephemeral_descriptor is not None:
                    os.close(ephemeral_descriptor)
        derived = set(expected) - loaded
        for name in sorted(derived):
            if name.endswith("vision_model.embeddings.position_ids"):
                length = expected[name].shape[-1]
                value = torch.arange(length, device=self.device).expand(1, -1)
            elif name.endswith("rotary_emb.inv_freq"):
                # Newer Transformers exposes this deterministic, non-persistent
                # buffer in named_buffers(); older checkpoints do not serialize it.
                module_name, _ = name.rsplit(".", 1)
                rotary = planner.get_submodule(module_name)
                value, _ = rotary.rope_init_fn(rotary.config, self.device)
            elif name.endswith("rotary_emb._cos_cached") or name.endswith("rotary_emb._sin_cached"):
                module_name, buffer_name = name.rsplit(".", 1)
                rotary = planner.get_submodule(module_name)
                length = expected[name].shape[0]
                steps = torch.arange(length, device=self.device, dtype=rotary.inv_freq.dtype)
                frequencies = torch.outer(steps / rotary.scaling_factor, rotary.inv_freq)
                angles = torch.cat((frequencies, frequencies), dim=-1)
                value = angles.cos() if buffer_name == "_cos_cached" else angles.sin()
            else:
                continue
            set_module_tensor_to_device(planner, name, self.device, value=value)
            loaded.add(name)
        missing = set(expected) - loaded
        if missing:
            raise RuntimeError(f"official planner checkpoint is incomplete: {sorted(missing)}")
        if any(value.is_meta for value in planner.parameters()):
            raise RuntimeError("planner still contains meta parameters")
        planner.config.use_cache = False
        self._record_checkpoint(
            CHECKPOINT_SHARDS[-1][0], CHECKPOINT_SHARDS[-1][1], CHECKPOINT_SHARDS[-1][2],
            "pinned_remote_memfd_stream" if not (
                self.checkpoint / "llava_ckpt_safetensors" / CHECKPOINT_SHARDS[-1][0]
            ).is_file() else "resident_local_file",
            strict_planner_load={
                "expected_named_tensors": len(expected),
                "loaded_or_deterministically_derived_named_tensors": len(loaded),
                "missing_named_tensors": [],
                "meta_parameters_remaining": False,
            },
        )
        image_processor = CLIPImageProcessor.from_pretrained(self.vision)
        return planner.eval(), tokenizer, image_processor

    def _load_policy(self):
        from diffuser_actor.trajectory_optimization.diffuser_actor_act_simple import (
            DiffuserActorACTS,
        )
        from utils.common_utils import get_gripper_loc_bounds

        bounds = get_gripper_loc_bounds(
            str(self.source / "tasks/calvin_rel_traj_location_bounds_task_ABC_D.json"),
            buffer=0.01,
        )
        policy = DiffuserActorACTS(
            backbone="clip",
            image_size=(256, 256),
            embedding_dim=192,
            num_vis_ins_attn_layers=2,
            use_instruction=True,
            fps_subsampling_factor=3,
            gripper_loc_bounds=bounds,
            rotation_parametrization="6D",
            quaternion_format="wxyz",
            diffusion_timesteps=DIFFUSION_STEPS,
            nhist=1,
            relative=True,
            lang_enhanced=True,
        )
        policy_path = self.checkpoint / "policy.pth"
        policy_name, policy_size, policy_hash = POLICY_CHECKPOINT
        if (
            not policy_path.is_file()
            or policy_path.stat().st_size != policy_size
            or self._sha256(policy_path) != policy_hash
        ):
            raise RuntimeError("OpenHelix policy checkpoint differs from pinned bytes")
        checkpoint = torch.load(
            policy_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )["weight"]
        state = {name[7:]: value for name, value in checkpoint.items()}
        policy.load_state_dict(state, strict=True)
        self._record_checkpoint(
            policy_name, policy_size, policy_hash, "resident_local_file",
            strict_policy_load={
                "checkpoint_state_tensors": len(state),
                "missing_keys": [],
                "unexpected_keys": [],
            },
        )
        return policy.to(self.device).eval()

    def load_model(self):
        self._validate()
        self._add_source_path()
        planner, tokenizer, image_processor = self._load_planner()
        policy = self._load_policy()
        return _Models(planner, policy, tokenizer, image_processor).eval()

    def load_example_inputs(self, count: int):
        if self.replay_manifest_path.is_file():
            return [self.load_replay_input(index, 0) for index in range(count)]
        files = self._validate()
        self._add_source_path()
        observations = [_calvin_observation(path) for path in files]
        examples = []
        for index in range(count):
            observation = observations[index % len(observations)]
            generator = torch.Generator(device="cuda").manual_seed(91_000 + index)
            noise = torch.randn(
                INTERPOLATION_LENGTH - 1,
                9,
                generator=generator,
                device=self.device,
            ).cpu()
            position_step_noise = []
            rotation_step_noise = []
            for _ in range(DIFFUSION_STEPS):
                position_step_noise.append(
                    torch.randn(
                        INTERPOLATION_LENGTH - 1,
                        3,
                        generator=generator,
                        device=self.device,
                    ).cpu()
                )
                rotation_step_noise.append(
                    torch.randn(
                        INTERPOLATION_LENGTH - 1,
                        6,
                        generator=generator,
                        device=self.device,
                    ).cpu()
                )
            examples.append(
                {
                    **observation,
                    "instruction": INSTRUCTION,
                    "noise": noise,
                    "position_step_noise": torch.stack(position_step_noise),
                    "rotation_step_noise": torch.stack(rotation_step_noise),
                }
            )
        return examples

    def _load_replay_metadata(self) -> None:
        if self.replay_manifest is not None:
            return
        if not self.replay_manifest_path.is_file():
            raise FileNotFoundError(self.replay_manifest_path)
        manifest = json.loads(self.replay_manifest_path.read_text())
        noise_path = Path(manifest["noise_file"])
        if not noise_path.is_absolute():
            noise_path = self.replay_manifest_path.parent / noise_path
        noise = np.load(noise_path, mmap_mode="r", allow_pickle=False)
        if list(noise.shape) != [1000, 3, 4446] or noise.dtype != np.float32:
            raise ValueError("OpenHelix replay noise shape/dtype differs")
        self.replay_manifest = manifest
        self.replay_noise = noise

    def load_replay_input(self, replay_index: int, noise_id: int = 0):
        self._load_replay_metadata()
        assert self.replay_manifest is not None and self.replay_noise is not None
        rows = self.replay_manifest["observations"]
        if not 0 <= replay_index < len(rows) or not 0 <= noise_id < 3:
            raise IndexError((replay_index, noise_id))
        row = rows[replay_index]
        path = Path(self.replay_manifest["fixture_root"]) / row["source_file"]
        observation = _calvin_observation(path)
        value = np.asarray(self.replay_noise[replay_index, noise_id], dtype=np.float32)
        return {
            **observation,
            "instruction": row["instruction"],
            "noise": torch.from_numpy(value[:171].copy()).reshape(19, 9),
            "position_step_noise": torch.from_numpy(value[171:1596].copy()).reshape(25, 19, 3),
            "rotation_step_noise": torch.from_numpy(value[1596:4446].copy()).reshape(25, 19, 6),
        }

    @staticmethod
    def request_noise(seed: int) -> dict[str, torch.Tensor]:
        """Create one request-local stochastic input, independent of batching.

        The released policy samples an initial trajectory and per-step DDPM
        variance noise internally.  Closed-loop pairing must instead bind those
        random values to the request identity so singleton and dynamic execution
        receive byte-identical stochastic inputs.
        """
        if not 0 <= seed < 2**63:
            raise ValueError(f"OpenHelix request-noise seed is out of range: {seed}")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return {
            "noise": torch.randn(
                INTERPOLATION_LENGTH - 1, 9, generator=generator, dtype=torch.float32
            ),
            "position_step_noise": torch.randn(
                DIFFUSION_STEPS, INTERPOLATION_LENGTH - 1, 3,
                generator=generator, dtype=torch.float32,
            ),
            "rotation_step_noise": torch.randn(
                DIFFUSION_STEPS, INTERPOLATION_LENGTH - 1, 6,
                generator=generator, dtype=torch.float32,
            ),
        }

    def load_live_calvin_input(
        self, observation: dict[str, Any], env: Any, instruction: str, noise_seed: int
    ) -> dict[str, Any]:
        """Convert one live official CALVIN observation into policy inputs.

        This deliberately calls the checkpoint-matched evaluator's camera and
        proprioception preprocessing rather than approximating point clouds from
        serialized calibration values.  Copies prevent evaluator preprocessing
        from mutating the simulator-owned observation used for trajectory logs.
        """
        from online_evaluation_calvin.evaluate_utils import (
            prepare_proprio_states,
            prepare_visual_states,
        )

        raw_robot = np.asarray(observation["robot_obs"]).copy()
        raw_static = _planner_image_uint8(observation["rgb_obs"]["rgb_static"])
        prepared = {
            "rgb_obs": {
                "rgb_static": raw_static.copy(),
                "rgb_gripper": np.asarray(
                    observation["rgb_obs"]["rgb_gripper"]
                ).copy(),
            },
            "depth_obs": {
                "depth_static": np.asarray(
                    observation["depth_obs"]["depth_static"]
                ).copy(),
                "depth_gripper": np.asarray(
                    observation["depth_obs"]["depth_gripper"]
                ).copy(),
            },
            "robot_obs": raw_robot.copy(),
        }
        prepare_visual_states(prepared, env)
        prepare_proprio_states(prepared, env)
        proprio = np.asarray(prepared["proprio"][-1], dtype=np.float32)
        if proprio.shape != (8,):
            raise RuntimeError(f"OpenHelix live CALVIN proprio shape differs: {proprio.shape}")
        return {
            "planner_image": raw_static,
            "rgb": np.stack([
                prepared["rgb_obs"]["rgb_static"],
                prepared["rgb_obs"]["rgb_gripper"],
            ]).astype(np.float32).transpose(0, 3, 1, 2),
            "pcd": np.stack([
                prepared["pcd_obs"]["pcd_static"],
                prepared["pcd_obs"]["pcd_gripper"],
            ]).astype(np.float32).transpose(0, 3, 1, 2),
            "proprio": proprio,
            "robot_euler": raw_robot[:6].astype(np.float32),
            "instruction": instruction,
            **self.request_noise(noise_seed),
        }

    @staticmethod
    def compose_batch(samples):
        return {
            "planner_images": [sample["planner_image"] for sample in samples],
            "rgb": torch.from_numpy(np.stack([sample["rgb"] for sample in samples])),
            "pcd": torch.from_numpy(np.stack([sample["pcd"] for sample in samples])),
            "proprio": torch.from_numpy(np.stack([sample["proprio"] for sample in samples])),
            "robot_euler": torch.from_numpy(
                np.stack([sample["robot_euler"] for sample in samples])
            ),
            "instructions": [sample["instruction"] for sample in samples],
            "noise": torch.stack([sample["noise"] for sample in samples]),
            "position_step_noise": torch.stack(
                [sample["position_step_noise"] for sample in samples]
            ),
            "rotation_step_noise": torch.stack(
                [sample["rotation_step_noise"] for sample in samples]
            ),
        }

    def _planner_features(self, models: _Models, images, instructions):
        from datasets.calvin_dataset import transfer
        from model.llava.constants import IMAGE_TOKEN_INDEX
        from model.llava.mm_utils import tokenizer_image_token

        from model.llava import conversation as conversation_lib

        conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
        conv = conversation_lib.default_conversation.copy()
        separator = conv.sep + conv.roles[1] + ":"
        token_rows = []
        for instruction in instructions:
            # The official prompt templates are randomly selected.  Bind the
            # selection to each request (rather than its batch position) so the
            # same instruction has the same prompt in every arrangement.
            random_state = random.getstate()
            random.seed(31_415)
            conversation = transfer([instruction])[0][0]
            random.setstate(random_state)
            first_round = conversation.split(conv.sep2)[0]
            prompt = first_round.split(separator)[0] + separator + " " + "<ACT>"
            token_rows.append(
                tokenizer_image_token(prompt, models.tokenizer, return_tensors="pt")
            )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            token_rows,
            batch_first=True,
            padding_value=models.tokenizer.pad_token_id,
        ).to(self.device)
        attention_mask = input_ids.ne(models.tokenizer.pad_token_id)
        if (input_ids == IMAGE_TOKEN_INDEX).sum(dim=1).ne(1).any():
            raise RuntimeError("each OpenHelix prompt must contain one image token")

        pil_images = [Image.fromarray(image) for image in images]
        pixels = models.image_processor.preprocess(pil_images, return_tensors="pt")[
            "pixel_values"
        ].to(self.device, dtype=torch.bfloat16)
        self._trace("planner.input_ids", input_ids)
        self._trace("planner.pixels", pixels)
        _, features = models.planner.evaluate(pixels, input_ids, attention_mask)
        features = features.reshape(len(images), 1, 512)
        self._trace("planner.action_token", features)
        return features

    @staticmethod
    def _scheduler_step(scheduler, model_output, timestep, sample, variance_noise):
        from diffusers.schedulers import scheduling_ddpm

        original = scheduling_ddpm.randn_tensor
        scheduling_ddpm.randn_tensor = lambda *args, **kwargs: variance_noise
        try:
            return scheduler.step(model_output, timestep, sample).prev_sample
        finally:
            scheduling_ddpm.randn_tensor = original

    def _sample_policy(
        self,
        policy,
        fixed_inputs,
        noise,
        position_step_noise,
        rotation_step_noise,
    ):
        batch_size = noise.shape[0]
        policy.position_noise_scheduler.set_timesteps(DIFFUSION_STEPS)
        policy.rotation_noise_scheduler.set_timesteps(DIFFUSION_STEPS)
        condition = torch.zeros_like(noise)
        mask = torch.zeros_like(noise, dtype=torch.bool)
        initial_timestep = torch.full(
            (batch_size,),
            policy.position_noise_scheduler.timesteps[0],
            device=self.device,
            dtype=torch.long,
        )
        noisy_position = policy.position_noise_scheduler.add_noise(
            condition[..., :3], noise[..., :3], initial_timestep
        )
        noisy_rotation = policy.rotation_noise_scheduler.add_noise(
            condition[..., 3:9], noise[..., 3:9], initial_timestep
        )
        noisy_condition = torch.cat((noisy_position, noisy_rotation), dim=-1)
        trajectory = torch.where(mask, noisy_condition, noise)
        self._trace("policy.noise", trajectory)

        output = None
        for index, timestep in enumerate(policy.position_noise_scheduler.timesteps):
            step = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
            output = policy.policy_forward_pass(trajectory, step, fixed_inputs)[0][-1]
            position = self._scheduler_step(
                policy.position_noise_scheduler,
                output[..., :3],
                timestep,
                trajectory[..., :3],
                position_step_noise[:, index],
            )
            rotation = self._scheduler_step(
                policy.rotation_noise_scheduler,
                output[..., 3:9],
                timestep,
                trajectory[..., 3:9],
                rotation_step_noise[:, index],
            )
            trajectory = torch.cat((position, rotation), dim=-1)
            self._trace(f"policy.diffusion.{index}", trajectory)
            self._trace(f"flow.step_{index:02d}.state", trajectory)
        assert output is not None
        return torch.cat((trajectory, output[..., 9:]), dim=-1)

    def run_model(self, models: _Models, batch):
        from utils.pytorch3d_transforms import matrix_to_euler_angles, quaternion_to_matrix

        planner_features = self._planner_features(
            models, batch["planner_images"], batch["instructions"]
        ).float()
        rgb = batch["rgb"].to(self.device, dtype=torch.float32)[..., 20:180, 20:180]
        pcd = batch["pcd"].to(self.device, dtype=torch.float32)[..., 20:180, 20:180]
        proprio = batch["proprio"].to(self.device, dtype=torch.float32)[:, None]
        noise = batch["noise"].to(self.device, dtype=torch.float32)
        position_step_noise = batch["position_step_noise"].to(self.device, dtype=torch.float32)
        rotation_step_noise = batch["rotation_step_noise"].to(self.device, dtype=torch.float32)
        self._trace("policy.rgb", rgb)
        self._trace("policy.pcd", pcd)
        self._trace("policy.proprio", proprio)

        relative_pcd, relative_proprio = models.policy.convert2rel(pcd, proprio)
        relative_pcd = relative_pcd.clone()
        relative_proprio = relative_proprio.clone()
        relative_pcd = torch.permute(
            models.policy.normalize_pos(torch.permute(relative_pcd, [0, 1, 3, 4, 2])),
            [0, 1, 4, 2, 3],
        )
        relative_proprio[..., :3] = models.policy.normalize_pos(relative_proprio[..., :3])
        relative_proprio = models.policy.convert_rot(relative_proprio)
        fixed_inputs = models.policy.encode_inputs(
            rgb, relative_pcd, planner_features, relative_proprio
        )
        trajectory = self._sample_policy(
            models.policy,
            fixed_inputs,
            noise,
            position_step_noise,
            rotation_step_noise,
        )
        trajectory = models.policy.unconvert_rot(trajectory)
        trajectory[..., :3] = models.policy.unnormalize_pos(trajectory[..., :3])
        trajectory[..., 7] = trajectory[..., 7].sigmoid()
        self._trace("policy.relative_trajectory", trajectory)

        euler = matrix_to_euler_angles(quaternion_to_matrix(trajectory[..., 3:7]), "XYZ")
        gripper = 2 * trajectory[..., 7:].ge(0.5).to(trajectory.dtype) - 1
        relative_action = torch.cat((trajectory[..., :3], euler, gripper), dim=-1)
        current = batch["robot_euler"].to(self.device, dtype=torch.float32)
        absolute = relative_action.clone()
        absolute[..., :6] += current[:, None]
        self._trace("actions.absolute", absolute)
        return absolute

    @staticmethod
    def extract_robot_output(output: Any):
        return output


adapter = OpenHelixCalvinAdapter()
