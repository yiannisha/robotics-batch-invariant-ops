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
import sys
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

    def __init__(self) -> None:
        self.source = Path(os.environ.get("OPENHELIX_SOURCE", ""))
        self.checkpoint = Path(os.environ.get("OPENHELIX_CHECKPOINT", ""))
        self.base = Path(os.environ.get("OPENHELIX_BASE", ""))
        self.vision = Path(os.environ.get("OPENHELIX_VISION", ""))
        self.sample_dir = Path(os.environ.get("OPENHELIX_SAMPLE_DIR", ""))
        self.device = torch.device("cuda")
        self.trace_callback = None

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.trace_callback is not None:
            self.trace_callback(label, value)

    def _validate(self) -> list[Path]:
        files = sorted(self.sample_dir.glob("episode_*.npz"))
        required = [
            self.source / "planer.py",
            self.checkpoint / "policy.pth",
            self.base / "config.json",
            self.base / "tokenizer.model",
            self.vision / "config.json",
            self.vision / "preprocessor_config.json",
        ]
        required += sorted((self.checkpoint / "llava_ckpt_safetensors").glob("*.safetensors"))
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing OpenHelix artifacts: {missing}")
        if len(required) < 13:
            raise FileNotFoundError("the seven official LLM shards are required")
        if len(files) < 3:
            raise FileNotFoundError("three public CALVIN transition fixtures are required")
        return files

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
            planner.resize_token_embeddings(len(tokenizer))

        parameter_names = set(dict(planner.named_parameters()))
        expected = dict(planner.named_parameters()) | dict(planner.named_buffers())
        loaded: set[str] = set()
        shards = sorted((self.checkpoint / "llava_ckpt_safetensors").glob("*.safetensors"))
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name not in expected:
                        continue
                    value = handle.get_tensor(name)
                    set_module_tensor_to_device(
                        planner,
                        name,
                        self.device,
                        value=value,
                        dtype=torch.bfloat16 if name in parameter_names else None,
                    )
                    loaded.add(name)
        derived = set(expected) - loaded
        for name in sorted(derived):
            if name.endswith("vision_model.embeddings.position_ids"):
                length = expected[name].shape[-1]
                value = torch.arange(length, device=self.device).expand(1, -1)
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
        checkpoint = torch.load(
            self.checkpoint / "policy.pth",
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )["weight"]
        state = {name[7:]: value for name, value in checkpoint.items()}
        policy.load_state_dict(state, strict=True)
        return policy.to(self.device).eval()

    def load_model(self):
        self._validate()
        self._add_source_path()
        planner, tokenizer, image_processor = self._load_planner()
        policy = self._load_policy()
        return _Models(planner, policy, tokenizer, image_processor).eval()

    def load_example_inputs(self, count: int):
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

        random_state = random.getstate()
        random.seed(31_415)
        conversation = transfer([instructions[0]])[0][0]
        random.setstate(random_state)

        from model.llava import conversation as conversation_lib

        conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
        conv = conversation_lib.default_conversation.copy()
        separator = conv.sep + conv.roles[1] + ":"
        first_round = conversation.split(conv.sep2)[0]
        prompt = first_round.split(separator)[0] + separator + " " + "<ACT>"
        token_ids = tokenizer_image_token(prompt, models.tokenizer, return_tensors="pt")
        input_ids = token_ids[None].repeat(len(images), 1).to(self.device)
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
