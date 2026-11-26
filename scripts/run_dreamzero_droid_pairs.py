"""Preliminary native singleton/companion/repeat checks in official DROID sim.

Requires the separate IsaacLab environment; inference uses the NPZ queue worker.
All trajectories start from one saved scene state and one saved initial image.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch


def flatten_state(tree, prefix=''):
    out = {}
    for key, value in tree.items():
        path = prefix + key
        if isinstance(value, dict):
            out.update(flatten_state(value, path+'/'))
        else:
            out[path] = value.detach().cpu().numpy().copy()
    return out


def digest(arrays):
    h = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.asarray(value)
        h.update(key.encode())
        h.update(str((value.shape, str(value.dtype))).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def observation(obs, frames, prompt):
    import cv2
    policy = obs['policy']
    keys = {'external_cam': 'video.exterior_image_1_left',
            'external_cam_2': 'video.exterior_image_2_left', 'wrist_cam': 'video.wrist_image_left'}
    result = {}
    for key, output in keys.items():
        image = policy[key][0, ..., :3].detach().cpu().numpy()
        image = cv2.resize(image, (320,180), interpolation=cv2.INTER_LINEAR)
        frames.setdefault(output, []).append(image)
        selected = frames[output][-4:]
        if len(frames[output]) == 1:
            selected = frames[output]
        else:
            selected = [selected[0]] * (4-len(selected)) + selected
        result[output] = np.stack(selected)
    result['state.joint_position'] = policy['arm_joint_pos'].detach().cpu().numpy().reshape(1,7).astype(np.float64)
    result['state.gripper_position'] = policy['gripper_pos'].detach().cpu().numpy().reshape(1,1).astype(np.float64)
    result['annotation.language.action_text'] = np.asarray(prompt)
    return result


class Queue:
    def __init__(self, path):
        self.path = path
        self.index = 0

    def infer(self, target, companion, reset):
        if companion is not None:
            companion = dict(companion)
            for key in target:
                if key.startswith('video.'):
                    companion[key] = np.repeat(companion[key][-1:], target[key].shape[0], axis=0)
        members = [target] + ([] if companion is None else [companion])
        batch = {key: np.stack([x[key] for x in members]) for key in target}
        batch['_reset'] = np.asarray(reset)
        stem = self.path / f'{self.index:06d}'
        temporary = Path(str(stem)+'.tmp.npz')
        np.savez(temporary, **batch)
        temporary.replace(Path(str(stem)+'.request.npz'))
        deadline = time.monotonic()+3600
        while not Path(str(stem)+'.json').exists():
            if time.monotonic()>deadline:
                raise TimeoutError(f'Inference request still unresolved: {stem}')
            time.sleep(0.2)
        meta = json.loads(Path(str(stem)+'.json').read_text())
        with np.load(Path(str(stem)+'.result.npz')) as data:
            joints = data['action.joint_position'][0]
            gripper = data['action.gripper_position'][0].reshape(-1,1)
            action = np.concatenate([joints,gripper],axis=-1)
        assert meta['finite'] and action.shape == (24,8), (meta, action.shape)
        self.index += 1
        return action, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.queue.mkdir(parents=True, exist_ok=True)
    # Initialize CUDA and pip Warp before Kit avoids stale device enumeration
    # and Kit's older embedded Warp on this Blackwell host.
    torch.cuda.init()
    import warp
    warp.init()
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, device='cuda:0',
        kit_args='--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0').app
    import gymnasium as gym
    import sim_evals.environments
    from isaaclab_tasks.utils import parse_env_cfg
    cfg = parse_env_cfg('DROID', device='cuda:0', num_envs=1, use_fabric=True)
    cfg.set_scene(1)
    cfg.seed = 42
    env = gym.make('DROID', cfg=cfg)
    base = env.unwrapped
    obs,_ = env.reset(seed=42)
    obs,_ = env.reset(seed=42)
    snapshot = copy.deepcopy(base.scene.get_state())
    # PhysX normalizes USD quaternions on write. Canonicalize before freezing
    # the snapshot so repeated restores can be compared bit for bit.
    base.scene.reset_to(snapshot)
    base.sim.forward()
    base.scene.update(base.physics_dt)
    snapshot = copy.deepcopy(base.scene.get_state())
    initial = observation(obs, {}, 'put the cube in the bowl')
    np.savez(args.output/'initial_observation.npz', **initial)
    np.savez(args.output/'initial_state.npz', **flatten_state(snapshot))
    # A distinct valid simulated request: the same robot after moving joint 1.
    alternative = np.concatenate([initial['state.joint_position'][0].copy(), [1.0]])
    alternative[0] += 0.2
    for _ in range(8):
        obs,_,_,_,_ = env.step(torch.tensor(alternative,device=base.device,dtype=torch.float32)[None])
    # Keep prompt lengths equal: native encode_prompt masks all batch rows in
    # a loop over lengths; unequal lengths would introduce a semantic confound.
    companion = observation(obs, {}, 'put the cube in the bowl')
    np.savez(args.output/'companion_observation.npz', **companion)

    realized_reference = None

    def restore():
        nonlocal realized_reference
        env.reset(seed=42)
        base.scene.reset_to(snapshot)
        base.sim.forward()
        base.scene.update(base.physics_dt)
        current = flatten_state(base.scene.get_state())
        # Compare repeated realized states, rather than the write command:
        # PhysX can round a normalized quaternion by one float32 ULP on write.
        if realized_reference is None:
            realized_reference = {k:v.copy() for k,v in current.items()}
            np.savez(args.output/'realized_initial_state.npz', **realized_reference)
        reference = realized_reference
        error = max(float(np.max(np.abs(current[k]-reference[k]))) for k in reference)
        assert error == 0, f'Initial scene differs by {error}'
        return digest(current)

    identity = restore()
    (args.output/'setup.json').write_text(json.dumps({'initial_state_sha256':identity,
        'initial_observation_sha256':digest(initial), 'companion_sha256':digest(companion),
        'scene':1,'seed':42,'steps':args.steps,'torch':torch.__version__,
        'initial_image_reused':True,'dit_cache':False}, indent=2))
    print('SIMULATION_SETUP_VERIFIED', identity, flush=True)
    if args.collect_only:
        env.close()
        app.close()
        return
    queue = Queue(args.queue)
    conditions = [('native_singleton',False),('native_dynamic',True),
                  ('native_singleton_repeat',False),('native_dynamic_repeat',True)]
    fixed = {}
    for condition, dynamic in conditions:
        action, meta = queue.infer(initial, companion if dynamic else None, True)
        fixed[condition] = {'actions':action.tolist(),'meta':meta}
        (args.output/'fixed_input.json').write_text(json.dumps(fixed,indent=2))
        print('FIXED_INPUT_COMPLETED',condition,flush=True)
    first = fixed['native_singleton']['meta']
    for condition in fixed:
        meta = fixed[condition]['meta']
        assert first['normalized_target_sha256']==meta['normalized_target_sha256'], condition
        assert [x['target_sha256'] for x in first['noise']]==[x['target_sha256'] for x in meta['noise']], condition
    for condition, dynamic in conditions:
        restored = restore()
        frames = {}
        target = initial
        trajectory = []
        for step in range(args.steps):
            if step % 8 == 0:
                if step:
                    target = observation(obs, frames, 'put the cube in the bowl')
                else:
                    # Populate the frame history with the exact saved initial frame.
                    frames = {k:[v[0].copy()] for k,v in initial.items() if k.startswith('video.')}
                actions, meta = queue.infer(target, companion if dynamic else None, step==0)
            action = actions[step%8].copy()
            action[-1] = float(action[-1]>0.5)
            obs,reward,term,trunc,_ = env.step(torch.tensor(action,device=base.device,dtype=torch.float32)[None])
            state = flatten_state(base.scene.get_state())
            trajectory.append({'step':step,'action':action.tolist(),'reward':float(reward.item()),
                'terminated':bool(term.item()),'truncated':bool(trunc.item()),
                'state':{k:v.tolist() for k,v in state.items()},'inference_index':meta['index']})
        (args.output/f'{condition}.json').write_text(json.dumps({'condition':condition,
            'initial_state_sha256':restored,'trajectory':trajectory},indent=2))
        print('TRAJECTORY_COMPLETED',condition,flush=True)
    env.close()
    app.close()


if __name__ == '__main__':
    main()
