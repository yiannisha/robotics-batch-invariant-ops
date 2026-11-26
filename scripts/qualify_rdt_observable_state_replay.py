"""Full observable-state fixed-action qualification for RDT's five tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from scripts.maniskill_state_evidence import capture
from scripts.qualify_simpler_fixed_action_replay import digest


TASKS=['PegInsertionSide-v1','PickCube-v1','StackCube-v1','PlugCharger-v1','PushCube-v1']


def make_env(task):
    import gymnasium as gym
    import mani_skill.envs
    return gym.make(task,obs_mode='rgb',control_mode='pd_joint_pos',render_mode='rgb_array',
        reward_mode='sparse' if task=='PlugCharger-v1' else 'dense',sim_backend='cpu',
        sensor_configs={'shader_pack':'default'},human_render_camera_configs={'shader_pack':'default'},
        viewer_camera_configs={'shader_pack':'default'})


def render(env):return np.ascontiguousarray(env.render().squeeze(0).cpu().numpy(),dtype=np.uint8)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tasks',nargs='+',default=TASKS)
    p.add_argument('--initializations',type=int,default=3);p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if not set(args.tasks)<=set(TASKS) or args.initializations<1 or args.repeats<2:raise ValueError('Invalid schedule')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'frames').mkdir()
    torch.set_num_threads(8);torch.cuda.init()
    files=[Path(__file__).resolve(),Path(__file__).with_name('maniskill_state_evidence.py').resolve(),Path(__file__).with_name('qualify_simpler_fixed_action_replay.py').resolve()]
    runtime=dict(scope='full_observable_state_fixed_action_replay',tasks=args.tasks,initializations=args.initializations,
        repeats=args.repeats,max_steps=400,reset='fresh_scene_reconstruction_verified_against_canonical_snapshot',
        rng='Python_NumPy_Torch_CPU_CUDA_and_environment_generators',
        reward_mode_deviation='PlugCharger_sparse_only_on_ManiSkill3.0.1',
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},model_evaluated=False)
    (args.output/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    for file in files:(args.output/file.name).write_bytes(file.read_bytes())
    blocks=[]
    for task in args.tasks:
        for initialization in range(args.initializations):
            reference=None
            for repeat in range(args.repeats):
                seed=20241201+initialization
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                env=make_env(task)
                try:
                    obs,_=env.reset(seed=seed);rgb=render(env)
                    joint_start=obs['agent']['qpos'][0,:-2].detach().cpu().numpy().copy()
                    trajectory=[];steps=[]
                    def retain():
                        evidence,raw=capture(env,obs,rgb)
                        identity=digest(raw);path=args.output/'frames'/(identity+'.pt')
                        if not path.exists():torch.save(raw,path)
                        trajectory.append(dict(evidence,raw_frame=path.name))
                    retain()
                    for i in range(400):
                        action=np.concatenate([joint_start+0.01*np.sin(i/13),[1. if (i//20)%2 else -1.]]).astype(np.float32)
                        obs,reward,terminated,truncated,info=env.step(action);rgb=render(env);retain()
                        success=bool(info['success'])
                        steps.append(dict(action=action.tolist(),reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),success=success))
                        if (bool(terminated) or bool(truncated)) and success:break
                    row=dict(task=task,initialization=initialization,repeat=repeat,seed=seed,trajectory=trajectory,steps=steps)
                    with (args.output/'replays.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
                    current=digest(dict(trajectory=trajectory,steps=steps))
                    if reference is not None and current!=reference:raise RuntimeError(f'Observable replay differs: {task}/{initialization}/{repeat}')
                    reference=current
                    print(f'EXACT {task}/{initialization}/{repeat}: {len(steps)} controls',flush=True)
                finally:env.close()
            blocks.append(dict(task=task,initialization=initialization,repeats=args.repeats,exact=True))
    (args.output/'summary.json').write_text(json.dumps(dict(blocks=blocks,all_prescribed_blocks_exact=True,
        independent_audit_pending=True,model_evaluated=False),indent=2)+'\n')


if __name__=='__main__':main()
