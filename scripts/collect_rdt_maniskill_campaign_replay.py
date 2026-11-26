"""Collect native complete episodes and checkpoint-matched real replay inputs.

Five tasks x20 starts x10 evenly spaced pre-action observations. All episode
outcomes are retained. Starts0..3 are diagnostic;4..19 are held out per task.
This collection is not a paired success-rate study or a numerical freeze.
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch

from qualify_rdt_maniskill_replay import canonical
from qualify_simpler_fixed_action_replay import digest

TASKS=('PegInsertionSide-v1','PickCube-v1','StackCube-v1','PlugCharger-v1','PushCube-v1')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','siglip','source','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    os.environ.update(RDT_CHECKPOINT=str(args.checkpoint),RDT_SIGLIP=str(args.siglip),RDT_SOURCE=str(args.source))
    from batch_invariant_ops import set_batch_invariant_mode
    from scripts.model_invariance.adapters.rdt_maniskill import RDTManiSkillAdapter
    from scripts.run_rdt_paired_server import PairedRDTBackend,verify_artifacts
    import gymnasium as gym
    import mani_skill.envs
    class Adapter(RDTManiSkillAdapter):
        def _validate_paths(self):
            for path in (self.checkpoint/'rdt/mp_rank_00_model_states.pt',self.siglip/'model.safetensors',self.source/'configs/base.yaml'):
                if not path.is_file():raise FileNotFoundError(path)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    adapter=Adapter();policy=adapter.load_model()
    backend=PairedRDTBackend(adapter,policy,condition='native_dynamic',checkpoint=args.checkpoint,
        language=adapter.language_embeddings,artifacts=verify_artifacts(args.checkpoint,args.siglip),fixed_language_tokens=30,maximum_batch=32)
    runtime=dict(scope='native_replay_collection',tasks=TASKS,initializations=20,observations_per_episode=10,
                 language_tokens=30,executed_action_stride=4,max_steps=400,runner_sha256=sha(Path(__file__)),
                 torch=torch.__version__,gpu=torch.cuda.get_device_name(),paired_success_rate_campaign_complete=False)
    (args.output/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    observations=[];episodes=[];noises=[]
    for task in TASKS:
        for initialization in range(20):
            seed=20241201+initialization
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            reward_mode='sparse' if task=='PlugCharger-v1' else 'dense'
            env=gym.make(task,obs_mode='rgb',control_mode='pd_joint_pos',render_mode='rgb_array',reward_mode=reward_mode,
                sim_backend='cpu',sensor_configs={'shader_pack':'default'},human_render_camera_configs={'shader_pack':'default'},
                viewer_camera_configs={'shader_pack':'default'})
            directory=args.output/'episodes'/task/str(initialization);directory.mkdir(parents=True)
            frames=[];calls=[];steps=[];done=False
            try:
                obs,_=env.reset(seed=seed)
                history=deque([None,None],maxlen=2)
                def capture():
                    rgb=np.ascontiguousarray(env.render().squeeze(0).cpu().numpy(),dtype=np.uint8)
                    history.append(rgb.copy())
                    path=directory/f'frame-{len(frames):04d}.png';Image.fromarray(rgb).save(path)
                    frames.append(dict(image_file=str(path.relative_to(args.output)),image_sha256=sha(path),
                        proprio=np.ascontiguousarray(canonical(obs['agent']['qpos'])[:,:-1].reshape(-1),dtype=np.float32).tolist(),
                        observation_sha256=digest(canonical(obs)),physics_sha256=digest(canonical(env.unwrapped.get_state_dict()))))
                capture()
                while len(steps)<400 and not done:
                    call=len(calls);noise_id=f'{task}/{initialization}/policy-call-{call}'
                    noise_seed=int.from_bytes(hashlib.sha256(noise_id.encode()).digest()[:8],'little')
                    target=dict(task_name=task,paired_request_id='target',history_valid=[v is not None for v in history],
                        image_history=np.stack([np.zeros_like(history[-1]) if v is None else v for v in history]),
                        proprio=np.asarray(frames[-1]['proprio'],dtype=np.float32),
                        noise=torch.randn(64,128,generator=torch.Generator().manual_seed(noise_seed)).numpy())
                    adapter.reset_model(policy)
                    with torch.inference_mode(),set_batch_invariant_mode(False):actions=backend.infer({'examples':[target]})['actions'][0]
                    if not np.isfinite(actions).all():raise RuntimeError('Nonfinite native action')
                    calls.append(dict(policy_call=call,frame_index=len(frames)-1,noise_id=noise_id,noise_seed=noise_seed,raw_action_chunk=actions.tolist()))
                    for action in actions[::4]:
                        obs,reward,terminated,truncated,info=env.step(action)
                        success=bool(info['success']);done=(bool(terminated) or bool(truncated)) and success
                        steps.append(dict(policy_call=call,action=action.tolist(),reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),success=success))
                        capture()
                        if done or len(steps)>=400:break
                episode=dict(task=task,initialization=initialization,seed=seed,reward_mode=reward_mode,success=done,frames=frames,calls=calls,steps=steps)
                raw=directory/'episode.json';raw.write_text(json.dumps(episode)+'\n')
                if len(frames)-1<10:raise RuntimeError('Fewer than ten pre-action observations; do not silently duplicate')
                selected=np.linspace(0,len(frames)-2,10,dtype=int).tolist()
                for frame_index in selected:
                    record=frames[frame_index];rgb=np.asarray(Image.open(args.output/record['image_file']).convert('RGB'))
                    previous=np.asarray(Image.open(args.output/frames[frame_index-1]['image_file']).convert('RGB')) if frame_index else np.zeros_like(rgb)
                    index=len(observations);noise=torch.stack([torch.randn(64,128,generator=torch.Generator().manual_seed(980000+3*index+j)) for j in range(3)]).numpy();noises.append(noise)
                    request=dict(task_name=task,paired_request_id=f'{task}/{initialization}/{frame_index}',history_valid=[frame_index>0,True],
                                 image_history=np.stack([previous,rgb]),proprio=np.asarray(record['proprio'],dtype=np.float32))
                    target_path=directory/f'replay-{frame_index:04d}.pt';torch.save(request,target_path)
                    observations.append(dict(observation_index=index,task=task,episode_id=initialization,frame_index=frame_index,
                        split='diagnostic' if initialization<4 else 'heldout',request_file=str(target_path.relative_to(args.output)),request_sha256=sha(target_path),
                        image_files=[frames[frame_index-1]['image_file'] if frame_index else None,record['image_file']],
                        source_episode_file=str(raw.relative_to(args.output)),source_episode_sha256=sha(raw),noise_sha256=[hashlib.sha256(v.tobytes()).hexdigest() for v in noise]))
                episodes.append(dict(task=task,initialization=initialization,source_episode_file=str(raw.relative_to(args.output)),sha256=sha(raw),success=done))
                print(f'COLLECTED {task}/{initialization}: success={done}, steps={len(steps)}, observations={len(observations)}',flush=True)
            finally:env.close()
    noise_path=args.output/'noise.npy';np.save(noise_path,np.stack(noises))
    manifest=dict(schema_version=1,status='collected_pending_independent_audit',model='rdt_maniskill',runtime_sha256=sha(args.output/'runtime.json'),
                  observations=observations,episodes=episodes,noise_file='noise.npy',noise_file_sha256=sha(noise_path),noise_shape=list(np.stack(noises).shape))
    (args.output/'replay_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
