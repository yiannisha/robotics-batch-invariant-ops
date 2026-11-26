"""Replay retained controls and compare raw observations/derived MuJoCo state.

Diagnostic evidence only: this does not relax the study's exact-repeat gate.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import random

import numpy as np
import torch

from scripts.qualify_simpler_fixed_action_replay import digest


def differences(a,b,path='observation'):
    result=[]
    if isinstance(a,dict):
        for key in a:
            result.extend(differences(a[key],b[key],path+'.'+str(key)))
    elif isinstance(a,np.ndarray):
        if a.dtype!=b.dtype or a.shape!=b.shape:return [dict(field=path,shape_changed=True)]
        mask=a!=b
        if mask.any():
            indices=np.argwhere(mask)
            result.append(dict(field=path,dtype=str(a.dtype),shape=list(a.shape),changed_elements=int(mask.sum()),
                max_absolute_error=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))),
                first_indices=indices[:8].tolist()))
    elif a!=b:result.append(dict(field=path,left=a,right=b))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--forward-before-observe',action='store_true')
    p.add_argument('--disable-dither',action='store_true');p.add_argument('--make-current-before-render',action='store_true')
    p.add_argument('--offscreen-samples',type=int);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv
    from robosuite.utils.binding_utils import MjRenderContext
    original_context=MjRenderContext._set_mujoco_context_and_buffers
    def context(self):
        if args.offscreen_samples is not None:self.model.vis.quality.offsamples=args.offscreen_samples
        original_context(self)
    MjRenderContext._set_mujoco_context_and_buffers=context
    if args.disable_dither or args.make_current_before_render:
        original_render=MjRenderContext.render
        def render(self,*positional,**kwargs):
            self.gl_ctx.make_current()
            if args.disable_dither:
                from OpenGL import GL
                GL.glDisable(GL.GL_DITHER)
            return original_render(self,*positional,**kwargs)
        MjRenderContext.render=render
    class EpisodeEnv(LiberoEnv):
        def step(self,action):
            obs,reward,done,info=self._env.step(action)
            if args.forward_before_observe:
                self._env.sim.forward();obs=self._env._get_observations(force_update=True)
            success=self._env.check_success();terminated=done or success
            info.update(task=self.task,task_id=self.task_id,done=done,is_success=success)
            return self._format_raw_obs(obs),reward,terminated,False,info
    with gzip.open(args.episode,'rt') as f:episode=json.load(f)
    task,initial=episode['task'],episode['initialization'];suite=benchmark.get_benchmark_dict()['libero_10']()
    torch.set_num_threads(8);torch.cuda.init()
    warm=EpisodeEnv(suite,0,'libero_10',obs_type='pixels_agent_pos',episode_index=0)
    try:warm.init_state_id=0;warm.reset(seed=20261108)
    finally:warm.close()
    runtime=dict(scope='fixed_controls_render_and_derived_state_diagnostic',task=task,initialization=initial,repeats=args.repeats,
        forward_before_observe=args.forward_before_observe,source_episode_sha256=hashlib.sha256(args.episode.read_bytes()).hexdigest(),
        disable_dither=args.disable_dither,make_current_before_render=args.make_current_before_render,offscreen_samples=args.offscreen_samples,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_evaluated=False)
    (args.output/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n');(args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    reference=None;reports=[]
    for repeat in range(args.repeats):
        seed=20261108+initial;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        env=EpisodeEnv(suite,task,'libero_10',obs_type='pixels_agent_pos',episode_index=initial)
        try:
            env.init_state_id=initial;obs,_=env.reset(seed=seed);frames=[];raws=[]
            def capture():
                sim=env._env.sim
                derived={name:np.array(getattr(sim.data,name),copy=True) for name in (
                    'geom_xpos','geom_xmat','cam_xpos','cam_xmat','site_xpos','site_xmat','body_xpos','body_xmat',
                    'light_xpos','light_xdir','qacc','qacc_warmstart','qfrc_constraint') if hasattr(sim.data,name)}
                raw=dict(observation=obs,physics=np.array(sim.get_state().flatten(),copy=True),derived=derived)
                identity=digest(raw);path=args.output/'frames'/(identity+'.pt.gz');path.parent.mkdir(exist_ok=True)
                if not path.exists():
                    buffer=io.BytesIO();torch.save(raw,buffer);path.write_bytes(gzip.compress(buffer.getvalue(),compresslevel=1,mtime=0))
                frames.append(dict(observation_sha256=digest(obs),physics_sha256=digest(raw['physics']),derived_sha256=digest(derived),raw_file=path.name))
                raws.append(raw)
            capture()
            for step in episode['steps']:
                obs,_,_,_,_=env.step(np.asarray(step['action'],dtype=np.float32));capture()
            first=None;detail=[]
            if reference is not None:
                first=next((i for i,(a,b) in enumerate(zip(reference,raws)) if digest(a)!=digest(b)),None)
                if first is not None:detail=differences(reference[first],raws[first],'raw')
            else:reference=raws
            original_obs_mismatches=[i for i,(frame,old) in enumerate(zip(frames,episode['trajectory'])) if frame['observation_sha256']!=old['observation_sha256']]
            original_physics_mismatches=[i for i,(raw,old) in enumerate(zip(raws,episode['trajectory'])) if digest(raw['physics'])!=old['sim_state_sha256']]
            row=dict(repeat=repeat,frames=frames,first_repeat_difference=first,difference_detail=detail,
                original_observation_mismatches=original_obs_mismatches,original_physics_mismatches=original_physics_mismatches)
            with (args.output/'replays.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            reports.append({k:v for k,v in row.items() if k!='frames'})
            print(json.dumps(reports[-1]),flush=True)
        finally:env.close()
    (args.output/'summary.json').write_text(json.dumps(dict(runtime=runtime,repeats=reports,all_raw_repeats_exact=all(row['first_repeat_difference'] is None for row in reports)),indent=2)+'\n')


if __name__=='__main__':main()
