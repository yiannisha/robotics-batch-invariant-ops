"""Qualify all ten tasks and retained failing controls under a pinned renderer."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import random

import numpy as np
import torch

from scripts.libero_deterministic_renderer import install
from scripts.qualify_simpler_fixed_action_replay import digest
from scripts.sim_state_evidence import observation_evidence


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-episodes',nargs='+',type=Path,required=True)
    p.add_argument('--tasks',nargs='+',type=int,default=list(range(10)))
    p.add_argument('--initializations',type=int,default=3);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--regression-repeats',type=int,default=5);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);install()
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv
    class EpisodeEnv(LiberoEnv):
        def step(self,action):
            obs,reward,done,info=self._env.step(action);success=self._env.check_success();terminated=done or success
            info.update(task=self.task,task_id=self.task_id,done=done,is_success=success)
            return self._format_raw_obs(obs),reward,terminated,False,info
    suite=benchmark.get_benchmark_dict()['libero_10']();torch.set_num_threads(8);torch.cuda.init()
    paths=[Path(__file__).resolve(),Path(__file__).with_name('libero_deterministic_renderer.py').resolve(),Path(__file__).with_name('sim_state_evidence.py').resolve(),Path(__file__).with_name('qualify_simpler_fixed_action_replay.py').resolve()]
    runtime=dict(scope='no_MSAA_complete_observable_fixed_action_replay_qualification',tasks=args.tasks,initializations=args.initializations,
        repeats=args.repeats,regression_repeats=args.regression_repeats,offscreen_samples=0,max_steps=520,
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
        renderer_protocol_changed=True,default_renderer_study_results_must_not_be_pooled=True,model_evaluated=False)
    for file in paths:(args.output/file.name).write_bytes(file.read_bytes())
    (args.output/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    blocks=[]
    def run(task,initial,repeat,kind,actions,original=None):
        seed=20261108+initial;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        env=EpisodeEnv(suite,task,'libero_10',obs_type='pixels_agent_pos',episode_index=initial)
        try:
            env.init_state_id=initial;obs,_=env.reset(seed=seed);trajectory=[];steps=[]
            def retain():
                evidence,raw=observation_evidence(env,obs,include_rng=True)
                raw=dict(raw,observation=obs,derived={name:np.array(getattr(env._env.sim.data,name),copy=True) for name in (
                    'geom_xpos','geom_xmat','cam_xpos','cam_xmat','site_xpos','site_xmat','body_xpos','body_xmat',
                    'light_xpos','light_xdir','qacc','qacc_warmstart','qfrc_constraint') if hasattr(env._env.sim.data,name)})
                identity=digest(raw);path=args.output/'frames'/(identity+'.pt.gz');path.parent.mkdir(exist_ok=True)
                if not path.exists():
                    buffer=io.BytesIO();torch.save(raw,buffer);path.write_bytes(gzip.compress(buffer.getvalue(),compresslevel=1,mtime=0))
                trajectory.append(dict(evidence,derived_sha256=digest(raw['derived']),raw_frame=path.name))
            retain()
            for index,action in enumerate(actions):
                obs,reward,terminated,truncated,info=env.step(action);retain()
                steps.append(dict(action=action.tolist(),reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),success=bool(info['is_success'])))
                if terminated or truncated:break
            row=dict(kind=kind,task=task,initialization=initial,repeat=repeat,trajectory=trajectory,steps=steps)
            if original is not None:
                row['source_episode_sha256']=original['sha256']
                old=original['episode']['trajectory']
                row['original_physics_exact']=len(trajectory)==len(old) and all(frame['sim_state_sha256']==expected['sim_state_sha256'] for frame,expected in zip(trajectory,old))
                if not row['original_physics_exact']:raise RuntimeError('Renderer protocol changed retained-control physics')
            with (args.output/'replays.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
            print(f'REPLAY {kind}/{task}/{initial}/{repeat}: {len(steps)} controls',flush=True)
            return digest(dict(trajectory=trajectory,steps=steps))
        finally:env.close()
    for task in args.tasks:
        for initial in range(args.initializations):
            reference=None
            for repeat in range(args.repeats):
                actions=[np.array([0.01*np.sin(i/7),0.01*np.cos(i/9),0.01*np.sin(i/11),0.,0.,0.,-1. if (i//20)%2 else 1.],dtype=np.float32) for i in range(520)]
                value=run(task,initial,repeat,'fixed_wave',actions)
                if reference is not None and reference!=value:raise RuntimeError(f'Exact fixed replay failed: {task}/{initial}/{repeat}')
                reference=value
            blocks.append(dict(kind='fixed_wave',task=task,initialization=initial,repeats=args.repeats,exact=True))
    for path in args.source_episodes:
        with gzip.open(path,'rt') as f:episode=json.load(f)
        original=dict(episode=episode,sha256=hashlib.sha256(path.read_bytes()).hexdigest());reference=None
        for repeat in range(args.regression_repeats):
            value=run(episode['task'],episode['initialization'],repeat,'retained_failing_controls',
                [np.asarray(step['action'],dtype=np.float32) for step in episode['steps']],original)
            if reference is not None and reference!=value:raise RuntimeError('Retained failing controls do not reproduce exactly')
            reference=value
        blocks.append(dict(kind='retained_failing_controls',task=episode['task'],initialization=episode['initialization'],repeats=args.regression_repeats,exact=True,source_episode_sha256=original['sha256']))
    (args.output/'summary.json').write_text(json.dumps(dict(blocks=blocks,all_prescribed_replays_exact=True,independent_audit_pending=True,model_evaluated=False),indent=2)+'\n')


if __name__=='__main__':main()
