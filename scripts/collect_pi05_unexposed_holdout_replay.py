"""Collect the prescribed 100-episode/1,000-observation PI05 LIBERO corpus.

Uses native official inference and fixed request-local noise. Failed episodes
remain in the corpus. This preparation run is not a paired success-rate study.
"""
from __future__ import annotations

import argparse
import gzip
import io
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch

from verify_pi05_libero_checkpoint import load_verified_policy
from qualify_simpler_fixed_action_replay import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization-offset",type=int,default=0)
    for name in ("checkpoint","tokenizer","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--reset-audit",type=Path,required=True)
    args=parser.parse_args()
    if not 0 <= args.initialization_offset <= 40:parser.error("ten starts must fit LIBERO's 50 initialization IDs")
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    from scripts.libero_deterministic_renderer import install
    install()
    reset=json.loads(args.reset_audit.read_text())
    if reset['status']!='complete_no_msaa_observable_replay_independently_verified' or not reset['all_observable_state_and_rgb_repeats_exact']:raise ValueError('Unqualified no-MSAA runtime')
    reset_runtime=json.loads((args.reset_audit.parent/'runtime.json').read_text())
    for file,expected in reset_runtime['source_sha256'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=expected:raise ValueError('Qualified source changed')
    from scripts.sim_state_evidence import observation_evidence
    from batch_invariant_ops import set_batch_invariant_mode
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep
    from lerobot.policies.factory import make_pre_post_processors

    class EpisodeEnv(LiberoEnv):
        def step(self,action):
            obs,reward,done,info=self._env.step(action);success=self._env.check_success()
            info.update(task=self.task,task_id=self.task_id,done=done,is_success=success)
            return self._format_raw_obs(obs),reward,done or success,False,info

    def capture(env,obs):
        evidence,raw=observation_evidence(env,obs,include_rng=True)
        raw=dict(raw,observation=obs,derived={name:np.array(getattr(env._env.sim.data,name),copy=True) for name in (
            'geom_xpos','geom_xmat','cam_xpos','cam_xmat','site_xpos','site_xmat','body_xpos','body_xmat',
            'light_xpos','light_xdir','qacc','qacc_warmstart','qfrc_constraint') if hasattr(env._env.sim.data,name)})
        identity=digest(raw);path=args.output/'frames'/(identity+'.pt.gz');path.parent.mkdir(exist_ok=True)
        if not path.exists():
            buffer=io.BytesIO();torch.save(raw,buffer);temporary=path.with_suffix('.tmp')
            temporary.write_bytes(gzip.compress(buffer.getvalue(),compresslevel=1,mtime=0));temporary.replace(path)
        return dict(evidence,derived_sha256=digest(raw['derived']),raw_frame=str(path.relative_to(args.output))),identity

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    policy,loading=load_verified_policy(args.checkpoint)
    policy.to("cuda").eval()
    pre,post=make_pre_post_processors(policy.config,pretrained_path=args.checkpoint,
        preprocessor_overrides={"tokenizer_processor":{"tokenizer_name":str(args.tokenizer)},
                                "device_processor":{"device":"cpu"}})
    suite=benchmark.get_benchmark_dict()["libero_10"]()
    runtime={"scope":"native_real_observation_corpus_collection","suite":"libero_10",
             "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
             "mujoco_egl_device_id":os.environ.get("MUJOCO_EGL_DEVICE_ID"),
             "tasks":list(range(10)),"initializations_per_task":10,"observations_per_episode":10,
             "diagnostic_initializations":list(range(args.initialization_offset,args.initialization_offset+2)),
             "heldout_initializations":list(range(args.initialization_offset+2,args.initialization_offset+10)),
             "observation_selection":"10 evenly spaced pre-action control frames1..steps-1, excluding reset and terminal observations; no success/error selection",
             "renderer":"qualified_zero_offscreen_samples",
             "reset_audit_sha256":hashlib.sha256(args.reset_audit.read_bytes()).hexdigest(),
             "reset_source_sha256":reset_runtime['source_sha256'],
             "heldout_acceptance_pending_prepared_input_disjointness":True,
             "rollout_noise_namespace":"fresh-no-msaa-004",
             "checkpoint_loading":loading,"action_steps":policy.config.n_action_steps,"max_steps":520,
             "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "paired_success_rate_campaign_complete":False}
    (args.output/"runtime.json").write_text(json.dumps(runtime,indent=2,default=str)+"\n")
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())

    def add_batch(value):
        if isinstance(value,np.ndarray):return value[None]
        if isinstance(value,dict):return {k:add_batch(v) for k,v in value.items()}
        return value

    def process(obs,instruction):
        batch=LiberoProcessorStep().observation(preprocess_observation(add_batch(obs)))
        batch["task"]=[instruction]
        return {k:v.cpu() for k,v in pre(batch).items() if isinstance(v,torch.Tensor)}

    observations=[]
    noises=np.empty((1000,3,50,32),dtype=np.float32)
    for task in range(10):
        for initialization in range(args.initialization_offset,args.initialization_offset+10):
            seed=20261108+initialization
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            env=EpisodeEnv(suite,task,"libero_10",obs_type="pixels_agent_pos",episode_index=initialization)
            try:
                env.init_state_id=initialization
                obs,_=env.reset(seed=seed);policy.reset()
                first_evidence,first_identity=capture(env,obs)
                env.close()
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                env=EpisodeEnv(suite,task,"libero_10",obs_type="pixels_agent_pos",episode_index=initialization)
                env.init_state_id=initialization;obs,_=env.reset(seed=seed);policy.reset()
                initial_evidence,initial_identity=capture(env,obs)
                if first_identity!=initial_identity:raise RuntimeError('Cold initial physics/controller/RNG/RGB/derived mismatch')
                raw_frames=[initial_evidence]
                # Copy retained arrays: simulator observation buffers may be reused.
                def copy_obs(value):
                    if isinstance(value,np.ndarray):return value.copy()
                    if isinstance(value,dict):return {k:copy_obs(v) for k,v in value.items()}
                    return value
                frames=[copy_obs(obs)]
                steps=[];calls=[];call=0;success=False
                while len(steps)<520 and not success:
                    inputs=process(obs,env.task_description)
                    key=f"fresh-no-msaa-004/libero_10/{task}/{initialization}/policy-call-{call}"
                    noise_seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"little")
                    noise=torch.randn(1,50,32,generator=torch.Generator().manual_seed(noise_seed))
                    with torch.inference_mode(),torch.random.fork_rng(devices=[torch.cuda.current_device()]),set_batch_invariant_mode(False):
                        normalized=policy.predict_action_chunk({k:v.to("cuda") for k,v in inputs.items()},noise=noise.to("cuda"))
                        actions=post(normalized.cpu()).cpu()[0]
                    if not torch.isfinite(actions).all():raise RuntimeError("nonfinite actions")
                    calls.append({"policy_call":call,"noise_id":key,"noise_sha256":digest(noise.numpy()),
                                  "input_sha256":digest({k:v.numpy() for k,v in inputs.items()}),
                                  "predicted_actions":actions.tolist()})
                    for action in actions[:policy.config.n_action_steps]:
                        obs,reward,terminated,truncated,info=env.step(action.numpy())
                        success=bool(info["is_success"])
                        steps.append({"policy_call":call,"action":action.tolist(),"reward":float(reward),
                                      "terminated":bool(terminated),"truncated":bool(truncated),"success":success})
                        frames.append(copy_obs(obs))
                        raw_frames.append(capture(env,obs)[0])
                        if terminated or truncated or len(steps)>=520:break
                    call+=1
                    if terminated or truncated:break
                if len(steps)<11:raise RuntimeError("episode has fewer than ten distinct observation times")
                indices=np.linspace(1,len(steps)-1,10,dtype=int).tolist()
                episode_dir=args.output/f"task-{task:02d}"/f"episode-{initialization:02d}"
                episode_dir.mkdir(parents=True)
                instruction=env.task_description
                for frame_index,step_index in enumerate(indices):
                    observation=frames[step_index]
                    index=len(observations)
                    inputs=process(observation,instruction)
                    input_path=episode_dir/f"observation-{frame_index:02d}.pt"
                    torch.save(inputs,input_path)
                    raw_images={}
                    for camera,image in observation["pixels"].items():
                        image_path=episode_dir/f"observation-{frame_index:02d}-{camera}.png"
                        Image.fromarray(image).save(image_path)
                        raw_images[camera]=str(image_path.relative_to(args.output))
                    request_ids=[];noise_records=[]
                    for noise_index in range(3):
                        request_id=f"fresh-no-msaa-004/libero_10/{task}/{initialization}/observation-{frame_index}/noise-{noise_index}"
                        noise_seed=int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:8],"little")
                        noises[index,noise_index]=torch.randn(50,32,generator=torch.Generator().manual_seed(noise_seed)).numpy()
                        request_ids.append(request_id)
                        noise_records.append({"sha256":hashlib.sha256(noises[index,noise_index].tobytes()).hexdigest()})
                    observations.append({"replay_index":index,"task":task,"episode_id":initialization,
                        "source_step":step_index,"instruction":instruction,"split":"diagnostic" if initialization<args.initialization_offset+2 else "heldout",
                        "input_file":str(input_path.relative_to(args.output)),
                        "input_file_sha256":hashlib.sha256(input_path.read_bytes()).hexdigest(),
                        "images":raw_images,"robot_state":{k:{j:np.asarray(v).tolist() for j,v in vals.items()} for k,vals in observation["robot_state"].items()},
                        "observation_sha256":digest(observation),"request_ids":request_ids,"noise":noise_records})
                episode={"task":task,"initialization":initialization,"seed":seed,"success":success,
                         "selected_step_indices":indices,"steps":steps,"calls":calls,"raw_frames":raw_frames,
                         "cold_initial_raw_identity":initial_identity,"cold_initial_repeat_raw_identity":first_identity}
                (episode_dir/"episode.json").write_text(json.dumps(episode)+"\n")
                # A completed episode is retained immediately; interrupted collection
                # remains explicitly partial and never receives a final manifest.
                with (args.output/"observations.partial.jsonl").open("a") as stream:
                    for row in observations[-10:]:stream.write(json.dumps(row)+"\n")
                print(f"COLLECTED task={task} initialization={initialization} steps={len(steps)} success={success} observations={len(observations)}",flush=True)
            finally:env.close()
    if len(observations)!=1000:raise RuntimeError("replay cardinality mismatch")
    noise_path=args.output/"noise.npy"
    np.save(noise_path,noises)
    manifest={"schema_version":1,"model_id":"pi05_libero","scope":runtime["scope"],
              "initialization_ids":list(range(args.initialization_offset,args.initialization_offset+10)),
              "diagnostic_initialization_ids":runtime["diagnostic_initializations"],
              "fixture_root":str(args.output.resolve()),"noise_file":str(noise_path.resolve()),
              "noise_shape":list(noises.shape),"noise_sha256":hashlib.sha256(noise_path.read_bytes()).hexdigest(),
              "observations":observations,"checkpoint_revision":"7152f13e34c6c115d808aea0e5d08aca74eac14b",
              "numerical_freeze":None,"paired_success_rate_campaign_complete":False}
    (args.output/"replay_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("COMPLETED 100 episodes / 1000 observations; numerical qualification pending",flush=True)


if __name__=="__main__":main()
