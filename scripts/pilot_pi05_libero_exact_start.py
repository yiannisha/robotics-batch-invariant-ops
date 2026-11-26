"""Eight-condition exact-start complete-episode PI05 LIBERO pilot.

Pilot evidence does not satisfy numerical freeze or full campaign gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from verify_pi05_libero_checkpoint import load_verified_policy
from qualify_simpler_fixed_action_replay import digest
from probe_pi05_libero_real_inputs import tensor_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "tokenizer", "companions", "output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--tasks",nargs="+",type=int,default=[0])
    parser.add_argument("--initializations",type=int,default=1)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    from batch_invariant_ops import set_batch_invariant_mode
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep
    from lerobot.policies.factory import make_pre_post_processors

    class EpisodeEnv(LiberoEnv):
        def step(self,action):
            # Preserve the released terminal observation and success semantics,
            # but defer its unused auto-reset to the next fresh-scene episode.
            obs,reward,done,info=self._env.step(action)
            success=self._env.check_success()
            terminated=done or success
            info.update({"task":self.task,"task_id":self.task_id,"done":done,"is_success":success})
            return self._format_raw_obs(obs),reward,terminated,False,info

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    policy,loading=load_verified_policy(args.checkpoint)
    policy.to("cuda").eval()
    preprocess,postprocess=make_pre_post_processors(policy.config,pretrained_path=args.checkpoint,
        preprocessor_overrides={"tokenizer_processor":{"tokenizer_name":str(args.tokenizer)},
                                "device_processor":{"device":"cpu"}})
    companions=torch.load(args.companions,map_location="cpu",weights_only=True)
    suite=benchmark.get_benchmark_dict()["libero_10"]()
    conditions={"native_singleton":(False,False),"native_dynamic":(False,True),
                "patched_singleton":(True,False),"patched_dynamic":(True,True)}
    conditions.update({key+"_repeat":value for key,value in list(conditions.items())})
    with set_batch_invariant_mode(True):pass

    def batch_raw(value):
        if isinstance(value,np.ndarray):return value[None]
        if isinstance(value,dict):return {key:batch_raw(item) for key,item in value.items()}
        return value

    def process(obs,instruction):
        batch=LiberoProcessorStep().observation(preprocess_observation(batch_raw(obs)))
        batch["task"]=[instruction]
        return {key:val for key,val in preprocess(batch).items() if isinstance(val,torch.Tensor)}

    def observation_record(env,obs):
        return {"observation_sha256":digest(obs),
                "sim_state_sha256":digest(env._env.sim.get_state().flatten()),
                "eef":np.asarray(obs["robot_state"]["eef"]["pos"]).tolist(),
                "gripper":np.asarray(obs["robot_state"]["gripper"]["qpos"]).tolist()}

    runtime={"scope":"complete_episode_pilot","suite":"libero_10","tasks":args.tasks,
             "initializations":args.initializations,"conditions":conditions,
             "checkpoint_loading":loading,"action_steps":policy.config.n_action_steps,
             "max_steps":520,"dynamic_cycle":[1,2,4,8],
             "noise_scheme":"sha256-indexed-task-initialization-policy-call",
             "reset_protocol":"fresh_scene_explicit_initialization_id",
             "terminal_auto_reset":"deferred_until_next_episode",
             "companion_sha256":hashlib.sha256(args.companions.read_bytes()).hexdigest(),
             "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "success_rate_campaign_complete":False}
    (args.output/"runtime.json").write_text(json.dumps(runtime,indent=2,default=str)+"\n")
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    blocks=[]
    for task in args.tasks:
        for initialization in range(args.initializations):
            episodes={}
            for condition,(invariant,dynamic) in conditions.items():
                seed=20261108+initialization
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                env=EpisodeEnv(suite,task,"libero_10",obs_type="pixels_agent_pos",episode_index=initialization)
                try:
                    env.init_state_id=initialization
                    obs,_=env.reset(seed=seed)
                    policy.reset()
                    trajectory=[observation_record(env,obs)]
                    calls=[];steps=[];success=False;call=0
                    while len(steps)<520 and not success:
                        inputs=process(obs,env.task_description)
                        key=f"libero_10/{task}/{initialization}/policy-call-{call}"
                        noise_seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"little")
                        noise=torch.randn(1,policy.config.chunk_size,policy.config.max_action_dim,
                                          generator=torch.Generator().manual_seed(noise_seed))
                        input_sha=digest({name:value.numpy() for name,value in inputs.items()}|{"noise":noise.numpy()})
                        size=(1,2,4,8)[call%4] if dynamic else 1
                        position=call%size
                        samples=[companions[(task+call+j)%len(companions)] for j in range(size-1)]
                        samples.insert(position,{"inputs":inputs,"noise":noise})
                        batch={name:torch.cat([sample["inputs"][name] for sample in samples]).to("cuda") for name in inputs}
                        noises=torch.cat([sample["noise"] for sample in samples]).to("cuda")
                        with torch.inference_mode(),set_batch_invariant_mode(invariant):
                            normalized=policy.predict_action_chunk(batch,noise=noises.clone())[position:position+1].cpu()
                            actions=postprocess(normalized).cpu()[0]
                        if not torch.isfinite(actions).all():raise RuntimeError("nonfinite actions")
                        calls.append({"call":call,"batch_size":size,"position":position,"noise_id":key,
                                      "target_input_sha256":input_sha,"normalized_sha256":tensor_digest(normalized),
                                      "normalized_actions":normalized.tolist(),"actions":actions.tolist()})
                        for action in actions[:policy.config.n_action_steps]:
                            action=action.numpy()
                            obs,reward,terminated,truncated,info=env.step(action)
                            success=bool(info["is_success"])
                            steps.append({"policy_call":call,"action":action.tolist(),"reward":float(reward),
                                          "terminated":bool(terminated),"truncated":bool(truncated),"success":success})
                            trajectory.append(observation_record(env,obs))
                            if terminated or truncated or len(steps)>=520:break
                        call+=1
                        if terminated or truncated:break
                    record={"task":task,"initialization":initialization,"condition":condition,"calls":calls,
                            "steps":steps,"trajectory":trajectory,"success":success}
                    with (args.output/"episodes.jsonl").open("a") as stream:stream.write(json.dumps(record)+"\n")
                    episodes[condition]=record
                    print(f"FINISHED task={task} initialization={initialization} {condition}: success={success}, steps={len(steps)}",flush=True)
                finally:env.close()
            initial={digest(item["trajectory"][0]) for item in episodes.values()}
            first_inputs={item["calls"][0]["target_input_sha256"] for item in episodes.values()}
            if len(initial)!=1 or len(first_inputs)!=1:raise RuntimeError("initial state/input mismatch")
            for name in list(conditions)[:4]:
                for field in ("calls","steps","trajectory","success"):
                    if episodes[name][field]!=episodes[name+"_repeat"][field]:raise RuntimeError(f"A/A mismatch: {name}/{field}")
            for field in ("steps","trajectory","success"):
                if episodes["patched_singleton"][field]!=episodes["patched_dynamic"][field]:raise RuntimeError(f"invariant singleton/dynamic mismatch: {field}")
            left,right=episodes["native_singleton"],episodes["native_dynamic"]
            mismatch=next((i for i,(a,b) in enumerate(zip(left["steps"],right["steps"])) if a["action"]!=b["action"]),None)
            attribution=None
            if mismatch is not None:
                a=left["calls"][left["steps"][mismatch]["policy_call"]]
                b=right["calls"][right["steps"][mismatch]["policy_call"]]
                attribution={"step":mismatch,"same_target_input":a["target_input_sha256"]==b["target_input_sha256"],
                             "same_trajectory_prefix":left["trajectory"][:mismatch+1]==right["trajectory"][:mismatch+1],
                             "dynamic_batch_size":b["batch_size"]}
                if not attribution["same_target_input"] or not attribution["same_trajectory_prefix"]:raise RuntimeError("native first-action attribution failed")
            blocks.append({"task":task,"initialization":initialization,"initial_state_and_input_exact":True,
                           "all_four_AA_repeats_exact":True,"invariant_singleton_dynamic_exact":True,
                           "native_first_action_divergence":attribution,
                           "success":{key:val["success"] for key,val in episodes.items()}})
            print(f"QUALIFIED block {task}/{initialization}",flush=True)
    (args.output/"summary.json").write_text(json.dumps({"scope":"complete_episode_pilot","blocks":blocks,
        "success_rate_campaign_complete":False},indent=2)+"\n")


if __name__=="__main__":main()
