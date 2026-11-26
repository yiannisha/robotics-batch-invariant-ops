"""RDT complete-episode A/A and batch-variation pilot on fresh ManiSkill scenes."""
from __future__ import annotations

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

from qualify_simpler_fixed_action_replay import digest
from qualify_rdt_maniskill_replay import canonical


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","siglip","source","companions","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--tasks",nargs="+",default=["PickCube-v1"])
    parser.add_argument("--initializations",type=int,default=1)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    os.environ.update({"RDT_CHECKPOINT":str(args.checkpoint),"RDT_SIGLIP":str(args.siglip),"RDT_SOURCE":str(args.source)})
    from batch_invariant_ops import set_batch_invariant_mode
    from scripts.model_invariance.adapters.rdt_maniskill import RDTManiSkillAdapter
    from scripts.run_rdt_paired_server import PairedRDTBackend,verify_artifacts
    import gymnasium as gym
    import mani_skill.envs

    class RealAdapter(RDTManiSkillAdapter):
        def _validate_paths(self):
            for path in (self.checkpoint/"rdt/mp_rank_00_model_states.pt",self.siglip/"model.safetensors",self.source/"configs/base.yaml"):
                if not path.is_file():raise FileNotFoundError(path)

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    artifacts=verify_artifacts(args.checkpoint,args.siglip)
    adapter=RealAdapter();policy=adapter.load_model()
    backends={invariant:PairedRDTBackend(adapter,policy,condition="patched_dynamic" if invariant else "native_dynamic",
        checkpoint=args.checkpoint,language=adapter.language_embeddings,artifacts=artifacts,fixed_language_tokens=30,maximum_batch=8)
        for invariant in (False,True)}
    released=PairedRDTBackend(adapter,policy,condition="native_dynamic",checkpoint=args.checkpoint,
        language=adapter.language_embeddings,artifacts=artifacts)
    companions=torch.load(args.companions,map_location="cpu",weights_only=False)
    conditions={"native_singleton":(False,False),"native_dynamic":(False,True),
                "patched_singleton":(True,False),"patched_dynamic":(True,True)}
    conditions.update({name+"_repeat":value for name,value in list(conditions.items())})
    runtime={"scope":"complete_episode_pilot","model":"rdt_1b","tasks":args.tasks,"initializations":args.initializations,
             "conditions":conditions,"dynamic_cycle":[1,2,4,8],"language_extent":30,"primary_language_padding":"fixed_right_in_both_modes",
             "max_steps":400,"model_control_frequency":25,"executed_action_stride":4,
             "termination":"released evaluator: stop on terminal/truncated success or 400 executed controls",
             "reset_protocol":"fresh_scene_per_condition_explicit_indexed_seed",
             "artifact_files":artifacts,"runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "backend_sha256":hashlib.sha256(Path(__import__('scripts.run_rdt_paired_server',fromlist=['x']).__file__).read_bytes()).hexdigest(),
             "paired_success_rate_campaign_complete":False}
    (args.output/"runtime.json").write_text(json.dumps(runtime,indent=2)+"\n")
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    with set_batch_invariant_mode(True):pass

    def td(value):
        return {"dtype":str(value.dtype),"shape":list(value.shape),
                "sha256":hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}

    def image(env):return np.ascontiguousarray(env.render().squeeze(0).cpu().numpy(),dtype=np.uint8)
    def state_record(env,obs,rgb):
        controller=env.unwrapped.agent.controller
        controller_state=controller.get_state() if hasattr(controller,"get_state") else None
        return {"observation_sha256":digest(canonical(obs)),"policy_rgb_sha256":digest(rgb),
                "physics_state_sha256":digest(canonical(env.unwrapped.get_state_dict())),
                "controller_state_sha256":digest(canonical(controller_state)),
                "qpos":canonical(obs["agent"]["qpos"]).tolist()}

    blocks=[]
    for task in args.tasks:
        for initialization in range(args.initializations):
            episodes={}
            for name,(invariant,dynamic) in conditions.items():
                seed=20241201+initialization
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                reward_mode="sparse" if task=="PlugCharger-v1" else "dense"
                env=gym.make(task,obs_mode="rgb",control_mode="pd_joint_pos",render_mode="rgb_array",reward_mode=reward_mode,
                    sim_backend="cpu",sensor_configs={"shader_pack":"default"},human_render_camera_configs={"shader_pack":"default"},
                    viewer_camera_configs={"shader_pack":"default"})
                try:
                    obs,_=env.reset(seed=seed);rgb=image(env)
                    history=deque([None,rgb.copy()],maxlen=2)
                    trajectory=[state_record(env,obs,rgb)];calls=[];steps=[];done=False;fidelity=None
                    while len(steps)<400 and not done:
                        call=len(calls)
                        key=f"{task}/{initialization}/policy-call-{call}"
                        noise_seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"little")
                        noise=torch.randn(64,128,generator=torch.Generator().manual_seed(noise_seed)).numpy()
                        target={"task_name":task,"paired_request_id":"target","history_valid":[value is not None for value in history],
                                "image_history":np.stack([np.zeros_like(rgb) if value is None else value for value in history]),
                                "proprio":np.ascontiguousarray(canonical(obs["agent"]["qpos"])[:,:-1].reshape(-1),dtype=np.float32),"noise":noise}
                        size=[1,2,4,8][call%4] if dynamic else 1;position=call%size
                        indices=[(initialization*97+call*11+j)%len(companions) for j in range(size-1)]
                        examples=[dict(companions[index],paired_request_id=f"companion-{j}") for j,index in enumerate(indices)]
                        examples.insert(position,target)
                        for history_index,picture in enumerate(history):
                            if picture is not None:Image.fromarray(picture).save(args.output/f"{task}-{initialization}-{name}-call-{call:03d}-history-{history_index}.png")
                        inputs={"history_valid":target["history_valid"],"control_frequency":25,
                                "language_mask":[index<adapter.language_embeddings[task].shape[1] for index in range(30)]}
                        def vision_input(module,values):inputs["vision_pixels"]=td(values[0][position*6:(position+1)*6])
                        def trace(label,tensor):
                            if label in ("input.language","input.state","input.noise"):
                                inputs[label]=td(tensor[position:position+1])
                        hook=policy.vision.register_forward_pre_hook(vision_input)
                        adapter.trace_callback=trace;adapter.reset_model(policy)
                        try:
                            with torch.inference_mode(),set_batch_invariant_mode(invariant):
                                actions=backends[invariant].infer({"examples":examples})["actions"][position]
                        finally:hook.remove();adapter.trace_callback=None
                        if not np.isfinite(actions).all():raise RuntimeError("nonfinite RDT action")
                        if name=="native_singleton" and call==0:
                            adapter.reset_model(policy)
                            with torch.inference_mode(),set_batch_invariant_mode(False):
                                original=released.infer({"examples":[target]})["actions"][0]
                            fidelity={"comparison":"fixed30_native_singleton_vs_released_longest_padding_singleton",
                                      "exact":actions.tobytes()==original.tobytes(),"max_error":float(np.max(np.abs(actions-original)))}
                        calls.append({"policy_call":call,"batch_size":size,"position":position,"companion_indices":indices,
                            "noise_id":key,"target_input_sha256":digest(inputs),"model_inputs":inputs,
                            "raw_action_chunk":actions.tolist()})
                        for action in actions[::4]:
                            obs,reward,terminated,truncated,info=env.step(action)
                            rgb=image(env);history.append(rgb.copy())
                            step_success=bool(info["success"])
                            done=(bool(terminated) or bool(truncated)) and step_success
                            steps.append({"policy_call":call,"action":action.tolist(),"reward":float(reward),
                                "terminated":bool(terminated),"truncated":bool(truncated),"success":step_success})
                            trajectory.append(state_record(env,obs,rgb))
                            if done or len(steps)>=400:break
                    record={"task":task,"initialization":initialization,"condition":name,"seed":seed,"reward_mode":reward_mode,
                            "success":bool(done),"calls":calls,"steps":steps,"trajectory":trajectory,"singleton_padding_fidelity":fidelity}
                    with (args.output/"episodes.jsonl").open("a") as stream:stream.write(json.dumps(record)+"\n")
                    episodes[name]=record
                    print(f"FINISHED {task}/{initialization}/{name}: success={done}, steps={len(steps)}",flush=True)
                finally:env.close()
            if len({digest(row["trajectory"][0]) for row in episodes.values()})!=1 or len({row["calls"][0]["target_input_sha256"] for row in episodes.values()})!=1:
                raise RuntimeError("initial state/model input differs")
            for name in list(conditions)[:4]:
                for field in ("calls","steps","trajectory","success"):
                    if episodes[name][field]!=episodes[name+"_repeat"][field]:raise RuntimeError(f"A/A differs: {name}/{field}")
            for field in ("steps","trajectory","success"):
                if episodes["patched_singleton"][field]!=episodes["patched_dynamic"][field]:raise RuntimeError(f"invariant batching changes {field}")
            left,right=episodes["native_singleton"],episodes["native_dynamic"]
            first=next((i for i,(a,b) in enumerate(zip(left["steps"],right["steps"])) if a["action"]!=b["action"]),None)
            if first is not None:
                a=left["calls"][left["steps"][first]["policy_call"]];b=right["calls"][right["steps"][first]["policy_call"]]
                if a["target_input_sha256"]!=b["target_input_sha256"] or left["trajectory"][:first+1]!=right["trajectory"][:first+1]:raise RuntimeError("native first-action attribution failed")
            blocks.append({"task":task,"initialization":initialization,"all_AA_exact":True,"invariant_batching_exact":True,
                "native_first_action_divergence":first,"success":{name:row["success"] for name,row in episodes.items()},
                "singleton_padding_fidelity":episodes["native_singleton"]["singleton_padding_fidelity"]})
            print(f"QUALIFIED block {task}/{initialization}",flush=True)
    (args.output/"summary.json").write_text(json.dumps({"scope":runtime["scope"],"blocks":blocks,
        "paired_success_rate_campaign_complete":False},indent=2)+"\n")


if __name__=="__main__":main()
