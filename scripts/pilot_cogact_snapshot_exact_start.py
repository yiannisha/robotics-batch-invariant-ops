"""Complete-episode CogACT exact-snapshot pilot with fixed target input extent.

Native and invariant operators both consume fixed-64-token inputs. Fidelity to
the released longest-padding singleton is recorded separately. No full campaign
or success-rate completion is claimed by this pilot.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from qualify_simpler_fixed_action_replay import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","source","openvla-source","llama-metadata","pizero-project",
                 "simpler-root","simpler-python","vulkan-icd","companions","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--tasks",nargs="+",default=["pick_can"])
    parser.add_argument("--initializations",type=int,default=1)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    os.environ.update({"COGACT_CHECKPOINT":str(args.checkpoint),"COGACT_SOURCE":str(args.source),
        "COGACT_OPENVLA_SOURCE":str(args.openvla_source),"COGACT_LLAMA_METADATA":str(args.llama_metadata)})
    from batch_invariant_ops import set_batch_invariant_mode
    from scripts.model_invariance.adapters import cogact_small as implementation
    from scripts.run_cogact_paired_server import RequestState

    class SnapshotAdapter(implementation.CogACTSmallAdapter):
        def _validate_paths(self):
            for path in (self.checkpoint/"checkpoints/CogACT-Small.pt",self.checkpoint/"config.json",
                         self.source/"vla/cogactvla.py",self.openvla_source/"prismatic/models/vlms/prismatic.py",
                         self.llama_metadata/"config.json",self.llama_metadata/"tokenizer.model"):
                if not path.is_file():raise FileNotFoundError(path)
            return self.checkpoint/"checkpoints/CogACT-Small.pt"

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    adapter=SnapshotAdapter()
    policy=adapter.load_model()
    from sim_cogact.adaptive_ensemble import AdaptiveEnsembler
    sys.path.insert(0,str(args.pizero_project))
    from experiments.simpler_support import make_simpler_env,observation_state
    tree=ast.parse((args.pizero_project/"experiments/prepare_replay.py").read_text())
    tasks=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
               and any(isinstance(target,ast.Name) and target.id=="TASKS" for target in node.targets))
    implementation.CANONICAL_LANGUAGE_TOKENS=64
    companions=torch.load(args.companions,map_location="cpu",weights_only=False)
    conditions={"native_singleton":(False,False),"native_dynamic":(False,True),
                "patched_singleton":(True,False),"patched_dynamic":(True,True)}
    conditions.update({name+"_repeat":value for name,value in list(conditions.items())})
    with set_batch_invariant_mode(True):pass
    runtime={"scope":"complete_episode_pilot","model":"cogact_small","tasks":args.tasks,
             "initializations":args.initializations,"conditions":conditions,"max_steps":200,
             "language_padding":"fixed_right_64_in_both_primary_operator_modes",
             "native_label":"controlled native operators with fixed input extent; not unmodified longest-padding serving",
             "noise_scheme":"sha256-task-initialization-policy-call","dynamic_cycle":[1,2,4,8],
             "controller":"released_adaptive_ensemble_horizon2_alpha0.1_and_google_sticky_gripper",
             "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "companion_sha256":hashlib.sha256(args.companions.read_bytes()).hexdigest(),
             "source_sha256":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                 Path(implementation.__file__),Path(sys.modules[RequestState.__module__].__file__),
                 args.pizero_project/"experiments/simpler_worker.py",args.pizero_project/"experiments/simpler_support.py")},
             "paired_success_rate_campaign_complete":False}
    (args.output/"runtime.json").write_text(json.dumps(runtime,indent=2)+"\n")
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())

    def td(value):
        return {"dtype":str(value.dtype),"shape":list(value.shape),
                "sha256":hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}

    blocks=[]
    for task in args.tasks:
        env=make_simpler_env(tasks[task],args.simpler_python,args.simpler_root,args.vulkan_icd)
        try:
            for initialization in range(args.initializations):
                env.reset(seed=20261108+initialization,options={"obj_init_options":{"episode_id":initialization}})
                snapshot=env.capture_snapshot()
                episodes={}
                for condition,(invariant,dynamic) in conditions.items():
                    restored=env.restore_snapshot()
                    obs=restored["observation"]
                    controller=RequestState("google_robot",AdaptiveEnsembler)
                    trajectory=[{"observation_sha256":digest(obs),"state":observation_state(obs)}]
                    calls=[];success=False;truncated=False;fidelity=None
                    while not truncated:
                        call=len(calls)
                        if call>=200:raise RuntimeError("simulator did not reach its declared horizon")
                        instruction=env.get_language_instruction()
                        image=next(iter(obs["image"].values()))["rgb"]
                        image_path=args.output/f"{task}-{initialization}-{condition}-call-{call:03d}.png"
                        Image.fromarray(image).save(image_path)
                        noise_id=f"{task}/{initialization}/policy-call-{call}"
                        seed=int.from_bytes(hashlib.sha256(noise_id.encode()).digest()[:8],"little")
                        target={"image":Image.fromarray(image),"instruction":instruction,
                                "noise":torch.randn(1,16,7,generator=torch.Generator().manual_seed(seed))}
                        size=[1,2,4,8][call%4] if dynamic else 1
                        position=call%size
                        companion_indices=[(initialization*97+call*11+j)%len(companions) for j in range(size-1)]
                        requests=[companions[index] for index in companion_indices]
                        requests.insert(position,target)
                        inputs={};normalized={}
                        def trace(label,tensor):
                            if label.startswith("input.") or label=="diffusion.noise":inputs[label]=td(tensor[position:position+1])
                            elif label=="actions.normalized":normalized["value"]=tensor[position:position+1].detach().cpu().clone()
                        adapter.trace_callback=trace
                        try:
                            with torch.inference_mode(),set_batch_invariant_mode(invariant):
                                outputs=adapter.run_model(policy,adapter.compose_fixed_batch(requests))
                        finally:adapter.trace_callback=None
                        raw=outputs[position].numpy()
                        if not np.isfinite(raw).all():raise RuntimeError("nonfinite model action")
                        if condition=="native_singleton" and call==0:
                            with torch.inference_mode(),set_batch_invariant_mode(False):
                                released=adapter.run_model(policy,adapter.compose_batch([target]))[0].numpy()
                            fidelity={"comparison":"fixed64_native_singleton_vs_released_longest_padding_singleton",
                                      "exact":raw.tobytes()==released.tobytes(),"max_absolute_error":float(np.max(np.abs(raw-released)))}
                        selected,action=controller.process(raw)
                        obs,reward,success,truncated,info=env.step(action)
                        calls.append({"policy_call":call,"batch_size":size,"position":position,"companion_indices":companion_indices,
                                      "noise_id":noise_id,"request_noise":td(target["noise"]),"target_input_sha256":digest(inputs),
                                      "model_inputs":inputs,"normalized_actions":normalized["value"].tolist(),
                                      "raw_action_chunk":raw.tolist(),"selected_action":selected.tolist(),
                                      "environment_action":action.tolist(),"image_path":image_path.name,
                                      "reward":float(reward),"success":bool(success),"truncated":bool(truncated),"info":str(info)})
                        trajectory.append({"observation_sha256":digest(obs),"state":observation_state(obs)})
                    record={"task":task,"initialization":initialization,"condition":condition,"snapshot":snapshot,
                            "restore_checks":restored["checks"],"calls":calls,"trajectory":trajectory,
                            "success":bool(success),"singleton_padding_fidelity":fidelity}
                    with (args.output/"episodes.jsonl").open("a") as stream:stream.write(json.dumps(record)+"\n")
                    episodes[condition]=record
                    print(f"FINISHED {task}/{initialization}/{condition}: success={success}, steps={len(calls)}",flush=True)
                if len({digest(row["trajectory"][0]) for row in episodes.values()})!=1:raise RuntimeError("initial observation/state mismatch")
                if len({row["calls"][0]["target_input_sha256"] for row in episodes.values()})!=1:raise RuntimeError("initial model input mismatch")
                # File names differ between repeats; every measured value must agree.
                def canonical_calls(row):return [{key:value for key,value in call.items() if key!="image_path"} for call in row["calls"]]
                for name in list(conditions)[:4]:
                    if canonical_calls(episodes[name])!=canonical_calls(episodes[name+"_repeat"]) or episodes[name]["trajectory"]!=episodes[name+"_repeat"]["trajectory"] or episodes[name]["success"]!=episodes[name+"_repeat"]["success"]:
                        raise RuntimeError(f"complete episode A/A mismatch: {name}")
                left,right=episodes["patched_singleton"],episodes["patched_dynamic"]
                for field in ("trajectory","success"):
                    if left[field]!=right[field]:raise RuntimeError(f"invariant batching changes {field}")
                if [call["environment_action"] for call in left["calls"]]!=[call["environment_action"] for call in right["calls"]]:raise RuntimeError("invariant actions differ")
                left,right=episodes["native_singleton"],episodes["native_dynamic"]
                first=next((index for index,(a,b) in enumerate(zip(left["calls"],right["calls"])) if a["environment_action"]!=b["environment_action"]),None)
                if first is not None and (left["calls"][first]["target_input_sha256"]!=right["calls"][first]["target_input_sha256"] or left["trajectory"][:first+1]!=right["trajectory"][:first+1]):raise RuntimeError("native first-action attribution failed")
                blocks.append({"task":task,"initialization":initialization,"all_AA_exact":True,"invariant_batching_exact":True,
                    "native_first_action_divergence":first,"success":{name:row["success"] for name,row in episodes.items()},
                    "singleton_padding_fidelity":episodes["native_singleton"]["singleton_padding_fidelity"]})
                print(f"QUALIFIED block {task}/{initialization}",flush=True)
        finally:env.close()
    (args.output/"summary.json").write_text(json.dumps({"scope":runtime["scope"],"blocks":blocks,
        "paired_success_rate_campaign_complete":False},indent=2)+"\n")


if __name__=="__main__":main()
