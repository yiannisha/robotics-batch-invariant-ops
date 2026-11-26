"""Independently verify CogACT/RDT raw pilot episodes, noise and attribution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def sha(value):
    # Canonical serialization retains signed zeros, unlike Python equality.
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def tensor_identity(value):
    return {"dtype":str(value.dtype),"shape":list(value.shape),
            "sha256":hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def audit(directory,model,controller_class=None,ensemble_class=None):
    runtime=json.loads((directory/"runtime.json").read_text())
    rows=[json.loads(line) for line in (directory/"episodes.jsonl").read_text().splitlines()]
    wanted={(task,initialization,name) for task in runtime["tasks"]
            for initialization in range(runtime["initializations"]) for name in runtime["conditions"]}
    indexed={(row["task"],row["initialization"],row["condition"]):row for row in rows}
    if len(indexed)!=len(rows) or set(indexed)!=wanted:raise RuntimeError("episode coverage mismatch")
    blocks=[]
    for task in runtime["tasks"]:
        for initialization in range(runtime["initializations"]):
            block={name:indexed[task,initialization,name] for name in runtime["conditions"]}
            for name,row in block.items():
                controller=controller_class("google_robot",ensemble_class) if model=="cogact" else None
                count=len(row["calls"]) if model=="cogact" else len(row["steps"])
                if len(row["trajectory"])!=count+1:raise RuntimeError("trajectory coverage mismatch")
                for call_index,call in enumerate(row["calls"]):
                    size=[1,2,4,8][call_index%4] if "dynamic" in name else 1
                    if call["policy_call"]!=call_index or call["batch_size"]!=size or call["position"]!=call_index%size:raise RuntimeError("batch schedule mismatch")
                    if sha(call["model_inputs"])!=call["target_input_sha256"]:raise RuntimeError("recorded input hash mismatch")
                    key=f"{task}/{initialization}/policy-call-{call_index}"
                    if call["noise_id"]!=key:raise RuntimeError("noise depends on condition/batch")
                    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"little")
                    if model=="cogact":
                        noise=torch.randn(1,16,7,generator=torch.Generator().manual_seed(seed))
                        if tensor_identity(noise)!=call["request_noise"] or tensor_identity(noise)!=call["model_inputs"]["diffusion.noise"]:raise RuntimeError("actual noise differs")
                        selected,action=controller.process(np.asarray(call["raw_action_chunk"],dtype=np.float64))
                        if sha(selected.tolist())!=sha(call["selected_action"]) or action.tobytes()!=np.asarray(call["environment_action"],dtype=np.float64).tobytes():raise RuntimeError("released controller action mismatch")
                        if not (directory/call["image_path"]).is_file():raise FileNotFoundError(call["image_path"])
                    else:
                        noise=torch.randn(64,128,generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16).unsqueeze(0)
                        if tensor_identity(noise)!=call["model_inputs"]["input.noise"]:raise RuntimeError("actual noise differs")
                        steps=[step for step in row["steps"] if step["policy_call"]==call_index]
                        if sha([step["action"] for step in steps])!=sha(call["raw_action_chunk"][::4][:len(steps)]):raise RuntimeError("released stride/execution mismatch")
            if len({sha(row["trajectory"][0]) for row in block.values()})!=1 or len({row["calls"][0]["target_input_sha256"] for row in block.values()})!=1:raise RuntimeError("initial state/model input differs")
            for name in ("native_singleton","native_dynamic","patched_singleton","patched_dynamic"):
                left,right=block[name],block[name+"_repeat"]
                for field in ["calls","trajectory","success"]+(["steps"] if model=="rdt" else []):
                    a,b=left[field],right[field]
                    if field=="calls" and model=="cogact":
                        a=[{key:value for key,value in call.items() if key!="image_path"} for call in a]
                        b=[{key:value for key,value in call.items() if key!="image_path"} for call in b]
                    if sha(a)!=sha(b):raise RuntimeError(f"A/A mismatch: {name}/{field}")
            def actions(row):
                return [call["environment_action"] for call in row["calls"]] if model=="cogact" else [step["action"] for step in row["steps"]]
            left,right=block["patched_singleton"],block["patched_dynamic"]
            if sha(left["trajectory"])!=sha(right["trajectory"]) or left["success"]!=right["success"] or sha(actions(left))!=sha(actions(right)):raise RuntimeError("invariant batching mismatch")
            left,right=block["native_singleton"],block["native_dynamic"]
            first=next((index for index,(a,b) in enumerate(zip(actions(left),actions(right))) if sha(a)!=sha(b)),None)
            if first is not None:
                a=first if model=="cogact" else left["steps"][first]["policy_call"]
                b=first if model=="cogact" else right["steps"][first]["policy_call"]
                if left["calls"][a]["target_input_sha256"]!=right["calls"][b]["target_input_sha256"] or sha(left["trajectory"][:first+1])!=sha(right["trajectory"][:first+1]):raise RuntimeError("first native mismatch lacks attribution")
            blocks.append({"task":task,"initialization":initialization,"all_AA_exact":True,
                           "invariant_batching_exact":True,"native_first_action_divergence":first})
    return {"status":"pilot_raw_evidence_independently_verified","model":model,"scope":"complete_episode_pilot",
            "episodes":len(rows),"blocks":blocks,"request_noise_regenerated_and_verified":True,
            "controller_actions_recomputed":model=="cogact",
            "episode_bytes_sha256":hashlib.sha256((directory/"episodes.jsonl").read_bytes()).hexdigest(),
            "auditor_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "paired_success_rate_campaign_complete":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--model",choices=("cogact","rdt"),required=True)
    parser.add_argument("--cogact-source",type=Path)
    parser.add_argument("--openvla-source",type=Path)
    args=parser.parse_args()
    controller=ensemble=None
    if args.model=="cogact":
        if not args.cogact_source or not args.openvla_source:parser.error("CogACT audit needs both pinned source paths")
        for path in (args.openvla_source,args.cogact_source):sys.path.insert(0,str(path))
        from scripts.run_cogact_paired_server import RequestState
        from sim_cogact.adaptive_ensemble import AdaptiveEnsembler
        controller,ensemble=RequestState,AdaptiveEnsembler
    report=audit(args.directory,args.model,controller,ensemble)
    (args.directory/"independent_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
