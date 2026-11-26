"""Qualify LingBot on actual three-camera RoboTwin observations and joint states."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

from qualify_simpler_fixed_action_replay import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","checkpoint","qwen-metadata","robotwin-source","replay","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    os.environ.update({"QWEN3VL_PATH":str(args.qwen_metadata),"LINGBOT_VLA2_CHECKPOINT":str(args.checkpoint)})
    sys.path.insert(0,str(args.source))
    from batch_invariant_ops import set_batch_invariant_mode,deterministic_token_choice_moe
    from scripts.model_invariance.adapters.lingbot_vla2 import LingBotVLA2Adapter
    from lingbotvla.models.vla.lingbot_vla import qwen2_action_expert
    released_moe=qwen2_action_expert.robby_moe_forward

    class RealAdapter(LingBotVLA2Adapter):
        def _validate_paths(self):
            if not (self.checkpoint/"model.safetensors.index.json").is_file():raise FileNotFoundError(self.checkpoint)
            if not (self.checkpoint.parent.parent.parent/"lingbotvla_cli.yaml").is_file():raise FileNotFoundError("training config")

    torch.set_num_threads(8)
    adapter=RealAdapter()
    model=adapter.load_model()
    summary_path=args.replay/"summary.json"
    replay_summary=json.loads(summary_path.read_text())
    if not replay_summary.get("fixed_action_replay_pilot_exact"):raise RuntimeError("RoboTwin replay pilot not qualified")
    description_path=args.robotwin_source/"description/task_instruction/adjust_bottle.json"
    instruction=json.loads(description_path.read_text())["full_description"]
    paths=[args.replay/"initialization-0.pt",args.replay/"repeat-0-step-0009.pt",args.replay/"repeat-0-step-0019.pt"]
    transformed=[];requests=[];provenance=[]
    for index,path in enumerate(paths):
        raw=torch.load(path,map_location="cpu",weights_only=False)["observation"]
        observation={"task":instruction,"observation.state":raw["joint_action"]["vector"],
            "observation.images.cam_high":raw["observation"]["head_camera"]["rgb"],
            "observation.images.cam_left_wrist":raw["observation"]["left_camera"]["rgb"],
            "observation.images.cam_right_wrist":raw["observation"]["right_camera"]["rgb"]}
        applied=adapter.server._prepare_model_input(observation)
        transformed.append(applied)
        request={key:applied[key] for key in adapter.model_keys}
        request["noise"]=torch.randn(50,55,generator=torch.Generator().manual_seed(82000+index))
        requests.append(request)
        provenance.append({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "raw_observation_sha256":digest(raw),"instruction":instruction,"state":np.asarray(observation["observation.state"]).tolist()})
    torch.save(requests,args.output/"requests.pt")
    runtime={"scope":"real_robotwin_same_episode_numerical_probe","observations":provenance,
             "companion_contract":"different real observation times from one qualified fixed-action replay; not unrelated-task evidence",
             "released_instruction_source":str(description_path),"released_instruction_sha256":hashlib.sha256(description_path.read_bytes()).hexdigest(),
             "source_replay_sha256":hashlib.sha256(summary_path.read_bytes()).hexdigest(),
             "deterministic_moe_in_both_primary_modes":True,"precision":"float32",
             "tf32_matmul":torch.backends.cuda.matmul.allow_tf32,"torch":torch.__version__,
             "gpu":torch.cuda.get_device_name(),"paired_success_rate_campaign_complete":False}
    (args.output/"runtime.json").write_text(json.dumps(runtime,indent=2)+"\n")
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    with set_batch_invariant_mode(True):pass

    def run(indices,invariant):
        adapter.reset_model(model)
        with torch.inference_mode(),set_batch_invariant_mode(invariant):
            normalized=adapter.run_model(model,adapter.compose_batch([requests[index] for index in indices]))
            physical=adapter.server._unapply_batched_actions([transformed[index] for index in indices],normalized)
        return normalized,physical

    # The released batching wrapper allocates noise internally. Intercept only
    # the exact allocation and compare against its unchanged sampler entrypoint.
    original_randn=torch.randn;consumed=[]
    def explicit_randn(*shape,**kwargs):
        geometry=kwargs.get("size",shape[0] if len(shape)==1 and isinstance(shape[0],(tuple,list)) else shape)
        if tuple(geometry)==(1,50,55):
            consumed.append(tuple(geometry))
            return requests[0]["noise"].unsqueeze(0).to(device=kwargs.get("device","cpu"),dtype=kwargs.get("dtype",torch.float32)).clone()
        return original_randn(*shape,**kwargs)
    adapter.reset_model(model)
    with torch.inference_mode(),set_batch_invariant_mode(False),patch.object(torch,"randn",side_effect=explicit_randn):
        official=adapter.server.vla.sample_actions_batch(adapter.compose_batch([requests[0]]),use_bf16=False,use_compile=False,
            sample_compile_fn=adapter.server.sample_actions_fn).cpu()
    reference,_=run([0],False)
    if consumed!=[(1,50,55)] or official.numpy().tobytes()!=reference.numpy().tobytes():raise RuntimeError("official LingBot batch sampler equivalence failed")
    print("Official LingBot B1 batching wrapper equivalence verified",flush=True)
    records=[]
    for invariant in (False,True):
        for target in range(3):
            reference,reference_physical=run([target],invariant)
            repeat,repeated_physical=run([target],invariant)
            if reference.numpy().tobytes()!=repeat.numpy().tobytes() or digest(reference_physical)!=digest(repeated_physical):raise RuntimeError("controlled B1 A/A mismatch")
            for size in (1,2,4):
                for kind in ("duplicate","different_real_frames"):
                    indices=[target if kind=="duplicate" else (target+j+1)%3 for j in range(size-1)]+[target]
                    normalized,physical=run(indices,invariant)
                    selected=normalized[-1:]
                    selected_physical={key:value[-1:] for key,value in physical.items()}
                    exact=selected.numpy().tobytes()==reference.numpy().tobytes() and digest(selected_physical)==digest(reference_physical)
                    record={"invariant":invariant,"target":target,"batch_size":size,"composition":kind,"position":size-1,
                            "exact":exact,"max_normalized_error":float((selected-reference).abs().max()),
                            "physical_actions":{key:value.tolist() for key,value in selected_physical.items()}}
                    with (args.output/"comparisons.jsonl").open("a") as stream:stream.write(json.dumps(record)+"\n")
                    records.append(record)
                    print(f"invariant={invariant} target={target} B={size} {kind}: exact={exact}",flush=True)
    qwen2_action_expert.robby_moe_forward=released_moe
    try:
        first,_=run([0],False);second,_=run([0],False)
        released={"status":"executed","repeat_exact":first.numpy().tobytes()==second.numpy().tobytes(),"max_error":float((first-second).abs().max())}
    except Exception as error:
        released={"status":"execution_error","error":str(error)}
    finally:qwen2_action_expert.robby_moe_forward=deterministic_token_choice_moe
    summary={"scope":runtime["scope"],"official_B1_wrapper_equivalent":True,"controlled_B1_repeats_exact":True,
             "comparisons":len(records),"native_failures":sum(not row["exact"] for row in records if not row["invariant"]),
             "invariant_failures":sum(not row["exact"] for row in records if row["invariant"]),
             "released_atomic_moe_repeatability":released,"paired_success_rate_campaign_complete":False}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    if summary["invariant_failures"]:raise RuntimeError("invariant LingBot real-input probe failed")


if __name__=="__main__":main()
