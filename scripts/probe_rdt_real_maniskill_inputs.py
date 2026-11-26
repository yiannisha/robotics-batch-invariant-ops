"""Check released RDT preprocessing/sampler and invariance on actual ManiSkill resets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import yaml

from qualify_simpler_fixed_action_replay import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","siglip","source","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    os.environ.update({"RDT_CHECKPOINT":str(args.checkpoint),"RDT_SIGLIP":str(args.siglip),"RDT_SOURCE":str(args.source)})
    from batch_invariant_ops import set_batch_invariant_mode
    from scripts.model_invariance.adapters.rdt_maniskill import RDTManiSkillAdapter
    from scripts.run_rdt_paired_server import PairedRDTBackend,verify_artifacts,STATE_MIN,STATE_MAX
    import gymnasium as gym
    import mani_skill.envs

    class RealAdapter(RDTManiSkillAdapter):
        def _validate_paths(self):
            for path in (self.checkpoint/"rdt/mp_rank_00_model_states.pt",self.siglip/"model.safetensors",
                         self.siglip/"config.json",self.source/"configs/base.yaml"):
                if not path.is_file():raise FileNotFoundError(path)

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    artifacts=verify_artifacts(args.checkpoint,args.siglip)
    adapter=RealAdapter()
    policy=adapter.load_model()
    print("RDT and SigLIP loaded",flush=True)
    backends={False:PairedRDTBackend(adapter,policy,condition="native_dynamic",checkpoint=args.checkpoint,
                                   language=adapter.language_embeddings,artifacts=artifacts),
              True:PairedRDTBackend(adapter,policy,condition="patched_dynamic",checkpoint=args.checkpoint,
                                  language=adapter.language_embeddings,artifacts=artifacts)}
    requests=[];provenance=[]
    for index,task in enumerate(("PickCube-v1","PushCube-v1","StackCube-v1")):
        env=gym.make(task,obs_mode="rgb",control_mode="pd_joint_pos",render_mode="rgb_array",reward_mode="dense",sim_backend="cpu",
                     sensor_configs={"shader_pack":"default"},human_render_camera_configs={"shader_pack":"default"},viewer_camera_configs={"shader_pack":"default"})
        try:
            obs,_=env.reset(seed=20241201)
            image=np.ascontiguousarray(env.render().squeeze(0).cpu().numpy(),dtype=np.uint8)
            proprio=obs["agent"]["qpos"][:,:-1].reshape(-1).cpu().numpy().astype(np.float32)
            noise=torch.randn(64,128,generator=torch.Generator().manual_seed(73000+index)).numpy()
            requests.append({"task_name":task,"paired_request_id":f"{task}/initialization-0",
                "image_history":np.stack([np.zeros_like(image),image]),"history_valid":[False,True],"proprio":proprio,"noise":noise})
            provenance.append({"task":task,"seed":20241201,"image_sha256":digest(image),"proprio":proprio.tolist(),
                               "environment_control_frequency":env.unwrapped.control_freq})
        finally:env.close()
    torch.save(requests,args.output/"requests.pt")
    runtime={"scope":"real_maniskill_numerical_integration_probe","source_observations":provenance,
             "artifact_files":artifacts,"native_backend":backends[False].metadata(),"patched_backend":backends[True].metadata(),
             "strict_rdt_checkpoint_load":True,"paired_success_rate_campaign_complete":False}
    (args.output/"runtime.json").write_text(json.dumps(runtime,indent=2)+"\n")
    with set_batch_invariant_mode(True):pass

    def run(indices,invariant):
        # Request IDs identify slots, including duplicate semantic requests.
        examples=[dict(requests[index],paired_request_id=f"slot-{slot}/source-{index}") for slot,index in enumerate(indices)]
        adapter.reset_model(policy)
        with torch.inference_mode(),set_batch_invariant_mode(invariant):
            return backends[invariant].infer({"examples":examples})["actions"]

    # Exercise the unchanged released ManiSkill wrapper over these exact weights.
    import importlib.util
    spec=importlib.util.spec_from_file_location("rdt_released_maniskill_wrapper",args.source/"scripts/maniskill_model.py")
    official_wrapper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official_wrapper)
    RoboticDiffusionTransformerModel=official_wrapper.RoboticDiffusionTransformerModel
    DATA_STAT=official_wrapper.DATA_STAT
    wrapper=RoboticDiffusionTransformerModel.__new__(RoboticDiffusionTransformerModel)
    wrapper.args=yaml.safe_load((args.source/"configs/base.yaml").read_text())
    wrapper.dtype=torch.bfloat16;wrapper.device="cuda";wrapper.image_size=None;wrapper.control_frequency=25
    wrapper.image_processor=policy.processor;wrapper.vision_model=policy.vision;wrapper.policy=policy.model
    for name in ("state_min","state_max","action_min","action_max"):
        setattr(wrapper,name,torch.tensor(DATA_STAT[name],device="cuda"))
    consumed=[];original_randn=torch.randn;request=requests[0]
    def explicit_randn(*shape,**kwargs):
        geometry=kwargs.get("size",shape[0] if len(shape)==1 and isinstance(shape[0],(tuple,list)) else shape)
        if tuple(geometry)==(1,64,128):
            consumed.append(tuple(geometry))
            return torch.from_numpy(request["noise"]).unsqueeze(0).to(device=kwargs.get("device","cpu"),dtype=kwargs.get("dtype",torch.float32)).clone()
        return original_randn(*shape,**kwargs)
    adapter.reset_model(policy)
    with torch.inference_mode(),set_batch_invariant_mode(False),patch.object(torch,"randn",side_effect=explicit_randn):
        official=wrapper.step(torch.from_numpy(request["proprio"]).unsqueeze(0),
            [None,None,None,Image.fromarray(request["image_history"][1]),None,None],adapter.language_embeddings[request["task_name"]]).cpu().numpy()
    controlled=run([0],False)
    if consumed!=[(1,64,128)] or official.tobytes()!=controlled.tobytes():
        raise RuntimeError(f"released RDT wrapper differs from controlled native backend; max_error={np.max(np.abs(official-controlled))}")
    print("Official RDT ManiSkill B1 wrapper equivalence verified",flush=True)
    records=[]
    for invariant in (False,True):
        for target in range(3):
            reference=run([target],invariant)
            if reference.tobytes()!=run([target],invariant).tobytes():raise RuntimeError("B1 A/A mismatch")
            for size in (1,2,4):
                for kind in ("duplicate","unrelated"):
                    indices=[target if kind=="duplicate" else (target+j+1)%3 for j in range(size-1)]+[target]
                    selected=run(indices,invariant)[-1:]
                    record={"invariant":invariant,"target":target,"batch_size":size,"position":size-1,"composition":kind,
                            "exact":reference.tobytes()==selected.tobytes(),"max_error":float(np.max(np.abs(reference-selected))),
                            "actions":selected.tolist()}
                    with (args.output/"comparisons.jsonl").open("a") as stream:stream.write(json.dumps(record)+"\n")
                    records.append(record)
                    print(f"invariant={invariant} target={target} B={size} {kind}: exact={record['exact']}",flush=True)
    summary={"scope":runtime["scope"],"official_B1_wrapper_equivalent":True,"B1_repeats_exact":True,
             "comparisons":len(records),"native_failures":sum(not row["exact"] for row in records if not row["invariant"]),
             "invariant_failures":sum(not row["exact"] for row in records if row["invariant"]),"paired_success_rate_campaign_complete":False}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    if summary["invariant_failures"]:raise RuntimeError("RDT invariant real-input probe failed")


if __name__=="__main__":main()
