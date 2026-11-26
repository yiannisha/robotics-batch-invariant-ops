"""Independently audit raw episodes, images, preprocessing and all replay noises."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def sha(path):
    result=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):result.update(chunk)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","checkpoint","tokenizer","output","audited-manifest"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.audited_manifest.exists():raise FileExistsError("audit output exists")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep
    from lerobot.policies.factory import make_pre_post_processors
    from scripts.run_model_replay_invariance import split_local_companion_rows
    from scripts.qualify_simpler_fixed_action_replay import digest

    torch.set_num_threads(8)
    manifest=json.loads(args.manifest.read_text())
    initialization_ids=manifest.get("initialization_ids",list(range(10)))
    diagnostic_ids=manifest.get("diagnostic_initialization_ids",initialization_ids[:2])
    if len(initialization_ids)!=10 or len(set(initialization_ids))!=10 or any(i not in range(50) for i in initialization_ids) or diagnostic_ids!=initialization_ids[:2]:
        raise ValueError("unexpected initialization membership")
    rows=manifest["observations"]
    root=Path(manifest["fixture_root"])
    if len(rows)!=1000 or any(row["replay_index"]!=i for i,row in enumerate(rows)):raise RuntimeError("observation cardinality/order mismatch")
    for i,row in enumerate(rows):row["observation_index"]=i%10
    config=PreTrainedConfig.from_pretrained(args.checkpoint)
    if not isinstance(config,PI05Config):raise TypeError("not PI05")
    config.device="cpu";config.compile_model=False;config.gradient_checkpointing=False
    pre,_=make_pre_post_processors(config,pretrained_path=args.checkpoint,
        preprocessor_overrides={"tokenizer_processor":{"tokenizer_name":str(args.tokenizer)},
                                "device_processor":{"device":"cpu"}})
    noise_path=Path(manifest["noise_file"])
    if sha(noise_path)!=manifest["noise_sha256"]:raise RuntimeError("noise archive changed")
    noises=np.load(noise_path,mmap_mode="r")
    if noises.shape!=(1000,3,50,32) or noises.dtype!=np.float32:raise RuntimeError("noise geometry/dtype mismatch")
    counts=Counter();episodes={};request_ids=set()
    for index,row in enumerate(rows):
        task,episode=row["task"],row["episode_id"]
        if task not in range(10) or episode not in initialization_ids:raise RuntimeError("unexpected task/episode")
        expected_split="diagnostic" if episode in diagnostic_ids else "heldout"
        if row["split"]!=expected_split:raise RuntimeError("episode split changed")
        counts[expected_split]+=1
        key=(task,episode)
        if key not in episodes:
            path=root/f"task-{task:02d}"/f"episode-{episode:02d}"/"episode.json"
            raw=json.loads(path.read_text())
            if raw["task"]!=task or raw["initialization"]!=episode or raw["seed"]!=20261108+episode:raise RuntimeError("source episode identity mismatch")
            if not raw["steps"] or len(raw["steps"])>520:raise RuntimeError("invalid episode length")
            final=raw["steps"][-1]
            if not final["terminated"] and not final["truncated"] and len(raw["steps"])!=520:raise RuntimeError("incomplete source episode")
            if raw["success"]!=final["success"]:raise RuntimeError("episode outcome mismatch")
            selection=np.linspace(0,len(raw["steps"]),10,dtype=int).tolist()
            if raw["selected_step_indices"]!=selection or len(set(selection))!=10:raise RuntimeError("observation selection changed")
            flattened=[]
            for call_index,call in enumerate(raw["calls"]):
                if call["policy_call"]!=call_index or call["noise_id"]!=f"libero_10/{task}/{episode}/policy-call-{call_index}":raise RuntimeError("collection noise/call identity mismatch")
                expected_seed=int.from_bytes(hashlib.sha256(call["noise_id"].encode()).digest()[:8],"little")
                source_noise=torch.randn(1,50,32,generator=torch.Generator().manual_seed(expected_seed)).numpy()
                if digest(source_noise)!=call["noise_sha256"]:raise RuntimeError("collection noise mismatch")
                executed=[step["action"] for step in raw["steps"] if step["policy_call"]==call_index]
                if not executed or len(executed)>config.n_action_steps or executed!=call["predicted_actions"][:len(executed)]:raise RuntimeError("source action execution mismatch")
                if call_index<len(raw["calls"])-1 and len(executed)!=config.n_action_steps:raise RuntimeError("short nonterminal chunk")
                flattened.extend(executed)
            if len(flattened)!=len(raw["steps"]):raise RuntimeError("source action coverage mismatch")
            episodes[key]={"path":str(path),"sha256":sha(path),"raw":raw}
        observation_index=index%10
        if row["source_step"]!=episodes[key]["raw"]["selected_step_indices"][observation_index]:raise RuntimeError("source frame index mismatch")
        images={name:np.asarray(Image.open(root/path).convert("RGB")) for name,path in row["images"].items()}
        robot_state={name:{key:np.asarray(value,dtype=np.float64) for key,value in entry.items()} for name,entry in row["robot_state"].items()}
        raw_obs={"pixels":images,"robot_state":robot_state}
        if digest(raw_obs)!=row["observation_sha256"]:raise RuntimeError(f"raw observation bytes differ: {index}")
        batched={"pixels":{name:image[None] for name,image in images.items()},
                 "robot_state":{name:{key:value[None] for key,value in entry.items()} for name,entry in robot_state.items()}}
        batch=LiberoProcessorStep().observation(preprocess_observation(batched))
        batch["task"]=[row["instruction"]]
        expected={name:value for name,value in pre(batch).items() if isinstance(value,torch.Tensor)}
        input_path=root/row["input_file"]
        if sha(input_path)!=row["input_file_sha256"]:raise RuntimeError("input file changed")
        actual=torch.load(input_path,map_location="cpu",weights_only=True)
        if set(actual)!=set(expected):raise RuntimeError("preprocessed input keys differ")
        for name in actual:
            left,right=actual[name],expected[name]
            if left.dtype!=right.dtype or left.shape!=right.shape or not torch.equal(left.contiguous().view(torch.uint8),right.contiguous().view(torch.uint8)):
                raise RuntimeError(f"preprocessed tensor differs: observation={index}, tensor={name}")
        for noise_index in range(3):
            request_id=f"libero_10/{task}/{episode}/observation-{observation_index}/noise-{noise_index}"
            if row["request_ids"][noise_index]!=request_id or request_id in request_ids:raise RuntimeError("request identity mismatch/duplicate")
            request_ids.add(request_id)
            seed=int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:8],"little")
            expected_noise=torch.randn(50,32,generator=torch.Generator().manual_seed(seed)).numpy()
            value=noises[index,noise_index]
            if value.tobytes()!=expected_noise.tobytes() or hashlib.sha256(value.tobytes()).hexdigest()!=row["noise"][noise_index]["sha256"]:raise RuntimeError("request-local noise differs")
        row["observation_index"]=observation_index
        row["image_file_sha256"]={name:sha(root/path) for name,path in row["images"].items()}
        row["source_image_rgb_sha256"]={name:hashlib.sha256(image.tobytes()).hexdigest() for name,image in images.items()}
        row["source_episode_sha256"]=episodes[key]["sha256"]
        row["companion_replay_indices"]=[item["replay_index"] for item in split_local_companion_rows(rows,row,index,31)]
        companions=[rows[i] for i in row["companion_replay_indices"]]
        if len({r["replay_index"] for r in companions})!=31 or any(r["split"]!=row["split"] or (r["task"],r["episode_id"])==(task,episode) for r in companions):
            raise RuntimeError("companion split/episode isolation failed")
        if (index+1)%100==0:print(f"AUDITED {index+1}/1000 observations",flush=True)
    if counts!={"diagnostic":200,"heldout":800} or len(episodes)!=100:raise RuntimeError("episode/split coverage mismatch")
    report={"status":"complete_corpus_independently_verified","source_manifest_sha256":sha(args.manifest),
            "episodes":100,"observations":1000,"request_noise_pairs":3000,"splits":dict(counts),
            "exact_reconstructed_preprocessed_inputs":1000,"all_raw_rgb_state_and_noise_hashes_verified":True,
            "source_episodes":[{k:v for k,v in value.items() if k!="raw"} for value in episodes.values()],
            "auditor_sha256":sha(Path(__file__)),"numerical_inference_qualified":False,"closed_loop_complete":False}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    manifest["corpus_audit_path"]=str(args.output.resolve())
    manifest["corpus_audit_sha256"]=sha(args.output)
    manifest["companion_schedule"]={"version":"split_local_unrelated_observations_v1","max_companions":31,
        "description":"Same split and outside target episode; prioritize distinct episodes then distinct observations."}
    args.audited_manifest.write_text(json.dumps(manifest,indent=2)+"\n")
    print("COMPLETE: raw corpus and every preprocessed input/noise independently verified",flush=True)


if __name__=="__main__":main()
