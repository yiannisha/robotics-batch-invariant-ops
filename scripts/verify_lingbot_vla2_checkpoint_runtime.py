"""Strict released LingBot construction and all-tensor checkpoint load audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from safetensors import safe_open


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","checkpoint","qwen-metadata","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    os.environ["QWEN3VL_PATH"]=str(args.qwen_metadata)
    sys.path.insert(0,str(args.source))
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from lingbotvla.models.vla.lingbot_vla import qwen2_action_expert
    from batch_invariant_ops import deterministic_token_choice_moe
    qwen2_action_expert.robby_moe_forward=deterministic_token_choice_moe
    os.chdir(args.source)
    load_dtypes={}
    original_load=LingbotVLAv2Server.load_model_weights
    def record_official_load(self,*values,**kwargs):
        result=original_load(self,*values,**kwargs)
        load_dtypes.update({name:value.dtype for name,value in self.vla.state_dict().items()})
        return result
    # The official server loads into constructor dtypes before its final FP32
    # conversion. Preserve and audit that exact sequence, including rounding.
    from unittest.mock import patch
    with patch.object(LingbotVLAv2Server,"load_model_weights",record_official_load):
        server=LingbotVLAv2Server(path_to_pi_model=str(args.checkpoint),chunk_ret=True,
                                use_bf16=False,use_fp32=True,use_compile=False)
    server.reset("robotwin")
    state=server.vla.state_dict()
    seen=set();count=0
    for path in sorted(args.checkpoint.glob("*.safetensors")):
        with safe_open(path,framework="pt",device="cpu") as source:
            for name in source.keys():
                if name in seen or name not in state:raise RuntimeError(f"duplicate/unexpected tensor: {name}")
                original=source.get_tensor(name)
                actual=state[name].detach().cpu()
                expected=original.to(load_dtypes[name]).to(actual.dtype)
                if not torch.equal(actual,expected):raise RuntimeError(f"loaded tensor differs after official dtype sequence: {name}")
                seen.add(name);count+=original.numel()
    if seen!=set(state):raise RuntimeError("checkpoint does not cover complete state dict")
    if len(seen)!=1708 or count!=6375907511:raise RuntimeError("unexpected official tensor/element counts")
    report={"status":"all_tensors_loaded_and_verified","tensor_count":len(seen),"source_elements":count,
            "strict_official_loader":True,"all_tensors_exact_after_declared_dtype_conversion":True,
            "load_dtype_counts":{str(dtype):sum(value==dtype for value in load_dtypes.values()) for dtype in set(load_dtypes.values())},
            "dtype_sequence":"checkpoint -> official constructor parameter dtype -> released FP32 deployment conversion",
            "torch":torch.__version__,"gpu":torch.cuda.get_device_name(),"precision":"float32",
            "deterministic_moe":"shared_with_native_and_invariant_conditions",
            "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "numerical_inference_qualified":False,"closed_loop_qualified":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
