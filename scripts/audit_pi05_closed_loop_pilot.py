"""Independently reconstruct exact-start pilot acceptance from raw episodes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    def source_bytes(name):
        path=args.directory/name
        if path.exists():return path.read_bytes()
        return gzip.decompress((args.directory/(name+".gz")).read_bytes())
    runtime=json.loads((args.directory/"runtime.json").read_text())
    rows=[json.loads(line) for line in source_bytes("episodes.jsonl").decode().splitlines()]
    conditions={"native_singleton":(False,False),"native_dynamic":(False,True),
                "patched_singleton":(True,False),"patched_dynamic":(True,True)}
    conditions.update({key+"_repeat":value for key,value in list(conditions.items())})
    expected={(task,initialization,name) for task in runtime["tasks"]
              for initialization in range(runtime["initializations"]) for name in conditions}
    indexed={(row["task"],row["initialization"],row["condition"]):row for row in rows}
    if len(indexed)!=len(rows) or set(indexed)!=expected:raise RuntimeError("duplicate/missing/extra episodes")
    checks=[]
    for task in runtime["tasks"]:
        for initialization in range(runtime["initializations"]):
            block={name:indexed[task,initialization,name] for name in conditions}
            for name,(_,dynamic) in conditions.items():
                row=block[name]
                if len(row["trajectory"])!=len(row["steps"])+1:raise RuntimeError("trajectory cardinality mismatch")
                if not row["steps"] or len(row["steps"])>runtime["max_steps"]:raise RuntimeError("invalid episode horizon")
                if not row["steps"][-1]["terminated"] and not row["steps"][-1]["truncated"] and len(row["steps"])!=runtime["max_steps"]:
                    raise RuntimeError("episode stopped early")
                if row["success"]!=row["steps"][-1]["success"]:raise RuntimeError("outcome disagrees with raw steps")
                for call_index,call in enumerate(row["calls"]):
                    size=[1,2,4,8][call_index%4] if dynamic else 1
                    if call["call"]!=call_index or call["batch_size"]!=size or call["position"]!=call_index%size:
                        raise RuntimeError("batch schedule mismatch")
                    if call["noise_id"]!=f"libero_10/{task}/{initialization}/policy-call-{call_index}":
                        raise RuntimeError("noise is not indexed independently of condition")
                    steps=[item for item in row["steps"] if item["policy_call"]==call_index]
                    if not steps or len(steps)>runtime["action_steps"]:raise RuntimeError("invalid executed action chunk")
                    if [item["action"] for item in steps]!=call["actions"][:len(steps)]:raise RuntimeError("executed action differs from model output")
                    if call_index<len(row["calls"])-1 and len(steps)!=runtime["action_steps"]:raise RuntimeError("short nonterminal action chunk")
            initial=[row["trajectory"][0] for row in block.values()]
            if any(value!=initial[0] for value in initial):raise RuntimeError("initial state or observations differ")
            if len({row["calls"][0]["target_input_sha256"] for row in block.values()})!=1:raise RuntimeError("initial model inputs differ")
            for name in list(conditions)[:4]:
                for field in ("calls","steps","trajectory","success"):
                    if block[name][field]!=block[name+"_repeat"][field]:raise RuntimeError(f"A/A differs: {name}/{field}")
            left,right=block["patched_singleton"],block["patched_dynamic"]
            for field in ("steps","trajectory","success"):
                if left[field]!=right[field]:raise RuntimeError(f"invariant batching changes {field}")
            if [call["normalized_sha256"] for call in left["calls"]]!=[call["normalized_sha256"] for call in right["calls"]]:
                raise RuntimeError("normalized invariant chunks differ")
            left,right=block["native_singleton"],block["native_dynamic"]
            mismatch=next((i for i,(a,b) in enumerate(zip(left["steps"],right["steps"])) if a["action"]!=b["action"]),None)
            if mismatch is not None:
                a=left["calls"][left["steps"][mismatch]["policy_call"]]
                b=right["calls"][right["steps"][mismatch]["policy_call"]]
                if a["target_input_sha256"]!=b["target_input_sha256"] or left["trajectory"][:mismatch+1]!=right["trajectory"][:mismatch+1] or b["batch_size"]==1:
                    raise RuntimeError("first native mismatch lacks attribution")
            checks.append({"task":task,"initialization":initialization,"eight_conditions_verified":True,
                           "AA_exact":True,"invariant_batching_exact":True,"native_first_action_mismatch":mismatch,
                           "success":{name:row["success"] for name,row in block.items()}})
    report={"status":"pilot_raw_evidence_verified","scope":"complete_episode_pilot",
            "episodes":len(rows),"blocks":checks,"success_rate_campaign_complete":False,
            "source_sha256":{name:hashlib.sha256(source_bytes(name)).hexdigest()
                             for name in ("runtime.json","episodes.jsonl")},
            "auditor_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
