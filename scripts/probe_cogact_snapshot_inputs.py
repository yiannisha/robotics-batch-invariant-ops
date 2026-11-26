"""Qualify CogACT-Small on canonical SIMPLER inputs before closed-loop work."""
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
from PIL import Image

from qualify_simpler_fixed_action_replay import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "source", "openvla-source", "llama-metadata",
                 "pizero-project", "simpler-root", "simpler-python", "vulkan-icd", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    os.environ.update({"COGACT_CHECKPOINT": str(args.checkpoint),
                       "COGACT_SOURCE": str(args.source),
                       "COGACT_OPENVLA_SOURCE": str(args.openvla_source),
                       "COGACT_LLAMA_METADATA": str(args.llama_metadata)})
    from batch_invariant_ops import set_batch_invariant_mode
    from scripts.model_invariance.adapters import cogact_small as implementation

    class SnapshotAdapter(implementation.CogACTSmallAdapter):
        def _validate_paths(self):
            # Real SIMPLER observations replace the historical LIBERO fixture.
            for path in (self.checkpoint / "checkpoints/CogACT-Small.pt",
                         self.checkpoint / "config.json", self.checkpoint / "dataset_statistics.json",
                         self.source / "vla/cogactvla.py", self.openvla_source / "prismatic/models/vlms/prismatic.py",
                         self.llama_metadata / "config.json", self.llama_metadata / "tokenizer.model"):
                if not path.is_file():
                    raise FileNotFoundError(path)
            return self.checkpoint / "checkpoints/CogACT-Small.pt"

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    adapter = SnapshotAdapter()
    policy = adapter.load_model()
    print("CogACT loaded", flush=True)
    sys.path.insert(0, str(args.pizero_project))
    from experiments.simpler_support import make_simpler_env
    # Read the reference mapping without importing its policy-specific runner.
    import ast
    tree = ast.parse((args.pizero_project / "experiments/prepare_replay.py").read_text())
    TASKS = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "TASKS" for target in node.targets))
    requests = []
    provenance = []
    for index, task in enumerate(("pick_can", "move_near", "open_drawer")):
        env = make_simpler_env(TASKS[task], args.simpler_python, args.simpler_root, args.vulkan_icd)
        try:
            env.reset(seed=20261108, options={"obj_init_options": {"episode_id": 0}})
            snapshot = env.capture_snapshot()
            restored = env.restore_snapshot()
            observation = restored["observation"]
            image = next(iter(observation["image"].values()))["rgb"]
            instruction = env.get_language_instruction()
            requests.append({"image": Image.fromarray(image), "instruction": instruction,
                             "noise": torch.randn(1,16,7,generator=torch.Generator().manual_seed(74000+index))})
            provenance.append({"task": task, "snapshot": snapshot, "restore_checks": restored["checks"],
                               "observation_sha256": digest(observation), "instruction": instruction})
        finally:
            env.close()
    tokenizer = policy.model.vlm.llm_backbone.tokenizer
    lengths = []
    for request in requests:
        prompt = policy.model.vlm.get_prompt_builder()
        prompt.add_turn(role="human", message=f"What action should the robot take to {request['instruction'].lower()}?")
        lengths.append(len(tokenizer(prompt.get_prompt()).input_ids)+2)
    implementation.CANONICAL_LANGUAGE_TOKENS = max(lengths)
    runtime = {"scope": "snapshot_real_input_numerical_probe", "model": "cogact_small",
               "observations": provenance, "canonical_language_tokens": max(lengths),
               "execution_configuration": adapter.execution_configuration(),
               "adapter_sha256": hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest(),
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
               "success_rate_campaign_complete": False}
    (args.output / "runtime.json").write_text(json.dumps(runtime,indent=2)+"\n")
    torch.save(requests, args.output / "requests.pt")
    with set_batch_invariant_mode(True):
        pass

    def run(samples, invariant):
        with torch.inference_mode(), set_batch_invariant_mode(invariant):
            batch = adapter.compose_batch(samples)
            return adapter.run_model(policy, batch).cpu()

    # Check the unchanged official B1 implementation with exactly the same noise.
    request = requests[0]
    original_randn = torch.randn
    consumed = []
    def explicit_randn(*shape, **kwargs):
        geometry = tuple(shape[0]) if len(shape)==1 and isinstance(shape[0],(tuple,list)) else tuple(shape)
        if geometry == (1,16,7):
            consumed.append(geometry)
            return request["noise"].to(device=kwargs.get("device","cpu"), dtype=kwargs.get("dtype",torch.float32)).clone()
        return original_randn(*shape, **kwargs)
    with torch.inference_mode(), set_batch_invariant_mode(False), patch.object(torch,"randn",side_effect=explicit_randn):
        official, _ = policy.model.predict_action_batch(
            image=[request["image"]], instruction=[request["instruction"]],
            unnorm_key=implementation.UNNORM_KEY, cfg_scale=implementation.CFG_SCALE,
            use_ddim=True, num_ddim_steps=implementation.DDIM_STEPS, do_sample=False)
    controlled = run([request], False)
    if consumed != [(1,16,7)] or not np.array_equal(official, controlled.numpy()):
        raise RuntimeError("controlled CogACT B1 does not match official batch sampler")
    print("Official B1 sampler equivalence verified", flush=True)
    records = []
    for invariant in (False,True):
        for target in range(3):
            reference = run([requests[target]],invariant)
            if not torch.equal(reference,run([requests[target]],invariant)):
                raise RuntimeError("B1 A/A mismatch")
            for size in (1,2,4):
                for composition in ("duplicate","unrelated"):
                    samples=[requests[target if composition=="duplicate" else (target+j+1)%3] for j in range(size-1)]
                    position=size-1
                    samples.insert(position,requests[target])
                    actions=run(samples,invariant)[position:position+1]
                    record={"invariant":invariant,"target":target,"batch_size":size,"position":position,
                            "composition":composition,"exact":digest(actions.numpy())==digest(reference.numpy()),
                            "max_error":float((actions-reference).abs().max()),"actions":actions.tolist()}
                    with (args.output/"comparisons.jsonl").open("a") as stream:stream.write(json.dumps(record)+"\n")
                    records.append(record)
                    print(f"invariant={invariant} target={target} B={size} {composition}: exact={record['exact']}",flush=True)
    summary={"scope":runtime["scope"],"official_b1_equivalent":True,"same_condition_repeats_exact":True,
             "comparisons":len(records),"native_failures":sum(not r["exact"] for r in records if not r["invariant"]),
             "invariant_failures":sum(not r["exact"] for r in records if r["invariant"]),"success_rate_campaign_complete":False}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    if summary["invariant_failures"]:raise RuntimeError("CogACT invariant probe failed")


if __name__ == "__main__":
    main()
