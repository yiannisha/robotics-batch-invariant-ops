"""Fail-closed all-tensor load audit for the selected LeRobot LIBERO policy."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_verified_policy(checkpoint):
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.configs.policies import PreTrainedConfig

    acquisition = json.loads((checkpoint / "acquisition.json").read_text())
    if acquisition["revision"] != "7152f13e34c6c115d808aea0e5d08aca74eac14b":
        raise RuntimeError("unexpected checkpoint revision")
    weight_path = checkpoint / "model.safetensors"
    digest = hashlib.sha256()
    with weight_path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    expected = next(row["sha256"] for row in acquisition["files"] if row["path"] == "model.safetensors")
    if digest.hexdigest() != expected:
        raise RuntimeError("checkpoint changed after acquisition")
    config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(config, PI05Config):
        raise RuntimeError("checkpoint is not a PI05 policy")
    config.device = "cpu"
    config.compile_model = False
    config.gradient_checkpointing = False
    policy = PI05Policy(config).eval()
    original = load_file(weight_path, device="cpu")
    fixed = policy._fix_pytorch_state_dict_keys(original, config)
    remapped = {key if key.startswith("model.") else "model." + key: value
                for key, value in fixed.items()}
    # Safetensors omits redundant tied parameters. Only restore aliases proven
    # by the constructed official model's actual Parameter identity.
    parameters = dict(policy.named_parameters(remove_duplicate=False))
    tied_aliases = {}
    for name in policy.state_dict():
        if name in remapped:
            continue
        parameter = parameters.get(name)
        aliases = [key for key, candidate in parameters.items()
                   if parameter is not None and candidate is parameter and key in remapped]
        if not aliases:
            raise RuntimeError(f"missing learned tensor without a loaded tied alias: {name}")
        remapped[name] = remapped[aliases[0]]
        tied_aliases[name] = aliases[0]
    result = policy.load_state_dict(remapped, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(str(result))
    loaded = policy.state_dict()
    if loaded.keys() != remapped.keys():
        raise RuntimeError("state-dict coverage mismatch")
    for name, value in loaded.items():
        if not torch.equal(value, remapped[name].to(value.dtype)):
            raise RuntimeError(f"tensor mismatch after declared dtype conversion: {name}")
    report = {"status": "all_tensors_loaded_and_verified", "checkpoint": acquisition,
              "torch": torch.__version__,
              "transformers": importlib.metadata.version("transformers"),
              "lerobot": importlib.metadata.version("lerobot"),
              "tensor_count": len(loaded),
              "source_elements": sum(value.numel() for value in original.values()),
              "strict": True, "all_tensors_exact_after_declared_dtype_conversion": True,
              "verified_tied_aliases": tied_aliases,
              "compile_model": False, "dtype": config.dtype,
              "config": config.to_dict() if hasattr(config, "to_dict") else str(config),
              "numerical_inference_qualified": False, "closed_loop_qualified": False}
    return policy, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    policy, report = load_verified_policy(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"VERIFIED {report['tensor_count']} tensors", flush=True)


if __name__ == "__main__":
    main()
