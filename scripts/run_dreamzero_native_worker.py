"""Two-rank native DreamZero inference for a local, serialized NPZ request queue.

Run with torchrun and PYTHONPATH pointing at the official DreamZero checkout.
No numerical operators are patched. Each request contains already converted
DROID observations, with an explicit leading batch dimension.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from tianshou.data import Batch


def digest(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous()
        header = str((tuple(value.shape), str(value.dtype))).encode()
        payload = value.view(torch.uint8).numpy().tobytes()
    else:
        value = np.asarray(value)
        header = str((value.shape, str(value.dtype))).encode()
        payload = value.tobytes()
    return hashlib.sha256(header + payload).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--queue', type=Path, required=True)
    args = parser.parse_args()
    os.environ['ENABLE_DIT_CACHE'] = 'false'
    os.environ['ATTENTION_BACKEND'] = 'TE'
    # Match the official server: scheduler steps specialize on Python state.
    torch._dynamo.config.recompile_limit = 800
    from socket_test_optimized_AR import init_mesh
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
    from groot.vla.data.schema import EmbodimentTag
    mesh = init_mesh()
    rank = dist.get_rank()
    group = dist.new_group(backend='gloo')
    policy = GrootSimPolicy(embodiment_tag=EmbodimentTag('oxe_droid'),
                            model_path=args.model_path, device='cuda', device_mesh=mesh)
    head = policy.trained_model.action_head
    noise_records = []
    original_noise = head.generate_noise

    def capture_noise(shape, **kwargs):
        noise = original_noise(shape, **kwargs)
        noise_records.append({'shape': list(noise.shape), 'target_sha256': digest(noise[:1])})
        return noise

    head.generate_noise = capture_noise
    input_records = {}
    original_apply = policy.apply

    def capture_apply(batch, **kwargs):
        batch = original_apply(batch, **kwargs)
        for key, value in batch.normalized_obs.items():
            if torch.is_tensor(value) or isinstance(value, np.ndarray):
                input_records[key] = digest(value[:1])
        return batch

    policy.apply = capture_apply
    args.queue.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.queue / 'ready.json').write_text(json.dumps({'pid': os.getpid(), 'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(), 'cache': False, 'noise_seed': head.seed}))
    index = 0
    while True:
        request = args.queue / f'{index:06d}.request.npz'
        objects = [None]
        if rank == 0:
            while not request.exists():
                time.sleep(0.1)
            with np.load(request, allow_pickle=False) as data:
                objects[0] = {key: data[key] for key in data.files}
        dist.broadcast_object_list(objects, src=0, group=group)
        obs = objects[0]
        reset = bool(obs.pop('_reset'))
        if reset:
            head.language = None
            head.current_start_frame = 0
            head.clip_feas = head.ys = None
            head.kv_cache1 = head.kv_cache_neg = None
            head.crossattn_cache = head.crossattn_cache_neg = None
        obs['annotation.language.action_text'] = obs['annotation.language.action_text'].tolist()
        noise_records.clear()
        input_records.clear()
        t0 = time.monotonic()
        with torch.inference_mode():
            result, _ = policy.lazy_joint_forward_causal(Batch(obs=obs))
        torch.cuda.synchronize()
        if rank == 0:
            arrays = {key: value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
                      for key, value in result.act.items()}
            temporary = args.queue / f'{index:06d}.result.tmp.npz'
            np.savez(temporary, **arrays)
            temporary.replace(args.queue / f'{index:06d}.result.npz')
            meta = {'index': index, 'seconds': time.monotonic()-t0,
                    'normalized_target_sha256': dict(input_records), 'noise': list(noise_records),
                    'raw_target_sha256': {k: digest(v[:1]) for k,v in obs.items() if isinstance(v,np.ndarray)},
                    'finite': all(bool(np.isfinite(v).all()) for v in arrays.values())}
            (args.queue / f'{index:06d}.json').write_text(json.dumps(meta, indent=2))
            print('COMPLETED', json.dumps(meta), flush=True)
        dist.barrier(group=group)
        index += 1


if __name__ == '__main__':
    main()
