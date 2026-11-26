"""Snapshot-SIMPLER CogACT corpus adapter with a declared fixed input extent."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from scripts.model_invariance.adapters import cogact_small as implementation


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class CogACTSimplerExactAdapter(implementation.CogACTSmallAdapter):
    name='cogact-small-simpler-fixed64-controlled-inputs'

    def __init__(self):
        super().__init__()
        self.replay_manifest_path=Path(os.environ.get('COGACT_SIMPLER_EXACT_MANIFEST',''))

    def _validate_paths(self):
        for path in (self.checkpoint/'checkpoints/CogACT-Small.pt',self.checkpoint/'config.json',self.checkpoint/'dataset_statistics.json',
                     self.source/'vla/cogactvla.py',self.openvla_source/'prismatic/models/vlms/prismatic.py',
                     self.llama_metadata/'config.json',self.llama_metadata/'tokenizer.model'):
            if not path.is_file():raise FileNotFoundError(path)
        return self.checkpoint/'checkpoints/CogACT-Small.pt'

    def load_model(self):
        if not self.replay_manifest_path.is_file():raise FileNotFoundError(self.replay_manifest_path)
        self.replay_manifest=json.loads(self.replay_manifest_path.read_text())
        if self.replay_manifest.get('model_id')!='cogact_simpler_exact' or len(self.replay_manifest['observations'])!=1000:
            raise ValueError('Expected complete snapshot-SIMPLER corpus')
        audit=Path(self.replay_manifest['corpus_audit_path'])
        if sha(audit)!=self.replay_manifest['corpus_audit_sha256'] or json.loads(audit.read_text())['status']!='complete_corpus_independently_verified':
            raise ValueError('Corpus audit is missing or changed')
        self.replay_noise=np.load(self.replay_manifest['noise_file'],mmap_mode='r',allow_pickle=False)
        if self.replay_noise.shape!=(1000,3,1,16,7):raise ValueError('Noise geometry differs')
        implementation.CANONICAL_LANGUAGE_TOKENS=64
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        return super().load_model()

    @lru_cache(maxsize=64)
    def _image(self,index):
        row=self.replay_manifest['observations'][index]
        path=Path(self.replay_manifest['fixture_root'])/row['image_file']
        if sha(path)!=row['image_sha256']:raise ValueError('Raw image changed')
        with Image.open(path) as handle:return handle.convert('RGB').copy()

    def load_replay_input(self,replay_index,noise_id=0):
        if self.replay_manifest is None:raise RuntimeError('load_model must precede replay requests')
        if replay_index not in range(1000) or noise_id not in (0,1,2):raise IndexError((replay_index,noise_id))
        row=self.replay_manifest['observations'][replay_index]
        value=np.array(self.replay_noise[replay_index,noise_id],copy=True)
        if hashlib.sha256(value.tobytes()).hexdigest()!=row['noise'][noise_id]['sha256']:raise ValueError('Request noise changed')
        return dict(image=self._image(replay_index),instruction=row['instruction'],noise=torch.from_numpy(value))

    def load_example_inputs(self,count):
        rows=[r for r in self.replay_manifest['observations'] if r['split']=='diagnostic']
        rows.sort(key=lambda r:(r['observation_index'],r['episode_id'],r['task']))
        if count>len(rows):raise ValueError('Insufficient diagnostic examples')
        return [self.load_replay_input(r['replay_index'],0) for r in rows[:count]]

    compose_batch=staticmethod(implementation.CogACTSmallAdapter.compose_fixed_batch)
    compose_fixed_batch=compose_batch
    compose_released_batch=staticmethod(implementation.CogACTSmallAdapter.compose_batch)

    def run_model(self,policy,batch):
        callback=self.trace_callback;normalized={}
        def capture(label,value):
            if label=='actions.normalized':normalized['value']=value.detach().clone()
            if callback is not None:
                callback(label,value)
                if label.startswith('diffusion.'):
                    parts=label.split('.',2)
                    if len(parts)==3 and parts[1].isdigit():
                        # DDIM carries [conditional B, unconditional B]. Keep
                        # both branches under each logical request so generic
                        # singleton/target slicing uses the same leading axis.
                        if value.shape[0]%2:raise ValueError('Invalid classifier-free branch count')
                        per_request=torch.stack(value.chunk(2,dim=0),dim=1)
                        callback(f'flow.step_{int(parts[1]):02d}.{parts[2]}',per_request)
        self.trace_callback=capture
        try:physical=super().run_model(policy,batch)
        finally:self.trace_callback=callback
        if 'value' not in normalized:raise RuntimeError('Normalized diffusion output missing')
        return dict(normalized=normalized['value'],physical=physical.to(self.device))

    @staticmethod
    def execution_configuration():
        return dict(checkpoint_revision='ca302a300542393a0b06e291465076a7d9105c23',suite='simpler_google_four',
                    language_padding='fixed_right_in_both_primary_modes',language_tokens=64,
                    precision='vlm_bfloat16_diffusion_float32',tf32=False,chunk_size=16,action_dim=7,
                    diffusion_steps=implementation.DDIM_STEPS,cfg_scale=implementation.CFG_SCALE,execution='eager',
                    released_default_fidelity='separate_longest_padding_singleton_measurement_required')

    @staticmethod
    def fixed_batch_shape_metadata(batch):
        return dict(language_padding='fixed_right_in_both_primary_modes',language_tokens=64,canonical_length=64)


adapter=CogACTSimplerExactAdapter()
