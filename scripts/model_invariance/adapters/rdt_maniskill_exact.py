"""Audited ManiSkill corpus adapter including genuinely batched SigLIP encoding."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.model_invariance.adapters.rdt_maniskill import RDTManiSkillAdapter, MANISKILL_INDICES
from scripts.run_rdt_paired_server import PairedRDTBackend, verify_artifacts


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class RDTManiSkillExactAdapter:
    name='rdt-1b-maniskill-entire-policy-real-sim-corpus'

    def __init__(self):
        self.replay_manifest_path=Path(os.environ.get('RDT_MANISKILL_EXACT_MANIFEST',''))
        self.trace_callback=None
        self.inner=None
        self.backend=None
        self.manifest=None
        self.noise=None

    def load_model(self):
        if not self.replay_manifest_path.is_file():raise FileNotFoundError(self.replay_manifest_path)
        self.manifest=json.loads(self.replay_manifest_path.read_text())
        if self.manifest.get('model_id')!='rdt_maniskill_exact' or len(self.manifest['observations'])!=1000:
            raise ValueError('Expected audited complete RDT corpus')
        audit=Path(self.manifest['corpus_audit_path'])
        if sha(audit)!=self.manifest['corpus_audit_sha256'] or json.loads(audit.read_text())['status']!='complete_corpus_independently_verified':
            raise ValueError('Corpus audit is missing or changed')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        class Inner(RDTManiSkillAdapter):
            def _validate_paths(self):
                for path in (self.checkpoint/'rdt/mp_rank_00_model_states.pt',self.siglip/'model.safetensors',self.source/'configs/base.yaml'):
                    if not path.is_file():raise FileNotFoundError(path)
        self.inner=Inner();policy=self.inner.load_model()
        self.backend=PairedRDTBackend(self.inner,policy,condition='native_dynamic',checkpoint=self.inner.checkpoint,
            language=self.inner.language_embeddings,artifacts=verify_artifacts(self.inner.checkpoint,self.inner.siglip),
            fixed_language_tokens=30,maximum_batch=64)
        self.noise=np.load(self.manifest['noise_file'],mmap_mode='r',allow_pickle=False)
        if self.noise.shape!=(1000,3,64,128):raise ValueError('Noise shape differs')
        return policy

    @lru_cache(maxsize=64)
    def _request(self,index):
        row=self.manifest['observations'][index]
        path=Path(self.manifest['fixture_root'])/row['request_file']
        if sha(path)!=row['request_sha256']:raise ValueError('Request bytes changed')
        return torch.load(path,map_location='cpu',weights_only=False)

    def load_replay_input(self,replay_index,noise_id=0):
        if self.manifest is None:raise RuntimeError('load_model must precede requests')
        if replay_index not in range(1000) or noise_id not in (0,1,2):raise IndexError((replay_index,noise_id))
        row=self.manifest['observations'][replay_index]
        value=np.array(self.noise[replay_index,noise_id],copy=True)
        if hashlib.sha256(value.tobytes()).hexdigest()!=row['noise'][noise_id]['sha256']:raise ValueError('Request noise changed')
        return dict(self._request(replay_index),noise=value,paired_request_id=row['request_ids'][noise_id])

    def load_example_inputs(self,count):
        rows=[r for r in self.manifest['observations'] if r['split']=='diagnostic']
        rows.sort(key=lambda r:(r['observation_index'],r['episode_id'],r['task']))
        if count>len(rows):raise ValueError('Insufficient diagnostic examples')
        return [self.load_replay_input(r['replay_index'],0) for r in rows[:count]]

    @staticmethod
    def compose_batch(samples):return list(samples)
    compose_fixed_batch=compose_batch

    @staticmethod
    def reset_model(policy):RDTManiSkillAdapter.reset_model(policy)

    def run_model(self,policy,batch):
        normalized={}
        def trace(label,value):
            if label=='diffusion.4.sample':normalized['value']=value[:,:,list(MANISKILL_INDICES)].detach().clone()
            if self.trace_callback is not None:
                self.trace_callback(label,value)
                if label.startswith('diffusion.'):
                    _,step,kind=label.split('.',2)
                    self.trace_callback(f'flow.step_{int(step):02d}.{kind}',value)
        self.inner.trace_callback=trace
        def pixels(module,inputs):
            if self.trace_callback is not None:
                value=inputs[0]
                self.trace_callback('input.vision_pixels',value.reshape(len(batch),6,*value.shape[1:]))
        hook=policy.vision.register_forward_pre_hook(pixels)
        # Slot names satisfy the serving protocol even for duplicate requests;
        # names do not enter preprocessing, model tensors, or indexed noise.
        examples=[dict(request,paired_request_id=f'slot-{i}:{request["paired_request_id"]}') for i,request in enumerate(batch)]
        try:
            physical=self.backend.infer({'examples':examples})['actions']
        finally:
            hook.remove();self.inner.trace_callback=None
        if 'value' not in normalized:raise RuntimeError('Final diffusion trace missing')
        return dict(normalized=normalized['value'],physical=torch.from_numpy(physical).to(self.inner.device))

    @staticmethod
    def extract_robot_output(output):return output

    @staticmethod
    def execution_configuration():
        return dict(checkpoint_revision='9622afab2b7ce2312a6cf1febc526928589b77eb',suite='maniskill_five',
                    language_padding='fixed_right_in_both_modes',language_tokens=30,image_history=2,camera_slots=3,
                    vision_encoding='genuinely_batched_six_slots_per_request',diffusion_steps=5,chunk_size=64,
                    precision='bfloat16',tf32=False,control_frequency=25,action_dim=8,noise_dim=128,
                    sampler='official_maniskill_wrapper_equivalent_DPM_solver',execution='eager')

    @staticmethod
    def fixed_batch_shape_metadata(batch):
        return dict(language_padding='fixed_right_in_both_modes',language_tokens=30,canonical_length=30)


adapter=RDTManiSkillExactAdapter()
