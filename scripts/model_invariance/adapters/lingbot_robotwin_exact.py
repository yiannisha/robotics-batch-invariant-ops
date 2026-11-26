"""Audited RoboTwin campaign replay adapter for the pinned LingBot checkpoint."""
import gzip
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

from scripts.model_invariance.adapters.lingbot_vla2 import LingBotVLA2Adapter


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


class LingBotRobotwinExactAdapter(LingBotVLA2Adapter):
    def __init__(self):
        super().__init__()
        self.replay_manifest_path=Path(os.environ.get('LINGBOT_ROBOTWIN_EXACT_MANIFEST','missing-lingbot-campaign-manifest'))
        self.source=Path(os.environ.get('LINGBOT_VLA2_SOURCE','missing-lingbot-source'))

    def _validate_paths(self):
        for path in (self.checkpoint/'model.safetensors.index.json',self.checkpoint.parents[2]/'lingbotvla_cli.yaml',self.source/'deploy/lingbot_vla_v2_policy.py'):
            if not path.is_file():raise FileNotFoundError(path)

    def _load_replay_metadata(self):
        if self.replay_manifest is not None:return
        m=json.loads(self.replay_manifest_path.read_text())
        if m['model_id']!='lingbot_robotwin_exact':raise ValueError('Foreign replay corpus')
        ap=Path(m['corpus_audit_path'])
        if sha(ap)!=m['corpus_audit_sha256']:raise ValueError('Corpus audit changed')
        a=json.loads(ap.read_text())
        if a['status']!='native_corpus_completed_episode_inputs_and_actions_independently_rebuilt' or not a['all_prescribed_native_episodes_rebuilt'] or a['completed_native_episodes']!=200:raise ValueError('Complete independent native corpus reconstruction required')
        if sha(Path(m['source_manifest_snapshot']))!=m['source_manifest_sha256'] or m['source_manifest_sha256']!=a['manifest_snapshot_sha256']:raise ValueError('Source corpus changed')
        if len(m['observations'])!=a['selected_inputs_rebuilt']:raise ValueError('Source observation coverage differs')
        original=json.loads(Path(m['source_manifest_snapshot']).read_text())
        self._source_episode_hashes={e['source_episode_file']:e['sha256'] for e in original['episodes']}
        for row,source in zip(m['observations'],original['observations']):
            if any(row.get(key)!=value for key,value in source.items()):raise ValueError('Numerical input row differs from audited source')
            path=(Path(m['fixture_root'])/row['request_file']).resolve()
            if a['raw_files_sha256'].get(str(path))!=row['request_sha256']:raise ValueError('Request file was not independently audited')
        noise_path=Path(m['noise_file'])
        if sha(noise_path)!=m['noise_sha256']:raise ValueError('Replay noise changed')
        noise=np.load(noise_path,mmap_mode='r',allow_pickle=False)
        if list(noise.shape)!=[len(m['observations']),3,50,55] or noise.dtype!=np.float32:raise ValueError('Noise geometry differs')
        self.replay_manifest=m;self.replay_noise=noise

    def load_model(self):
        self._load_replay_metadata();self._validate_paths();sys.path.insert(0,str(self.source))
        acquisition_path=self.checkpoint.parents[2]/'acquisition.json';a=json.loads(acquisition_path.read_text())
        if a['revision']!='0451855729ec904f970600e0aec8b84661423afe':raise ValueError('Foreign checkpoint revision')
        for row in a['files']:
            if sha(self.checkpoint.parents[2]/row['path'])!=row['sha256']:raise ValueError('Checkpoint asset changed')
        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
        from safetensors import safe_open
        original=LingbotVLAv2Server.load_model_weights;dtypes={}
        def observe(server,*args,**kwargs):
            result=original(server,*args,**kwargs);dtypes.update({k:v.dtype for k,v in server.vla.state_dict().items()});return result
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        with patch.object(LingbotVLAv2Server,'load_model_weights',observe):model=super().load_model()
        state=self.server.vla.state_dict();seen=set();elements=0
        for path in sorted(self.checkpoint.glob('*.safetensors')):
            with safe_open(path,framework='pt',device='cpu') as shard:
                for name in shard.keys():
                    if name in seen or name not in state:raise ValueError('Foreign/duplicate checkpoint tensor')
                    actual=state[name].detach().cpu().contiguous();expected=shard.get_tensor(name).to(dtypes[name]).to(actual.dtype).contiguous()
                    if not torch.equal(actual.reshape(-1).view(torch.uint8),expected.reshape(-1).view(torch.uint8)):raise ValueError('Official dtype load differs: '+name)
                    seen.add(name);elements+=actual.numel()
        if seen!=set(state) or len(seen)!=1708 or elements!=6375907511:raise ValueError('Incomplete checkpoint tensor coverage')
        self.startup_weight_audit=dict(tensors=len(seen),elements=elements,all_byte_exact_after_official_dtype_sequence=True)
        return model

    def load_replay_input(self,replay_index,noise_id=0):
        self._load_replay_metadata()
        if replay_index not in range(len(self.replay_manifest['observations'])) or noise_id not in (0,1,2):raise IndexError((replay_index,noise_id))
        row=self.replay_manifest['observations'][replay_index]
        if row['replay_index']!=replay_index:raise ValueError('Replay index differs')
        root=Path(self.replay_manifest['fixture_root']).resolve();path=(root/row['request_file']).resolve()
        if not path.is_relative_to(root) or sha(path)!=row['request_sha256']:raise ValueError('Prepared request changed')
        with gzip.open(path,'rb') as stream:values=torch.load(stream,map_location='cpu',weights_only=False)
        if set(values)!=set(self.model_keys):raise ValueError('Prepared model input keys differ')
        for key,value in values.items():
            signature=dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
            if signature!=row['input_signatures'][key]:raise ValueError('Prepared tensor differs: '+key)
        noise=np.array(self.replay_noise[replay_index,noise_id],copy=True)
        if hashlib.sha256(noise.tobytes()).hexdigest()!=row['noise'][noise_id]['sha256']:raise ValueError('Indexed noise differs')
        if self.server is None:raise RuntimeError('Load the official model/transform server before replay inputs')
        frame_path=(root/row['raw_frame']).resolve()
        if not frame_path.is_relative_to(root):raise ValueError('Raw frame escapes source corpus')
        with gzip.open(frame_path,'rb') as stream:raw=torch.load(stream,map_location='cpu',weights_only=False)
        from scripts.qualify_simpler_fixed_action_replay import digest
        if digest(raw)!=frame_path.name.split('.')[0]:raise ValueError('Raw source frame content address differs')
        obs=raw['observation']
        transformed=self.server._prepare_model_input(dict(task=self._episode_instruction(root,row),
            **{'observation.state':obs['joint_action']['vector'],
               'observation.images.cam_high':obs['observation']['head_camera']['rgb'],
               'observation.images.cam_left_wrist':obs['observation']['left_camera']['rgb'],
               'observation.images.cam_right_wrist':obs['observation']['right_camera']['rgb']}))
        for key,value in values.items():
            actual=transformed[key].detach().cpu().contiguous()
            if actual.dtype!=value.dtype or actual.shape!=value.shape or not torch.equal(actual.reshape(-1).view(torch.uint8),value.contiguous().reshape(-1).view(torch.uint8)):
                raise ValueError('Official action-transform context changes prepared model input: '+key)
        return dict(values,noise=torch.from_numpy(noise),transform_context=transformed)

    def _episode_instruction(self,root,row):
        path=(root/row['source_episode_file']).resolve()
        if not path.is_relative_to(root):raise ValueError('Episode path escapes source corpus')
        if sha(path)!=self._source_episode_hashes[row['source_episode_file']]:raise ValueError('Source episode changed')
        with gzip.open(path,'rt') as stream:episode=json.load(stream)
        return episode['instruction']

    @staticmethod
    def compose_batch(samples):
        return dict(LingBotVLA2Adapter.compose_batch(samples),transform_contexts=[sample['transform_context'] for sample in samples])

    compose_fixed_batch=compose_batch

    def physical_actions(self,contexts,normalized):
        # Feature transforms may mutate dictionaries/tensors; each inference
        # receives a fresh context, including duplicate-request arrangements.
        physical=self.server._unapply_batched_actions([copy.deepcopy(context) for context in contexts],normalized)
        value=np.asarray(physical['action'],dtype=np.float32)
        if value.shape!=(len(contexts),50,14) or not np.isfinite(value).all():raise ValueError('Invalid released physical action output')
        return torch.from_numpy(value.copy()).to(self.device)

    def load_example_inputs(self,count):
        self._load_replay_metadata();rows=[r for r in self.replay_manifest['observations'] if r['split']=='diagnostic']
        if count>len(rows):raise ValueError('Insufficient diagnostic inputs')
        return [self.load_replay_input(r['replay_index'],0) for r in rows[:count]]

    def execution_configuration(self):
        return dict(checkpoint_revision='0451855729ec904f970600e0aec8b84661423afe',suite='robotwin_released_50',
            precision='official_constructor_dtype_then_fp32',flow_steps=10,action_chunk_size=50,model_action_dim=55,
            language_tokens=72,deterministic_moe='deterministic_token_choice_moe_shared_native_invariant',tf32=False,
            released_atomic_moe_baseline='separate_nonrepeatable_supplement',vision_shape_cache='cleared_each_inference',
            output_domains=['normalized_55','official_physical_14'])

    def run_model(self,model,batch):
        self.reset_model(model)
        normalized=super().run_model(model,batch)
        physical=self.physical_actions(batch['transform_contexts'],normalized)
        normalized=normalized.to(self.device)
        self._trace('actions.normalized',normalized);self._trace('actions.physical',physical)
        return dict(normalized=normalized,physical=physical)


adapter=LingBotRobotwinExactAdapter()
