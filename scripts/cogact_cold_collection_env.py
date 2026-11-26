"""Legacy collection API backed by two qualified cold SIMPLER processes.

`restore_snapshot` here recreates the canonical snapshot in a fresh process and
checks every expanded state hash. The protocol is declared in returned metadata.
"""
import gzip
import io
from pathlib import Path

import torch

from scripts.cogact_simpler_state_worker import RemoteStateEnv
from scripts.qualify_simpler_fixed_action_replay import digest

PROTOCOL='fresh_seeded_worker_reconstruction_exact_canonical_snapshot_comparison'


class ColdCollectionEnv:
    def __init__(self,environment,python,simpler_root,reference_worker,vulkan_icd,output):
        self.environment=environment;self.python=python;self.simpler_root=simpler_root
        self.reference_worker=reference_worker;self.vulkan_icd=vulkan_icd;self.output=Path(output)
        self.remote=None;self.sequence=0
    def spawn(self):
        self.close();logs=self.output/'cold_workers';logs.mkdir(parents=True,exist_ok=True)
        log=logs/f'{self.environment}-{self.seed}-{self.sequence}.log';self.sequence+=1
        self.remote=RemoteStateEnv(self.environment,self.python,self.simpler_root,self.reference_worker,self.vulkan_icd,log,process_seed=self.seed)
        self.remote.request(dict(command='reset',seed=self.seed,options=self.options))
    def retain(self,observable):
        raw=observable['raw'];identity=digest(raw);path=self.output/'raw_frames'/(identity+'.pt.gz');path.parent.mkdir(exist_ok=True)
        if not path.exists():
            buffer=io.BytesIO();torch.save(raw,buffer);temporary=path.with_suffix('.tmp')
            temporary.write_bytes(gzip.compress(buffer.getvalue(),compresslevel=1,mtime=0));temporary.replace(path)
        self.current=observable;self.current_raw_frame=str(path.relative_to(self.output))
    def reset(self,seed,options):
        self.seed=seed;self.options=options;self.spawn()
    def capture_snapshot(self):
        observable=self.remote.request(dict(command='capture_snapshot'))['observable'];self.retain(observable)
        self.canonical=dict(observable['hashes'])
        return dict(reset_protocol=PROTOCOL,canonical_hashes=self.canonical,canonical_raw_frame=self.current_raw_frame)
    def restore_snapshot(self):
        self.spawn();observable=self.remote.request(dict(command='capture_snapshot'))['observable'];self.retain(observable)
        checks={key:observable['hashes'].get(key)==value for key,value in self.canonical.items()}
        if set(observable['hashes'])!=set(self.canonical) or not all(checks.values()):
            raise RuntimeError('Fresh-worker initial physics/controller/task/RNG/observations differ')
        return dict(reset_protocol=PROTOCOL,observation=observable['raw']['observation'],checks=checks,
            reconstructed_raw_frame=self.current_raw_frame)
    def get_language_instruction(self):return self.current['raw']['instruction']
    def step(self,action):
        result=self.remote.request(dict(command='step_with_state',action=action.tolist()));self.retain(result['observable'])
        return result['observation'],result['reward'],result['success'],result['truncated'],result['info']
    def close(self):
        if self.remote is not None:self.remote.close();self.remote=None
