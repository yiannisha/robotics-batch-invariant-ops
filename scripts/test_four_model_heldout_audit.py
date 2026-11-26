"""Synthetic full-validator tests; no model, simulator or scientific acceptance."""
import contextlib
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.audit_four_model_heldout import main
from scripts.freeze_four_model_numerics import MODELS,required_arrangements


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


class HeldoutAudit(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.paths={key:self.root/(key+'.json') for key in ('manifest','freeze','disjointness-audit','runtime','output')}
        self.paths.update(records=self.root/'records.jsonl.gz',fidelity=self.root/'fidelity.jsonl.gz')
        self.model='lingbot_robotwin_exact';adapter,revision,steps,kinds=MODELS[self.model]
        config=dict(checkpoint_revision=revision,tf32=False)
        observations=[dict(split='heldout',task='task'+str(i),episode_id=str(i),observation_index=0,
            request_ids=[f'request-{i}-{j}' for j in range(3)],noise=[dict(sha256=hashlib.sha256(f'noise-{i}-{j}'.encode()).hexdigest()) for j in range(3)],
            image_sha256=hashlib.sha256(str(i).encode()).hexdigest(),companion_replay_indices=[j for j in range(8) if j!=i]) for i in range(8)]
        self.paths['manifest'].write_text(json.dumps(dict(model_id=self.model,observations=observations)))
        self.paths['disjointness-audit'].write_text(json.dumps(dict(status='complete_observation_disjoint_heldout_independently_verified',
            candidate_manifest_sha256=sha(self.paths['manifest']),identity_kind='prepared_neural_inputs_before_first_model_operation',prepared_input_disjointness_verified=True)))
        self.source=self.root/'model.py';self.source.write_text('# synthetic source fixture\n')
        bound={str(p.resolve()):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [self.paths['manifest'],self.paths['disjointness-audit'],self.source]}
        self.paths['freeze'].write_text(json.dumps(dict(model_id=self.model,content_addressed_freeze=bound,adapter_execution_configuration=config,
            prepared_input_disjointness_verified=True,invariant_diagnostic_exact=True,full_traced_operator_qualification_complete=True,large_batch_feasible={})))
        self.paths['runtime'].write_text(json.dumps(dict(adapter=adapter,split='heldout',execution_configuration=config,
            replay_manifest_sha256=sha(self.paths['manifest']),numerical_freeze=str(self.paths['freeze']),tf32_matmul=False,tf32_cudnn=False,
            source_sha256={str(self.source.resolve()):sha(self.source)})))
        tensor=dict(exact=True,nonfinite_count=0,unequal_element_count=0,reference_sha256='a'*64,candidate_sha256='a'*64)
        metric=dict(exact=True,nonfinite_count=0,unequal_element_count=0,max_absolute_error=0.,mean_absolute_error=0.,rmse=0.,relative_l2_error=0.,tensors={'output':tensor})
        self.rows=[];self.fidelity=[]
        for source in observations:
            for noise_id,request in enumerate(source['request_ids']):
                common=dict(request_id=request,task=source['task'],episode_id=source['episode_id'],observation_index=0,noise_id=noise_id,
                    request_noise_sha256=source['noise'][noise_id]['sha256'],source_image_sha256=source['image_sha256'],adapter_execution_configuration=config,
                    replay_manifest_sha256=sha(self.paths['manifest']),numerical_freeze_sha256=sha(self.paths['freeze']))
                self.fidelity.append(dict(common,comparison='batch_invariant_singleton_vs_native_singleton',output=metric))
                for mode in ('native','batch_invariant'):
                    for transform,size,position,kind in sorted(required_arrangements((1,2,4,8))):
                        index={'first':0,'middle':size//2,'last':size-1}[position]
                        companions=[observations[j]['request_ids'][noise_id] for j in source['companion_replay_indices'][:size-1]]
                        if kind=='duplicate':companions=[request]*(size-1)
                        if transform=='singleton':order=[request]
                        elif transform=='batch_position':order=list(companions);order.insert(index,request)
                        elif transform=='permutation':order=list(reversed([request,*companions]))
                        else:order=[request,*companions]
                        self.rows.append(dict(common,adapter=adapter,implementation=mode,transformation=transform,
                            logical_batch_size=size,batch_size=max(1,size//2) if transform=='partition' else size,
                            target_batch_position=position,target_batch_index=index,companion_type=kind,request_ordering=order,
                            companion_ids=[v for i,v in enumerate(order) if i!=index],tf32_matmul=False,tf32_cudnn=False,
                            output=metric,flow_steps={f'flow.step_{i:02d}.{kind}':metric for i in range(steps) for kind in kinds}))

    def run_audit(self):
        for name,rows in [('records',self.rows),('fidelity',self.fidelity)]:
            with gzip.open(self.paths[name],'wt') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
        argv=['audit','--model-id',self.model]
        for name,path in self.paths.items():argv+=['--'+name,str(path)]
        with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):main()

    def test_complete_synthetic_matrix_and_fidelity(self):
        self.run_audit();r=json.loads(self.paths['output'].read_text())
        self.assertEqual(r['request_noise_pairs'],24);self.assertEqual(r['arrangement_records'],1104)
        self.assertEqual(r['singleton_fidelity_records'],24)

    def test_changed_frozen_source_rejected(self):
        self.source.write_text('# changed\n')
        with self.assertRaisesRegex(ValueError,'Frozen input changed'):self.run_audit()
        self.assertFalse(self.paths['output'].exists())

    def test_indexed_noise_corruption_rejected(self):
        self.rows[0]['request_noise_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'Input/noise fingerprints'):self.run_audit()

    def test_missing_denoising_stage_rejected(self):
        self.rows[0]['flow_steps']=copy.copy(self.rows[0]['flow_steps']);self.rows[0]['flow_steps'].pop('flow.step_00.state')
        with self.assertRaisesRegex(ValueError,'Denoising stage missing'):self.run_audit()

    def test_missing_singleton_fidelity_rejected(self):
        self.fidelity.pop()
        with self.assertRaisesRegex(ValueError,'Incomplete heldout singleton fidelity'):self.run_audit()


if __name__=='__main__':unittest.main()
