import gzip
import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_diagnostic_batch_stability import accumulator,add,expected_arrangements,summarize,table


def metric(size=2,error=1.):
    return dict(shape=[size],dtype='torch.float32',nonfinite_count=0,rmse=error,max_absolute_error=error,
        unequal_element_count=size if error else 0,exact=not bool(error),reference_sha256='a'*64,
        candidate_sha256=('b' if error else 'a')*64)


class DiagnosticSummaryTests(unittest.TestCase):
    def test_mse_weights_elements_without_calling_it_centered_variance(self):
        value=accumulator();add(value,metric(2,1.));add(value,metric(8,3.))
        row=table({('native',):value},['mode'])[0]
        self.assertEqual(row['mean_squared_singleton_batch_difference'],7.4)
        self.assertEqual(row['elements'],10)

    def test_incomplete_request_excluded_and_corrupt_gzip_rejected(self):
        runtime=dict(split='diagnostic',batch_sizes=[1,2],modes=['native','batch_invariant'],
            adapter='test',replay_manifest_sha256='manifest',execution_configuration={},tf32_matmul=False,tf32_cudnn=False)
        rows=[]
        for mode in runtime['modes']:
            for transform,size,position,companions in sorted(expected_arrangements([1,2])):
                m=metric(error=0. if mode=='batch_invariant' else 1.)
                rows.append(dict(implementation=mode,transformation=transform,logical_batch_size=size,
                    batch_size=1 if transform=='partition' else size,target_batch_position=position,companion_type=companions,
                    request_id='complete',task='task',request_noise_sha256='noise',adapter='test',
                    replay_manifest_sha256='manifest',adapter_execution_configuration={},tf32_matmul=False,tf32_cudnn=False,
                    output={'tensors':{'output.normalized':m}},flow_steps={'flow.step_00.sample':{'tensors':{'output':m}}}))
        rows.append({**rows[0],'request_id':'incomplete'})
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'records.jsonl.gz';path.write_bytes(gzip.compress(''.join(json.dumps(r)+'\n' for r in rows).encode()))
            Path(str(path)+'.runtime.json').write_text(json.dumps(runtime))
            result=summarize(path)
            self.assertEqual(result['included_complete_request_noise_pairs'],1)
            self.assertEqual(result['excluded_partial_requests'][0]['request_id'],'incomplete')
            self.assertTrue(all(r['nonexact_comparisons']==0 for r in result['normalized_by_task_batch'] if r['implementation']=='batch_invariant'))
            corrupt=bytearray(path.read_bytes());corrupt[-8]^=1;path.write_bytes(corrupt)
            with self.assertRaises(gzip.BadGzipFile):summarize(path)


if __name__=='__main__':unittest.main()
