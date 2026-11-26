"""Synthetic-only timing audit controls; no GPU or model results are produced."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import unittest
import zlib

from scripts.audit_isolated_operator_performance_v2 import audit, sha


def write(path, value):
    path.write_text(json.dumps(value) + '\n')


def desc(shape, stride, dtype='torch.float32'):
    return dict(shape=shape, stride=stride, dtype=dtype)


def signature(value):
    if isinstance(value, dict) and 'shape' in value:
        return dict(**value, logical_bytes_sha256='1' * 64)
    if isinstance(value, list):
        return [signature(x) for x in value]
    if isinstance(value, dict):
        return {k: signature(v) for k, v in value.items()}
    return value


class OperatorTimingAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-operator-timing-audit-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scripts = Path(__file__).resolve().parent
        self.plan_path = self.root / 'plan.json'
        self.directory = self.root / 'timing'
        self.directory.mkdir()
        cfg = dict(tf32=False)
        adapter = 'scripts.model_invariance.adapters.rdt_maniskill_exact'
        payload = b'\x01\x00' * 8
        cap = dict(data=base64.b64encode(zlib.compress(payload)).decode(), dtype='torch.bool',
                   encoding='zlib_base64_original_dtype_bytes_v1', original_dtype_bytes_count=len(payload),
                   original_dtype_bytes_sha256=hashlib.sha256(payload).hexdigest(),
                   values_shape=[8, 1, 1, 2], shape=[8, 1, 2, 2], stride=[2, 2, 0, 1])
        entries = [dict(aten_operation='aten::matmul', arguments=[desc([1, 2, 3], [6, 3, 1]), desc([1, 3, 2], [6, 2, 1])], keyword_arguments={}),
                   dict(aten_operation='aten::scaled_dot_product_attention',
                        arguments=[desc([8, 1, 2, 3], [6, 6, 3, 1])] * 3 + [desc([8, 1, 2, 2], [2, 2, 0, 1], 'torch.bool')],
                        keyword_arguments={}, attention_mask=cap)]
        sources, cases = {}, []
        for batch, entry in zip([1, 8], entries):
            p = self.root / f'trace_B{batch}.json'
            write(p, dict(adapter=adapter, adapter_execution_configuration=cfg, batch_size=batch, request_ids=['synthetic'], operations=[entry]))
            sources[str(p)] = sha(p)
            cases.append(dict(id=f'B{batch}_source_000', source_trace=str(p), source_trace_sha256=sha(p),
                              source_operation_index=0, logical_batch_size=batch, entry=entry,
                              modes=['native', 'batch_invariant', 'explicit_invariant_per_matrix'],
                              variants=['whole'] if batch == 1 else ['whole', 'attention_qk', 'attention_pv'],
                              mask_binding=None if batch == 1 else dict(capture=cap, selector=None)))
        coverage = self.root / 'coverage.json'
        write(coverage, dict(status='source_operator_inventory_reconciled', full_traced_operator_qualification_complete=True, source_traces=sources.copy()))
        sources[str(coverage)] = sha(coverage)
        self.plan = dict(model='rdt', adapter=adapter, adapter_execution_configuration=cfg, cases=cases,
                         source_sha256=sources, timing_protocol=dict(sessions=5, warmup_calls=20, timed_calls=200))
        write(self.plan_path, self.plan)
        diagnostic = self.root / 'invariance.jsonl.gz.runtime.json'
        write(diagnostic, dict(torch='synthetic-torch', cuda='synthetic-cuda', gpu='synthetic GPU', adapter=adapter))
        freeze = self.root / 'freeze.json'
        bound = {p: dict(sha256=h) for p, h in sources.items()}
        bound[str(diagnostic)] = dict(sha256=sha(diagnostic))
        write(freeze, dict(adapter_execution_configuration=cfg, content_addressed_freeze=bound))
        self.raw = self.directory / 'operator_timings.jsonl'
        self.rows = []
        samples = [.001 + i * 1e-7 for i in range(200)]
        tail = samples[189] + (samples[190] - samples[189]) * .05
        for session in range(5):
            for index, case in enumerate(cases):
                for variant_index, variant in enumerate(case['variants']):
                    shift = (index + session + variant_index) % 3
                    order = case['modes'][shift:] + case['modes'][:shift]
                    for position, mode in enumerate(order):
                        source_sig = signature([case['entry']['arguments'], {}])
                        if index == 1:
                            source_sig[0][3]['logical_bytes_sha256'] = hashlib.sha256(b'\x01\x00' * 16).hexdigest()
                        sig = copy.deepcopy(source_sig)
                        if variant == 'attention_qk':
                            sig = signature([[desc([8, 2, 3], [6, 3, 1]), desc([8, 3, 2], [6, 1, 3])], {}])
                        if variant == 'attention_pv':
                            sig = signature([[desc([8, 2, 2], [4, 2, 1]), desc([8, 2, 3], [6, 3, 1])], {}])
                        seed = int.from_bytes(hashlib.sha256(f"rdt/{case['id']}/{session}/20261109".encode()).digest()[:8], 'big') % (2 ** 63)
                        self.rows.append(dict(case_id=case['id'], session=session, variant=variant, mode=mode, mode_position=position, seed=seed,
                                          probabilities_seed=(seed + 1) % (2 ** 63) if variant == 'attention_pv' else None,
                                          plan_sha256=sha(self.plan_path), model='rdt', gpu_uuid='GPU-SYNTHETIC',
                                          timing_protocol=self.plan['timing_protocol'], source_trace_sha256=case['source_trace_sha256'],
                                          source_operation_index=0, operation=case['entry']['aten_operation'], logical_batch_size=case['logical_batch_size'],
                                          measured_operation=case['entry']['aten_operation'] if variant == 'whole' else 'aten::bmm',
                                          status='timed', same_case_output_repeat_exact=True, operand_signatures=sig, source_operand_signatures=source_sig,
                                          cuda_seconds=samples, synchronized_wall_seconds=samples,
                                          median_cuda_seconds=statistics.median(samples), p95_cuda_seconds=tail,
                                          median_synchronized_wall_seconds=statistics.median(samples), p95_synchronized_wall_seconds=tail,
                                          whole_operation_calls_per_second=1 / statistics.median(samples), kernel_launch_count=None,
                                          output_signature=dict(shape=[1], stride=[1], dtype='torch.float32', logical_bytes_sha256='2' * 64),
                                          peak_incremental_allocated_bytes=64))
        worker_sources = [self.scripts / n for n in ('benchmark_qualified_operator_cases.py', 'operator_performance_operands.py', 'build_operator_performance_plans.py')]
        worker_sources += [self.scripts / 'benchmark_qualified_operator_cases_v2.py']
        worker_sources += [self.scripts.parent / 'batch_invariant_ops' / n for n in ('__init__.py', 'batch_invariant_ops.py')]
        isolation_sources = worker_sources[:3] + [self.scripts / n for n in ('run_isolated_operator_benchmark.py', 'run_isolated_operator_benchmark_v2.py', 'benchmark_qualified_operator_cases_v2.py', 'gpu_isolation.py', 'run_isolated_frozen_benchmark.py')]
        def archive(name, paths):
            target = self.root / name
            target.mkdir()
            pins = {str(p): sha(p) for p in paths}
            for p in paths:
                (target / pins[str(p)]).write_bytes(p.read_bytes())
            return pins, str(target)
        pins, folder = archive('worker_sources', worker_sources)
        self.runtime = dict(status='completed_raw_operator_timings_independent_acceptance_pending', model='rdt', gpu_uuid='GPU-SYNTHETIC',
                            plan_sha256=sha(self.plan_path), adapter_execution_configuration=cfg, tf32=False, autocast=False, cuda_events_initialized_before_timing=True,
                            timing_protocol=self.plan['timing_protocol'], source_sha256=pins, source_archive=folder,
                            torch='synthetic-torch', cuda='synthetic-cuda', gpu_name='synthetic GPU')
        pins, folder = archive('isolation_sources', isolation_sources)
        self.isolation = dict(accepted=True, child_exit_code=0, status='completed_operator_timing_isolation_verified_independent_audit_pending',
                              plan_sha256=sha(self.plan_path), gpu_uuid='GPU-SYNTHETIC', raw=str(self.raw),
                              freeze=str(freeze), freeze_sha256=sha(freeze), source_sha256=pins, source_archive=folder,
                              benchmark_pid=123, observation_count=3)
        self.journal = self.directory / 'gpu_observations.jsonl'
        self.observations = [dict(gpu_uuid='GPU-SYNTHETIC', gpu_name='synthetic GPU', processes=processes)
                             for processes in [[], [dict(pid=123)], []]]
        self.flush()

    def flush(self):
        self.raw.write_text(''.join(json.dumps(r) + '\n' for r in self.rows))
        self.runtime.update(raw_sha256=sha(self.raw), raw_records=len(self.rows))
        runtime = Path(str(self.raw) + '.runtime.json')
        write(runtime, self.runtime)
        self.journal.write_text(''.join(json.dumps(r) + '\n' for r in self.observations))
        self.isolation.update(raw_sha256=sha(self.raw), runtime_sha256=sha(runtime),
                              observation_journal=str(self.journal), observation_journal_sha256=sha(self.journal))
        write(self.directory / 'isolation.json', self.isolation)

    def test_positive_complete_synthetic_factorial(self):
        result = audit(self.plan_path, self.directory)
        self.assertEqual(result['timing_slots'], 60)
        self.assertTrue(result['all_planned_modes_timed'])

    def test_unprimed_cuda_event_protocol_rejected(self):
        self.runtime['cuda_events_initialized_before_timing'] = False
        self.flush()
        with self.assertRaises(ValueError): audit(self.plan_path, self.directory)

    def test_rehashed_wrong_median_rejected(self):
        self.rows[0]['median_cuda_seconds'] = 9.
        self.flush()
        with self.assertRaises(ValueError): audit(self.plan_path, self.directory)

    def test_rehashed_missing_slot_rejected(self):
        self.rows.pop()
        self.flush()
        with self.assertRaises(ValueError): audit(self.plan_path, self.directory)

    def test_duplicate_slot_rejected(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        self.flush()
        with self.assertRaises(ValueError): audit(self.plan_path, self.directory)

    def test_source_stride_change_rejected(self):
        self.rows[0]['operand_signatures'][0][0]['stride'][0] += 1
        self.flush()
        with self.assertRaises(ValueError): audit(self.plan_path, self.directory)

    def test_mask_byte_change_rejected(self):
        row = next(r for r in self.rows if r['logical_batch_size'] == 8)
        row['source_operand_signatures'][0][3]['logical_bytes_sha256'] = '0' * 64
        self.flush()
        with self.assertRaises(ValueError): audit(self.plan_path, self.directory)

    def test_foreign_gpu_process_rejected(self):
        self.observations[1]['processes'].append(dict(pid=999))
        self.flush()
        with self.assertRaises(ValueError): audit(self.plan_path, self.directory)

    def test_missing_source_inventory_rejected(self):
        self.runtime['source_sha256'].pop(str(self.scripts / 'operator_performance_operands.py'))
        self.flush()
        with self.assertRaises(ValueError): audit(self.plan_path, self.directory)

    def test_capacity_failure_retained_without_fake_timings(self):
        row = self.rows[0]
        for key in list(row):
            if key in ('cuda_seconds', 'synchronized_wall_seconds', 'median_cuda_seconds', 'p95_cuda_seconds',
                       'median_synchronized_wall_seconds', 'p95_synchronized_wall_seconds', 'output_signature',
                       'whole_operation_calls_per_second', 'kernel_launch_count', 'peak_incremental_allocated_bytes',
                       'same_case_output_repeat_exact'):
                row.pop(key)
        row.update(status='cuda_capacity_failure', phase='operator_execution', error='synthetic CUDA OOM fixture')
        self.flush()
        result = audit(self.plan_path, self.directory)
        self.assertFalse(result['all_planned_modes_timed'])
        self.assertEqual(result['statuses']['cuda_capacity_failure'], 1)

    def test_rehashed_shrunken_qualified_case_roster_rejected(self):
        removed = self.plan['cases'].pop()
        self.rows = [r for r in self.rows if r['case_id'] != removed['id']]
        write(self.plan_path, self.plan)
        value = sha(self.plan_path)
        self.runtime['plan_sha256'] = self.isolation['plan_sha256'] = value
        for row in self.rows:
            row['plan_sha256'] = value
        self.flush()
        with self.assertRaises(ValueError): audit(self.plan_path, self.directory)

    def test_missing_attention_matrix_components_rejected(self):
        self.plan['cases'][1]['variants'] = ['whole']
        self.rows = [r for r in self.rows if r['variant'] == 'whole']
        write(self.plan_path, self.plan)
        value = sha(self.plan_path)
        self.runtime['plan_sha256'] = self.isolation['plan_sha256'] = value
        for row in self.rows:
            row['plan_sha256'] = value
        self.flush()
        with self.assertRaises(ValueError): audit(self.plan_path, self.directory)

    def test_component_stride_corruption_rejected(self):
        row = next(r for r in self.rows if r['variant'] == 'attention_qk')
        row['operand_signatures'][0][1]['stride'][-1] += 1
        self.flush()
        with self.assertRaises(ValueError): audit(self.plan_path, self.directory)


if __name__ == '__main__':
    unittest.main()
