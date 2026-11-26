"""Accepted reset evidence can be reused without mutating its byte bindings."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.qualify_lingbot_robotwin_reset_roster_v2 import reusable_entries, main, sha, HELPERS
from scripts.lingbot_rigid_qualification import CONTRACT


class ResetReuse(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.helpers=self.root/'helpers';self.helpers.mkdir()
        for name in HELPERS:(self.helpers/name).write_text('pinned '+name)
        self.controls=self.root/'controls.jsonl';self.controls.write_text('retained controls\n')

    def entry(self,task):
        directory=self.root/task;directory.mkdir();(directory/'replays.jsonl').write_text('retained replay bytes\n')
        runtime=dict(seed_start=100000,capture_source_task_state=True,capture_cuda_rng=True,
                     source_episode_sha256=sha(self.controls),source_sha256={str(self.helpers/'robotwin_exact_worker.py'):sha(self.helpers/'robotwin_exact_worker.py')})
        (directory/'runtime.json').write_text(json.dumps(runtime))
        (directory/'robotwin_exact_worker.py').write_bytes((self.helpers/'robotwin_exact_worker.py').read_bytes())
        proof=dict(status='complete_fixed_action_replay_independently_verified',task=task,seed=100000,
                   repeats=3,controls_per_repeat=1,raw_frames_verified=6,initial_state_and_full_trajectories_exact=True,
                   cpu_and_cuda_rng_verified=True,source_defined_task_state_frames_verified=6,
                   rigid_body_state_frames_verified=6,entity_references_bound_to_physics=0,
                   raw_replays_sha256=sha(directory/'replays.jsonl'),auditor_sha256=sha(self.helpers/'audit_robotwin_benchmark_replay.py'))
        (directory/'independent_audit.json').write_text(json.dumps(proof))
        return dict(directory=str(directory),independent_audit_sha256=sha(directory/'independent_audit.json'),
                    raw_replays_sha256=proof['raw_replays_sha256'])

    def record(self):
        return dict(physics_capture_contract=CONTRACT,prescribed_tasks=['task'],qualified_tasks={'task':self.entry('task')})

    def test_valid_reuse_is_read_only(self):
        record=self.record();before={str(p):sha(p) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(set(reusable_entries(record,['task'],sha(self.controls))),{'task'})
        self.assertEqual(before,{str(p):sha(p) for p in self.root.rglob('*') if p.is_file()})

    def test_changed_reports_replays_or_sources_are_rejected(self):
        record=self.record();directory=Path(record['qualified_tasks']['task']['directory'])
        for path in [directory/'independent_audit.json',directory/'replays.jsonl',directory/'robotwin_exact_worker.py',
                     self.helpers/'robotwin_exact_worker.py',self.helpers/'audit_robotwin_benchmark_replay.py']:
            payload=path.read_bytes();path.write_bytes(payload+b' ')
            with self.subTest(path=path.name),self.assertRaises((ValueError,KeyError)):
                reusable_entries(record,['task'],sha(self.controls))
            path.write_bytes(payload)

    def test_legacy_or_wrong_control_contract_is_rejected(self):
        record=self.record();changed=copy.deepcopy(record);changed['physics_capture_contract']='legacy'
        with self.assertRaises(ValueError):reusable_entries(changed,['task'],sha(self.controls))
        with self.assertRaises(ValueError):reusable_entries(record,['task'],'different')
        path=Path(record['qualified_tasks']['task']['directory'])/'independent_audit.json'
        proof=json.loads(path.read_text());del proof['rigid_body_state_frames_verified'];path.write_text(json.dumps(proof))
        record['qualified_tasks']['task']['independent_audit_sha256']=sha(path)
        with self.assertRaises(ValueError):reusable_entries(record,['task'],sha(self.controls))

    def test_full_reuse_never_launches_or_rewrites_audits(self):
        tasks=['task_'+str(i) for i in range(50)]
        record=dict(physics_capture_contract=CONTRACT,prescribed_tasks=tasks,qualified_tasks={t:self.entry(t) for t in tasks})
        prior=self.root/'prior.json';prior.write_text(json.dumps(record));roster=self.root/'roster.json';roster.write_text(json.dumps({'prescribed_tasks':tasks}));out=self.root/'output'
        hashes={t:sha(Path(e['directory'])/'independent_audit.json') for t,e in record['qualified_tasks'].items()}
        args=['test','--source',str(self.root),'--python','python','--source-episodes',str(self.controls),
              '--task-roster',str(roster),'--output',str(out),'--helper-directory',str(self.helpers),
              '--reuse-qualification-union',str(prior)]
        with patch('sys.argv',args),patch('subprocess.run',side_effect=AssertionError('Reused audits must not be rerun')),patch('builtins.print'):
            main()
        result=json.loads((out/'qualification_union.json').read_text());self.assertEqual(result['qualified_task_count'],50)
        self.assertEqual(hashes,{t:sha(Path(e['directory'])/'independent_audit.json') for t,e in record['qualified_tasks'].items()})


if __name__=='__main__':unittest.main()
