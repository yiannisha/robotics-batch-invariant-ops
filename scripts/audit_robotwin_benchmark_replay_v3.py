"""Version 3: verify full replay, task callbacks, and source-bound Action records."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .qualify_simpler_fixed_action_replay import digest
else:
    from qualify_simpler_fixed_action_replay import digest


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_task_actions(task, directory, cache):
    def actions(value):
        if isinstance(value, dict):
            if value.get('type') == 'robotwin.Action':
                yield value
            for nested in value.values():
                yield from actions(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from actions(nested)
    count = 0
    for value in actions(task['fields']):
        source = Path(value['source']).resolve()
        require(source == Path(task['source']).resolve().parent / 'utils' / 'action.py',
                'Action source escapes released task utilities')
        expected = value['source_sha256']
        if str(source) not in cache:
            payload = source.read_bytes()
            require(hashlib.sha256(payload).hexdigest() == expected, 'Action source changed')
            archive = directory / 'audited_action_sources' / expected
            archive.parent.mkdir(exist_ok=True)
            if not archive.exists():archive.write_bytes(payload)
            require(sha(archive) == expected, 'Archived Action source differs')
            tree = ast.parse(payload.decode())
            classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Action']
            require(len(classes) == 1 and not classes[0].bases, 'Action definition differs')
            defaults = {}
            for n in classes[0].body:
                if isinstance(n, ast.AnnAssign) and n.value is not None:
                    require(isinstance(n.target, ast.Name), 'Uncovered Action class assignment')
                    defaults[n.target.id] = ast.literal_eval(n.value)
                elif isinstance(n, ast.Assign):
                    require(all(isinstance(t, ast.Name) for t in n.targets), 'Uncovered Action class assignment')
                    for t in n.targets:defaults[t.id] = ast.literal_eval(n.value)
            fields = {n.attr for n in ast.walk(classes[0]) if isinstance(n, ast.Attribute)
                      and isinstance(n.value, ast.Name) and n.value.id == 'self'
                      and isinstance(n.ctx, ast.Store)}
            cache[str(source)] = (expected, defaults, fields)
        identity, defaults, fields = cache[str(source)]
        require(identity == expected and digest(value['class_defaults']) == digest(defaults),
                'Action class defaults differ')
        instance = value['instance_fields']
        require(isinstance(instance, dict) and set(instance) <= fields and
                {'arm_tag', 'action', 'args'} <= set(instance), 'Action instance field coverage differs')
        require(set(instance) | set(defaults) == fields, 'Action effective field coverage differs')
        require(instance['action'] in ('move', 'gripper'), 'Unknown Action kind')
        arm = instance['arm_tag']
        require(isinstance(arm, dict) and arm == {'type': 'ArmTag', 'arm': arm.get('arm')}
                and arm['arm'] in ('left', 'right'), 'Action arm identity differs')
        require(isinstance(instance['args'], dict), 'Action arguments omitted')
        target = 'target_pose' if instance['action'] == 'move' else 'target_gripper_pos'
        require(target in instance and instance[target] is not None, 'Action target omitted')
        count += 1
    return count


def verify_task_callbacks(task, directory, cache):
    """Check stable callback identity against captured task and source ASTs."""
    def callbacks(value):
        if isinstance(value, dict):
            if value.get('type') == 'task_bound_method':
                yield value
            else:
                for nested in value.values():
                    yield from callbacks(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from callbacks(nested)
    task_path = Path(task['source']).resolve()
    owner = 'envs.' + task_path.stem + '.' + task_path.stem
    count = 0
    for value in callbacks(task['fields']):
        require(value['owner_class'] == owner, 'Callback owner class differs')
        require(Path(value['owner_source']).resolve() == task_path and
                value['owner_source_sha256'] == task['source_sha256'], 'Callback owner source differs')
        source = Path(value['function_source']).resolve()
        require(source.parent == task_path.parent and source.stem in (task_path.stem, '_base_task'),
                'Callback function source escapes task inheritance')
        expected = value['function_source_sha256']
        if str(source) not in cache:
            payload = source.read_bytes()
            require(hashlib.sha256(payload).hexdigest() == expected, 'Callback function source changed')
            archive = directory / 'audited_callback_sources' / expected
            archive.parent.mkdir(exist_ok=True)
            if not archive.exists():
                archive.write_bytes(payload)
            require(sha(archive) == expected, 'Archived callback source differs')
            cache[str(source)] = (expected, ast.parse(payload.decode()))
        identity, tree = cache[str(source)]
        require(identity == expected, 'Callback function hash differs between frames')
        require(value['function_module'] == 'envs.' + source.stem, 'Callback module differs')
        class_name, separator, name = value['function_qualname'].rpartition('.')
        require(separator and name in task['inventory']['referenced'], 'Callback selection is not source referenced')
        definitions = [method for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == class_name
                       for method in cls.body if isinstance(method, ast.FunctionDef) and method.name == name]
        require(len(definitions) == 1, 'Callback definition missing or ambiguous')
        fn = definitions[0]
        require(not fn.args.defaults and not any(x is not None for x in fn.args.kw_defaults)
                and value['defaults'] is None and value['kwdefaults'] is None,
                'Callback defaults need independent coverage')
        count += 1
    return count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--source-episodes',type=Path,required=True);args=p.parse_args()
    d=args.directory;runtime=json.loads((d/'runtime.json').read_text());selected=json.loads((d/'selected_start.json').read_text())
    require(runtime['seed_start']==100000 and runtime['instruction_pool_limit']==100 and runtime['capture_cuda_rng'],'Released seed/language/RNG protocol differs')
    require(sha(args.source_episodes)==runtime['source_episode_sha256'],'Retained controls changed')
    for path,expected in runtime['source_sha256'].items():require(sha(d/Path(path).name)==expected,'Archived execution source changed')
    source=next(json.loads(line) for line in args.source_episodes.read_text().splitlines() if json.loads(line)['condition']=='native_singleton')
    actions=np.asarray([a for c in source['calls'] for a in c['executed_actions']],dtype=np.float32)
    require(hashlib.sha256(actions.tobytes()).hexdigest()==runtime['source_action_bytes_sha256'],'Control bytes differ')
    candidates=[json.loads(line) for line in (d/'expert_seeds.jsonl').read_text().splitlines()]
    require([r['request']['seed'] for r in candidates]==list(range(100000,selected['seed']+1)),'Missing/reordered eligibility candidates')
    for i,row in enumerate(candidates):
        request,response=row['request'],row['response']
        require(request['task']==runtime['task'] and request['test_num']==100 and request['instruction_type']=='unseen','Eligibility protocol differs')
        eligible=response['ok'] and response['result']['eligible']
        require(eligible==(i==len(candidates)-1),'Did not choose first expert-eligible seed')
    canonical={k:v for k,v in selected['selection']['canonical'].items() if k!='path'}
    rows=[json.loads(line) for line in (d/'replays.jsonl').read_text().splitlines()]
    require(len(rows)==runtime['repeats'] and len(rows)>=2,'Missing repeats')
    frame_count=0
    expanded_task_frames=0
    callback_records=0;callback_sources={};action_records=0;action_sources={}
    for repeat,row in enumerate(rows):
        require(row['repeat']==repeat and row['seed']==selected['seed'] and row['trajectory'][0]==canonical,'Repeat seed/canonical state differs')
        controls=np.asarray(row['actions'],dtype=np.float32)
        expected=actions[np.minimum(np.arange(len(controls)),len(actions)-1)]
        require(controls.shape==expected.shape and controls.tobytes()==expected.tobytes(),'Executed controls differ from fixed source')
        trajectory=row['trajectory'];require(len(trajectory)==len(controls)+1==len(row['raw_frame_paths']),'Frame/control coverage differs')
        final=trajectory[-1]['controller']
        require(final['eval_success'] or final['take_action_count']==final['step_limit'],'Replay is incomplete')
        for step,(frame,path) in enumerate(zip(trajectory,row['raw_frame_paths'])):
            raw=torch.load(path,map_location='cpu',weights_only=False)
            for key,field in (('observation','observation_sha256'),('physics','physics_sha256'),('rng','rng_sha256'),('task_state','task_state_sha256')):
                require(digest(raw[key])==frame[field],f'Changed raw {key} at repeat{repeat}/step{step}')
            require(raw['controller']==frame['controller'] and raw['controller']['take_action_count']==step,'Raw controller/index differs')
            require('cuda' in raw['rng'] and 'cuda_initialized' in raw['rng'],'CUDA RNG state omitted')
            require(raw['controller']['instruction']==selected['selection']['instruction'],'Instruction differs')
            if runtime.get('capture_source_task_state'):
                task=raw['task_state']
                require(task.get('schema')=='source_defined_task_fields_v1','Missing expanded task state')
                task_source=Path(task['source'])
                require(sha(task_source)==task['source_sha256'],'Task source changed since capture')
                # Keep source bytes bound to the hash recorded by the worker.
                archive=d/'audited_task_sources'/task['source_sha256']
                if not archive.exists():
                    archive.parent.mkdir(exist_ok=True)
                    archive.write_bytes(task_source.read_bytes())
                require(sha(archive)==task['source_sha256'],'Archived task source changed')
                expected=set()
                for node in ast.walk(ast.parse(archive.read_text())):
                    if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='self' and isinstance(node.ctx,(ast.Store,ast.Del)):
                        expected.add(node.attr)
                    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('setattr','delattr') and node.args and isinstance(node.args[0],ast.Name) and node.args[0].id=='self':
                        require(len(node.args)>1 and isinstance(node.args[1],ast.Constant) and isinstance(node.args[1].value,str),'Dynamic task mutation is uncovered')
                        expected.add(node.args[1].value)
                require(set(task['fields'])==set(task['inventory']['assigned'])==expected,'Task field coverage differs from source assignments')
                require(all(isinstance(v.get('present'),bool) and ('value' in v)==v['present'] for v in task['fields'].values()),'Invalid field presence evidence')
                callback_records+=verify_task_callbacks(task,d,callback_sources)
                action_records+=verify_task_actions(task,d,action_sources)
                expanded_task_frames+=1
            frame_count+=1
        require(digest(trajectory)==digest(rows[0]['trajectory']),'Repeated raw trajectory differs')
    for path,(identity,tree) in callback_sources.items():require(sha(Path(path))==identity,'Callback source changed during audit')
    for path,(identity,defaults,fields) in action_sources.items():require(sha(Path(path))==identity,'Action source changed during audit')
    report=dict(status='complete_fixed_action_replay_independently_verified',task=runtime['task'],seed=selected['seed'],
                repeats=len(rows),controls_per_repeat=len(rows[0]['actions']),raw_frames_verified=frame_count,
                initial_state_and_full_trajectories_exact=True,cpu_and_cuda_rng_verified=True,
                source_defined_task_state_frames_verified=expanded_task_frames,
                task_owned_callback_records_verified=callback_records,callback_source_sha256={p:h for p,(h,_) in callback_sources.items()},
                task_action_records_verified=action_records,action_source_sha256={p:h for p,(h,_,_) in action_sources.items()},
                model_evaluated=False,full_50_task_qualification_complete=False,
                raw_replays_sha256=sha(d/'replays.jsonl'),auditor_sha256=sha(Path(__file__)))
    (d/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
