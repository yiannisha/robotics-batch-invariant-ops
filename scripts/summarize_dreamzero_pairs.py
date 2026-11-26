"""Audit preliminary native DreamZero pairs from raw actions and scene states."""
import argparse
import json
from pathlib import Path
import numpy as np


def difference(a,b):
    a,b = np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    d = np.abs(a-b)
    return {'exact_equal':bool(np.array_equal(a,b)), 'max_abs':float(d.max()),
            'mean_abs':float(d.mean()), 'different_elements':int(np.count_nonzero(d))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    root=args.directory
    fixed=json.loads((root/'fixed_input.json').read_text())
    names=['native_singleton','native_dynamic','native_singleton_repeat','native_dynamic_repeat']
    trajectories={name:json.loads((root/f'{name}.json').read_text()) for name in names}
    reference=fixed[names[0]]['meta']
    result={'scope':'one scene, one initial state, batch sizes 1 and 2, preliminary native inference',
            'normalized_target_inputs_identical':all(x['meta']['normalized_target_sha256']==reference['normalized_target_sha256'] for x in fixed.values()),
            'target_noise_identical':all([n['target_sha256'] for n in x['meta']['noise']]==[n['target_sha256'] for n in reference['noise']] for x in fixed.values()),
            'initial_states_identical':len({x['initial_state_sha256'] for x in trajectories.values()})==1,
            'comparisons':{}}
    for label,a,b in [('batch_effect',names[0],names[1]),('singleton_repeat',names[0],names[2]),('dynamic_repeat',names[1],names[3])]:
        ta,tb=trajectories[a]['trajectory'],trajectories[b]['trajectory']
        assert len(ta)==len(tb)>0
        state_keys=list(ta[0]['state'])
        states={key:difference([x['state'][key] for x in ta],[x['state'][key] for x in tb]) for key in state_keys}
        result['comparisons'][label]={'fixed_input_actions':difference(fixed[a]['actions'],fixed[b]['actions']),
            'executed_actions':difference([x['action'] for x in ta],[x['action'] for x in tb]),
            'first_chunk_actions':difference([x['action'] for x in ta[:8]],[x['action'] for x in tb[:8]]),
            'first_chunk_arm_joint_position':difference(
                [x['state']['articulation/robot/joint_position'][0][:7] for x in ta[:8]],
                [x['state']['articulation/robot/joint_position'][0][:7] for x in tb[:8]]),
            'scene_state':states,
            'first_different_action_step':next((i for i,(x,y) in enumerate(zip(ta,tb)) if x['action']!=y['action']),None)}
    result['controls_pass']=all(result[k] for k in ['normalized_target_inputs_identical','target_noise_identical','initial_states_identical'])
    comparisons=result['comparisons']
    repeat=max(comparisons[k]['fixed_input_actions']['max_abs'] for k in ['singleton_repeat','dynamic_repeat'])
    result['fixed_batch_effect_exceeds_repeat']=comparisons['batch_effect']['fixed_input_actions']['max_abs']>repeat
    joint_key='articulation/robot/joint_position'
    repeat_joint=max(comparisons[k]['scene_state'][joint_key]['max_abs'] for k in ['singleton_repeat','dynamic_repeat'])
    result['robot_joint_effect_exceeds_repeat']=comparisons['batch_effect']['scene_state'][joint_key]['max_abs']>repeat_joint
    result['first_chunk_repeat_trajectories_exact']=all(comparisons[k]['first_chunk_arm_joint_position']['exact_equal'] for k in ['singleton_repeat','dynamic_repeat'])
    result['preliminary_batch_dependent_behavior_observed']=bool(result['controls_pass'] and result['fixed_batch_effect_exceeds_repeat'] and result['first_chunk_repeat_trajectories_exact'] and not comparisons['batch_effect']['first_chunk_arm_joint_position']['exact_equal'])
    (root/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
