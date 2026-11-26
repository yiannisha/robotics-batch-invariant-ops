"""Prevent legacy pose-only replay reports from qualifying primary paired runs."""
CONTRACT = 'rigid_components_v2_entities_v4'


def require_rigid_replay_proof(proof):
    frames = proof.get('raw_frames_verified', 0)
    if not (proof.get('status') == 'complete_fixed_action_replay_independently_verified'
            and proof.get('repeats') == 3 and type(frames) is int and frames > 0
            and frames == 3 * (proof.get('controls_per_repeat', -1) + 1)
            and proof.get('initial_state_and_full_trajectories_exact') is True
            and proof.get('cpu_and_cuda_rng_verified') is True
            and proof.get('source_defined_task_state_frames_verified') == frames
            and proof.get('rigid_body_state_frames_verified') == frames
            and type(proof.get('entity_references_bound_to_physics')) is int
            and proof['entity_references_bound_to_physics'] >= 0):
        raise ValueError('Explicit rigid-body-state replay qualification required; legacy pose reports are insufficient')
