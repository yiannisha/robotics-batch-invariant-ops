"""Extract dual-arm stability from the independently audited full LingBot cohort.

No pilot/partial-cohort substitution and no neural inference. The resulting
magnitudes require their own independent recount before acceptance.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_lingbot_campaign_corpus import content_hash
from scripts.measure_lingbot_pilot_stability import dual_arm_poses, dual_arm_drift
from scripts.qualify_simpler_fixed_action_replay import digest
from scripts.summarize_paired_stability import numerical_pair

CONDITIONS = ('native_singleton', 'native_dynamic', 'patched_singleton', 'patched_dynamic')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def require_complete_proof(proof, lock):
    require(proof['status'] == 'lingbot_paired_raw_inputs_actions_and_observable_trajectories_independently_verified',
            'Complete independent LingBot proof required')
    require(proof['cohort_rollouts_complete'] and proof['completed_paired_starts'] == 2500
            and proof['episodes'] == 20000
            and proof['all_completed_blocks_pass_reset_AA_and_invariant_trajectory_checks'],
            'Complete2500-start/20000-episode cohort required')
    require(proof.get('episode_files_bound_to_verified_bytes') is True
            and len(proof.get('episode_file_sha256', {})) == 20000,
            'Independent episode-byte binding required')
    tasks = lock['global_tasks']
    require(len(tasks) == len(set(tasks)) == 50 and lock['initialization_ids'] == list(range(50)),
            'Wrong task/start roster')
    observed = {(b['task'], b['initialization']) for b in proof['blocks']}
    require(len(observed) == len(proof['blocks']) and observed == {(t, i) for t in tasks for i in range(50)},
            'Missing, duplicate or foreign independently audited start')


def compare_actions(left, right):
    count = min(len(left['steps']), len(right['steps']))
    require(count > 0, 'Empty control horizon')
    x = np.asarray([s['action'] for s in left['steps'][:count]], dtype=np.float32)
    y = np.asarray([s['action'] for s in right['steps'][:count]], dtype=np.float32)
    require(x.shape == y.shape == (count, 14) and np.isfinite(x).all() and np.isfinite(y).all(),
            'Invalid physical action geometry')
    different = np.any(x.view(np.uint32) != y.view(np.uint32), axis=1)
    delta = x.astype(np.float64) - y.astype(np.float64)
    fixed, b1, feedback = [], 0, 0
    for a, b in zip(left['calls'], right['calls']):
        require(a['call'] == b['call'], 'Policy call alignment differs')
        if a['target_input_sha256'] != b['target_input_sha256']:
            feedback += 1
            continue
        require(a['noise_id'] == b['noise_id'], 'Equal-input comparison has different noise')
        if b['batch_size'] == 1:
            b1 += 1
            continue
        fixed.append(dict(call=a['call'], batch_size=b['batch_size'], target_position=b['target_position'],
                          input_sha256=a['target_input_sha256'], **numerical_pair(a, b)))
    indices = np.flatnonzero(different)
    return dict(singleton_controls=len(left['steps']), batched_controls=len(right['steps']),
                aligned_controls=count, horizons_equal=len(left['steps']) == len(right['steps']),
                action_disagreement_controls=int(different.sum()),
                first_action_divergence_control_index=int(indices[0]) if len(indices) else None,
                action_component_max_absolute_difference=np.abs(delta).max(axis=0).tolist(),
                action_component_rms_difference=np.sqrt(np.mean(delta ** 2, axis=0)).tolist(),
                fixed_identical_input_nonsingleton_pairs=fixed, identical_input_B1_controls=b1,
                later_calls_with_different_inputs=feedback,
                batched_policy_calls=sum(call['batch_size'] > 1 for call in right['calls']),
                fixed_input_batching_error_observed=bool(fixed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('directory', 'paired-audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve prior magnitudes')
    torch.set_num_threads(8)
    directory = args.directory.resolve()
    lock_path = directory / 'candidate_lock.json'
    proof_hash, lock_hash = sha(args.paired_audit), sha(lock_path)
    proof, lock = json.loads(args.paired_audit.read_text()), json.loads(lock_path.read_text())
    require_complete_proof(proof, lock)
    require(proof['candidate_lock_sha256'] == lock_hash, 'Candidate identity changed')
    raw_hashes, starts = {}, []
    for block in proof['blocks']:
        task, initial = block['task'], block['initialization']
        folder = directory / 'blocks' / task / f'start-{initial:02d}'
        require(folder.resolve().is_relative_to(directory / 'blocks'), 'Episode path escapes study')
        episodes, episode_hashes, checks = {}, {}, {}
        for condition in CONDITIONS:
            for suffix in ('', '_repeat'):
                name = condition + suffix
                path = folder / (name + '.json.gz')
                episode_hashes[name] = sha(path)
                require(proof['episode_file_sha256'].get(str(path.resolve())) == episode_hashes[name],
                        'Episode differs from independently audited bytes')
                with gzip.open(path, 'rt') as stream:
                    episode = json.load(stream)
                require(episode['task'] == task and episode['initialization'] == initial
                        and episode['candidate_lock_sha256'] == lock_hash, 'Foreign episode')
                require(episode['success'] == block['outcomes'][condition], 'Audited outcome changed')
                episodes[name] = episode
            checks[condition] = all(digest(episodes[condition][key]) == digest(episodes[condition + '_repeat'][key])
                                    for key in ('calls', 'steps', 'trajectory', 'success'))
        require(all(checks.values()), 'Same-condition repeats differ')
        require(len({digest(e['trajectory'][0]) for e in episodes.values()}) == 1, 'Reset state differs')
        cache = {}
        def poses(episode, count):
            require(len(episode['trajectory']) == len(episode['steps']) + 1, 'Trajectory/control alignment differs')
            result = []
            for frame in episode['trajectory'][:count]:
                path = (directory / frame['raw_frame']).resolve()
                require(path.parent == directory / 'frames', 'Raw frame path escapes study')
                if path not in cache:
                    identity = sha(path)
                    require(proof['raw_files_sha256'].get(str(path)) == identity, 'Raw frame not independently accepted')
                    with gzip.open(path, 'rb') as stream:
                        raw = torch.load(stream, map_location='cpu', weights_only=False)
                    require(content_hash(raw) == path.name.split('.')[0], 'Raw frame content address changed')
                    require(digest(raw['observation']) == frame['observation_sha256'], 'Raw observation changed')
                    cache[path] = dual_arm_poses(raw)
                    raw_hashes[str(path)] = identity
                result.append(cache[path])
            require(len(result) == count, 'Missing common frames')
            return result
        start = dict(task=task, initialization=initial, seed=block['seed'],
                     same_condition_repeat_controls_exact=checks, episode_file_sha256=episode_hashes, comparisons={})
        for mode, prefix in [('native', 'native'), ('invariant', 'patched')]:
            left, right = episodes[prefix + '_singleton'], episodes[prefix + '_dynamic']
            measurement = compare_actions(left, right)
            common_frames = measurement['aligned_controls'] + 1
            drift = dual_arm_drift(poses(left, common_frames), poses(right, common_frames))
            measurement.update(common_frames=common_frames, arms=drift['arms'],
                               eef_position_max_m=max(v['position_max_m'] for v in drift['arms'].values()),
                               eef_position_scalar_definition='Maximum of the two separately reported arm position peaks')
            if mode == 'native':
                require(measurement['first_action_divergence_control_index'] == block['native_first_action_divergence'],
                        'First action divergence differs from independent paired proof')
            else:
                require(measurement['horizons_equal'] and measurement['action_disagreement_controls'] == 0
                        and all(pair['exact'] for pair in measurement['fixed_identical_input_nonsingleton_pairs'])
                        and all(value == 0 for arm in drift['arms'].values() for value in arm.values()),
                        'Invariant action or dual-arm trajectory drift')
            start['comparisons'][mode] = measurement
        starts.append(start)
        print(f'Measured LingBot {task}/{initial}: both arms; independent magnitude recount pending', flush=True)
    require(sha(args.paired_audit) == proof_hash and sha(lock_path) == lock_hash, 'Source proof changed')
    result = dict(primary_endpoint='singleton_vs_batched_stability', model='lingbot',
                  scope='Complete50-task/2500-start independently audited paired cohort',
                  paired_audit=str(args.paired_audit.resolve()), paired_audit_sha256=proof_hash,
                  candidate_lock_sha256=lock_hash, source_sha256=sha(Path(__file__)),
                  helper_sha256={str(Path(__file__).with_name(name).resolve()): sha(Path(__file__).with_name(name))
                                 for name in ('measure_lingbot_pilot_stability.py', 'summarize_paired_stability.py',
                                              'audit_lingbot_campaign_corpus.py', 'qualify_simpler_fixed_action_replay.py')},
                  physical_drift_scope='Separate left/right EEF position and rotation; common horizon only; no extrapolation',
                  numerical_scope='Identical prepared input plus indexed noise only; post-feedback calls excluded',
                  raw_frame_sha256=raw_hashes, starts=starts,
                  independent_magnitude_audit_pending=True, full_campaign_complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
