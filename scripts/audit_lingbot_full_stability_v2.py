"""Independently rederive full LingBot action and dual-arm drift magnitudes.

Consumes the episode-byte-bound full paired proof. Does not import the magnitude
reporter or its numerical/drift helpers, construct a model or perform inference.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def close(value, expected, name, *, atol=1e-14):
    check(math.isfinite(value) and math.isclose(value, expected, rel_tol=1e-9, abs_tol=atol),
          'Incorrect ' + name)


def arm_distances(left, right):
    """Relative-quaternion product rather than the reporter's chord formula."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    check(left.shape == right.shape and left.ndim == 3 and left.shape[1:] == (2, 7), 'Invalid arm pose arrays')
    check(np.isfinite(left).all() and np.isfinite(right).all(), 'Nonfinite pose')
    positions = np.sqrt(np.sum((left[:, :, :3] - right[:, :, :3]) ** 2, axis=-1))
    q, r = left[:, :, 3:], right[:, :, 3:]
    qnorm, rnorm = np.sqrt(np.sum(q * q, axis=-1, keepdims=True)), np.sqrt(np.sum(r * r, axis=-1, keepdims=True))
    check(np.all(qnorm > 0) and np.all(rnorm > 0), 'Zero quaternion')
    q, r = q / qnorm, r / rnorm
    scalar = np.sum(q * r, axis=-1)
    vector = q[:, :, :1] * r[:, :, 1:] - r[:, :, :1] * q[:, :, 1:] - np.cross(q[:, :, 1:], r[:, :, 1:])
    rotations = 2 * np.arctan2(np.sqrt(np.sum(vector * vector, axis=-1)), np.abs(scalar))
    return positions, rotations


def fixed_errors(left, right):
    x, y = np.asarray(left['normalized_actions'], dtype=np.float32), np.asarray(right['normalized_actions'], dtype=np.float32)
    check(x.shape == y.shape == (50, 55) and np.isfinite(x).all() and np.isfinite(y).all(), 'Invalid normalized actions')
    for call, value in ((left, x), (right, y)):
        check(hashlib.sha256(value.tobytes()).hexdigest() == call['normalized_sha256'], 'Normalized fingerprint differs')
    a, b = x.view(np.uint32), y.view(np.uint32)
    delta = x.astype(np.float64) - y.astype(np.float64)
    error = np.abs(delta)
    ordered_a = np.where(a >= 2 ** 31, (2 ** 32 - 1) - a.astype(np.int64), a.astype(np.int64) + 2 ** 31)
    ordered_b = np.where(b >= 2 ** 31, (2 ** 32 - 1) - b.astype(np.int64), b.astype(np.int64) + 2 ** 31)
    return dict(exact=bool(np.array_equal(a, b)), elements=x.size, different_elements=int(np.count_nonzero(a != b)),
                max_absolute_error=float(error.max()), squared_error_sum=float(np.sum(delta * delta)),
                rms_error=float(np.sqrt(np.sum(delta * delta) / x.size)),
                max_symmetric_relative_error=float((error / np.maximum(np.maximum(np.abs(x.astype(np.float64)),
                                                                                 np.abs(y.astype(np.float64))),
                                                                        np.finfo(np.float64).tiny)).max()),
                max_ulp=int(np.abs(ordered_a - ordered_b).max()), source_float_bits=32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), 'Preserve existing audit')
    torch.set_num_threads(8)
    report_hash = sha(args.report)
    report = json.loads(args.report.read_text())
    check(report['model'] == 'lingbot' and report['primary_endpoint'] == 'singleton_vs_batched_stability',
          'Foreign magnitude report')
    paired_path = Path(report['paired_audit'])
    paired_hash = sha(paired_path)
    check(paired_hash == report['paired_audit_sha256'], 'Paired proof changed')
    proof = json.loads(paired_path.read_text())
    check(proof['status'] == 'lingbot_paired_raw_inputs_actions_and_observable_trajectories_independently_verified'
          and proof['episode_files_bound_to_verified_bytes'] is True and proof['cohort_rollouts_complete']
          and proof['completed_paired_starts'] == 2500 and proof['episodes'] == 20000
          and proof['all_completed_blocks_pass_reset_AA_and_invariant_trajectory_checks'], 'Incomplete bound paired acceptance')
    check(len(proof['episode_file_sha256']) == 20000, 'Incomplete episode byte bindings')
    check(proof['auditor_sha256'] == sha(Path(__file__).with_name('audit_lingbot_paired_study_bound_v2.py')),
          'Bound reconstruction auditor source differs')
    directory = args.directory.resolve()
    lock_path = directory / 'candidate_lock.json'
    lock_hash = sha(lock_path)
    check(lock_hash == proof['candidate_lock_sha256'] == report['candidate_lock_sha256'], 'Candidate differs')
    lock = json.loads(lock_path.read_text())
    check(lock.get('runtime',{}).get('adapter')=='scripts.model_invariance.adapters.lingbot_robotwin_exact_v2','Corrected paired adapter required')
    check(lock['runtime'].get('tf32_matmul') is False and lock['runtime'].get('tf32_cudnn') is False,'Corrected paired precision required')
    tasks = lock['global_tasks']
    check(len(tasks) == len(set(tasks)) == 50 and lock['initialization_ids'] == list(range(50)), 'Foreign task roster')
    roster = {(task, initial) for task in tasks for initial in range(50)}
    verified = {(b['task'], b['initialization']): b for b in proof['blocks']}
    starts = {(s['task'], s['initialization']): s for s in report['starts']}
    check(verified.keys() == starts.keys() == roster and len(report['starts']) == len(proof['blocks']) == 2500,
          'Missing/duplicate start')
    for path, identity in report['helper_sha256'].items():
        check(sha(path) == identity, 'Reporter helper changed')
    reporter = Path(__file__).with_name('summarize_lingbot_full_paired_stability.py')
    check(sha(reporter) == report['source_sha256'], 'Magnitude reporter changed')
    frame_hashes, episode_files, fixed_count = {}, {}, 0
    for (task, initial), start in sorted(starts.items()):
        folder = directory / 'blocks' / task / f'start-{initial:02d}'
        check(folder.resolve().is_relative_to(directory / 'blocks'), 'Escaping episode path')
        episodes = {}
        for condition in ('native_singleton', 'native_dynamic', 'patched_singleton', 'patched_dynamic'):
            check(start['same_condition_repeat_controls_exact'].get(condition) is True, 'Missing exact repeat control')
            for suffix in ('', '_repeat'):
                name = condition + suffix
                path = folder / (name + '.json.gz')
                identity = sha(path)
                check(identity == proof['episode_file_sha256'].get(str(path.resolve()))
                      == start['episode_file_sha256'].get(name), 'Episode bytes differ')
                episode_files[str(path.resolve())] = identity
                if not suffix:
                    with gzip.open(path, 'rt') as stream:
                        episodes[condition] = json.load(stream)
        cache = {}
        def pose_array(episode, count):
            result = []
            for frame in episode['trajectory'][:count]:
                path = (directory / frame['raw_frame']).resolve()
                check(path.parent == directory / 'frames', 'Escaping frame path')
                if path not in cache:
                    identity = sha(path)
                    check(identity == proof['raw_files_sha256'].get(str(path))
                          == report['raw_frame_sha256'].get(str(path)), 'Unaccepted raw pose frame')
                    with gzip.open(path, 'rb') as stream:
                        value = torch.load(stream, map_location='cpu', weights_only=False)
                    endpose = value['observation']['endpose']
                    cache[path] = np.asarray([endpose['left_endpose'], endpose['right_endpose']], dtype=np.float64)
                    frame_hashes[str(path)] = identity
                result.append(cache[path])
            check(len(result) == count, 'Missing pose frames')
            return np.asarray(result)
        for mode, prefix in [('native', 'native'), ('invariant', 'patched')]:
            left, right = episodes[prefix + '_singleton'], episodes[prefix + '_dynamic']
            measurement = start['comparisons'][mode]
            n = min(len(left['steps']), len(right['steps']))
            x = np.asarray([s['action'] for s in left['steps'][:n]], dtype=np.float32)
            y = np.asarray([s['action'] for s in right['steps'][:n]], dtype=np.float32)
            check(n > 0 and x.shape == y.shape == (n, 14), 'Invalid control alignment')
            different = np.any(x.view(np.uint32) != y.view(np.uint32), axis=1)
            indices = np.flatnonzero(different)
            first = int(indices[0]) if len(indices) else None
            check(measurement['singleton_controls'] == len(left['steps'])
                  and measurement['batched_controls'] == len(right['steps'])
                  and measurement['aligned_controls'] == n and measurement['common_frames'] == n + 1
                  and measurement['horizons_equal'] == (len(left['steps']) == len(right['steps'])), 'Wrong horizons')
            check(measurement['action_disagreement_controls'] == int(different.sum())
                  and measurement['first_action_divergence_control_index'] == first, 'Wrong physical divergence')
            delta = x.astype(np.float64) - y.astype(np.float64)
            for actual, expected in [(measurement['action_component_max_absolute_difference'], np.abs(delta).max(axis=0)),
                                     (measurement['action_component_rms_difference'], np.sqrt(np.mean(delta * delta, axis=0)))]:
                check(np.asarray(actual).shape == (14,) and np.allclose(actual, expected, rtol=1e-10, atol=1e-14),
                      'Physical component magnitudes differ')
            if mode == 'native':
                check(first == verified[task, initial]['native_first_action_divergence'], 'Audited divergence differs')
            positions, rotations = arm_distances(pose_array(left, n + 1), pose_array(right, n + 1))
            for arm, label in enumerate(('left', 'right')):
                values = measurement['arms'][label]
                for key, expected in [('position_max_m', positions[:, arm].max()),
                                      ('position_rms_m', np.sqrt(np.mean(positions[:, arm] ** 2))),
                                      ('position_last_common_m', positions[-1, arm]),
                                      ('rotation_max_rad', rotations[:, arm].max()),
                                      ('rotation_rms_rad', np.sqrt(np.mean(rotations[:, arm] ** 2)))]:
                    close(values[key], float(expected), label + ' ' + key, atol=1e-12)
            close(measurement['eef_position_max_m'], float(positions.max()), 'maximum arm position drift')
            expected_fixed, b1, feedback = [], 0, 0
            for a, b in zip(left['calls'], right['calls']):
                check(a['call'] == b['call'], 'Calls misaligned')
                if a['target_input_sha256'] != b['target_input_sha256']:
                    feedback += 1
                    continue
                check(a['noise_id'] == b['noise_id'], 'Indexed noise differs')
                if b['batch_size'] == 1:
                    b1 += 1
                    continue
                expected_fixed.append((a, b))
            check(len(expected_fixed) == len(measurement['fixed_identical_input_nonsingleton_pairs'])
                  and measurement['identical_input_B1_controls'] == b1
                  and measurement['later_calls_with_different_inputs'] == feedback
                  and measurement['fixed_input_batching_error_observed'] == bool(expected_fixed)
                  and measurement['batched_policy_calls'] == sum(b['batch_size'] > 1 for b in right['calls']),
                  'Fixed-input selection differs')
            for (a, b), actual in zip(expected_fixed, measurement['fixed_identical_input_nonsingleton_pairs']):
                check(actual['call'] == a['call'] and actual['batch_size'] == b['batch_size']
                      and actual['target_position'] == b['target_position']
                      and actual['input_sha256'] == a['target_input_sha256'], 'Fixed-input metadata differs')
                expected = fixed_errors(a, b)
                for key, value in expected.items():
                    if isinstance(value, (bool, int)):
                        check(actual[key] == value, 'Incorrect fixed-input ' + key)
                    else:
                        close(actual[key], value, 'fixed-input ' + key)
                fixed_count += 1
            if mode == 'invariant':
                check(first is None and measurement['horizons_equal'] and np.all(positions == 0)
                      and np.all(rotations == 0)
                      and all(p['exact'] for p in measurement['fixed_identical_input_nonsingleton_pairs']),
                      'Invariant drift')
    check(frame_hashes == report['raw_frame_sha256'], 'Raw pose provenance inventory differs')
    check(episode_files == proof['episode_file_sha256'], 'Episode byte inventory differs')
    check(sha(args.report) == report_hash and sha(paired_path) == paired_hash and sha(lock_path) == lock_hash,
          'Source evidence changed')
    result = dict(status='full_lingbot_dual_arm_stability_magnitudes_independently_verified',
                  report_sha256=report_hash, paired_proof_sha256=paired_hash, candidate_lock_sha256=lock_hash,
                  paired_starts_verified=2500, episode_files_verified=len(episode_files),
                  raw_pose_frame_files_verified=len(frame_hashes), fixed_input_pairs_verified=fixed_count,
                  auditor_sha256=sha(Path(__file__)), full_campaign_complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('Verified full2500-start LingBot dual-arm magnitudes', flush=True)


if __name__ == '__main__':
    main()
