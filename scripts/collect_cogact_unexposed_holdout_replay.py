"""Collect100 fresh native CogACT episodes using qualified cold reconstruction.

Four tasks x25 fresh starts50..74 x10 pre-action observations. Starts50..54
are unused diagnostic candidates;55..74 prospective heldout. No BI inference. Both collection and later controlled comparisons
use fixed64 language tokens; released-default fidelity is a separate measure.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from qualify_simpler_fixed_action_replay import digest

TASKS=('pick_can','move_near','open_drawer','close_drawer')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','source','openvla-source','llama-metadata','pizero-project','simpler-root','simpler-python','vulkan-icd','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--initialization-start',type=int,default=50)
    p.add_argument('--reset-audit',type=Path,required=True)
    args=p.parse_args()
    if args.initialization_start<50:raise ValueError('Fresh starts must exclude paired0..49')
    acquisition=json.loads((args.checkpoint/'acquisition.json').read_text())
    if acquisition['revision']!='ca302a300542393a0b06e291465076a7d9105c23':raise ValueError('Foreign checkpoint revision')
    assets={}
    for row in acquisition['files']:
        file=args.checkpoint/row['path']
        if sha(file)!=row['sha256']:raise ValueError('Changed checkpoint asset')
        assets[str(file.resolve())]=row['sha256']
    for file in args.llama_metadata.rglob('*'):
        if file.is_file() and '.cache' not in file.parts:assets[str(file.resolve())]=sha(file)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    os.environ.update(COGACT_CHECKPOINT=str(args.checkpoint),COGACT_SOURCE=str(args.source),
                      COGACT_OPENVLA_SOURCE=str(args.openvla_source),COGACT_LLAMA_METADATA=str(args.llama_metadata))
    from batch_invariant_ops import set_batch_invariant_mode
    from scripts.model_invariance.adapters import cogact_small as implementation
    from scripts.run_cogact_paired_server import RequestState
    class Adapter(implementation.CogACTSmallAdapter):
        def _validate_paths(self):
            for path in (self.checkpoint/'checkpoints/CogACT-Small.pt',self.checkpoint/'config.json',self.source/'vla/cogactvla.py',
                         self.openvla_source/'prismatic/models/vlms/prismatic.py',self.llama_metadata/'config.json',self.llama_metadata/'tokenizer.model'):
                if not path.is_file():raise FileNotFoundError(path)
            return self.checkpoint/'checkpoints/CogACT-Small.pt'
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    implementation.CANONICAL_LANGUAGE_TOKENS=64
    adapter=Adapter();policy=adapter.load_model()
    from sim_cogact.adaptive_ensemble import AdaptiveEnsembler
    sys.path.insert(0,str(args.pizero_project))
    from experiments.simpler_support import observation_state
    from scripts.cogact_cold_collection_env import ColdCollectionEnv,PROTOCOL
    reset=json.loads(args.reset_audit.read_text())
    if reset['status']!='complete_observable_state_fixed_replay_independently_verified' or not reset['packed_scene_physics_controller_task_rng_observations_exact']:raise ValueError('Cold reset qualification missing')
    reset_runtime=json.loads((args.reset_audit.parent/'runtime.json').read_text())
    for path,identity in reset_runtime['source_sha256'].items():
        if sha(Path(path))!=identity:raise ValueError('Qualified simulator source changed')
    def make_simpler_env(environment,python,simpler_root,vulkan):
        return ColdCollectionEnv(environment,python,simpler_root,args.pizero_project/'experiments/simpler_worker.py',vulkan,args.output)
    tree=ast.parse((args.pizero_project/'experiments/prepare_replay.py').read_text())
    mapping=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TASKS' for t in n.targets))
    runtime=dict(scope='controlled_native_replay_collection',tasks=TASKS,initializations_per_task=25,observations_per_episode=10,
        diagnostic_initializations=list(range(args.initialization_start,args.initialization_start+5)),heldout_initializations=list(range(args.initialization_start+5,args.initialization_start+25)),language_tokens=64,
        initialization_ids=list(range(args.initialization_start,args.initialization_start+25)),reset_protocol=PROTOCOL,
        reset_audit_sha256=sha(args.reset_audit),batch_invariant_model_evaluated=False,heldout_acceptance_pending_disjointness_audit=True,
        acquisition_sha256=sha(args.checkpoint/'acquisition.json'),assets_sha256=assets,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_name=torch.cuda.get_device_name(),
        frame_selection='Ten evenly spaced pre-action frames; prospective heldout excludes reset frame0. No selection by success or numerical error.',
        native_contract='native operators with fixed64 language extent; not unmodified released longest-padding policy',
        controller='released adaptive ensemble horizon2 alpha0.1 and Google sticky gripper10',
        runner_sha256=sha(Path(__file__)),source_sha256={str(f):sha(f) for f in (Path(implementation.__file__),
            Path(sys.modules[RequestState.__module__].__file__),args.pizero_project/'experiments/simpler_worker.py',args.pizero_project/'experiments/simpler_support.py')},
        paired_success_rate_campaign_complete=False)
    runtime['source_sha256'].update(reset_runtime['source_sha256'])
    runtime['source_sha256'][str(Path(__file__).with_name('cogact_cold_collection_env.py').resolve())]=sha(Path(__file__).with_name('cogact_cold_collection_env.py'))
    (args.output/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    observations=[];episodes=[];noise_records=[]
    for task in TASKS:
        env=make_simpler_env(mapping[task],args.simpler_python,args.simpler_root,args.vulkan_icd)
        try:
            for initialization in range(args.initialization_start,args.initialization_start+25):
                env.reset(seed=20261108+initialization,options={'obj_init_options':{'episode_id':initialization}})
                snapshot=env.capture_snapshot();restored=env.restore_snapshot()
                if not all(restored['checks'].values()):raise RuntimeError('Inexact simulator snapshot restoration')
                obs=restored['observation'];controller=RequestState('google_robot',AdaptiveEnsembler)
                directory=args.output/'episodes'/task/str(initialization);directory.mkdir(parents=True)
                frames=[];calls=[];truncated=False;success=False
                def capture():
                    rgb=next(iter(obs['image'].values()))['rgb'];path=directory/f'frame-{len(frames):04d}.png'
                    Image.fromarray(rgb).save(path)
                    frames.append(dict(image_file=str(path.relative_to(args.output)),image_sha256=sha(path),
                                       raw_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),observation_sha256=digest(obs),state=observation_state(obs),raw_frame=env.current_raw_frame,observable_hashes=env.current['hashes']))
                    return rgb
                rgb=capture()
                while not truncated:
                    call=len(calls)
                    if call>=200:raise RuntimeError('Simulator did not reach declared horizon')
                    instruction=env.get_language_instruction();noise_id=f'{task}/{initialization}/policy-call-{call}'
                    seed=int.from_bytes(hashlib.sha256(noise_id.encode()).digest()[:8],'little')
                    target=dict(image=Image.fromarray(rgb),instruction=instruction,noise=torch.randn(1,16,7,generator=torch.Generator().manual_seed(seed)))
                    # Adapter inference has no persistent session/cache state;
                    # generate() constructs its cache per call. Controller is
                    # recreated above for every episode; explicit noise and
                    # fork_rng isolate inference from ambient RNG state.
                    with torch.inference_mode(),torch.random.fork_rng(devices=[torch.cuda.current_device()]),set_batch_invariant_mode(False):raw=adapter.run_model(policy,adapter.compose_fixed_batch([target]))[0].numpy()
                    if not np.isfinite(raw).all():raise RuntimeError('Nonfinite model output')
                    selected,action=controller.process(raw)
                    obs,reward,success,truncated,info=env.step(action)
                    calls.append(dict(policy_call=call,frame_index=len(frames)-1,instruction=instruction,noise_id=noise_id,noise_seed=seed,
                        raw_action_chunk=raw.tolist(),selected_action=selected.tolist(),environment_action=action.tolist(),reward=float(reward),
                        success=bool(success),truncated=bool(truncated),info=str(info)))
                    rgb=capture()
                if len(calls)<10:raise RuntimeError('Fewer than ten distinct pre-action observations')
                episode=dict(task=task,initialization=initialization,reset_protocol=PROTOCOL,snapshot=snapshot,restore_checks=restored['checks'],frames=frames,calls=calls,success=bool(success))
                path=directory/'episode.json';path.write_text(json.dumps(episode)+'\n')
                first_frame=0 if initialization<args.initialization_start+5 else 1
                selected_frames=np.linspace(first_frame,len(calls)-1,10,dtype=int).tolist()
                if len(set(selected_frames))!=10:raise RuntimeError('Fewer than ten distinct eligible observations')
                for frame_index in selected_frames:
                    index=len(observations);frame=frames[frame_index];instruction=calls[frame_index]['instruction']
                    request_id=f'simpler/{task}/{initialization}/observation-{frame_index}'
                    noises=np.stack([torch.randn(1,16,7,generator=torch.Generator().manual_seed(1990000+3*index+j)).numpy() for j in range(3)])
                    noise_records.append(noises)
                    observations.append(dict(observation_index=index,task=task,episode_id=initialization,frame_index=frame_index,
                        split='diagnostic' if initialization<args.initialization_start+5 else 'heldout',instruction=instruction,image_file=frame['image_file'],
                        image_sha256=frame['image_sha256'],raw_rgb_sha256=frame['raw_rgb_sha256'],
                        source_episode_file=str(path.relative_to(args.output)),source_episode_sha256=sha(path),
                        request_ids=[f'{request_id}/noise-{j}' for j in range(3)],noise_sha256=[hashlib.sha256(v.tobytes()).hexdigest() for v in noises]))
                episodes.append(dict(task=task,initialization=initialization,source_episode_file=str(path.relative_to(args.output)),sha256=sha(path),success=bool(success)))
                print(f'COLLECTED {task}/{initialization}: success={success}, steps={len(calls)}, observations={len(observations)}',flush=True)
        finally:env.close()
    values=np.stack(noise_records);noise_path=args.output/'noise.npy';np.save(noise_path,values)
    manifest=dict(schema_version=1,status='collected_pending_independent_audit',model='cogact_simpler',runtime_sha256=sha(args.output/'runtime.json'),
                  observations=observations,episodes=episodes,noise_file='noise.npy',noise_file_sha256=sha(noise_path),noise_shape=list(values.shape))
    (args.output/'replay_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
