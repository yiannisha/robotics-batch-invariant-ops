"""Independently verify frozen held-out replay for the four exact adapters."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from scripts.freeze_four_model_numerics_v2 import MODELS,read,require,sha256,stream,required_arrangements
from scripts.audit_pi05_actual_operators import check_metric


def schedule(row,source,observations,noise_id):
    size=row['logical_batch_size'];index=row['target_batch_index'];request=row['request_id']
    require(size in (1,2,4,8,16,32) and isinstance(index,int) and 0<=index<size,'Invalid logical batch/target')
    companions=[observations[i]['request_ids'][noise_id] for i in source['companion_replay_indices'][:size-1]]
    require(len(companions)==size-1,'Missing frozen companions')
    require(all(observations[i]['split']=='heldout' for i in source['companion_replay_indices'][:size-1]),'Companion crosses heldout split')
    if row['companion_type']=='duplicate':companions=[request]*(size-1)
    transformation=row['transformation']
    if transformation=='singleton':expected=[request]
    elif transformation=='batch_position':expected=list(companions);expected.insert(index,request)
    elif transformation=='permutation':expected=list(reversed([request,*companions]))
    elif transformation=='partition':expected=[request,*companions]
    else:raise ValueError('Unknown batching transformation')
    require(row['request_ordering']==expected,'Frozen heldout request ordering differs')
    require(row['companion_ids']==[v for i,v in enumerate(expected) if i!=index],'Companion identities differ')
    require(expected[index]==request,'Target index does not identify selected request')
    physical=max(1,size//2) if transformation=='partition' else size
    require(row['batch_size']==physical,'Physical batch differs')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-id',choices=MODELS,required=True)
    for name in ('manifest','freeze','disjointness-audit','records','fidelity','runtime','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Preserve existing audits')
    paths=[a.manifest,a.freeze,a.disjointness_audit,a.records,a.fidelity,a.runtime]
    before={str(p.resolve()):sha256(p) for p in paths}
    freeze=read(a.freeze);bound=freeze['content_addressed_freeze'];manifest=read(a.manifest);runtime=read(a.runtime)
    require(freeze['model_id']==manifest['model_id']==a.model_id,'Wrong heldout model')
    require(freeze['prepared_input_disjointness_verified'] is True and freeze['invariant_diagnostic_exact'] is True and freeze['full_traced_operator_qualification_complete'] is True,'Incomplete numerical freeze')
    for path,identity in bound.items():
        file=Path(path);require(file.is_file() and file.stat().st_size==identity['bytes'] and sha256(file)==identity['sha256'],'Frozen input changed: '+path)
    for path in (a.manifest,a.disjointness_audit):
        require(bound.get(str(path.resolve()),{}).get('sha256')==before[str(path.resolve())],'Heldout manifest/freshness seal not frozen')
    seal=read(a.disjointness_audit)
    require(seal['status']=='complete_observation_disjoint_heldout_independently_verified' and seal['candidate_manifest_sha256']==before[str(a.manifest.resolve())],'Wrong final freshness seal')
    require(seal['identity_kind']=='prepared_neural_inputs_before_first_model_operation' and seal['prepared_input_disjointness_verified'] is True,'Effective input freshness missing')
    adapter,revision,steps,kinds=MODELS[a.model_id];configuration=freeze['adapter_execution_configuration']
    require(configuration.get('checkpoint_revision')==revision and configuration.get('tf32') is False,'Wrong numerical path')
    require(runtime['adapter']==adapter and runtime['split']=='heldout' and runtime['execution_configuration']==configuration,'Wrong heldout runtime')
    require(runtime['replay_manifest_sha256']==before[str(a.manifest.resolve())] and sha256(Path(runtime['numerical_freeze']))==before[str(a.freeze.resolve())],'Wrong runtime manifest/freeze')
    require(runtime['tf32_matmul'] is False and runtime['tf32_cudnn'] is False,'Runtime precision changed')
    for path,identity in runtime['source_sha256'].items():require(bound.get(str(Path(path).resolve()),{}).get('sha256')==identity and sha256(Path(path))==identity,'Heldout source not frozen')
    observations=manifest['observations'];selected=[r for r in observations if r['split']=='heldout']
    require(len(selected)==800 or a.model_id=='lingbot_robotwin_exact' and 0<len(selected)<=800,'Wrong heldout observation cohort')
    require(all(len(r['noise'])==len(r['request_ids'])==3 for r in selected),'Three indexed noises required')
    sources={key:(r,i) for r in selected for i,key in enumerate(r['request_ids'])}
    require(len(sources)==3*len(selected),'Duplicate heldout request IDs')
    required_steps={f'flow.step_{i:02d}.{kind}' for i in range(steps) for kind in kinds}
    def metadata(row):
        require(row['request_id'] in sources,'Unexpected heldout request')
        source,noise_id=sources[row['request_id']]
        require(row['replay_manifest_sha256']==before[str(a.manifest.resolve())] and row['numerical_freeze_sha256']==before[str(a.freeze.resolve())],'Record provenance differs')
        require(row['adapter_execution_configuration']==configuration,'Record numerical configuration differs')
        require(row['task']==source['task'] and row['episode_id']==source['episode_id'] and row['observation_index']==source['observation_index'] and row['noise_id']==noise_id,'Source metadata differs')
        require(row['request_noise_sha256']==source['noise'][noise_id]['sha256'] and row['source_image_sha256']==source.get('image_sha256',source.get('source_image_rgb_sha256')),'Input/noise fingerprints differ')
        return source,noise_id
    groups=defaultdict(set);count=0;native_nonexact=0
    for row in stream(a.records):
        source,noise_id=metadata(row);schedule(row,source,observations,noise_id)
        require(row['adapter']==adapter and row['tf32_matmul'] is False and row['tf32_cudnn'] is False,'Record runtime differs')
        mode=row['implementation'];require(mode in ('native','batch_invariant'),'Unexpected implementation')
        key=(row['transformation'],row['logical_batch_size'],row['target_batch_position'],row['companion_type'])
        group=groups[row['request_id'],mode];require(key not in group,'Duplicate heldout arrangement');group.add(key)
        require(required_steps.issubset(row['flow_steps']),'Denoising stage missing')
        for metric in [row['output'],*row['flow_steps'].values()]:
            check_metric(metric)
            if mode=='batch_invariant':require(metric['exact'],'Invariant heldout output/stage changed')
        count+=1;native_nonexact+=mode=='native' and not row['output']['exact']
    expected={(request,mode) for request in sources for mode in ('native','batch_invariant')}
    require(groups.keys()==expected,'Incomplete heldout modes/requests')
    for (_,mode),keys in groups.items():
        sizes=[1,2,4,8]+[b for b in (16,32) if freeze['large_batch_feasible'].get(f'{mode}|B{b}')]
        require(keys==required_arrangements(tuple(sizes)),'Incomplete heldout arrangements')
    fidelity=set();nonexact_fidelity=0
    for row in stream(a.fidelity):
        metadata(row);key=row['request_id'];require(key not in fidelity,'Duplicate fidelity request')
        require(row['comparison']=='batch_invariant_singleton_vs_native_singleton','Wrong singleton fidelity comparison')
        check_metric(row['output']);fidelity.add(key);nonexact_fidelity+=not row['output']['exact']
    require(fidelity==sources.keys(),'Incomplete heldout singleton fidelity')
    for path,identity in before.items():require(sha256(Path(path))==identity,'Audit input changed')
    result=dict(status='complete_four_model_heldout_independently_verified',model_id=a.model_id,input_sha256=before,
        request_noise_pairs=len(sources),implementation_groups=len(groups),arrangement_records=count,native_nonexact_arrangements=native_nonexact,
        invariant_output_and_all_recorded_stages_exact=True,singleton_fidelity_records=len(fidelity),singleton_fidelity_nonexact_records=nonexact_fidelity,
        auditor_sha256=sha256(Path(__file__)),full_model_campaign_complete=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(a.output)


if __name__=='__main__':main()
