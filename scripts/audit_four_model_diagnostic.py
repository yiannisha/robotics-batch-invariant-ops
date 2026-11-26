"""Independently recount complete four-model diagnostic/fidelity measurements."""
import argparse
import json
from pathlib import Path

from scripts.freeze_four_model_numerics import MODELS,capacity_audit,diagnostic_keys,read,require,sha256,stream
from scripts.audit_pi05_actual_operators import check_metric


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-id',choices=MODELS,required=True)
    for name in ('manifest','diagnostic','fidelity','runtime','capacity','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Audit output already exists')
    manifest=read(a.manifest);runtime=read(a.runtime);configuration=runtime['execution_configuration']
    adapter,revision,steps,kinds=MODELS[a.model_id];identity=sha256(a.manifest)
    require(runtime['adapter']==adapter and runtime['split']=='diagnostic' and runtime['replay_manifest_sha256']==identity,'Wrong diagnostic runtime')
    require(configuration and configuration.get('checkpoint_revision')==revision and configuration.get('tf32') is False,'Wrong pinned numerical configuration')
    require(runtime['tf32_matmul'] is False and runtime['tf32_cudnn'] is False,'Wrong runtime precision')
    for filename,h in runtime['source_sha256'].items():require(sha256(Path(filename))==h,'Numerical source changed')
    observations=[r for r in manifest['observations'] if r['split']=='diagnostic']
    require(len(observations)==200 or a.model_id=='lingbot_robotwin_exact' and 0<len(observations)<=200,'Wrong diagnostic observation count')
    require(all(len(r['noise'])==len(r['request_ids'])==3 for r in observations),'Three explicit noises required')
    sources={key:(row,i) for row in observations for i,key in enumerate(row['request_ids'])}
    require(len(sources)==3*len(observations),'Duplicate request/noise identities')
    issues=[];capacity=capacity_audit(a.capacity,a.manifest.parent,issues);require(capacity['complete'] and not issues,'Incomplete capacity: '+str(issues[:3]))
    for row in stream(a.capacity):require(row['adapter']==adapter and row['adapter_execution_configuration']==configuration,'Capacity configuration differs')
    groups=diagnostic_keys(a.diagnostic,manifest,set(sources),identity,capacity['feasible'],configuration)
    required_steps={f'flow.step_{i:02d}.{kind}' for i in range(steps) for kind in kinds}
    def metadata(row):
        source,noise=sources[row['request_id']]
        require(row['task']==source['task'] and row['episode_id']==source['episode_id'] and row['observation_index']==source['observation_index'] and row['noise_id']==noise,'Source observation/request metadata differs')
        require(row['request_noise_sha256']==source['noise'][noise]['sha256'],'Explicit noise fingerprint differs')
        require(row['source_image_sha256']==source.get('image_sha256',source.get('source_image_rgb_sha256')),'Source image fingerprint differs')
        require(row['adapter_execution_configuration']==configuration and row['replay_manifest_sha256']==identity and row.get('numerical_freeze_sha256') is None,'Configuration/manifest/freeze provenance differs')
    count=0;native_nonexact=0
    for row in stream(a.diagnostic):
        metadata(row);require(row['adapter']==adapter and row['tf32_matmul'] is False and row['tf32_cudnn'] is False,'Runtime identity/precision differs')
        logical=row['logical_batch_size'];physical=max(1,logical//2) if row['transformation']=='partition' else logical
        require(row['batch_size']==physical,'Physical batch extent differs')
        require(required_steps.issubset(row['flow_steps']),'Denoising stage omitted')
        for value in [row['output'],*row['flow_steps'].values()]:
            check_metric(value)
            if row['implementation']=='batch_invariant':require(value['exact'],'Invariant output or intermediate stage differs')
        native_nonexact+=row['implementation']=='native' and not row['output']['exact'];count+=1
    seen=set();fidelity_nonexact=0
    for row in stream(a.fidelity):
        key=row['request_id'];require(key in sources and key not in seen,'Unknown/duplicate fidelity request');metadata(row)
        require(row['comparison']=='batch_invariant_singleton_vs_native_singleton','Wrong fidelity comparison');check_metric(row['output']);seen.add(key);fidelity_nonexact+=not row['output']['exact']
    require(seen==set(sources),'Incomplete singleton fidelity')
    for filename,h in runtime['source_sha256'].items():require(sha256(Path(filename))==h,'Numerical source changed during audit')
    report=dict(status='complete_four_model_diagnostic_independently_verified',model_id=a.model_id,adapter_execution_configuration=configuration,
        manifest_sha256=identity,diagnostic_sha256=sha256(a.diagnostic),fidelity_sha256=sha256(a.fidelity),runtime_sha256=sha256(a.runtime),capacity_sha256=sha256(a.capacity),
        diagnostic_observations=len(observations),request_noise_pairs=len(sources),implementation_groups=len(groups),arrangement_records=count,
        native_nonexact_arrangements=native_nonexact,invariant_output_and_all_recorded_stages_exact=True,required_denoising_labels=sorted(required_steps),
        singleton_fidelity_records=len(seen),singleton_fidelity_nonexact_records=fidelity_nonexact,large_batch_feasible=capacity['feasible'],auditor_sha256=sha256(Path(__file__)),full_model_campaign_complete=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
