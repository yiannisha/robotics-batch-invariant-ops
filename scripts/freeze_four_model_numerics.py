"""Freeze the exact four-model campaign across distinct diagnostic/heldout corpora.

Requires complete diagnostics, capacity, source-operator coverage, and a final
prepared-input disjointness seal. A progress audit cannot authorize heldout work.
"""
import argparse
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import sys

# The retained legacy schedule validator imports its sibling by module name.
sys.path.insert(0,str(Path(__file__).resolve().parent))
from scripts.freeze_model_numerics import diagnostic_keys,capacity_audit,sha256,git,required_arrangements
from scripts.audit_pi05_actual_operators import check_metric

MODELS={
    'pi05_libero':('scripts.model_invariance.adapters.pi05_libero_exact','7152f13e34c6c115d808aea0e5d08aca74eac14b',10,('state','velocity')),
    'rdt_maniskill_exact':('scripts.model_invariance.adapters.rdt_maniskill_exact','9622afab2b7ce2312a6cf1febc526928589b77eb',5,('prediction','sample')),
    'cogact_simpler_exact':('scripts.model_invariance.adapters.cogact_simpler_exact','ca302a300542393a0b06e291465076a7d9105c23',10,('sample',)),
    'lingbot_robotwin_exact':('scripts.model_invariance.adapters.lingbot_robotwin_exact','0451855729ec904f970600e0aec8b84661423afe',10,('prediction','state')),
}


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(path.read_text())


def stream(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as source:
        for line in source:yield json.loads(line)


def disjointness_gate(manifest,audit):
    record=read(audit)
    require(record.get('status')=='complete_observation_disjoint_heldout_independently_verified'
        and record.get('candidate_manifest_sha256')==sha256(manifest),'Final manifest-bound disjointness audit required')
    require(record.get('identity_kind')=='prepared_neural_inputs_before_first_model_operation'
        and record.get('prepared_input_disjointness_verified') is True and record.get('noise_and_request_ids_excluded') is True
        and not record.get('collisions'),'Disjoint prepared neural inputs required')
    return record


def diagnostic_audit_gate(audit,model_id,configuration,files):
    record=read(audit)
    require(record.get('status')=='complete_four_model_diagnostic_independently_verified' and record.get('model_id')==model_id,'Complete independent diagnostic audit required')
    require(record.get('adapter_execution_configuration')==configuration and record.get('invariant_output_and_all_recorded_stages_exact') is True,'Diagnostic configuration/stage exactness differs')
    for key,path in files.items():require(record.get(key+'_sha256')==sha256(path),'Audited diagnostic input changed: '+key)
    require(record.get('auditor_sha256')==sha256(Path(__file__).with_name('audit_four_model_diagnostic.py')),'Independent diagnostic auditor source changed')
    return record


def operator_gate(path,model_id,configuration):
    record=read(path);paths=[path]
    if model_id=='pi05_libero':
        require(record.get('status')=='actual_operator_coverage_independently_verified' and record.get('complete') is True,'Complete PI05 operator audit required')
        for group in record['groups']:
            config=Path(group['configuration_file']);raw=Path(group['raw_file'])
            require(sha256(config)==group['configuration_sha256'] and sha256(raw)==group['raw_sha256'] and group['complete'],'Operator component changed/incomplete')
            require(read(config)['adapter_execution_configuration']==configuration,'Operator/model inference configuration differs')
            paths.extend((config,raw))
    else:
        require(record.get('status')=='source_operator_inventory_reconciled' and record.get('full_traced_operator_qualification_complete') is True,'Complete source inventory and measurements required')
        if model_id=='rdt_maniskill_exact':require(record.get('all_five_task_and_mixed_mask_measurements_complete') is True,'All task-mask measurements required')
        for filename,identity in record['source_traces'].items():
            trace=Path(filename);require(sha256(trace)==identity,'Source trace changed')
            value=read(trace);require(value['adapter']==MODELS[model_id][0] and value['adapter_execution_configuration']==configuration,'Foreign operator trace')
            paths.append(trace)
        for component in record['components']:
            config,raw,audit=map(Path,(component['configuration'],component['records'],component['audit']))
            value=read(audit)
            require(component['measurement_coverage_complete'] and sha256(config)==component['configuration_sha256'] and sha256(audit)==component['audit_sha256'],'Component changed/incomplete')
            require(sha256(raw)==value['raw_records_sha256'],'Measured raw component changed')
            require(read(config)['adapter_execution_configuration']==configuration,'Operator/model inference configuration differs')
            paths.extend((config,raw,audit))
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-id',choices=MODELS,required=True)
    for name in ('diagnostic-manifest','heldout-manifest','diagnostic','diagnostic-audit','fidelity','runtime','capacity','operator-audit','disjointness-audit','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--asset-root',type=Path,action='append',required=True)
    p.add_argument('--source-root',type=Path,action='append',required=True)
    a=p.parse_args();require(not a.output.exists(),'Freeze already exists')
    # Reject provisional seals before parsing large diagnostic archives.
    disjointness_gate(a.heldout_manifest,a.disjointness_audit)
    diagnostic,heldout,runtime=read(a.diagnostic_manifest),read(a.heldout_manifest),read(a.runtime)
    adapter,revision,steps,kinds=MODELS[a.model_id];configuration=runtime['execution_configuration']
    require(runtime['adapter']==adapter and runtime['split']=='diagnostic' and runtime['replay_manifest_sha256']==sha256(a.diagnostic_manifest),'Wrong diagnostic runtime')
    require(configuration and configuration.get('checkpoint_revision')==revision and configuration.get('tf32') is False,'Unqualified model revision/execution configuration')
    require(diagnostic['model_id']==heldout['model_id']==a.model_id,'Foreign replay corpus')
    observations=[r for r in diagnostic['observations'] if r['split']=='diagnostic']
    require(len(observations)==200 or a.model_id=='lingbot_robotwin_exact' and 0<len(observations)<=200,'Wrong diagnostic observation count')
    require(all(len(r['request_ids'])==len(r['noise'])==3 for r in observations),'Three explicit diagnostic noises required')
    expected={key for row in observations for key in row['request_ids']}
    require(len(expected)==3*len(observations),'Duplicate diagnostic request identities')
    issues=[];capacity=capacity_audit(a.capacity,a.diagnostic_manifest.parent,issues)
    require(capacity['complete'] and not issues,'Incomplete capacity evidence: '+str(issues[:3]))
    for row in stream(a.capacity):require(row['adapter']==adapter and row['adapter_execution_configuration']==configuration,'Capacity uses different numerical configuration')
    checked=diagnostic_audit_gate(a.diagnostic_audit,a.model_id,configuration,dict(manifest=a.diagnostic_manifest,diagnostic=a.diagnostic,fidelity=a.fidelity,runtime=a.runtime,capacity=a.capacity))
    required={f'flow.step_{i:02d}.{kind}' for i in range(steps) for kind in kinds}
    expected_records=len(expected)*sum(len(required_arrangements(tuple([1,2,4,8]+[size for size in (16,32) if capacity['feasible'].get(f'{mode}|B{size}')]))) for mode in ('native','batch_invariant'))
    require(set(checked['required_denoising_labels'])==required and checked['diagnostic_observations']==len(observations),'Independent denoising/observation coverage differs')
    require(checked['request_noise_pairs']==checked['singleton_fidelity_records']==len(expected) and checked['implementation_groups']==2*len(expected) and checked['arrangement_records']==expected_records,'Independent diagnostic cardinality differs')
    paths=[Path(__file__),Path(__file__).with_name('freeze_model_numerics.py'),Path(__file__).with_name('audit_four_model_diagnostic.py'),Path(__file__).with_name('audit_full_cross_model_campaign.py'),Path(__file__).with_name('audit_pi05_actual_operators.py'),
        a.diagnostic_manifest,a.heldout_manifest,a.diagnostic,a.diagnostic_audit,a.fidelity,a.runtime,a.capacity,a.disjointness_audit]
    paths.extend(operator_gate(a.operator_audit,a.model_id,configuration))
    for filename,identity in runtime['source_sha256'].items():
        path=Path(filename);require(sha256(path)==identity,'Diagnostic numerical source changed: '+filename);paths.append(path)
    revisions=set();weight_files=[]
    for root in a.asset_root:
        require(root.is_dir(),'Asset root missing')
        for path in root.rglob('*'):
            if not path.is_file() or any(v in ('.cache','.git','__pycache__') for v in path.parts):continue
            paths.append(path)
            if path.suffix in ('.safetensors','.pt','.bin'):weight_files.append(path)
            if path.name=='acquisition.json':
                acquisition=read(path);revisions.add(acquisition.get('revision'))
                for item in acquisition.get('files',[]):
                    require(sha256(path.parent/item['path'])==item['sha256'],'Checkpoint acquisition bytes changed')
    require(weight_files and revision in revisions,'Pinned checkpoint acquisition and learned weights required')
    for root in a.source_root:
        require(root.is_dir(),'Model dependency source root missing')
        paths.extend(path for path in root.rglob('*.py') if '.git' not in path.parts and '__pycache__' not in path.parts)
    for manifest in (diagnostic,heldout):
        audit=Path(manifest['corpus_audit_path']);require(sha256(audit)==manifest['corpus_audit_sha256'],'Corpus audit changed');paths.append(audit)
        corpus=read(audit)
        if a.model_id=='lingbot_robotwin_exact':require(corpus.get('all_prescribed_native_episodes_rebuilt') is True and corpus.get('completed_native_episodes')==200,'Complete LingBot corpus reconstruction required')
        else:require(corpus.get('status')=='complete_corpus_independently_verified' and corpus.get('observations')==1000 and corpus.get('episodes')==100,'Complete independently reconstructed corpus required')
        for key in ('noise_file','source_manifest_snapshot','source_replay_manifest_path'):
            if manifest.get(key):paths.append(Path(manifest[key]))
    sources=list(dict.fromkeys(path.resolve() for path in paths))
    require(all(path.is_file() for path in sources),'Missing freeze input')
    hashes={str(path):dict(bytes=path.stat().st_size,sha256=sha256(path)) for path in sources}
    report=dict(schema_version=1,model_id=a.model_id,frozen_at_utc=datetime.now(timezone.utc).isoformat(),git_head=git(['rev-parse','HEAD']),git_status=git(['status','--short']),
        content_addressed_freeze=hashes,adapter_execution_configuration=configuration,diagnostic_request_pairs=len(expected),diagnostic_implementation_groups=checked['implementation_groups'],
        diagnostic_arrangement_records=checked['arrangement_records'],singleton_fidelity_records=checked['singleton_fidelity_records'],large_batch_feasible=capacity['feasible'],invariant_diagnostic_exact=True,
        diagnostic_manifest_sha256=sha256(a.diagnostic_manifest),heldout_manifest_sha256=sha256(a.heldout_manifest),prepared_input_disjointness_verified=True,
        full_traced_operator_qualification_complete=True,full_four_model_campaign_complete=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);temporary=a.output.with_suffix(a.output.suffix+'.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(a.output)
    print(a.output)


if __name__=='__main__':main()
