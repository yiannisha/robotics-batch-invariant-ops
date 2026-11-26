"""Reconcile LingBot exact-path's complete traced operator inventory with audited components.

Inventory coverage and completed measurement coverage are separate. A partial
component cannot certify full operator qualification even if all shapes map.
"""
import argparse
import hashlib
import json
from pathlib import Path


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def pattern(row):
    def clean(value):
        if isinstance(value,dict):
            if 'shape' in value and 'dtype' in value:return {k:value[k] for k in ('shape','stride','dtype','device') if k in value}
            return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list):return [clean(v) for v in value]
        return value
    return json.dumps(dict(operation=row['aten_operation'],arguments=clean(row['arguments']),kwargs=clean(row['keyword_arguments'])),sort_keys=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('b1','b8','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--component',nargs=3,action='append',required=True,metavar=('CONFIG','RECORDS','AUDIT'))
    p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    traces={b:json.loads(path.read_text()) for b,path in ((1,a.b1),(8,a.b8))}
    require(all(t['adapter']=='scripts.model_invariance.adapters.lingbot_robotwin_exact' and t['batch_size']==b for b,t in traces.items()),'Wrong model/source batch')
    require(traces[1]['adapter_execution_configuration']==traces[8]['adapter_execution_configuration'],'Different inference protocols')
    expected={b:{pattern(row) for row in t['operations']} for b,t in traces.items()};covered={1:set(),8:set()};components=[]
    accepted={'selected_source_operator_measurements_independently_verified','source_linear_measured_arrangements_independently_verified','source_attention_measurements_independently_verified','shared_source_matmul_measurements_independently_verified','remaining_source_operator_measurements_independently_verified'}
    for config_file,records_file,audit_file in a.component:
        cp,rp,ap=map(Path,(config_file,records_file,audit_file));config=json.loads(cp.read_text());audit=json.loads(ap.read_text())
        require(config['adapter']=='scripts.model_invariance.adapters.lingbot_robotwin_exact' and config['adapter_execution_configuration']==traces[1]['adapter_execution_configuration'],'Foreign component protocol')
        require(audit['status'] in accepted and audit['configuration_sha256']==sha(cp),'Missing or foreign independent component audit')
        complete=audit.get('all_linear_arrangements_complete',audit.get('all_selected_arrangements_complete',False))
        raw_hash=hashlib.sha256()
        with rp.open('rb') as stream:
            for _ in range(audit['raw_records']):
                line=stream.readline();require(line.endswith(b'\n'),'Audited raw prefix truncated');raw_hash.update(line)
            if complete:require(not stream.read(1),'Complete component has additional unaudited records')
        require(raw_hash.hexdigest()==audit['raw_records_sha256'],'Audited measurement prefix changed')
        matched={1:set(),8:set()}
        for c in config['configurations']:
            entries=c.get('source_entries',c.get('source'))
            require(entries and 'b1' in entries and 'b8' in entries,'Unmapped source component')
            for batch in (1,8):
                key=pattern(entries['b'+str(batch)]);require(key in expected[batch],'Component source pattern not in complete inventory')
                matched[batch].add(key);covered[batch].add(key)
        components.append(dict(configuration=str(cp.resolve()),configuration_sha256=sha(cp),records=str(rp.resolve()),audit=str(ap.resolve()),audit_sha256=sha(ap),
            independently_verified_records=audit['raw_records'],measurement_coverage_complete=complete,source_patterns_b1=len(matched[1]),source_patterns_b8=len(matched[8])))
    missing={str(b):sorted(expected[b]-covered[b]) for b in (1,8)}
    inventory_complete=all(not v for v in missing.values());measurements_complete=all(c['measurement_coverage_complete'] for c in components)
    require(inventory_complete,'Source inventory contains unqualified/unmapped operator patterns')
    require(not a.require_complete or measurements_complete,'Mapped shapes are not completed measurements')
    report=dict(status='source_operator_inventory_reconciled',source_patterns_b1=len(expected[1]),source_patterns_b8=len(expected[8]),
        source_traces={str(path.resolve()):sha(path) for path in (a.b1,a.b8)},components=components,missing_patterns=missing,
        all_source_patterns_mapped=inventory_complete,all_component_measurements_complete=measurements_complete,
        full_traced_operator_qualification_complete=inventory_complete and measurements_complete,
        scope='Observed high-level source shapes/strides and captured mask variants; full-model numerical/fidelity/heldout QA remain separate.',performance_qualified=False,auditor_sha256=sha(Path(__file__)))
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('components','source_traces','missing_patterns')},indent=2))


if __name__=='__main__':main()
