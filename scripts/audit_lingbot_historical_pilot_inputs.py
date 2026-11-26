"""Independently bind regenerated pilot tensors to archived historical signatures."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import torch
from scripts.audit_lingbot_campaign_corpus_v2 import content_hash, KEYS


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_signatures(value):
    result={}
    for key,tensor in value.items():
        require(isinstance(tensor,torch.Tensor),'Non-tensor historical request field')
        array=tensor.detach().cpu().contiguous()
        result[key]=dict(dtype=str(array.dtype),shape=list(array.shape),
            sha256=hashlib.sha256(array.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    return result


def verify_request(value,expected):
    actual=tensor_signatures(value)
    require(actual==expected,'Recovered tensor bytes differ from archived historical signatures')
    return actual


def fingerprint(signatures):return json.dumps(signatures,sort_keys=True,separators=(',',':'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('episodes','history-index','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--recovery-directory',type=Path,action='append',required=True);a=p.parse_args()
    require(not a.output.exists(),'Preserve earlier audits')
    history=json.loads(a.history_index.read_text());payload=gzip.decompress(a.episodes.read_bytes())
    historical_path=next(k for k in history['sources_sha256'] if k.endswith('lingbot-closed-loop-pilot-001/episodes.jsonl'))
    pilot_sha=hashlib.sha256(payload).hexdigest();require(pilot_sha==history['sources_sha256'][historical_path],'Historical authority bytes differ')
    episodes=[json.loads(line) for line in payload.decode().splitlines()]
    old={(e['condition'],c['policy_call']):c['model_inputs'] for e in episodes for c in e['calls']}
    records=[];known=set();bindings={str(a.episodes.resolve()):sha(a.episodes),str(a.history_index.resolve()):sha(a.history_index)}
    for directory in a.recovery_directory:
        path=directory/'recovery_report.json';report=json.loads(path.read_text());bindings[str(path.resolve())]=sha(path)
        require(report['status']=='requested_historical_pilot_tensors_recovered_by_exact_archived_signatures' and
                report['historical_pilot_sha256']==pilot_sha and report['neural_inference_performed'] is False,'Incomplete or foreign recovery')
        for row in report['recovered_requests']:
            path=(directory/row['raw_request']).resolve();require(path.is_relative_to(directory.resolve()),'Recovered path escapes attempt')
            require(sha(path)==row['request_file_sha256'],'Recovered request container changed')
            with gzip.open(path,'rb') as f:value=torch.load(f,map_location='cpu',weights_only=False)
            require(content_hash(value)==path.name.split('.')[0],'Recovered content address changed')
            key=(row['condition'],row['policy_call']);require(key in old,'Foreign historical call')
            actual=verify_request(value,old[key]);require(actual==row['input_signatures'],'Recovery metadata differs')
            identity=hashlib.sha256(fingerprint({k:actual[k] for k in KEYS}).encode()).hexdigest()
            known.add(fingerprint(actual));bindings[str(path)]=sha(path)
            records.append(dict(path=str(path),sha256=sha(path),condition=key[0],policy_call=key[1],prepared_input_identity=identity))
    counts={e['condition']:sum(fingerprint(c['model_inputs']) in known for c in e['calls']) for e in episodes}
    report=dict(status='historical_pilot_request_tensors_independently_matched_to_archived_signatures',
        historical_pilot_sha256=pilot_sha,recovered_request_files=len({r['path'] for r in records}),
        reconstructed_call_records=len(records),unique_tensor_requests=len(known),
        covered_historical_call_occurrences=sum(counts.values()),historical_call_occurrences=len(old),coverage_by_condition=counts,
        complete_this_pilot_input_coverage=sum(counts.values())==len(old),request_files=records,raw_files_sha256=bindings,
        auditor_sha256=sha(Path(__file__)),complete_historical_exposure_coverage=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('request_files','raw_files_sha256')},indent=2))


if __name__=='__main__':main()
