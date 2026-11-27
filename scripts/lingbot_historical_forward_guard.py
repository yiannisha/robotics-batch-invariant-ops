"""Allow recovery inference only on byte-exact previously archived requests."""
import hashlib
import json
import os
from pathlib import Path

AUTHORITIES={
 '24d67091b45b1987e0410993a3daac57404f32c712bc4fff8581b2a8037ed267':1077,
 'b3134e2aece77a3ad009498186fa5f31552c46d2a79e7a3a992cf3e89434eaa5':82,
}
AUDITOR='809a9c4087c389e3e16c5f27b5ef88d77573065e70a8f71dd877b47308db5da6'


class UnarchivedRecoveryInput(ValueError):pass


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_authorities(paths):
    known={};bindings={}
    for path in paths:
        path=Path(path);h=sha(path)
        if h not in AUTHORITIES:raise ValueError('Unrecognized historical byte authority')
        report=json.loads(path.read_text())
        if report.get('auditor_sha256')!=AUDITOR:raise ValueError('Unrecognized historical reconstruction auditor')
        requests=[(file,expected) for file,expected in report['raw_files_sha256'].items() if Path(file).parent.name=='requests']
        if len(requests)!=AUTHORITIES[h]:raise ValueError('Historical request authority scope differs')
        for file,expected in requests:known.setdefault(Path(file).name,[]).append(dict(original_path=file,original_sha256=expected,authority_sha256=h))
        bindings[str(path.resolve())]=h
    if not known:raise ValueError('Historical input authority required')
    return known,bindings


def authorize(path,known):
    path=Path(path);h=sha(path)
    matches=[entry for entry in known.get(path.name,[]) if entry['original_sha256']==h]
    if not matches:raise UnarchivedRecoveryInput('Prepared input is not a byte-exact previously archived request; no inference permitted')
    return matches


def record_before_forward(ledger,path,known,request_id):
    matches=authorize(path,known)
    record=dict(request_id=request_id,path=str(Path(path).resolve()),sha256=sha(path),historical_authorities=matches,
                inference_attempt_authorized=True,prepared_input_already_archived=True)
    with Path(ledger).open('a') as stream:
        stream.write(json.dumps(record,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
    return record
