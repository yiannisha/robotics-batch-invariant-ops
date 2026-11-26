"""Extend real-policy operator traces with lossless attention-mask bytes.

Floating causal/padding masks must not be replaced with an unmasked test. Keep
the existing driver unchanged and archive this extension alongside its trace.
"""
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

import torch

from scripts import trace_model_operator_shapes as driver


def mask_record(value):
    if value is None:return None
    if not isinstance(value,torch.Tensor):raise TypeError('Attention mask must be a tensor or None')
    compact=value[tuple(slice(0,1) if stride==0 else slice(None) for stride in value.stride())]
    data=compact.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype),
        values_shape=list(compact.shape),encoding='zlib_base64_original_dtype_bytes_v1',
        original_dtype_bytes_sha256=hashlib.sha256(data).hexdigest(),original_dtype_bytes_count=len(data),
        data=base64.b64encode(zlib.compress(data,9)).decode('ascii'))


class MaskRecorder(driver.ShapeRecorder):
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        name=func._schema.name
        if name in driver.SELECTED:
            record=dict(aten_operation=name,arguments=driver.describe(args),keyword_arguments=driver.describe(kwargs or {}))
            if name=='aten::scaled_dot_product_attention':
                value=args[3] if len(args)>3 else (kwargs or {}).get('attn_mask')
                record['attention_mask']=mask_record(value)
            self.records[json.dumps(record,sort_keys=True)]+=1
        return func(*args,**(kwargs or {}))


def main():
    # Delegate input selection/model execution to the archived original driver.
    # It uses its own __file__ for provenance, so record this extension too.
    driver.ShapeRecorder=MaskRecorder;driver.main()
    index=sys.argv.index('--output');path=Path(sys.argv[index+1]);report=json.loads(path.read_text())
    report['mask_capture_extension_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['attention_masks_losslessly_captured']=True
    path.write_text(json.dumps(report,indent=2)+'\n');Path(str(path)+'.mask_driver.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
