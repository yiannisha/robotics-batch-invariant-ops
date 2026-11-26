"""CPU-only RDT input identity before the first neural operation, sans noise."""
from functools import lru_cache
import hashlib

import numpy as np
from PIL import Image
import torch

from scripts.qualify_simpler_fixed_action_replay import digest
from scripts.run_rdt_paired_server import STATE_MIN,STATE_MAX
from scripts.model_invariance.adapters.rdt_maniskill import MANISKILL_INDICES


def tensor_signature(value):
    value=value.detach().cpu().contiguous()
    return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def prepared_state(proprio):
    joints=np.asarray(proprio,dtype=np.float32).reshape(-1)
    if joints.shape!=(8,):raise ValueError('Expected eight source joints')
    state=torch.zeros(1,1,128,dtype=torch.bfloat16)
    state[0,0,list(MANISKILL_INDICES)]=torch.from_numpy((joints-STATE_MIN)/(STATE_MAX-STATE_MIN)*2-1).to(torch.bfloat16)
    return state


class PreparedIdentity:
    def __init__(self,processor,language):
        self.processor=processor;self.language=language
        self.color=tuple(int(v*255) for v in processor.image_mean)
        size=processor.size
        self.background=np.ones((size['height'],size['width'],3),dtype=np.uint8)*np.asarray(self.color,dtype=np.uint8)

    @lru_cache(maxsize=128)
    def pixels(self,shape,payload):
        image=Image.fromarray(np.frombuffer(payload,dtype=np.uint8).reshape(shape))
        width,height=image.size;side=max(width,height)
        square=Image.new(image.mode,(side,side),self.color);square.paste(image,((side-width)//2,(side-height)//2))
        return self.processor.preprocess(square,return_tensors='pt')['pixel_values'][0].to(torch.bfloat16)

    def tensors(self,request):
        history=np.asarray(request['image_history'],dtype=np.uint8);valid=request['history_valid']
        if history.ndim!=4 or history.shape[0]!=2 or history.shape[-1]!=3 or len(valid)!=2:raise ValueError('Wrong history geometry')
        slots=[history[0] if valid[0] else self.background,self.background,self.background,
               history[1] if valid[1] else self.background,self.background,self.background]
        pixels=torch.stack([self.pixels(tuple(v.shape),v.tobytes()) for v in slots]).unsqueeze(0)
        embedded=self.language[request['task_name']].to(torch.bfloat16)
        language=torch.zeros(1,30,4096,dtype=torch.bfloat16);language[:,:embedded.shape[1]]=embedded
        mask=torch.zeros(1,30,dtype=torch.bool);mask[:,:embedded.shape[1]]=True
        action_mask=torch.zeros(1,1,128,dtype=torch.bfloat16);action_mask[0,0,list(MANISKILL_INDICES)]=1
        return dict(vision_pixels=pixels,language=language,language_mask=mask,state=prepared_state(request['proprio']),
                    action_mask=action_mask,control_frequency=torch.tensor([25],dtype=torch.long))

    def __call__(self,request):
        return digest({key:tensor_signature(value) for key,value in self.tensors(request).items()})
