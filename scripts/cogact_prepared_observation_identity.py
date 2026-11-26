"""CPU-only CogACT image/token identity, excluding diffusion noise.

Identity conservatively includes the effective BF16 vision pixels after the
released autocast conversion, so Float32 differences erased by that conversion
cannot manufacture an unseen observation. `tensors` retains recorded FP32 pixels
for direct comparison with actual preprocessing traces.
"""
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from scripts.qualify_simpler_fixed_action_replay import digest


def signature(value):
    value=value.detach().cpu().contiguous()
    return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


class PreparedIdentity:
    def __init__(self,lock):
        from transformers import AutoTokenizer
        import timm
        from torchvision.transforms import Compose,Resize
        meta=next(Path(p).parent for p in lock['asset_sha256'] if Path(p).name=='tokenizer.model')
        for path,expected in lock['asset_sha256'].items():
            if Path(path).parent==meta and hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('Tokenizer asset changed')
        self.tokenizer=AutoTokenizer.from_pretrained(meta,model_max_length=2048,padding_side='right')
        self.tokenizer.add_special_tokens({'pad_token':'<PAD>'})
        path=next(Path(p) for p in lock['source_sha256'] if p.endswith('/llm/prompting/base_prompter.py'))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=lock['source_sha256'][str(path)]:raise ValueError('Prompt builder source changed')
        spec=importlib.util.spec_from_file_location('identity_pure_prompter',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.prompt_builder=module.PurePromptBuilder;self.transforms={}
        for key,name in [('dino','vit_large_patch14_reg4_dinov2.lvd142m'),('siglip','vit_so400m_patch14_siglip_224')]:
            cfg=timm.data.resolve_data_config(pretrained_cfg=timm.models.get_pretrained_cfg(name).to_dict());cfg['input_size']=(3,224,224)
            transform=timm.data.create_transform(**cfg,is_training=False)
            if not isinstance(transform,Compose) or not isinstance(transform.transforms[0],Resize):raise ValueError('Unexpected vision transform')
            self.transforms[key]=Compose([Resize((224,224),interpolation=transform.transforms[0].interpolation),*transform.transforms[1:]])

    @lru_cache(maxsize=128)
    def tokens(self,instruction):
        builder=self.prompt_builder('llama2');builder.add_turn(role='human',message=f'What action should the robot take to {instruction.lower()}?')
        values=self.tokenizer(builder.get_prompt(),truncation=True,return_tensors='pt').input_ids.squeeze(0)
        values=torch.cat([values,torch.tensor([29871,2])])
        if len(values)>64:raise ValueError('Instruction exceeds controlled extent')
        result=torch.full((1,64),self.tokenizer.pad_token_id,dtype=torch.long);result[0,:len(values)]=values
        return result

    @lru_cache(maxsize=128)
    def pixels(self,shape,payload):
        image=Image.fromarray(np.frombuffer(payload,dtype=np.uint8).reshape(shape))
        return {key:transform(image).unsqueeze(0) for key,transform in self.transforms.items()}

    def tensors(self,request):
        image=np.asarray(request['rgb'],dtype=np.uint8);ids=self.tokens(request['instruction'])
        result={'input.input_ids':ids,'input.attention_mask':ids.ne(self.tokenizer.pad_token_id)}
        result.update({'input.pixel_values.'+key:value for key,value in self.pixels(tuple(image.shape),image.tobytes()).items()})
        return result

    def __call__(self,request):
        return digest({key:signature(value.to(torch.bfloat16) if key.startswith('input.pixel_values.') else value) for key,value in self.tensors(request).items()})
