"""PI05 effective image/token inputs without constructing a neural model.

The official policy's image preprocessing runs on the actual selected device.
PI05 does not consume the separate observation.state tensor in sample_actions;
its state information is already encoded into language tokens by the processor.
"""
import hashlib
from types import SimpleNamespace

import numpy as np
import torch

from scripts.qualify_simpler_fixed_action_replay import digest


def signature(value):
    value=value.detach().cpu().contiguous()
    return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


class PreparedIdentity:
    def __init__(self,checkpoint,tokenizer,device='cuda'):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        self.config=PreTrainedConfig.from_pretrained(checkpoint)
        if not isinstance(self.config,PI05Config):raise ValueError('Not PI05')
        self.config.device='cpu';self.config.compile_model=False
        self.pre,_=make_pre_post_processors(self.config,pretrained_path=checkpoint,preprocessor_overrides={
            'tokenizer_processor':{'tokenizer_name':str(tokenizer)},'device_processor':{'device':'cpu'}})
        # _preprocess_images only reads config and the first parameter's device.
        # A zero-element tensor selects the device; no learned model is created.
        marker=torch.empty(0,device=device)
        self.context=SimpleNamespace(config=self.config,parameters=lambda:iter([marker]))
        self.preprocess_images=PI05Policy._preprocess_images

    def prepared(self,observation,instruction):
        from lerobot.envs.utils import preprocess_observation
        from lerobot.processor.env_processor import LiberoProcessorStep
        def batched(value):
            if isinstance(value,np.ndarray):return value[None]
            if isinstance(value,dict):return {k:batched(v) for k,v in value.items()}
            return value
        batch=LiberoProcessorStep().observation(preprocess_observation(batched(observation)));batch['task']=[instruction]
        return {k:v.cpu() for k,v in self.pre(batch).items() if isinstance(v,torch.Tensor)}

    def tensors(self,prepared):
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS,OBS_LANGUAGE_ATTENTION_MASK
        with torch.inference_mode():images,masks=self.preprocess_images(self.context,prepared)
        result={'language_tokens':prepared[OBS_LANGUAGE_TOKENS],'language_mask':prepared[OBS_LANGUAGE_ATTENTION_MASK]}
        for i,(image,mask) in enumerate(zip(images,masks,strict=True)):
            result[f'image.{i}']=image;result[f'image_mask.{i}']=mask
        return result

    def __call__(self,prepared):
        return digest({key:signature(value) for key,value in self.tensors(prepared).items()})
