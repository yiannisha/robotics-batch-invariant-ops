"""Explicit request-major projection inside independent leading-axis blocks.

The caller must establish the block layout from the model source. Shape
compatibility alone is not proof of semantic comparability.
"""
import torch


def project(reference, candidate, *, batch_size, target_index=0, blocks=1):
    if (batch_size < 2 or blocks < 1 or target_index not in range(batch_size)):
        raise ValueError('Invalid request projection specification')
    if (reference.ndim == 0 or candidate.ndim != reference.ndim
            or reference.shape[1:] != candidate.shape[1:]
            or reference.shape[0] % blocks != 0
            or reference.shape[0] == 0
            or candidate.shape[0] != reference.shape[0] * batch_size):
        return None
    stride = reference.shape[0] // blocks
    slices = [candidate[(block * batch_size + target_index) * stride:
                        (block * batch_size + target_index + 1) * stride]
              for block in range(blocks)]
    return torch.cat(slices, dim=0)


def source_blocks(adapter, module_name):
    if adapter == 'scripts.model_invariance.adapters.cogact_simpler_exact' and 'action_model.net' in module_name:
        return 2
    return 1
