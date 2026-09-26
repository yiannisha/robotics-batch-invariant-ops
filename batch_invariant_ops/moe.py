"""Deterministic reference operators for token-choice mixture-of-experts layers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .batch_invariant_ops import bmm_persistent


def deterministic_token_choice_moe(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    workspace: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Evaluate a top-k SwiGLU MoE without atomic routing or accumulation.

    ``gate_weight`` and ``up_weight`` have shape ``(E, I, D)`` and
    ``down_weight`` has shape ``(E, D, I)``.  The operator evaluates every
    expert with three fixed-schedule batched matrix multiplications, gathers
    the selected routes, and adds the small top-k dimension in a fixed
    left-to-right order.  This makes execution batch-invariant and repeatable,
    unlike fused implementations whose route packing or token accumulation
    uses atomics.

    The optional ``workspace`` argument is accepted for drop-in compatibility
    with inference kernels that cache temporary storage.  It is deliberately
    unused because reusing atomically populated buffers is exactly what this
    reference implementation avoids.

    Computing all experts is intentionally a deterministic inference fallback:
    its expert dimension and reduction schedules do not depend on routing or
    request batch composition.  The selected output has the same top-k SwiGLU
    semantics as sparse execution, at the cost of extra expert arithmetic.
    """

    del workspace
    if hidden_states.ndim != 2:
        raise ValueError(
            f"hidden_states must have shape (tokens, hidden), got {hidden_states.shape}"
        )
    if routing_weights.ndim != 2 or selected_experts.shape != routing_weights.shape:
        raise ValueError(
            "routing_weights and selected_experts must have the same (tokens, top_k) shape"
        )
    if routing_weights.shape[0] != hidden_states.shape[0]:
        raise ValueError("routing tensors and hidden_states have different token counts")
    if gate_weight.ndim != 3 or up_weight.shape != gate_weight.shape:
        raise ValueError(
            "gate_weight and up_weight must have the same (experts, intermediate, hidden) shape"
        )
    num_experts, intermediate_size, hidden_size = gate_weight.shape
    if hidden_states.shape[1] != hidden_size:
        raise ValueError("hidden_states and expert weights have incompatible hidden dimensions")
    if down_weight.shape != (num_experts, hidden_size, intermediate_size):
        raise ValueError("down_weight must have shape (experts, hidden, intermediate)")
    token_count, top_k = selected_experts.shape
    if token_count == 0:
        return hidden_states.new_empty((0, hidden_size))
    if top_k == 0:
        return hidden_states.new_zeros((token_count, hidden_size))

    torch._assert_async(
        (selected_experts >= 0).all(),
        "selected_experts contains a negative expert index",
    )
    torch._assert_async(
        (selected_experts < num_experts).all(),
        "selected_experts contains an out-of-range expert index",
    )
    expert_inputs = hidden_states.unsqueeze(0).expand(num_experts, -1, -1)
    if hidden_states.is_cuda:
        gate = bmm_persistent(expert_inputs, gate_weight.transpose(1, 2))
        up = bmm_persistent(expert_inputs, up_weight.transpose(1, 2))
        expert_outputs = bmm_persistent(F.silu(gate) * up, down_weight.transpose(1, 2))
    else:
        gate = torch.bmm(expert_inputs, gate_weight.transpose(1, 2))
        up = torch.bmm(expert_inputs, up_weight.transpose(1, 2))
        expert_outputs = torch.bmm(F.silu(gate) * up, down_weight.transpose(1, 2))
    expert_outputs = expert_outputs.permute(1, 0, 2)
    gather_indices = selected_experts.unsqueeze(-1).expand(-1, -1, hidden_size)
    route_outputs = torch.gather(expert_outputs, 1, gather_indices)
    route_outputs = route_outputs * routing_weights.unsqueeze(-1).to(route_outputs.dtype)

    output = route_outputs[:, 0]
    for slot_index in range(1, top_k):
        output = output + route_outputs[:, slot_index]
    return output
