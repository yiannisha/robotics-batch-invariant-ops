"""Cosmos 3 Edge Policy DROID adapter using NVIDIA's official PyTorch code.

This reuses the Cosmos 3 policy inference and batch-invariant attention repair
from the Nano adapter.  Set ``COSMOS3_EDGE_CHECKPOINT`` to a local snapshot of
``nvidia/Cosmos3-Edge-Policy-DROID`` and ``COSMOS3_DROID_SAMPLE_DIR`` to the
real DROID fixture described by the Nano adapter.

The JSON prompt and inclusive guidance interval below are the official Edge
serving arguments published by NVIDIA.
"""

from scripts.model_invariance.adapters.cosmos3_nano_policy import (
    Cosmos3NanoPolicyAdapter,
)


class Cosmos3EdgePolicyAdapter(Cosmos3NanoPolicyAdapter):
    name = "nvidia-cosmos3-edge-policy-droid-official-pytorch"
    checkpoint_environment_variable = "COSMOS3_EDGE_CHECKPOINT"
    format_prompt_as_json = True
    guidance_interval = (960.0, 1001.0)


adapter = Cosmos3EdgePolicyAdapter()
