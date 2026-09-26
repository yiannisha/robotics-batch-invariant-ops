# Cosmos 3 Edge Policy sources

- Official PyTorch framework: <https://github.com/nvidia-cosmos/cosmos-framework>
- Official DROID policy instructions: <https://github.com/nvidia-cosmos/cosmos-framework/blob/main/docs/action_policy_droid_server.md>
- Released Edge DROID policy: <https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID>
- Released observations: <https://huggingface.co/datasets/nvidia/Cosmos3-DROID>
- Wan2.2 VAE dependency: <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B>

The numerical investigation follows the official `RobolabPolicyService` and
`OmniMoTModel.generate_samples_from_batch` path. The relevant upstream source
boundaries are:

- `cosmos_framework/scripts/action_policy_server_robolab.py`: policy loading,
  DROID transforms, released Edge prompt/guidance configuration, and action
  decoding.
- `cosmos_framework/model/generator/omni_mot_model.py`: per-example seeds,
  four-step UniPC sampling, classifier-free guidance, and joint action/vision
  output handling.
- `cosmos_framework/model/generator/mot/attention.py`: B=1 dense shortcut and
  B>1 packed variable-length attention dispatch.
- `cosmos_framework/model/generator/mot/inference_text_kv_memory.py`: cached
  text-KV attention dispatch on later denoising steps.
- `cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py`:
  released causal video decoder and its channel-wise RMS normalization.

All repositories and snapshots are pinned in `model_commit.txt` and
`checkpoint.txt`. No JAX implementation or converted JAX checkpoint is used.
