# DeVA execution constraint audit

Audit date: 2026-09-26.

DeVA is eligible architecturally: the official repository is a PyTorch
implementation, exposes joint video/action inference, publishes a 5.11 GB
LIBERO guidance checkpoint, and documents evaluation on one GPU.

The official loader nevertheless requires the separately released
`nvidia/Cosmos-Predict2-2B-Video2World` base assets before it can construct the
video tokenizer and text-conditioning path. That Hugging Face repository is
gated. This environment has no accepted Hugging Face credential, and pinned
requests for the required `model-480p-16fps.pt` and text-encoder shards return
HTTP 401. The public DeVA post-training checkpoint alone is therefore not a
runnable learned-weight policy.

Pinned sources checked:

- Official PyTorch repository: <https://github.com/Mq-Zhang1/deva>, commit
  `19d1678df99fcf98569149708be65c1b63ba8861`.
- Public DeVA weights: <https://huggingface.co/mengqz9/DeVA>, commit
  `4f321201940669086389fbf144f62edc829cbbb5`.
- Required gated base: <https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World>,
  commit `f50c09f5d8ab133a90cac3f4886a6471e9ba3f18`.

No mirror, converted checkpoint, random initialization, or JAX substitute is
used. DeVA remains untested rather than being marked PASS or FAIL. It can be
resumed after the user accepts NVIDIA's model terms and exposes a read token.
