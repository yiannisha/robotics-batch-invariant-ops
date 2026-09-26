# DreamZero status: BLOCKED for the single-H100 benchmark

The current official DreamZero PyTorch release cannot be executed under this
plan's single-H100 constraint.

At official repository commit `ab790c1`, the inference prerequisites specify a
minimum of two GPUs and the documented command launches two distributed NCCL
processes. The released DreamZero-DROID model uses the 14B Wan2.1 backbone. Its
ten public PyTorch safetensors shards total 45,849,242,646 bytes; the complete
Hugging Face repository is 64,789,159,581 bytes once the optional ONNX and
TensorRT artifacts are included. DreamZero-AgiBot carries the same 45.85 GB
PyTorch shard set.

The repository contains a single-process Wan2.2 5B server, but the publisher
does not provide a trained DreamZero checkpoint for that path. The only public
publisher model repositories are DreamZero-DROID and DreamZero-AgiBot. Training
a smaller model is outside this inference investigation and would not validate
a released policy.

This is not merely an installation inconvenience:

- the official runnable release requires a multi-GPU setup, which the plan
  explicitly excludes;
- only one H100 is available;
- the workspace filesystem is 30 GB, smaller than either 45.85 GB PyTorch
  checkpoint, so the weights cannot be staged locally for an unsupported
  single-device experiment.

No model output, baseline result, or PASS is claimed. Revisit DreamZero if the
publisher releases a trained single-H100 checkpoint or a smaller public model.
