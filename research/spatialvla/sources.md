# SpatialVLA sources

- Official PyTorch repository: https://github.com/SpatialVLA/SpatialVLA
- Pinned source commit: `18fd74b2a633ec8d9ec7aadcd803969555cc9fbd`
- Official public checkpoint: https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt
- Pinned checkpoint revision: `886c4557dfde53b44c0f74fde7d89bf83a26c3cd`
- Published inference target: the checkpoint's `example.png`, prompt
  `What action should the robot take to pick the cup?`, and unnormalization key
  `bridge_orig/1.0.0`
- Unrelated companions and additional targets: real observations from the
  public LeRobot LIBERO episode-zero fixture already recorded by the other
  integrations in this repository

All learned tensors come from the two official PyTorch safetensors shards.
The checkpoint contains 4,027,854,731 parameter elements. No JAX artifact,
conversion, randomly initialized policy component, or substitute model is used.
