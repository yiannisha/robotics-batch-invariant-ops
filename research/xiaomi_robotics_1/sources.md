# Xiaomi-Robotics-1 sources

- Official native PyTorch source: <https://github.com/XiaomiRobotics/Xiaomi-Robotics-1>
- Official complete RoboCasa checkpoint: <https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa>
- Public real RoboCasa observations: <https://huggingface.co/datasets/daixianjie/robocasa_mg_lerobot>
- Official RoboCasa benchmark: <https://robocasa.ai/>

The checkpoint includes the complete learned Qwen3-VL vision/language model,
36-layer cached-KV action DiT, projections, tokenizer, image processor, custom
PyTorch model/processor code, and RoboCasa action statistics. No separate base
model, JAX implementation, or checkpoint conversion is involved.

The public fixture provides the same left-base, right-base, and wrist camera
roles used by Xiaomi's evaluator. Its stored low-dimensional state is EEF pose,
whereas Xiaomi consumes seven Panda joint positions plus one gripper value; the
adapter therefore uses a deterministic plausible joint-space state while
retaining real image frames and the real RoboCasa task instruction.
