# DreamZero DROID on the multi-GPU host

Updated 2026-11-08 UTC. Completed native inference and preliminary paired simulation. See `preliminary_results.md` and `results/dreamzero/preliminary/summary.json` for verified results. `blocked.md` describes the previous single-H100 host.

## Verified setup

Both requested experiment repositories are checked out. The official DreamZero source is the sibling `dreamzero` checkout at `ab790c198fbce33503358efbbd4187ce9a89adf3`, with no source modifications.

Scratch root: `/scratch/hadjiyiannis_3_gmail_com/dreamzero`.

- `checkpoint/`: pinned DROID revision `96ad344138c66e82536422432ad742f015784942`; all ten safetensors shards match their Hub content hashes. ONNX/TRT artifacts excluded.
- `venv/`: Python 3.11, torch 2.8.0+cu129, official editable DreamZero dependencies. UV install needs `--index-strategy unsafe-best-match` with the cu129 extra index.
- `sim-evals/`: official DROID simulation and downloaded public assets; `sim-venv/` contains IsaacLab 2.2 / IsaacSim 5.0. Its locked torch build lacks Blackwell kernels, so torch/vision/audio were replaced with 2.8.0/0.23.0/2.8.0 cu129. Flatdict needs setuptools<81 and `--no-build-isolation-package flatdict` when installing. System libglu1-mesa was also installed.
- Official scene 1 renders all three cameras. `results/dreamzero/preliminary/initial_cameras.png` records the views.
- Simulator restore produced the recorded starting-state digest `adbce4a159df15f58122612e7faf758221f3c4c0918672d08550b75614634d00`. All condition starts must match that digest. PhysX rounds quaternion components on write, so compare repeated realized states rather than the source write command.

## Completed run

All 12 inference requests and four 16-step trajectories completed. Fixed-input singleton and batch-2 repeats were exact. Both first-eight-step repeat trajectories were exact; adding a companion changed realized arm joints by up to 0.006523530 radians in that window. Native inference and simulator processes were stopped after artifact collection.

The completed worker used torchrun PID 2067636 (tool session 42546), and the simulator used PID 1921131 (session 55103). Their retained logs are `results/dreamzero/native-worker.log` and `results/dreamzero/simulation.log`. Original scratch logs are `native-worker2.log` and `pairs.log`. Do not interpret a stale scratch `queue/ready.json` as a live worker.

The first numeric-device launch reached a GPU occupied by training and ran out of memory. Use UUID selection on this host. A later worker needed the official server's compiler limit of 800; the completed run includes that setting.

## Protocol and scripts

`run_dreamzero_native_worker.py` calls official `GrootSimPolicy.lazy_joint_forward_causal` with two CFG ranks and unmodified native numerical operators. DiT cache is disabled. The source falls back from unavailable TE/FlashAttention to its native PyTorch SDPA implementation. Text/image/VAE use the official torch.compile setup. Target normalized-input and noise hashes are recorded for every request.

`run_dreamzero_droid_pairs.py` runs native singleton, native batch-2, and a repeat of each, first on fixed inputs then for 16 simulator steps with inference every 8 steps. A valid companion observation is collected after a small joint-1 motion. Both requests have the same instruction, avoiding the official encode_prompt loop's masking of all rows with each sample's text length. Each episode starts from the same saved physical scene and cached initial camera observation. Reset and hash checks must pass before interpreting differences.

The queue transport is local NPZ files in scratch `queue/`. It is serialized and starts at index zero. Do not restart a producer or worker into a partially consumed queue without inspecting the live handles and preserving existing evidence.

The simulator requires `OMNI_KIT_ACCEPT_EULA=YES`, `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json`, CUDA initialization and pip Warp initialization before Kit. Single-GPU rendering is configured in the harness.

`summarize_dreamzero_pairs.py results/dreamzero/preliminary` audits raw actions, target controls, simulator starts, robot joints and object states against both repeat conditions. Run after all four trajectory files exist. No task-success claim follows from short trajectory differences.

## Verification and scope

All ten checkpoint shard hashes match the pinned Hub content hashes. Twelve finite native outputs, all fixed target input/noise hashes, four matching initial scene states, and all four trajectories were audited. Source compilation and observation conversion checks passed; the result audit was also checked to reject mismatched target noise. No remaining requested experiment work is pending. This is a one-scene preliminary result, with no task-success or long-horizon determinism claim.

## Reproduce with the installed environments

Run these from the robotics-batch-invariant-ops root. Use a new empty queue and output directory for every run. The first command stays live; run the simulator in a second shell with the same DZ_RUN value. Stop the worker after the simulator finishes.

```bash
export DZ_ROOT=/scratch/hadjiyiannis_3_gmail_com/dreamzero
export DZ_RUN=rerun_001
mkdir -p "$DZ_ROOT/$DZ_RUN/queue"
CUDA_VISIBLE_DEVICES=GPU-94757602-37c3-0344-da17-6ac647c8325b,GPU-225bc55c-3ea0-e312-43ae-c673a57f5dde \
HF_HOME="$DZ_ROOT/hf" OMP_NUM_THREADS=4 PYTHONPATH="$PWD/../dreamzero" \
"$DZ_ROOT/venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=2 \
 scripts/run_dreamzero_native_worker.py --model-path "$DZ_ROOT/checkpoint" --queue "$DZ_ROOT/$DZ_RUN/queue"
```

```bash
export DZ_ROOT=/scratch/hadjiyiannis_3_gmail_com/dreamzero
export DZ_RUN=rerun_001
OMNI_KIT_ACCEPT_EULA=YES VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
CUDA_VISIBLE_DEVICES=GPU-e34fe940-5eed-cc24-23e4-18c32eacf887 \
"$DZ_ROOT/sim-venv/bin/python" scripts/run_dreamzero_droid_pairs.py \
 --queue "$DZ_ROOT/$DZ_RUN/queue" --output "results/dreamzero/$DZ_RUN"
"$DZ_ROOT/venv/bin/python" scripts/summarize_dreamzero_pairs.py "results/dreamzero/$DZ_RUN"
```
