# DreamZero DROID: preliminary native batch-variance result

The released DROID checkpoint ran successfully on this host with two RTX PRO 6000 Blackwell GPUs. Native singleton versus batch-2 inference changed the target's actions and produced different simulated robot motion. This is a preliminary result from one DROID scene and one initial state.

## Evidence

| Measurement | Singleton vs batch 2 | Singleton repeat | Batch-2 repeat |
|---|---:|---:|---:|
| Maximum joint-command difference, fixed 24-action chunk | 0.156030744 rad | 0 | 0 |
| Maximum executed arm-command difference, first 8 steps | 0.009406984 rad | 0 | 0 |
| Maximum realized arm-joint difference, first 8 steps | 0.006523530 rad | 0 | 0 |
| Maximum realized joint difference, all 16 steps | 0.035597801 rad | 0.008755267 rad | 0.011308564 rad |

The fixed-input comparison changed 159 of 168 target joint commands and 16 of 24 continuous gripper commands. All four fixed-input runs used identical target normalized tensors and identical initial video/action noise, verified by SHA-256. Repeated inference within each batch condition was bit-for-bit identical. All four trajectories began with the same realized simulator-state digest. The two repeat trajectories were exact through the first eight steps, making this the clearest controlled demonstration of a batch effect on physical robot motion.

Differences start at the first executed action. The full 24-action chunk has a larger maximum than the executed first-eight-action window. The second inference at step 8 consumes new rendered observations; repeat actions and trajectories then differ as well. Thus, the full 16-step effect is a preliminary observation with a measured repeat floor, not a fully deterministic long-horizon evaluation. No task-success difference is claimed.

## Protocol

The four conditions reproduce the native portion of PiZero's paired protocol: singleton, companion batch, singleton repeat, companion-batch repeat. Each short rollout executes 16 simulator steps, requesting a new action chunk every eight steps. Native DreamZero policy preprocessing, denoising, and action unnormalization are used. Numerical operators are unmodified. DiT caching is disabled consistently across conditions; the normal autoregressive caches are reset at episode start and retained between the two chunks.

The companion is another valid simulated observation, collected after a small joint-1 motion. It uses the same instruction as the target. This avoids an unrelated native text-masking issue: `encode_prompt` zeros all batch rows using every sample's sequence length. The target's batch slot is always zero.

Scene 1 is the official cube-and-bowl DROID simulation. A saved physical scene and cached initial three-camera observation are reused across conditions. Simulator reset equality is checked against the realized scene after PhysX quaternion normalization. Subsequent images are newly rendered. The original native seed, 1140, is retained; no alternative noise generator is substituted.

## Runtime identity

- DreamZero source: `ab790c198fbce33503358efbbd4187ce9a89adf3`, clean official checkout.
- DROID checkpoint: `GEAR-Dreams/DreamZero-DROID` revision `96ad344138c66e82536422432ad742f015784942`. All ten shards match Hub content hashes.
- Python 3.11; torch 2.8.0+cu129; two distributed classifier-free-guidance ranks.
- Native GPUs: nvidia-smi indices 4 and 5, selected by UUID. Simulator: GPU 0.
- Attention: official native PyTorch SDPA fallback because Transformer Engine and FlashAttention are absent. Results apply to this backend/software/hardware configuration; no cross-backend claim is made.
- Official text/image/VAE torch.compile path, with the official server's recompile limit of 800.

## Retained artifacts

All paths below are relative to the robotics-batch-invariant-ops checkout:

- `results/dreamzero/preliminary/summary.json`: comparison metrics and control audit.
- `results/dreamzero/preliminary/fixed_input.json`: four complete fixed-input action chunks and input/noise identities.
- `results/dreamzero/preliminary/native_*.json`: all four 16-step trajectories, actions, rewards, and scene states.
- `results/dreamzero/preliminary/initial_cameras.png`: verified rendered camera views.
- `results/dreamzero/preliminary/joint_divergence.png`: batch and repeat trajectory differences.
- `results/dreamzero/raw/`: all 12 request NPZs, result NPZs, and metadata JSONs; `raw_manifest.json` hashes all 36 files.
- `results/dreamzero/checkpoint_identity.json`, `provenance.json`, environment files, and runtime logs: source, weights, software, scripts, and inference evidence.

The runners are `scripts/run_dreamzero_native_worker.py`, `scripts/run_dreamzero_droid_pairs.py`, and `scripts/summarize_dreamzero_pairs.py`. Setup details and troubleshooting are in `research/dreamzero/local_setup_status.md`. The old `blocked.md` remains a historical record of the single-H100 restriction; that restriction is resolved for this experiment.

The requested setup and preliminary native pair checks are complete. No expanded campaign was run.
