# Cross-model PiZero-equivalent campaign

> Status correction (2026-10-05): the existing `results/cross_model/<model>/`
> measurements are an integration pilot, not a completed π0-equivalent
> campaign. The earlier `29/29 MEASUREMENT CORE COMPLETE` and passing
> `final_audit.json` used self-declared reduced schedules and must not be cited
> as full coverage. Run `python scripts/audit_full_cross_model_campaign.py`
> for the authoritative gate. Full reruns are written under each model's
> `full/` directory so the pilot evidence remains intact.
> After running that gate, `python scripts/report_full_coverage_status.py`
> writes progress-only per-path `status.json` files and aggregate JSON/CSV
> coverage tables. These tables cite the audit SHA and never substitute for a
> passing full-coverage audit.
> A 2026-10-05 artifact preflight found that 15 of the 29 paths no longer have
> every checkpoint artifact named in their pilot provenance on local disk.
> Those paths require exact pinned-artifact rehydration before full execution;
> a historical pilot record is not evidence that its checkpoint is runnable now.
> The exact π0-FAST `model.safetensors` was rehydrated on 2026-10-05 from
> upstream revision `840f4b503f4c09110421c33c810a85b6684fd658` and
> verified against the pilot-pinned SHA-256. This removes that one checkpoint
> blocker but does not complete π0-FAST experiments. The refreshed preflight
> now counts 14/29 paths with at least one missing local pinned artifact;
> the remaining filesystem space after this rehydration is about 6 GiB, so
> the other large checkpoints require additional storage before rehydration.
> OpenHelix's two missing checkpoint files were subsequently rehydrated from
> upstream revision `1a6fabbe148af045ed0710d007ed24c36fa83a53` and
> verified against the pinned SHA-256 values. The refreshed preflight is now
> 13/29 paths with missing artifacts, but only about 3.7 GiB remains; no
> OpenHelix full experiment has yet been run.

The exact π0-FAST checkpoint has also passed an offline CPU construction and
all-tensor load verification through the official LeRobot policy: all 604
checkpoint/state-dict keys matched, covering 7,729,724,352 source tensor
bytes, and every tensor was exactly equal after the policy's declared dtype
conversion. The pinned report is
`results/cross_model/pi_fast/full/checkpoint_cpu_load_verification.json`.
This rules out the official loader's silent random/partial-load fallback for
that invocation, but is preflight evidence only; no π0-FAST numerical,
timing, serving, held-out, or closed-loop experiment is complete.

OpenVLA-OFT now has a derived, hash-pinned deterministic replay manifest for
all 1,000 distinct real LIBERO-10 observations (100 source episodes, 200
diagnostic and 800 held out). Its adapter can preprocess those frames and
compose unrelated-task batches with native longest-in-batch or repaired
fixed-37-token right padding. This is input readiness only; no OpenVLA-OFT
full diagnostic, numerical freeze, held-out, or timing run has passed. The
checkpoint is LIBERO-Spatial fine-tuned while the frozen real-input corpus is
LIBERO-10, so these replay measurements probe inference behavior on real
LIBERO shapes/tasks and must not be described as matched-domain success rates.
The preflight now counts 8/29 paths with a frozen replay manifest.
For UVA action and joint/video paths, the 64-frame pilot fixture is not a
real source limit. A CPU-only inventory of the pinned public embedded-image
LIBERO parquet corpus found 1,000 distinct 16-frame contiguous real contexts
across 100 episodes, with 174–455 frames per episode. The full source-row
schedule and exact parquet hashes are recorded in
`results/cross_model/uva/full/context_source_inventory.json`. This is only
source feasibility evidence. The UVA adapter can now decode these contexts
directly from the pinned parquet rows; CPU checks of the first and last
contexts returned 16 real 256×256 RGB frames each, with their anchor frames
pixel-identical to the existing frozen single-frame replay. UVA still needs
diagnostic/held-out runs and the rest of the full campaign. The action path
now has a separate 390,528,128-byte frozen array holding three explicit
checkpoint-shaped VAE/action noises for every real context; its derived
replay manifest passes the independent source/window/noise checks, but lacks
a numerical freeze. The joint video/action path needs its own 100-step
video-diffusion noise tensor, bringing its required three-noise array to
20,247,936,128 bytes (about 18.9 GiB), beyond the currently available disk.
That path is not marked source-limited or complete. No UVA numerical result
is claimed here. The preflight now counts 9/29 paths with frozen replay
manifests.

This campaign treats the kernels in this repository as established,
batch-invariant infrastructure.  It measures complete pretrained policies; it
does not reopen kernel-correctness work.  New measurements live under
`results/cross_model/<model>/` and must not be confused with the earlier
`research/` evidence.

## Scope

The 23 additional runnable model families (excluding the already completed
π0 reference campaign) are:

1. pi-zero-pytorch π0.5
2. LeRobot π0-FAST
3. InternVLA-A1.5
4. MolmoAct2
5. GR00T N1.7
6. Cosmos 3 Nano
7. Cosmos 3 Edge
8. X-VLA
9. UVA
10. SmolVLA
11. OpenVLA-OFT
12. DexVLA
13. LingBot-VLA 2.0
14. BAAI UniVLA
15. SpatialVLA 4B
16. RDT-1B
17. CogACT-Small
18. OpenHelix
19. WSA Base
20. VITRA-VLA-3B
21. OpenDW DW05
22. OpenWAM Alpha
23. Xiaomi-Robotics-1

Families with materially different public execution paths are retained as
separate audit configurations: InternVLA standard/optimized, Cosmos Nano and
Edge policy/decoded-video, UVA action/joint, OpenDW action/joint, and OpenWAM
policy/decoded-video.  This yields 29 audited configurations across the 23
families.  A path is never counted as complete merely because another path in
the same family passed.

## Current full-evidence status (2026-10-05)

No path has passed the independent full-coverage audit. X-VLA completed an
initial four-condition, 50-reset-per-task LIBERO campaign, but the paired
reducer correctly rejected it: tasks 3, 8, and 9 had 49 initial-observation
mismatches each between B1 and B4 (147 total). The official `LiberoEnv`
increments its init-state cursor on Gym terminal auto-resets, making the next
manual reset depend on vector batch size despite matched reset seeds. That
entire 2,000-episode run and its machine-readable mismatch audit are preserved
under `paired_invalid_reset_stride/`; it is not paired evidence. A corrected
wrapper now assigns every vector slot the requested initialization ID before
the official reset and records that assignment in each raw rollout. A clean
four-condition rerun is active under `paired/`, with the continuation
supervisor requiring 500 audited resets and official metrics before launching
each next condition. The supervisor stops on any failed condition. X-VLA's
earlier
5×20×200 complete-policy and four-mode serving schedules used the first eight
frames of one replay episode. Their raw samples remain under
`xvla/full/performance/homogeneous_same_episode/` as single-task shape
evidence, but they do not establish cross-task performance coverage and are
excluded from the full gate. SmolVLA has older 5×20×200 timing for the
kernel-only dynamic-padding variant under
`smolvla/full/performance/kernel_only_dynamic_padding/`; it likewise does not
count as performance evidence for its repaired invariant path. Fresh
hash-pinned cross-task input selections are required for the main timing files.
An interrupted SmolVLA repaired-path timing run using eight frames from one
held-out episode is preserved without deletion under
`smolvla/full/performance/homogeneous_same_episode/policy_partial_timings.jsonl`;
its 5,674 warmup/timing records plus one environment record are not full
coverage. The replacement diagnostic-only cross-task policy timing completed
all 40 groups (8,000 timed calls, 800 warmups) and passed the independent
timing sub-audit. Its matched four-mode serving timing also completed all
20 groups (4,000 timed calls, 400 warmups), including fixed B8/48-token
padding, and passed its independent timing sub-audit. Twelve matched
kernel-launch profiles also passed their independent sub-audit. These
components do not imply completion of diagnostic, held-out, operator, queue,
or paired closed-loop requirements. The measurements use the explicit indices
listed below.
X-VLA's replacement cross-task full policy timing has also passed the
independent timing sub-audit (8,000 timed calls, 800 warmups, 40 groups).
Its matched four-mode serving timing passed the separate sub-audit (4,000
timed calls, 400 warmups, 20 groups), and all 12 matched kernel-launch
profiles passed their sub-audit. Its real request queue also passed: 4,000
verified requests over four modes × four offered-load bands, with a matching
summary. Diagnostic, held-out, operator, and paired closed-loop work remains
outstanding; its feasible B32 full timing and larger serving/queue
campaigns are also still required. These passed components do not complete
the X-VLA path.
Both X-VLA B16 complete-policy modes have since passed their separate
5×20×200 independent timing audits (1,000 timed calls and 100 warmups each).
B32 native and invariant also passed their separate 5×20×200 audits, each
with 1,000 timed calls and 100 warm-ups. The larger serving/queue schedules
remain outstanding.

SmolVLA, X-VLA, InternVLA-A1.5 standard/optimized, GR00T N1.7, and WSA Base
now have a frozen real-observation replay smoke with all 39 required
arrangements for one request/noise pair, plus B16/B32 capacity probes. The
invariant path is exact on that single pair for each; this is not a 600-pair
diagnostic completion or held-out evidence. X-VLA has five such completed
pairs and can resume its append-only diagnostic run.

The unrelated-task replay exposed prompt-length dependence in SmolVLA and
GR00T. Their official native paths retain longest-in-batch prompt padding;
the repaired invariant paths use, respectively, fixed right-padding to 48
tokens and fixed left-padding to 156 tokens. The full raw records include this
execution configuration and native-vs-repaired singleton fidelity. Earlier
dynamic-padding failures are preserved under `full/diagnostic/probes/`.
InternVLA's public optimized CUDA-graph cache crashed on B1→B2→B1 and
produced nonfinite actions when reused at the same size. Its adapter now
recaptures the public graph once per policy call. That avoids the observed
failures, but optimized-backend performance must include this recapture cost;
the failed variants remain in `full/diagnostic/probes/`.
For four-mode serving, fixed-size padded batches also fix the language extent:
48 right-padded tokens for SmolVLA and 156 left-padded tokens for GR00T. They
still run native operators and use their own same-position fixed-shape
reference. The measured fixed shape is recorded and checked alongside timing,
queue, and launch evidence.

The SmolVLA checkpoint-matched official LeRobot LIBERO evaluator completed
its independent feasibility probe: one episode on each of 10 LIBERO-10 tasks,
with 2/10 successes. The raw `eval_info.json`, 10 task videos, checkpoint
hash, command, and environment are under
`full/closed_loop/probe_output/` and `full/closed_loop/evaluator_attempt.json`.
This is **not** the required paired native-versus-repaired success-rate
campaign, and it proves SmolVLA is not eligible for an evaluator-unavailable
exception. Missing Python dependencies, LIBERO's interactive first-use
configuration, and missing EGL loader libraries were resolved in a separate
evaluator environment; do not cite those earlier setup failures as blockers.
The X-VLA checkpoint-matched LeRobot LIBERO evaluator also completed a
10-task, one-episode-per-task feasibility probe (0/10 successes). Its raw
metrics and videos are under `xvla/full/closed_loop/probe_output/`. The
zero-success probe is not an evaluator blocker and does not replace paired
native-versus-invariant success-rate testing.
An explicitly separate paired-evaluator pilot now has 10 native-singleton
episodes and 40 physical-B4 native-dynamic episodes (four per task). The
shared reset on each of 10 tasks had identical initial physical-observation
and first-noise hashes across singleton/dynamic, while its first action
differed by 1.348e-5 to 4.792e-5. These pilot rollouts do not count as the
required 50 resets/task × four conditions. A previous `n_episodes=1`
"dynamic" attempt was in fact capped to physical B1 by LeRobot and is
preserved under `paired_pilot/invalid_dynamic_episode_cap_1/`, not counted.
`scripts/run_lerobot_paired_libero.py` retains the official evaluator while
pinning per-reset noise and caching each rollout for resume; the pilot
verified identical reset/noise hashes across B1/B4. The separate reducer
`scripts/summarize_lerobot_paired_libero.py` requires all four conditions and,
outside a pilot directory, 50 matched resets on each of 10 tasks before it
will emit paired bootstrap intervals, disagreements, and trajectories. The
full four-condition runs and reducer output are still outstanding. A future
`completed` evaluator record must also pin the official evaluator source path
and SHA-256; the independent audit checks each condition manifest against that
source, its exact checkpoint, the official policy implementation, and the
paired wrapper before accepting the metrics.
The X-VLA full native-singleton 50-reset/task condition has been started with
the explicitly recorded official evaluator precision policy; the other three
conditions and the paired reducer are not complete.
`scripts/audit_lerobot_paired_prefix.py` independently checks each live
condition's cached task/reset keys, contiguous initialization prefix, physical
batch size, finite 20-state/7-action trajectories, and request-noise identity.
The X-VLA native-singleton prefix passed at 156/500 planned episodes; this
read-only checkpoint is not paired success-rate completion.
The cache-resume path has not yet been exercised in a completed evaluator run.

SmolVLA's operator manifest now includes the actual traced SigLIP
`aten::scaled_dot_product_attention` Q/K/V shape and noncontiguous strides,
in addition to matrix/Conv2d diagnostic cases. The qualifier records native,
explicit per-matrix reference, and invariant exactness and latency for each
seed. This is prepared scope, not a completed 100-seed operator sweep.
The full operator gate now requires the π0-plan sweep
`B={1,2,4,8,16,31,32,33,64}` for every configuration; policy-level batch
capacity does not silently waive a standalone operator size. Any genuine
operator OOM will need its own reproducible evidence and audit disposition.
The qualifier accepts `--configuration-ids` with `--resume` so the full
100-seed sweep can be run one configuration at a time without changing the
configuration-manifest hash or weakening the final audit.
X-VLA's operator manifest has been expanded from its five pilot Conv2d/MM
cases to include every one of the 10 distinct SDPA Q/K/V shape/stride
signatures in its pinned B8 ATen trace, plus QK/PV and the π0 action-query
diagnostic products. These are configured cases only; no X-VLA 100-seed
operator raw result is counted yet.

## Full-run timing commands

After its documented checkpoint, source, and fixture environment variables are
set, each adapter must use fresh `full/` outputs. The older directory-level
files are pilots, even if their metadata declares a 5×20×200 target.

```bash
python scripts/check_model_batch_invariance.py \
  --adapter ADAPTER \
  --batch-sizes 1,2,3,4,5,7,8,9,15,16,17,31,32,33,64 \
  --output results/cross_model/MODEL/full/diagnostic/baseline.json

python scripts/check_model_batch_invariance.py \
  --adapter ADAPTER \
  --batch-sizes 1,2,3,4,5,7,8,9,15,16,17,31,32,33,64 \
  --batch-invariant-ops \
  --output results/cross_model/MODEL/full/diagnostic/fixed.json

python scripts/check_model_batch_transformations.py \
  --adapter ADAPTER \
  --batch-sizes 2,4,8,16,32 \
  --output results/cross_model/MODEL/full/diagnostic/batch_transformations.json

python scripts/benchmark_model_policy.py \
  --adapter ADAPTER \
  --batch-sizes 1,2,4,8 \
  --replay-manifest results/cross_model/MODEL/full/replay_manifest.json \
  --replay-indices EIGHT_DISTINCT_DIAGNOSTIC_REPLAY_INDICES \
  --sessions 5 --warmup 20 --repetitions 200 --full-campaign \
  --output results/cross_model/MODEL/full/performance/policy_timings.jsonl

python scripts/benchmark_model_serving.py \
  --adapter ADAPTER --logical-batch-size 4 --fixed-batch-size 8 \
  --replay-manifest results/cross_model/MODEL/full/replay_manifest.json \
  --replay-indices THE_SAME_EIGHT_DISTINCT_DIAGNOSTIC_REPLAY_INDICES \
  --sessions 5 --warmup 20 --repetitions 200 --full-campaign \
  --output results/cross_model/MODEL/full/performance/serving_timings.jsonl
```

The benchmark is append-only and may be continued with `--resume`; completed
samples are retained and timing resumes after a fresh untimed warm-up. Full
timing now requires an explicit hash-pinned diagnostic replay selection: the
first eight rows of these frozen manifests can be held-out frames of one
episode and must not silently stand in for cross-task coverage. For the seven
currently frozen LIBERO manifests (SmolVLA, X-VLA, π0-FAST, InternVLA
standard/optimized, GR00T, and WSA Base), diagnostic indices
`80,190,330,340,410,450,510,550` represent eight distinct tasks and
episodes. Use the same selection for policy, serving, queue, and launch-profile
measurements. Model-specific
multi-input, trace, official-equivalence, operator-shape, joint-output, and
decoded-video commands remain in `scripts/` and are run in addition to this
shared surface.

For X-VLA's feasible larger policy shapes, the hash-pinned B16 selection is
`80,130,190,330,340,410,450,510,520,550,580,610,700,740,750,760`
(16 distinct episodes, all 10 tasks). The B32 selection appends
`800,870,890,940,81,131,191,331,341,411,451,511,521,551,581,611`
to that list (all 20 diagnostic episodes, all 10 tasks). Both selections
passed the independent diagnostic-input diversity check. Each size and
implementation has a separate full 5×20×200 timing artifact.

## SmolVLA reproduction

The first fresh campaign uses:

- LeRobot commit `e595b7902714ba51f91e47523f66f89c5181b649`;
- `HuggingFaceVLA/smolvla_libero` revision
  `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`;
- model SHA-256
  `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`;
- `HuggingFaceVLA/libero` revision
  `86958911c0f959db2bbbdb107eb3e17c5f9c798e`.

The public dataset parquet embeds its RGB PNG payloads.  Recreate the adapter
fixture with:

```bash
python scripts/prepare_lerobot_libero_fixture.py \
  --parquet /workspace/assets/smolvla/libero/data/chunk-000/file-000.parquet \
  --tasks /workspace/assets/smolvla/libero/meta/tasks.parquet \
  --output /workspace/assets/smolvla/fixture --frames 64

export LEROBOT_SMOLVLA_CHECKPOINT=/workspace/assets/smolvla/checkpoint
export LEROBOT_SMOLVLA_SAMPLE_DIR=/workspace/assets/smolvla/fixture
```

`results/cross_model/smolvla/provenance.json` binds the executed environment,
source revisions, checkpoint, and fixture hashes.  The baseline, invariant,
multi-input, transformation, first-divergence, singleton-fidelity, and raw
performance records are retained beside it.

For the full SmolVLA replay, `scripts/prepare_smolvla_full_replay.py` freezes
100 real LIBERO episodes (10 frames each) from the pinned dataset, separates
20 diagnostic from 80 held-out episodes with two diagnostic episodes per task,
and stores three explicit noise tensors per observation. The fixture is at
`/workspace/assets/smolvla/full_replay`; its manifest is
`results/cross_model/smolvla/full/replay_manifest.json`. The original 64-frame
fixture remains a pilot/performance input, not held-out evidence.

The lazy replay adapter reads one observation/noise pair at a time.

Each full replay manifest now freezes 31 companion indices per observation.
The selected observations are from distinct, unrelated episodes, with other
tasks prioritized. The audit checks that the recorded request order matches
this manifest schedule. Earlier one-request smoke records that incorrectly
used adjacent frames of the target episode are preserved under
`diagnostic/invalid_same_episode_smoke/` and excluded from full coverage.

Diagnostic comparisons can be resumed without rewriting raw records:

```bash
export LEROBOT_SMOLVLA_SAMPLE_DIR=/workspace/assets/smolvla/full_replay
python scripts/run_model_replay_invariance.py \
  --adapter scripts.model_invariance.adapters.lerobot_smolvla \
  --manifest results/cross_model/smolvla/full/replay_manifest.json \
  --split diagnostic --batch-sizes 1 2 4 8 \
  --output results/cross_model/smolvla/full/diagnostic/invariance.jsonl
```

Held-out execution uses the same command with `--split heldout`, a frozen
`--freeze` record, and separate `full/heldout/` outputs. A diagnostic subset
or a file's mere existence never satisfies the full audit.

`scripts/freeze_model_numerics.py` creates the numerical freeze only after
all 600 diagnostic request/noise pairs have both 23-arrangement modes and the
invariant path is exact. Because this shared worktree contains unrelated
uncommitted changes, the freeze records hashes of the exact adapter, kernel,
checkpoint, manifest, and diagnostic bytes in addition to the Git SHA. The
held-out runner rechecks those bytes before using the freeze.

After the four-mode 5×20×200 timing series finishes, the real-clock queue
experiment is run with `scripts/benchmark_model_request_queue.py` and
summarized with `scripts/summarize_model_request_queue.py`. The queue runner
replays each recorded batch schedule for numerical verification after the
timed queue, so checking output equality does not inflate queue latency.

## Shared real LIBERO replay variants

The pinned `HuggingFaceVLA/libero` embedded-image replay can be adapted to
other LIBERO checkpoints without inventing observations. π0-FAST has a
deterministic, one-request-per-observation manifest under
`results/cross_model/pi_fast/full/`; its hash-pinned upstream LeRobot decoder
uses argmax when temperature is zero. X-VLA has a stochastic manifest under
`results/cross_model/xvla/full/` with three stored 30×20 noises per
observation. Both adapters load one replay request lazily.
π0-FAST's six local PaliGemma tokenizer files were checked byte-for-byte
against the gated official Google repository's public immutable file hashes
at revision `35e4f46485b4d07967e7e9935bc3786aad50687c`; all eleven local
FAST action-tokenizer files match their public upstream revision
`79ae83e3cbd8786dcb84b628569f8d076ca8151e`. The generated
`pi_fast/full/tokenizer_provenance.json` records every hash and passed 17/17.
This verifies local assets, not π0-FAST's as-yet-unrun full experiments.
The separate `scripts/run_lerobot_pi0fast_paired_libero.py` is prepared for
checkpoint-matched official LIBERO runs. It records the temperature-zero
greedy-decoding contract as the deterministic request equivalent of a noise
hash, checks exact tokenizers and every loaded checkpoint tensor after the
policy's declared dtype conversion, and
reuses the official evaluator/rollout cache. It has only CPU hook tests so far;
its checkpoint/tokenizer/source CPU preflight also passed and is recorded in
`pi_fast/full/closed_loop/preflight/preflight.json`. The GPU evaluator probe
and all π0-FAST closed-loop episodes remain outstanding.

Source-audited replay manifests have also been prepared for InternVLA-A1.5
(standard and optimized backends), GR00T N1.7, and WSA Base using their
checkpoint-shaped explicit noises. These are preparation artifacts only:
adapter GPU smoke tests, B16/B32 capacity probes, diagnostic freezes,
held-out comparisons, and full performance series remain separate required
work for each path.

These are public real-input variants, not pixel-identical copies of the
separately compressed `lerobot/libero` MP4 frames used in those pilots. The
first 64 state/episode/frame/task metadata rows match, but one decoded camera
frame differs. The variant is labeled in each full manifest and diagnostic
note; any task-success evaluation must use its matched official evaluator.

## Pilot outputs

All 29 configurations across the 23 families have preliminary numerical and
performance evidence. Regenerate the pilot aggregate JSON, CSV, Markdown,
per-configuration reports, and SVG figures with:

```bash
python scripts/summarize_cross_model_campaign.py
```

The outputs are rooted at `results/cross_model/`:

- `aggregate_results.json` and `aggregate_results.csv` contain the full
  cross-model comparison;
- `CAMPAIGN_REPORT.md` is the aggregate narrative/table;
- `<configuration>/REPORT.md` provides a report for each audited path;
- `figures/` contains complete-policy and serving latency figures;
- each configuration retains provenance, baseline/fixed comparisons,
  transformation records, first-divergence evidence, raw policy/serving JSONL,
  summaries, and five real-shape operator sessions.

## Closed-loop execution gate

The checkpoint-matched simulator surface and source evidence for every path are
recorded by:

```bash
python scripts/probe_closed_loop_runtime.py
python scripts/audit_closed_loop_applicability.py
```

SAPIEN/ManiSkill still fails Vulkan instance creation with
`ErrorIncompatibleDriver`; exact stderr and host metadata are retained in
`results/cross_model/closed_loop/rdt_maniskill_probe.json`.  MuJoCo EGL was
subsequently made executable with the pinned LeRobot LIBERO environment, so
X-VLA is running the full four-condition, 10-task, 50-reset campaign.  Its
first completed 2,000-episode attempt was rejected after an independent audit
found that terminal auto-resets advanced LIBERO initialization IDs differently
at B1 and B4.  That evidence is preserved under
`xvla/full/closed_loop/paired_invalid_reset_stride/`.  The corrected runner
sets and records every initialization ID immediately before the official
reset; no offline fixture is claimed as an episode.

## Full-coverage gate

The earlier `final_audit_cross_model_campaign.py` is superseded. Check current
full-coverage status with:

```bash
python scripts/audit_full_cross_model_campaign.py
```

This gate has fixed π0 cardinalities and schedules and intentionally exits
nonzero while full held-out, operator, policy, serving-load, and evaluator
evidence is missing. Pilot files do not satisfy it.  For completed LeRobot
campaigns, the gate also reconstructs all 1,260 raw physical rollout batches
(500 per singleton condition and 130 per B4 condition), verifies requested
initialization IDs including the final B4 padding slots, and reconciles every
retained episode with its raw source.  `finalize_lerobot_paired_record.py`
stages a completed evaluator record and promotes it only when that independent
closed-loop audit passes.
