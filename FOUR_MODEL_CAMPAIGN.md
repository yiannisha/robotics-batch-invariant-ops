# Four-model exact-start campaign

Requested 2026-11-08. **Primary endpoint: singleton-versus-batched stability.**
The four-model campaign is active and incomplete. Success outcomes are secondary;
maximizing success is not an objective. WAMs are excluded.

### Audited stability snapshot

Snapshot expanded on 2026-11-10 after full CogACT magnitude, bootstrap and figure verification. Counts below refer to independent restored
starts, never repeat episodes or frames. Each condition uses the same canonical
start and indexed noise; numerical action errors are measured only when prepared
inputs match, while closed-loop trajectory drift uses the common horizon.

| Model | Audited starts | Native starts with action divergence | Invariant starts with action divergence |
|---|---:|---:|---:|
| π0.5 | 500 | 500/500 | 0/500 |
| RDT-1B | 250 | 250/250 | 0/250 |
| CogACT-Small | 200 | 200/200 | 0/200 |
| LingBot-VLA2 | Full cohort pending | Pending | Pending |

In these accepted cohorts, invariant inference has zero measured EEF position
drift and same-condition complete-episode repeat controls are exact. CogACT's
200-start magnitude evidence and its 20,000-resample paired-start bootstrap have passed independent audits (four tasks × 50 starts). All 200 native starts diverge; invariant action differences and EEF drift are zero. Drawer EEF positions use the robot-base frame; pick/move positions use world coordinates, with no pooling across frames. All 200 CogACT paired blocks (four tasks × 50 starts; 2,400 primary/repeat/supplement episode files) are now retained. The full independent raw audit has passed reset, repeat and invariant-trajectory checks across all 200 starts (234,000 raw frames). All 200 native starts first diverge at action 1 with identical prepared inputs. See [full raw audit evidence](results/four_model/current_evidence/cogact_full_200_raw_audit_completion_001.json). Final heldout observation-disjointness checks and the numerical freeze are complete, and heldout replay is running. The full-200 magnitude audit, bootstrap and figure review are accepted; later campaign stages remain incomplete. LingBot's retained one-start pilot
does not qualify its full campaign.
These measurements apply to the pinned controlled inference paths, including
fixed language extents for RDT/CogACT and shared deterministic MoE for LingBot;
released-native supplements are separate.

Accepted magnitude evidence is linked by hashes in
[`stability_audited_summary.json`](results/four_model/stability_audited_summary.json).
Task-specific paired-start intervals and their independent audits are indexed in
[`paired_stability_bootstrap_verified_summary_003.json`](results/four_model/paired_stability_bootstrap_verified_summary_003.json).
RMS/mean squared singleton-batch differences are paired deviations, not centered
sample variances. Tasks and numerical scales differ across models; do not rank
models using pooled errors.

Task-specific figures show action RMS differences, median per-start peak EEF
drift and action-divergence frequency for the accepted magnitude cohorts:
[π0.5, 500 starts](results/four_model/figures/paired_stability_pi05_500_003/pi05_paired_stability.png),
[RDT, 250 starts](results/four_model/figures/paired_stability_rdt_250_003/rdt_paired_stability.png),
[CogACT, 200 starts](results/four_model/figures/paired_stability_cogact_200_frame_v2_001/cogact_paired_stability.png).
PDFs and source/artifact hashes accompany each figure. The PI/RDT figure rows and eight full-cohort CogACT rows were checked against their independently audited sources; all three figures were visually inspected. The current [task table](results/four_model/singleton_batched_stability_task_table_003.csv) has 38 task/mode rows across 950 starts. Earlier 827- and 875-start artifacts remain historical snapshots. Promotion proof: [CogACT 200 acceptance](results/four_model/current_evidence/cogact_200_magnitude_acceptance_001.json).

Running/queued jobs, source pins and supersession history are recorded in
[`execution_state.json`](results/four_model/execution_state.json).
A queued script or a passing synthetic test is not accepted experimental evidence.
Full heldout/fidelity, timing/serving/queue, causal measurements and all remaining
paired audits must be verified before claiming campaign completion. Preserve the
recorded ordering of numerical freezes and later diagnostic additions; do not
retroactively claim missing ablations were complete before those freezes.
The [chronology reconciliation](results/four_model/current_evidence/final_diagnostic_chronology_reconciliation_002.json)
corrects an earlier overly strict interpretation: reference §14 freezes before
§16 causal ablations. Final acceptance still requires freeze-before-heldout,
unchanged numerical identities, complete causal/local audits, and disclosure of
the later scheduling here. An actual implementation bug or numerical-path
change requires requalification, a new freeze and full heldout re-evaluation.

## Checkpoint and benchmark selection

| Model | Checkpoint | Benchmark |
|---|---|---|
| π0.5 | `lerobot/pi05_libero_finetuned_v044` at `7152f13e34c6c115d808aea0e5d08aca74eac14b` | LIBERO |
| LingBot-VLA2 | `robbyant/lingbot-vla-v2-6b-robotwin` at `0451855729ec904f970600e0aec8b84661423afe` | RoboTwin 2.0 |
| RDT-1B | `robotics-diffusion-transformer/maniskill-model` at `9622afab2b7ce2312a6cf1febc526928589b77eb` | ManiSkill |
| CogACT-Small | `CogACT/CogACT-Small` at `ca302a300542393a0b06e291465076a7d9105c23` | SIMPLER |

π0.5 uses the [LeRobot-documented LIBERO fine-tune](https://huggingface.co/lerobot/pi05_libero_finetuned_v044).
Its old base-checkpoint numerical results cannot qualify this new deployment.
Initial source candidate: LeRobot v0.4.4 (`8fff0fde`), matching the checkpoint
name; strict loading, released preprocessing, sampler equivalence and numerical
invariance must be verified. Do not use a loader's silent random-weight fallback.

Other model sources retain the revisions and inference-only patches recorded
under `research/`. RoboTwin starts from the revision documented by the pinned
LingBot source: `13c3c47ff4312dd62484bcd51be034af55c062d1`.

## Acceptance requirements

Follow `../batch-invariant-pizero-exp/PLAN.md` and, for simulator controls, its
`open-pi-zero/experiments/simpler_snapshot_eval.py` and `simpler_worker.py`.
Use new records, never historical cross-model pilot records, for this campaign.

1. Verify checkpoint bytes, complete learned-tensor loading, upstream source,
   normalization, camera transforms, control rate, action representation,
   sampler, history, ensembles, and executed action horizon.
2. Qualify snapshot restore and fixed-action replay per simulator/task. Restore
   observable physics, velocities, controller state, task predicates/counters,
   elapsed time, RNGs and wrappers. Require exact initial observations including
   RGB and exact repeated per-step observations/outcomes. A same-seed reset alone
   does not qualify. Rebuild the physics scene when hidden solver state requires
   it. Preserve failed attempts and stop on mismatches.
3. Reset every request's policy caches, observation history, ensemble buffers,
   sticky grippers, and action queues. Supply explicit request-local noise indexed
   by task/initialization/policy-call; also control stochastic sampler increments
   if present. Padding, target position and companion activity must not change it.
4. Complete diagnostic operator qualification (100 seeded inputs/configuration),
   model ablations, local first-divergence attribution, transformations and
   native-versus-invariant singleton fidelity. Requalify kernels on Blackwell;
   H100 evidence alone is insufficient. Keep kernel changes evidence-driven.
5. Collect 1,000 benchmark observations across 100 episodes with 10 observations
   per episode, retain failures, freeze episode membership, reserve 200 diagnostic
   and 800 held-out observations, and store three request-local noises each.
   Distribute episodes across checkpoint-supported tasks and prescribe task
   membership before running. If a benchmark's standard suite changes these
   counts, document that change before collection; never reduce coverage silently.
6. Freeze sources/checkpoint/replay/runtime after numerical/operator diagnostic
   qualification gates pass, before heldout evaluation. The reference performs
   causal ablations after that freeze (§14 then §16); all causal/local analysis
   still must complete before final campaign acceptance. Do not change the
   numerical implementation based on heldout outcomes; an actual implementation
   bug or numerical-path change requires requalification, a new freeze and full
   properly exposure-qualified heldout re-evaluation.
   Run held-out comparisons at B=1,2,4,8, plus 16/32 where memory permits,
   duplicate/unrelated companions, target first/middle/last, permutations and
   partitions, following the reference's 23-arrangement core per mode (110,400
   held-out records and 2,400 singleton-fidelity records).
7. Run four closed-loop conditions, each twice: native singleton, native dynamic,
   invariant singleton, invariant dynamic. Dynamic request batches cycle 1,2,4,8
   and move the target position. One target environment advances independently;
   companions are frozen real replay requests. This avoids confusing vectorized
   environment resets with policy batching. Use 50 canonical starts per task.
   Require exact same-condition complete-episode repeats. Require exact invariant
   singleton/dynamic actions, trajectories and outcomes. Record the first native
   action mismatch and prove identical inputs and trajectory prefix before it.
   Freeze the checkpoint-supported task list before the full campaign. Four
   SIMPLER tasks imply 1,600 primary CogACT episodes; the separate released-native
   singleton/dynamic repeat supplement adds 800, for 2,400 retained episodes.
   The prescribed primary cohorts are π0.5: 10 tasks / 500 starts / 4,000 episodes;
   RDT: 5 tasks / 250 starts / 2,000 episodes; CogACT: 4 tasks / 200 starts;
   LingBot: 50 tasks / 2,500 starts / 20,000 episodes. Never count repeat controls
   as independent starts or merge released-native supplements into the primary
   controlled-native comparison.
8. LingBot's primary native and invariant paths both use the documented
   deterministic MoE fallback, isolating batch effects. Released atomic MoE
   repeatability is a separate diagnostic; do not silently drop failed repeats.
9. Run operator, complete-policy and serving/queue benchmarks independently of
   tracing. Retain five sessions, 20 warmups/session and 200 timed calls/session
   where specified by the reference, full raw timings and capacity failures.
10. Independently audit cardinality, source/runtime identities, reset and A/A
    qualification, raw outputs and resumability. Report task-specific success
    counts, paired confidence intervals/disagreements, action/trajectory
    differences, singleton fidelity, latency/throughput/memory and limitations.
    No claim extends to unobserved engine internals or other hardware/software.

## Local execution state

Large assets, isolated environments and execution logs live under:
`/scratch/hadjiyiannis_3_gmail_com/pi0-libero/four-model-campaign/`.
This is a new subdirectory; existing pi0 training assets/runs are untouched.
Root disk has only ~35 GiB free; scratch has ~1.9 TiB free at initial preflight.

The working pi-zero simulator is reused read-only from
`/home/hadjiyiannis_3_gmail_com/batch-invariance-experiments/`.
New runners are in this repository's `scripts/` directory:

- `prepare_four_model_assets.py`: immutable revisions, selected artifacts only,
  local SHA-256 and upstream LFS verification; acquisition is not qualification.
- `qualify_simpler_fixed_action_replay.py`: snapshot-worker fixed-action replay
  with exact returned observations (RGB/agent/task), actions and outcomes across
  three restores. Its default pilot uses three starts/task and runs to the
  environment horizon (or the explicit step cap). This is neither model
  evaluation nor a replacement for full closed-loop A/A qualification.

Execution log, 2026-11-08:

- Inspected live GPU processes; existing unrelated workloads were left running.
- Cloned pinned CogACT, its OpenVLA dependency, RDT and LingBot sources and the
  official RoboTwin source. Applied the existing CogACT inference-only/mmap
  patches to the new external source copies.
- Started checkpoint acquisition and supporting metadata/vision acquisition.
- Started exact fixed-action SIMPLER replay on the four reference tasks.
- Started isolated π0.5 and CogACT environment installation. The historical
  CogACT environment note lists nonexistent PyPI `draccus==0.3.1`; use upstream's
  supported `0.8.0`. SentencePiece 0.1.99 failed to build on Python 3.12;
  candidate 0.2.0 requires tokenizer-output qualification before acceptance.
- Completed the SIMPLER fixed-action replay pilot: four tasks × three starts ×
  three repeats = 36 rollouts / 3,474 steps. All returned observations, actions,
  rewards, outcomes and instructions were exact across repeats. Pick Can and
  Move Near reached the environment horizon at 80 steps; drawer tasks at 113.
  Retained summary/runtime and lossless raw archive:
  `results/four_model/qualification/simpler-fixed-actions-001/`.
  This qualifies these fixed-action pilot starts only, not policy episodes or
  unobserved physics internals. Full model A/A controls remain mandatory.
- π0.5, LingBot and RDT checkpoint acquisition/hash verification completed;
  CogACT acquisition remains in progress. Supporting RDT SigLIP weights and
  CogACT/LingBot construction metadata were acquired at recorded revisions.
- π0.5 and CogACT imports passed in their isolated environments. π0.5 strict
  loading is running. LIBERO installation required CMake 3.31.10 because
  `egl-probe` does not build under CMake 4's removed legacy compatibility.
- All four requested checkpoints are now acquired and hash-verified. π0.5's
  strict audit verified 813 state-dict entries exactly after declared dtype
  conversion. The safetensors file omits a redundant token embedding; the audit
  restores it only after proving actual Parameter identity with the loaded tied
  alias in the official model. No random missing learned tensor was accepted.
- LIBERO installed and suite imports passed. Started fresh-scene fixed-action
  replay qualification with explicit initial-state IDs (three tasks, three
  starts/task, three repeats). It currently downloads the simulator's assets.
- `results/four_model/execution_state.json` tracks continuation pointers;
  environment freezes and checkpoint identities are retained alongside the
  qualification evidence. These do not imply numerical or success-rate completion.

Continuation, 2026-11-08:

- LIBERO fixed-action pilot completed: three LIBERO-10 tasks × three starts ×
  three repeats, each 520 steps (27 rollouts / 14,040 steps), exact initial and
  per-step observed state/RGB across repeats.
- π0.5 real-input numerical probe: 72 comparisons on three actual LIBERO-10
  resets. Native actions differed in 30; invariant normalized/physical actions
  were exact in every comparison. Both modes passed B1 repeat checks. This is
  a probe, not the full diagnostic corpus or held-out result.
- π0.5 complete-episode pilot: eight conditions/repeats on task 0, initialization
  0. All initial states/inputs and A/A episodes were exact. Invariant singleton
  and dynamic actions/trajectories/outcomes matched exactly. Native first action
  mismatch was step 50, with an identical preceding trajectory and identical
  target model input. All eight episodes failed this particular start. The
  independent auditor reconstructed acceptance from raw episode records. No
  success-rate estimate is inferred from one start.
- CogACT loaded the complete official checkpoint. Its real SIMPLER-input probe
  matched the unchanged official B1 batch sampler with explicitly intercepted
  noise. All B1 repeats were exact; 12/36 total comparisons had native action
  differences and no invariant comparison failed. Canonical language extent was
  measured from these actual task prompts; full replay must recompute/freeze
  the maximum over its complete prescribed task roster before qualification.
- RDT fixed-action replay: PickCube, PushCube, StackCube and PegInsertionSide each
  passed three starts × three repeats at the released evaluator's 400-step
  horizon. The initial 50-step pilot is retained separately. The full-horizon
  runner follows upstream continuation beyond Gym's shorter time limit. Plug
  Charger rejected upstream's default dense reward on ManiSkill 3.0.1; that
  failed attempt is retained, and a separate sparse-reward run passed all three
  starts × three repeats at 400 steps. The reward-mode difference is explicit;
  no cross-reward or cross-version equivalence is claimed. Model inference and
  complete-episode A/A qualification remain required for RDT.
- Pinned RoboTwin textures, embodiments and objects were downloaded, hash-checked
  and extracted in the new external checkout. Isolated LingBot policy and
  RoboTwin simulator environments are installed. The official FlashAttention
  2.8.3 wheel matches PyTorch 2.8's C++ ABI and imports; Blackwell execution and
  the complete RoboTwin dependency stack are still unqualified.
- Started native π0.5 replay corpus collection: ten LIBERO-10 tasks × ten fixed
  starts/task × ten evenly spaced observations per complete episode. Starts 0–1
  of every task are diagnostic, 2–9 held out (200/800 observations). All failures
  are retained. Each selected observation receives three stored explicit noises.
  Collection is not a paired success-rate experiment or a numerical freeze.

Task rosters for this campaign are prescribed as LIBERO-10's ten tasks for
π0.5, RDT's five released ManiSkill tasks, the four reference Google-robot
SIMPLER tasks for CogACT, and LingBot's released RoboTwin 2.0 clean 50-task suite.
These are separate model-specific studies; do not pool model success rates or
claim coverage of other LIBERO suites, embodiments, or randomized RoboTwin.

- The π0.5 pilot auditor also rejected six deliberately corrupted copies of the
  raw evidence: missing episode, changed initial state, condition-dependent
  noise, incorrect batch schedule, differing invariant normalized chunk, and
  invalid first-divergence attribution. The portable audit reads raw JSONL or its
  lossless gzip archive and reverified the retained repository copy.
- π0.5 corpus collection is live at 55/100 episodes (550/1,000 observations).
  This progress is not completion and must be rechecked against the live process.
- LingBot's deploy import and a FlashAttention BF16 kernel executed successfully
  on Blackwell. Its strict load audit is in progress. The first audit compared
  checkpoint tensors directly to final FP32 tensors and detected a difference;
  the follow-up captures the official parameter dtype at load time to verify
  the entire checkpoint → constructor dtype → released FP32 conversion sequence.
  No weight or model computation was changed to make the audit pass.
- Curobo v0.7.8 is building for SM120 in the isolated RoboTwin environment.
  CUDA 12.8.93 compiler/runtime headers were installed locally under scratch.
  The first build's missing target-directory header path was corrected with
  explicit CPATH/LIBRARY_PATH; the retry is live. No system driver was changed.
- An unrestricted CUDA enumeration hit the host's unhealthy GPU 3; subsequent
  LingBot commands explicitly select a healthy visible device. Unrelated GPU
  workloads remain untouched.

### Updated evidence (2026-11-08)

Earlier live corpus/load/build entries are superseded: π0.5 corpus collection
and independent raw-input reconstruction completed (1,000 observations, three
explicit noises each); LingBot strict loading verified all 1,708 tensors through
the official dtype sequence; Curobo built for SM120. π0.5 full diagnostic replay
is live. Numerical freeze and held-out replay remain pending.

CogACT and RDT complete-episode pilots each contain eight conditions/repeats at
one start. Independent audits regenerated noise and checked input/prefix
identity, all A/A repeats, invariant singleton/dynamic trajectories and first
native action divergence (CogACT step1; RDT step16). These are pilot results,
not full paired success-rate studies. Archives and audits are retained under
results/four_model/qualification.

CogACT fixes language extent to64 for both modes, preserving identical target
inputs across batching. Singleton fidelity to the released unpadded runtime
fails (maximum physical action difference1.0, including gripper). Do not label
controlled-native outcomes as released-default success rates. RDT fixed30-token
pilot matched released singleton exactly. CogACT controller behavior separately
matched upstream over120 generated chunks.

RoboTwin cold initialization and a20-action/three-repeat pilot were exact for
adjust_bottle seed0, including RGB, physics and controller hashes. This is not
full trajectory qualification. OIDN reports unsupported CUDA denoising; actual
RGB frames were valid and repeatable. LingBot real-input probe passed36
comparisons using deterministic MoE shared by both modes; released atomic MoE
B1 repeats failed (maximum error0.0011393427848815918). Probe frames are from
one episode; unrelated-task batching coverage and full closed-loop qualification
remain outstanding.

π0.5 first-divergence localization completed with an exact invariant boundary.
Operator traces now include actual linear/matmul/conv2d calls; qualify observed
shapes and dtypes rather than reuse older flattened-camera assumptions.
No WAM work has been performed.

LingBot complete-episode pilot now passed independent audit: the released
expert accepted adjust_bottle seed0; the released unseen instruction was selected
after the second setup and reproduced at each reset. All eight episodes succeeded
in112 controls, all four A/A pairs were exact, and invariant singleton/dynamic
trajectories matched exactly. Native first-action divergence was step50 with
identical actual inputs and prior trajectory. The audit rehashed every raw
physics/controller/task/RNG/RGB frame and regenerated each request noise. This
is one expert-filtered start, not50-task success-rate evidence. Real companions
remain frames from one prior episode; unrelated-task coverage remains pending.

The first operator profiler run and two per-configuration retries terminated
on empty CUDA captures despite completed device execution. Successful raw
measurements were retained; the current sweep isolates each arrangement in a
new process and still requires actual CUDA kernel capture. No CPU-only profile
is accepted. Operand dtype generation preserves original strides.

### Companion split correction; primary π0.5 replay superseded

Source/manifest inspection found that the original unrelated-episode schedule
used held-out observations as diagnostic companions. The diagnostic split has
20 episodes, so31 mutually distinct non-target companion episodes cannot fit
B32. The old full diagnostic was deliberately interrupted (exit130), with all
raw records retained as provisional; it must not be resumed or accepted as
held-out evidence. This supersedes prior live/full-progress statements above.

The corrected split_local_unrelated_observations_v1 schedule keeps every
companion in the target split and outside its episode. It prioritizes distinct
episodes, then fills B32 with distinct other-episode observations:31 observations
from19 episodes in diagnostics,31 episodes in heldout. All1,000 rows passed
independent split/target-episode/uniqueness checks and sweep-prefix consistency
checks. This explicitly revises the earlier mutually-distinct-episode assumption.

To avoid reusing held-out inputs already forwarded as provisional companions,
the primary π0.5 corpus is being recollected from fresh initialization IDs10..19
per task:10,11 diagnostic and12..19 heldout. The raw corpus auditor now validates
explicit start membership and enforces the corrected companion schedule. New
numerical comparisons require the new audited manifest identity. Checkpoint and
model arithmetic have not changed. Original inputs and operator measurements
remain supplemental diagnostic evidence, not evidence of an untouched holdout.

Fresh collection launch002 failed before any episode because robosuite's EGL
setup parsed the CUDA UUID as an integer. Launch003 pins CUDA byUUID and sets
MUJOCO_EGL_DEVICE_ID=1 separately, retaining the failed log. RDT collection
continues; its audited manifest will use the corrected schedule from the start.

### Fresh corpora and complete-policy numerical qualification

The fresh π0.5 corpus (starts 10–19 per task) completed and passed independent
reconstruction of all 1,000 inputs and 3,000 noises. Diagnostic starts are 10,11;
held-out starts are 12–19. Its accepted diagnostic run is under
campaign/pi05_libero/full_fresh_split_v3, with split-local companions.

RDT collection completed 100 episodes across all five tasks, retaining failures.
Independent audit reconstructed all 1,000 RGB-history/proprio requests, regenerated
all noises and verified executed controls against the released stride. Its new
adapter batches all six vision slots per request through SigLIP. A three-noise
smoke produced 234 records: all 117 invariant final outputs were exact, native
outputs differed in 111 comparisons, and actual B32 ran in both modes. All
request/companion IDs were diagnostic. A separate 24-arrangement B16/B32 probe
measured memory and passed independent capacity audit.

The shared harness retains flow.step_* traces. DVA adapters originally emitted
diffusion.* labels; those are now also emitted as flow.step_* aliases. The
original RDT smoke proves final-output invariance, not intermediate-step coverage.
The first full RDT run was deliberately interrupted and retained as incomplete;
run002 retains all five prediction/sample pairs and has restarted diagnostics.
Model arithmetic did not change. CogACT uses the same corrected trace convention
before its full run.

New full replay runs save source hashes, model configuration, GPU UUID and
backend flags, and reject resumes against changed identities or missing runtime
provenance. Package-version snapshots for all four isolated policy environments
and a GPU/driver inventory are retained. Performance requires separate runs:
CogACT corpus collection initially shares our GPU1 with π0.5 numerical replay,
so those replay latency fields are not accepted performance measurements.

CogACT corpus collection is live: four tasks, 25 starts each, ten selected
observations per complete episode. Starts 0–4 are diagnostic, 5–24 held out.
The first complete episode independently reproduced all 80 controller actions
and 81 RGB byte hashes. Full corpus audit and numerical qualification are pending.

The original 27-configuration operator sweep reached terminal completion.
Expanded traces found two additional FP32 softmax shapes. Precision inspection
also found that the qualifier inherited cuDNN TF32True, while the actual model
uses TF32False. The F32 convolution native baseline is being repeated with both
TF32 flags explicitly disabled, alongside both softmax shapes. BF16 operations,
FP32 means and the default-TF32False matrix paths are unaffected. Original
convolution results remain TF32True supplemental evidence. The updated qualifier
records backend flags, GPU UUID and its own source hash per measurement.


2026-11-09 scheduling revision and verified milestones

The full varied-start PI05 paired study now runs concurrently with numerical
qualification on an immutable candidate source/checkpoint/runtime lock. This
candidate lock is not the numerical freeze. Every success-rate result remains
provisional until the same implementation passes all outstanding QA gates.
The prescribed cohort is all ten LIBERO-10 tasks, 50 official start IDs each,
four conditions and four repeat controls per start (4,000 episodes). Raw inputs,
indexed noise, physics trajectories, canonical initial snapshots and committed
per-condition records are retained under the external paired/pi05_libero_locked_v1
root. Repeats are controls, never additional independent success trials. A separate
auditor reconstructs policy inputs/noise and checks every completed block.

PI05 source operator coverage has independently passed all 29 observed patterns,
including true source matmul layouts, the two softmax patterns and the explicit
TF32False convolution supplement. This does not complete full-model QA.
CogACT's complete 100-episode corpus/controller/noise audit passed. Its corrected
three-request smoke retained both CFG branches at every diffusion step. All 24
B16/B32 capacity arrangements passed independent audit, and full diagnostic replay
has started. Fixed64 controlled-native remains distinct from released-default
unpadded inference; the outstanding fidelity comparison is still required.

The official-seed LingBot model pilot (first expert-eligible start100001; 100
instruction candidates) passed independent raw-evidence audit: all eight complete
episodes succeeded, A/A and invariant singleton/dynamic trajectories matched,
and native first-action divergence was attributed at control50. This is one start,
not a benchmark success rate. Expanded task-state capture now inventories all 50
released tasks (252 source-assigned fields), fails on unknown serialized state,
and has five passing unit tests. A new full fixed-action replay is running with
expanded task state and CPU/CUDA RNG. Static inventory is not runtime reset
qualification; no other task is accepted merely because it appears in the list.


RDT full observable reset gate and paired launch

Expanded cold reconstruction qualified all five task classes at three starts,
three repeats, 400 controls each: 45 episodes and 18,045 raw frames, independently
rehashed. Physics, numeric controller internals, actuator drives, source-assigned
task fields, local/global CPU/CUDA RNG, raw observations and policy RGB matched.
The full five-task x50-start paired runner has started on the qualified GPU5,
after reproducing both numerical-QA singleton outputs byte for byte. Its source
candidate remains immutable and provisional pending numerical/fidelity gates.
A second PI05 worker now covers tasks1..5 on newly free GPU7, after independently
matching QA singleton bytes and the existing global candidate lock. The renderer
remains EGL device1. These shared-device rollout timings are not performance
measurements.

The updated PI05 independent audit covers28 task/start blocks (224 episodes):
task0 has27 starts, native singleton18 successes, native dynamic17, invariant
singleton/dynamic18 each; task1 has one start with all conditions successful.
Do not pool these unbalanced task samples as an overall benchmark result.
Task-specific 10,000-resample paired percentile bootstrap statistics follow the
pi-zero analysis, with explicit repeat exclusion and conditional exact McNemar
counts. Partial samples remain provisional and cannot establish an overall gain.

RoboTwin roster001 was deliberately stopped after a rejected articulation-link
field. Its evidence is retained. Roster002 explicitly captures link pose,
velocities, mass/inertia, damping and related numeric state; lift_pot and
hanging_mug have independently passed full three-repeat fixed-action replay.
Other tasks remain unqualified until their own raw audits pass.


PI05 default-renderer stop and independently qualified no-MSAA rerun

Both default-renderer workers stopped on task2 because native-dynamic repeats
changed observation bytes at an intermediate control frame. Calls, actions,
physics state, controllers and captured derived geometry remained identical.
A retained-control diagnostic isolated one RGB pixel with one-level differences.
Disabling dithering did not solve it. Disabling multisampling gave exact raw
repeats and unchanged retained-control physics. A separate independent audit
verified all100 prescribed replays: ten tasks xthree starts xthree full fixed
waveforms plus five repeats of each failed native-dynamic control sequence.
The renderer setting is explicitly changed. Default-renderer v1 outcomes must
not be pooled with no-MSAA v2. The500-start v2 rerun has been launched on two
workers; one starts with task2. Every control now retains compressed raw RGB,
controller/RNG/physics and derived geometry for independent auditing. Startup
still must reproduce the frozen model QA singleton bytes on each actual GPU.

CogACT full paired study is live with four controlled-fixed64 primary conditions
and two released-native (singleton/dynamic) supplements, all repeated: twelve
episodes per start,200 starts,2400 episodes. Its first three starts (36 episodes)
passed independent raw-state, input preprocessing, noise and controller audits.
The fresh-worker SIMPLER reset qualification passed36 complete replays including
packed articulation state.27 matched the old worker;9 differed at initial
observation, which is retained as a protocol deviation. No old native collection
rates are pooled with the new full paired study.

The isolated RoboTwin click_bell callback-reference capture passed1203 raw-frame
checks. put_object_cabinet's fresh task instance lacked origin_z and arm_tag,
which upstream assigns before expert motion. An isolated worker initializes those
same fields from the initial object pose; its first eligible seed100026 matches
the old filter. Three700-control replays passed independent physics/controller/
task/RNG/observation audit. Original live roster sources remain unchanged.


Primary reporting endpoint clarified by user: singleton-versus-batched stability.
Success maximization is not an objective. Report fixed-input numerical variation
(exact disagreements, absolute/relative/ULP error, denoising traces) and paired
closed-loop action/trajectory variation, first divergence and physical drift.
Outcome flips are a secondary stability measure; equal aggregate success rates
do not imply stable inference. Preserve all source locks and the existing four-
model experiment scope. No checkpoint or policy tuning based on success results.

Quantitative stability evidence now includes the independently audited first12
PI05 no-MSAA starts,10 RDT PegInsertionSide starts and3 CogACT pick_can starts.
`results/four_model/paired_stability_quantitative.json` records per-component
physical action differences, first divergence, physics/observation disagreements,
and end-effector position/rotation drift over the common trajectory horizon.
Maximum native end-effector position drift per start ranges4.75--262.72mm for
PI05,14.92--191.19mm for RDT and8.38--15.88mm for CogACT's controlled-fixed64
baseline. Every invariant comparison has zero action/physics/observation drift,
equal horizons and exact repeat controls in these subsets. These are preliminary
task-limited measurements, not complete benchmark estimates.

Fixed-input numerical errors include only byte-identical preprocessed inputs
with matching indexed noise at B>1. Later feedback differences are excluded from
that metric. CogACT's released longest-padding comparison is retained separately;
padding extent and arithmetic can jointly affect that supplement. Candidate,
auditor and reporting sources are archived by hash. Seven scientific metric and
operator-output-precision checks pass. Source softmax output precision is now
preserved in RDT operator qualification, including a true FP64 reference. The
100-seed B1 case completed; further arrangements remain in progress, with failed
CUPTI captures retained and retried in isolated processes. Shared-GPU timing is
not accepted as performance evidence.

The isolated RoboTwin click_alarmclock fix passed three complete400-control
replays and independent audit of1203 raw frames, including task callbacks and
CPU/CUDA RNG. Bin-task task-state capture is still being qualified: released
Action objects and SAPIEN sphere entities require explicit capture. Failed probes
are retained; the latest helper was checked on actual dynamic/kinematic SAPIEN
CPU spheres before launching a fresh replay. The full LingBot task roster and
model campaign remain incomplete.

The fixed-input diagnostic summary now includes332 complete request/noise pairs
for PI05,142 for RDT and244 for CogACT. At B8 the element-weighted normalized
singleton/batch RMS deviations are0.0136731,0.00257037 and0.00635802 for native
inference, respectively; all invariant B8 comparisons have exactly zero error.
Every native B8 comparison is nonexact in this snapshot. B2/B4/B16/B32 and
duplicate/diverse, position, permutation/partition and per-denoising-stage tables
are retained separately in `results/four_model/diagnostic_batch_stability_progress.json`.
Only complete39-arrangement sets per implementation are accepted. Incomplete
trailing requests are excluded explicitly; corrupt gzip streams are rejected.
These errors are within-model paired deviations, not centered sample variance
or a cross-model quality ranking. Numerical QA is still in progress.

The bin task's corrected isolated helper completed three600-control replays at
the first eligible released seed100002 and independently verified1803 raw frames.
Task-state capture includes released Action defaults/kwargs and every instance
field, stable SAPIEN entity references, sphere collision/material state and
dynamic/kinematic body state. The unsupported-target and uncovered-entity probes
remain retained failures. Combined with the ongoing main roster and three other
isolated task fixes,28 of50 released tasks have independently audited first-seed
reset qualifications. The union report verifies the raw replay file hashes;
this does not establish broader-start coverage or LingBot model results.

The quantitative stability report passed an independent reconstruction of all25
reported starts:21,766 raw frame files and415 matching-input/noise numerical
comparisons. The auditor does not use the reporting script's pose cache or
numerical/rotation functions. Its independently recomputed task aggregates agree.

RDT actual high-level linear qualification now retains23 B1 source patterns and
25 B8 patterns, including shared-global and batched paths with the same B1 shape.
The full sweep prescribes425 first/last arrangements across B1,2,4,8,16,31,32,
33,64 and100 seeds each. The latest independent progress audit verifies54
complete arrangements/5400 records, with the remainder still running. The
sequence-major729x(6B)x1152 case preserves its actual logical batch axis, source
strides, and canonical singleton layout. RDT's BF16-to-FP32 source softmax passed
independent audit of700 records for B1,2,4,8; a separate larger-batch supplement
is running and requires a source/config equivalence audit before pooling.

CogACT's controlled-fixed64 B1 and all-four-task B8 source traces retain32
operator patterns each. An archived tracer extension now records exact original
additive attention-mask bytes, dtype, shape and strides using lossless compression;
serialization was independently checked. Its actual BF16 patch convolutions and
float32 VLM mean passed independent audit of5100 records across all51 prescribed
arrangements and100 seeds each, with zero invariant mismatches. Attention and
remaining operator classes are not complete. Sixteen numerical/layout/mask checks passed,
followed by two additional heldout-exposure gate checks.

RDT numerical heldout exposure correction: the full paired campaign's invariant
start4 initial observation matches the old heldout partition's task, RGB history,
history-validity flags and robot state exactly. Indexed noise differs. This is
retained in `results/four_model/heldout_observation_exposure.json`; it invalidates
an unseen-observation claim for that candidate partition, not the existing paired
stability comparisons. The original locked manifest remains unchanged while its
diagnostic job runs. Native-only collection of a replacement100-episode corpus
has started at initialization IDs50--69, outside the paired cohort's0--49.
Final heldout acceptance requires independent input reconstruction, observation
disjointness from prior corpora and the sealed paired exposure cohort, and a
numerical freeze binding the new manifest and disjointness audit. The new heldout
wrapper rejects the known exposed manifest even if its request noise is changed.
No replacement heldout inference has been run or accepted. Observation exposure
also needs checking for the other model cohorts before their holdout claims.


Stability audit update: CogACT independently audited paired cohort expanded to 18 starts, all native singleton/batched action-divergent, all invariant trajectories and same-condition repeats exact. Primary controlled comparison retains fixed64 language padding. RDT source-layout linear sweep independently verified complete for 425 arrangements ×100 seeds (42,500 records); source BF16→FP32 softmax smaller/larger sweeps independently verified and semantically equivalent, disjoint union 17 arrangements ×100 seeds (1,700 records). These are selected operator coverage, not full-model operator completion or qualified performance. Expanded magnitude report (91 π0.5,46 RDT,3 CogACT) has independent raw reconstruction running, session31417. LingBot native corpus v2 confirmed live, three completed episodes at observation; no LingBot batch-stability result yet.

CogACT fresh native collection started (session93233,pid1604201), pinned checkpoint assets rehashed before load, 100 episodes /1000 planned observations on GPU UUIDf391c6f3. Fresh cold-worker canonical-state reconstruction retained per episode. Prospective heldout55..74 excludes resetframe0 by predeclared evenly spaced selection; diagnostic50..54 retains frame0. Selection does not depend on success or numerical differences. Independent corpus auditor prepared with full raw-frame content-address/category checks, controller/noise reconstruction and cold initial comparison; it has not yet run against the completed corpus. Prepared-input disjointness and freeze still required before heldout acceptance.

CogACT fresh corpus002 stopped before first model call due nonexistent adapter.reset_model API. Stateless adapter contract inspected; controller recreated per episode, generate cache per call and RNG fork retained. Failed corpus/log preserved. Retry corpus003(session64506) completed pick_can50 first episode. RDT BF16 vision-head mean over dimension1, logical request group6, completed17 arrangements×100 seeds and independently audited; all invariant/reference checks exact. Auditor extended for grouped nonrequest reduction dimensions and independently regressed on all5,100 CogACT conv/mean records. Remaining matmul/BMM/masked attention coverage still outstanding.

RDT source matrix sweep launched session90677 with isolated processes, four actual source patterns (two rank4 group6 matmuls, two group96 pooling BMMs),68 arrangements/6,800 planned raw records. Specialized qualifier preserves observed B1/B8 stride scaling and logical request grouping without editing the generic qualifier or live full-model adapters. First B1 arrangement completed100 seeds. Independent source/record auditor prepared and running progress reconciliation. Shared timing remains unqualified; masked attention remains outstanding.

RDT source matrix sweep001 intentionally stopped after review identified target slices retaining full-batch stride for pooling BMM. Exploratory raw records retained and excluded from final qualification. Corrected sweep002(session78857) copies identical target values into traced B1 strides, asserts exact value equality and records singleton strides. Independent auditor now requires these fields. Small separate B2 pooling smoke covers both actual BMM patterns; full68-arrangement sweep remains pending.

π0.5 replacement no-MSAA native corpus launched session74242:100 episodes across10tasks×officialstarts0..9, new indexed rollout/replay noise namespace fresh-no-msaa-004, predeclared10 evenly spaced pre-action controlframes1..steps-1 excluding reset/terminal observation. Each start cold-reconstructed twice with full raw identity comparison; all control-frame physics/controller/RNG/observation/derived records retained. Independent auditor prepared to rebuild all source model inputs plus selected inputs/noises, raw states and frame selection. No heldout claim until prepared-input disjointness against all prior numerical/paired observations including rejected MSAA v1 and freeze. RDT corrected B2 pooling checks independently audited200 records, both observed BMM patterns; full68arrangements still running.

RDT attention source sweep launched session59307, three source QKV layouts including bool25-valid/30-token language mask with exact broadcast strides and source values in both B1/B8.51 arrangements×100 seeds prescribed. Independent auditor checked first132 raw records, confirming source QKV/mask values/layouts, dispatch, precision, FP64 references, invariant/reference exactness. No full operator completion claim: remaining arrangements and other task language-mask patterns still outstanding.

CogACT actual-source linear sweep launched session50899:24 traced layouts,408 arrangements,40,800 planned raw measurements. CFG doubled-batch rows explicitly mapped as conditional[i]/unconditional[B+i]; source singleton stride copied, explicit per-request reference restores branch-major full output ordering. Three CPU coordinate/reference tests pass. Independent auditor requires CFG target indices and all source layouts; first arrangement completed100 seeds. Shared timing remains unqualified. Remaining attention/matrix source operations still outstanding.

LingBot numerical input pipeline prepared for full native corpus: builder requires independent reconstruction of all200 episodes and binds source snapshot, request file hashes, explicit indexed noises and split-local companions; new exact adapter removes obsolete396-HDF5 requirement, verifies all1708 checkpoint tensors through official dtype sequence on model load, source-bound prepared fields and noise bytes, and clears vision metadata per inference. Actual first44 selected inputs share fixed72-token schema. Loader-only integration on one actual request (synthetic noise; no NN; deliberate in-memory metadata bypass confined to probe) passed noise selection and corrupt-signature rejection. Actual partial corpus sealing rejected before output/model creation. No LingBot full numerical result claimed. New corpus002 independent progress audit launched after correcting command invocation to module mode; initial direct-script import failure retained.

Heldout acceptance gate strengthened: requires disjoint actual prepared neural inputs (not merely distinct raw observations), explicitly verified and freeze-bound. RDT CPU identity reconstructs SigLIP-preprocessed BF16 pixels, BF16 normalized state, padded language/mask, action mask and control frequency; excludes noise/request IDs and metadata with no neural effect. Four tests pass, including distinct raw joints collapsing to identical prepared state and rejection of raw-only audit. RDT fresh-corpus progress exposure audit running session27291; progress status cannot unlock heldout. Final exposure seal still requires completed paired cohort. LingBot corpus002 independent progress audit is session77626.

CogACT four-layout attention qualification launched session58931:68arrangements/6,800 prescribed records. Captured BF16 additive causal/padding mask reconstructed losslessly, cycling source B8 rows and selecting matching singleton mask; byte checks atB1/2/8/33/64 passed. CFG split-branch target coordinates/source singleton strides preserved. Independent auditor verified first300 records, including target/fullmask bytes, precision/layout, dispatch, FP64 references and invariant/reference exactness. Full coverage still pending. LingBot corpus002 partial independent audit completed10 episodes,4,540rawframes,95calls,44selectedinputs, noNN in auditor; full200episode gate remains closed.

π0.5 fresh no-MSAA corpus collection finished100 episodes/1000observations, terminal0. Independent full raw-state/source-input/selection/preprocessing/noise audit launched session70310. CogACT CPU prepared-input identity reconstructed actual stored FP32 DINO/SigLIP pixels, token IDs and masks for three pick_can calls, noise/request IDs excluded. Freshness fingerprint conservatively uses effective BF16 vision pixels to prevent Float32-only differences erased by autocast from manufacturing disjointness. CogACT corresponding exposure auditor prepared; full new corpus audit/disjointness and final freeze remain pending. Initial loader probe used wrong raw field name image instead of actual rgb; corrected helper matches recorded tensors, no model/simulator path changed.

RDT prepared-input exposure progress audit completed:800 heldout observations against7,975 prior prepared-input identities, zero collisions. Explicit progress-only status cannot unlock heldout; final paired exposure cohort still must complete. π0.5 effective input identity now reproduces official GPU image resizing/normalization without constructing any learned model and excludes observation.state, which predict_action_chunk does not consume separately (state already encoded in prompt tokens). Actual prepared input integration passed and unused-state perturbation correctly leaves identity unchanged. Full exposure auditor prepared includes all retained pi05 paired sibling directories, notably rejected default-MSAA v1; uncommitted observations lacking task attribution conservatively checked with all10instructions. Candidate independent corpus audit still running, so no PI heldout result accepted.

RDT all-five-task diagnostic B8 trace completed session32682,36 observed patterns. Lossless bool attention masks show valid token counts[25,20,30,24,18,25,20,30]. Shape/stride/operation union unchanged from prior source trace (mask values aside). Existing source attention sweep covers only25-valid mask, so full attention qualification remains explicitly incomplete pending five-pattern and heterogeneous-mask coverage. Trace startup001 used incorrect environment variable names and failed before model load; corrected002 uses actual RDT_CHECKPOINT/RDT_SIGLIP/RDT_SOURCE and completed.

RDT source mask supplement prepared/launched:6 configurations (each of5 captured task-validity masks plus original heterogeneous batch),102arrangements/10,200 planned records. Bool compact mask rows expand to actual[B,1,67,30] stride[30,30,0,1]; singleton selects corresponding target mask. Per-request explicit reference retains all32sourceheads. New separate qualifier leaves currently running primary sweep unchanged; invariant/reference mismatches stop after retaining failed record. Independent supplement auditor still pending.

RDT all-task mask supplement independently audited first891rawrecords,8completearrangements/102. Separate actual-CUDA mask integration passed30 combinations across6maskconfigs×B1/2/8/33/64, checking exact bool dtype, actual[ B,1,67,30 ] shape,stride[30,30,0,1], full bytes and first/last singleton masks. π0.5 fresh corpus auditor001 intentionally stopped(exit143) after review identified field-loop variable collisions overwriting episode tuple key and source JSON path, causing repeated work/incorrect final episode index. Original auditor archived bySHA, log retained, data unchanged. Fixed frame_path/hash_field names and restarted independent audit002; no failed audit accepted.

Concise audited magnitude summary saved results/four_model/stability_audited_summary.json, bound to full raw-derived report and independent audit bySHA. Covers only existing25-start audited magnitude subset (12π05,10RDT,3CogACT), explicitly separate from larger paired-count cohorts. Includes fixed identical-input/noise normalized RMS/max output errors and physical EEF drift inmm; invariant zero throughout. Source metrics remain task-specific and normalized scales are model-specific. Corrected π05 corpus audit retry session44201 remains live.

CogACT four source attention layouts completed and independently verified:68arrangements/6,800records, captured additive mask bytes and CFG source singleton coordinates/strides retained. Shared rotary-position matmul independently qualified17 logical arrangements/1,700records with actual operands fixed[1,64,1]@[1,1,320], no fake batch-axis expansion. Linear sweep still pending; all32traced high-level patterns now mapped to qualification components. RDT inventory audit found remaining actual source group6 patch convolution at384px and group96 BF16-output pooling softmax, distinct from already qualified FP32-output vision softmax. Supplemental34arrangement/3,400record sweep launched with exact source grouping/output precision; independent auditor extension still pending. Full operator completion and isolated timing not claimed.

π0.5 fresh corpus independently verified complete after corrected auditor002:100episodes,29,228rawframes,632native source inputs,1000selected preprocessed inputs and3000indexednoises. Freshness still pending: started effective-image/token exposure progress check against bothprior001/003corpora and allretained pi05 pairedsiblings, including rejectedMSAA v1. CogACT independent source inventory reconciliation maps all32patterns atB1/B8 to components and verifies audited raw prefixes; linear measurement coverage incomplete, so full operator qualification remains false. Grouped-conv/default-softmax auditor extended and regressed successfully on priorCogACT5100conv/meanrecords and RDT700FP32softmaxrecords; newRDTconv/poolingsource progress audit passes.

Expanded magnitude report independent audit completed140starts(91π05/46RDT/3CogACT),136,786rawframefiles and1,536fixed-identical-input nonsingleton output pairs. Same-condition controls and invariant trajectories exact. Audited summary promoted to v2 source with prior25-start summary archived. Worst native EEF drift in this subset:318.826mmπ05,327.002mmRDT,15.884mmCogACT; task-specific normalized errors and common-horizon physical metrics remain separate. π05 prepared exposure progress audit completed800heldout inputs against9,868prior effective identities, zero collisions, including prior001/003corpora and rejectedMSAA v1/currentv2pairedrawinputs. Progress-only: final exposure cohort and freeze still pending. RDT source inventory reconciliation maps34B1/36B8patterns with all-five-task/mixedmask component required; allmeasurements notcomplete. CogACT18-start magnitude extraction started session18103 for later independent audit, avoiding reauditing unchangedπ/RDT subsets.

RDT remaining patch-convolution/BF16-pooling-softmax sweep terminal0 and independently complete:34arrangements×100seeds=3,400records. Actual group6 convolution384px and group96 pooling softmax output BF16 retained. Matrix and attention suites remain live; full operator qualification false. Scratch storage remains1.2TBfree at this observation.

RDT source matrix suite independently complete:68arrangements/6,800records, actual source grouping/strides and canonical singleton copies verified. Original attention head-split reference sweep intentionally stopped(exit241), data/logs preserved as exploratory. Full32head per-request native reference now used for both remaining unmasked layouts, consistent with all-task mask suite;34arrangements/3,400records prescribed unchanged batches/positions/100seeds. Actual GPU equivalence probe on both layouts atB2 proved identical operand values/strides and native/invariant outputs between old/new harnesses, and full-head reference matches actual source singleton. No model/operator weights or implementation changed. New sweep session28057; independent full-head auditor accepts recorded singleton strides/target indices/unmasked hashes. Current diagnostic progress575/600π05,245/600RDT,511/600CogACT; these are live progress, not final numerical acceptance.

RDT all-task/mixed bool-mask supplement terminal0 and independently complete:102arrangements/10,200records across5captured task masks plus heterogeneous batch. Unmasked source attention still pending; full operator qualification false. CogACT replacement native corpus003 finished100episodes/1000observations, terminal0; independent full corpus reconstruction running session26498. CogACT18-start magnitude report extraction completed, terminal0; independent raw-frame/numerical reconstruction running session45265. No new CogACT magnitude or heldout result accepted before these audits.

π05 capacity remeasured on actual fresh-v3 diagnostic manifest rather than stale001corpus:24B16/B32 native/BI duplicate/diverse first/middle/last arrangements allran, session86699 terminal0. Independent capacity audit validates frozen companion schedule, output/memory metadata, runtimeconfig matching current diagnostic and sourcebytes. LingBot numerical adapter extended before any numerical campaign to expose both normalized55 and official physical14 outputs, reconstructing full action-transform context from audited rawframe/instruction and checking model tensors against storedinput. Contexts cloned independently perrequest, including duplicates. Transforms-only probe(noNN) exactly reproduced retained native physical14actions forB1 andduplicateB8, with context tensor bytes unchanged(session69550). Complete native200episode gate remains required. Numerical freeze needs separate diagnostic/heldout manifest binding and actual adapter config; legacy static registry/defaultNone freeze must not be used for these exact adapters.

RDT complete traced source operator inventory independently reconciled:34B1/36B8patterns,69,700primary component records, all component measurements complete including5task/mixedmask102arrangements and full-head unmasked34arrangements. Full-model diagnostic/heldout and isolated performance remain separate/incomplete. π05 diagnostic producer completed600request/noisepairs terminal0; independent source/noise/companion/arrangement/physicalbatch/denoising/fidelity audit launched session55169. Four-model freeze implementation prepared with actual runtime adapter config and distinct diagnostic/heldout manifest binding, complete operator/capacity/diagnostic/fidelity gates, checkpoint acquisition bytes and final effective-input disjointness seal;4regressiontests pass, actualPI progress-only seal negative probe expectedrejects before freeze write. No numerical freeze or heldout acceptance yet.

π05 full diagnostic independently accepted:200observations/600explicitnoises,1200implementationgroups,46,800arrangements,10-step state/velocity traces and600singleton-fidelity records. Native22,200nonexactarrangements; invariant alloutputs andrecordedstages exact. BI-singleton/native-singleton fidelity nonexact600/600 is retained as a separate measurement, not hidden or success-tuned. CogACT fresh corpus independent audit passed100episodes/1000observations; prepared-input exposure progress audit running session81487. CogACT18-start magnitudes independently verified7,182rawfiles/1,109fixed-inputcomparisons:18/18native action/physics/observation divergent,0/18BI, worstnativeEEF288.528mm, median-startmax10.021mm, normalizedfixedinputRMS0.001954807. Consolidated audited summary now155starts(91PI/46RDT/18Cog); old140summary archived. Freeze now requires SHA-bound complete independent diagnostic report and verifies source modelconfig; actualPI diagnostic binding gate passes. Five freeze +four exposure regressiontests passed; final disjointness seal still pending, no freeze created.

CogACT source linear producer finished and no qualifier remains live; independent complete audit verified408arrangements/40,800records. Full source inventory independently reconciled32B1/32B8patterns and54,400records across linear,conv/mean,attention and shared rotary matmul; full_traced_operator_qualification_complete=True. Full-model diagnostics, heldout, fidelity and closed-loop campaigns remain separate. Prepared-input exposure progress completed795unique heldout effective inputs (800rows) versus15,477prior identities, zero overlaps; progress-only, no final freshness seal/freeze accepted. Primary endpoint remains singleton-versus-batched stability, never success optimization.

Expanded RDT paired independent audit v3 launched(session17882) with actual prepared-input reconstruction and full raw-frame checks; output independent_progress_audit_expanded_v3.json, no promotion before successful termination. Existing audited46-start RDT magnitude remains authoritative. Live diagnostic producers revalidated byPID: RDT275/600 andCogACT577/600; LingBot and all3paired studies also confirmedlive. Performance observer/wrapper syntax valid and4process-isolation tests pass; no isolated benchmark or freeze claimed.

Performance freeze guards validated with9tests: altered replay bytes, changed numerical configuration, wrong adapter and progress-only freshness rejected; compute/graphics isolation cases pass. No benchmark launched. Expanded RDT paired input/raw-state audit confirmedlivePID1484177. Recorded block progress475/500PI,199/250RDT,39/200Cog is explicitly separate from accepted155-start magnitude subset. LingBot native corpus51/200 andCogdiagnostic580/600, both live; no full acceptance implied.

Complete PI diagnostic fixed-input stability summary extracted600request/noisepairs. Separate direct raw-record recount verifies all12mode/batch output aggregates and binds rawSHAtoaccepted independent diagnostic audit; scope excludes denoising aggregate re-audit. Native B2/4/8/16/32 normalized outputRMS0.012729/0.011230/0.012242/0.011969/0.011886, every nonsingleton batch-position comparison nonexact; BIall0. RMSispaireddeviation fromsingleton, not centered samplevariance; no cross-model scale ranking. Reports pi05_complete_diagnostic_batch_stability.json and pi05_complete_diagnostic_batch_aggregate_audit.json.

Reusable independent output-aggregate auditor audit_diagnostic_batch_aggregates.py verified all120PI task/mode/batch groups, exact counts/dtypes/maxima/mean-squareddeviations/RMS from accepted raw diagnostic, independently of reporting module. Full raw/sourceaudit/report bytes checked before/after; v2audit binds auditorsourceSHA. Denosing aggregate tables explicitly outside this audit scope. Controlled corruption of actual reportRMS is checked separately; no model path changed.

Expanded CogACT paired independent audit v3 launchedsession3024 with preparedinput reconstruction, controller recomputation and raw observable state/observation checks, using newly committed blocks; no acceptance or replacement of18-start magnitude audit before successful terminal result. RDTexpandedv3 audit stillconfirmedlivePID1484177. Recorded block counts481PI/204RDT/39Cog, not accepted outcome counts. Cog full diagnostic586/600stilllive.

CogACT post-diagnostic pipeline launchedsession58556 after verifying live producerPID821764; tracks /proc starttick identity to avoid PIDreuse. Waits for actualproducerexit then complete independentdiagnosticaudit→batchstabilitysummary→independentoutputaggregaterecount with subprocess check=True. Incomplete producer data stops at fullaudit; sourceSHAguard pins4report/auditscripts during wait. Jobsource/log/state retained under logs/cogact-post-diagnostic-audit-001. No audit result accepted before terminalsuccess. Do not edit these pinned helper/auditor sources while waiting.

PI post-paired pipeline launchedsession80575 with liveworkers4179674/5 and captured/procstartidentities. Afterbothactualexit, requires full500start independentrawframe/input/repeat/invariant audit with --require-complete, then finaleffectiveinput disjointness audit using fresh004 and prior001/003 plus allpairedsiblings inclrejectedMSAA v1. Job script/log/state retained; pins4auditor/helperSHA. No freeze or final freshness accepted yet. Do not edit pinned sources while this job is waiting/running.

Isolated benchmark wrapper strengthened before any performance launch: archive selectedbenchmark/harness/wrapper/isolation sources, verify source/freeze/calibration bytes unchanged through completion, bind actualbenchmarkoutputSHA. Queue requires accepted completedserving isolation proof on samefreeze+GPU, matching exacttimingfilebytes; rejects contaminated, modified or otherGPU/freeze calibration. Childprocessgroup exit race handled without touching foreignjobs.12regressiontests pass. No performance claim or numerical freeze created.

CogACT full diagnostic producer and post-audit pipeline terminal0. Independent600input/noisepair/1200modegroup/46,800arrangement audit passed all recorded denoisingstage/output exactness forBI; native22,200nonexactarrangements. Separate600singletonfidelityrecords allnonexact BIvsnative, retained/disclosed. Independent48task/mode/batch outputaggregates verified: nativeB2/4/8/16/32 normalizedRMS0.009263914/0.007642052/0.008192352/0.008192305/0.008192397, invariantall0; allnative nonsingletonbatchpositioncomparisonsnonexact. Bothprimarymodes fixed64tokenpadding. Raw/report/runtime/capacity/fidelity bytes bound, copied accepteddiagproof current_evidence. Fullheldout, pairedtrajectory, finalfreshnessseal,freeze/performance stillseparate/incomplete.

PI full paired rollout collection completedall500prescribedstarts/4000episodes. Bothworker sessions70008/27852terminal0; processPIDsabsent. Postpipeline80575 correctlyadvanced to audit_pi05_no_msaa_paired_study --require-complete --rebuild-inputs, nowlive. Full500start audit andfinaleffective-inputseal stillpending; retain91-start acceptedmagnitude subset until newfullaudit+quantitativerecount pass. No successoptimization undertaken.

Actual CogACT completed operator+diagnostic components passed freeze-module gates on pinnedruntimeconfig (15operatorboundfiles,46,800diagnosticarrangements). Evidence current_evidence/cogact_actual_freeze_component_gates.json. This is componentqualification only: finaleffective-inputseal stillpending and nofreezecreated. PI500start completeaudit livePID1989063; RDT/Cogexpandedpaired audits live1484177/1647541; RDTfull diagnostic andLingnativecorpus bothlive.

Dedicated four-exact-model heldout auditor prepared, preserving old broadcampaign auditor/runtime. Validates frozenasset/sourcebytes, finaleffectiveinputseal, all800heldoutobservations/2400indexednoises (Lingvariable shortepisodes explicitly allowed), completepermodefeasiblebatchmatrix, frozencompanionorders/physicalpartitionbatch, output+denoisingmetrics andallheldoutsingletonfidelity.5schedule regressiontests pass;78actualCogdiagnosticarrangements integrate with syntheticin-memory splitlabels only (explicitnoheldoutclaim). Fullendtoendheldoutaudit remains untested until genuinefreeze+replayexist.

PI post-seal pipeline launchedsession2066; waitsonactual livepost-pairedaudit/sealPID1755007 with starttickguard. Requires precedingpipelinecompletedstatus; pins commandmanifest and stageauditors/helpers before wait. Threecheckedsequentialsubprocesses: fullfresh-v3/fresh004numericalfreeze including checkpoint/tokenizer/lerobotsources, exposurequalified2400pairheldoutB1/2/4/8/16/32 with2400singletonfidelityrecords on originaldiagnosticGPU1, independentheldoutaudit. Commands/source/log/state retained logs/pi05-post-seal-heldout-001*. NoheldoutNNorfreeze hasstarted. These stage sources nowpinned; do not edit whilequeued/running. Failure stopssequence, noautomaticrestart.

RDT fullcohortpost-paired pipeline launchedsession35254 afterrecorded245/250starts; waitsonliveproducer1421512 andexpandedprefixauditor1484177 with/procstartidentities. Requires prefixauditpass, thenfull250start independentaudit --require-complete --rebuild-inputs, thenfinaleffectiveinputdisjointnesscheck forfresh003againstcompletepairedcohortandoldfullmanifest. Pins4auditor/helpersources whilewaiting. Noacceptance/freezeclaimed; fullmodeldiagnostic stillpending.

RDT post-diagnostic followup launchedsession63668 afterlivePID2152926 verified289/600pairs. Reuses completedCogpost-audit sequence with explicitRDTmodel/path substitutions, auditedsourcehashguard and/procstartidentity: completeindependentdiagnostic→batchstabilitysummary→independentaggregatecheck. Jobstateconfirmswaiting; no newresultsaccepted. Lingnativecorpus62/200stilllive. Allfullgoalstages remainrequired.

RDT fullpaired recording completed250prescribedstarts/2000episodes; workerPID1421512absent andsession64204terminal0. Expandedprefixaudit1484177stilllive; fullcohort/finalexposure pipeline35254 waitsonthat audit beforecomplete250reconstruction. Accepted46-start magnitude unchanged. Lingpairedcode review confirms legacy pilot hardcodesadjust_bottle and singleepisodecompanions; full50task variedstart pairedrunner remainsneeded after fullsourcecorpus/numericalqualification, no scope reduction.

Full LingBot paired runner prepared separately from pinnedlivecollector/baseadapter:50qualifiedtasks×50expert-eligiblestarts×8conditions, explicitrequestindexednoise, unrelateddiagnosticcompanionrows, officialnormalized55+physical14outputs, source-selectedqualifiedworker, canonicalfullstate/inputresetchecks, fullrawretention, A/Arepeats andBItrajectoryguards, firstnativeactiondivergence equal-input/prefix attribution. Requirescomplete200nativecorpus+acceptedfullmodeldiagnostics beforeloadingstrict1708tensor model; sourcecorpuscheckpoint/tokenizermetadata/acquisitionbytes bound. Sixprotocoltests pass (noise/schedule uniqueness including short-corpusrowcounts, repeat/BIrejection, nativeinputattribution). No model/simintegration or actualpairedcampaign yet; independentLingpairedauditor stillneeded. No sourcepinnedbyexistingjobs changed.

LingBot sourcecorpus expandedindependent reconstruction launchedsession2692 on observed66completeepisodes, noNN(stubmodelraises). Outputindependent_corpus_progress_v3.json remainsprovisional untilterminalverification. Newpairedrunnerguards exercised on retainedofficial-seedpilot002 all8episodes via explicitsteps/hashschema conversion: repeatcontrols+BItrajectoriesexact, recoverednativefirstactiondivergencecontrol50. Integrationevidence savedcurrent_evidence/lingbot_paired_guards_actual_pilot_integration.json; this doesnotqualifyfull50taskrunner or newmodel/simexecution.

Independent LingBot paired auditor prepared separately fromproducer: validates50simqualificationproofs/rawreplays, source/assetarchives, canonicalexpertstart/eligiblecandidate sequence, fullrawphysics/controller/task/RNG/observations, CPUofficialpreparedinputs+indexednoise, unrelateddiagnosticcompanions, normalized55→physical14mapping, commandcoverage, A/Arepeats, BItrajectory/outputequality, nativefirstdivergenceequalinput/prefix. NoNNallowed. Physicalmappinghelper testedonactualretainedpilotcall50×14exact and deliberatelycorruptedphysicalmapping rejected evenwithrecomputedhash(session90784terminal0). Full newpairedformatendtoendintegration remains pending. Runner nowretainsGPUUUID/Torch/CUDA/simulatorruntime provenance. Existing pinnedjobsourcesunchanged.

LingBot post-native pipeline launchedsession39216 with livecollection1243919 andprogressauditor2489335 /procstartidentities. Afterbothterminal andprogressauditpass, fullindependent200episode corpusreconstruction(--require-complete) precedes strict numericalreplaymanifest/noise/companionbuilder. Bothauditor/buildersources SHA-pinned forwait; no partialcorpuspromotion or neuralnumericalqualification occurs. Fileslogs/lingbot-post-native-corpus-001*. All4fullcampaign goals remainactive.

Consolidated accepted PI+Cog fixed-input batchsize stability rows to fixed_input_batch_stability_verified_summary.json, binding independentaggregate andcompletediagnostic proofs bySHA/config. Each600paircohort complete; RDT/Ling excludeduntil fullacceptance. This is RMSpaireddeviation, not centeredsamplevariance; normalizedscaleskeptseparate. LiveRDTdiag300/600, allstateauditsstilllive; scratch1.1TBfree.

Dual-arm LingBotpilot metric extractor added/tested3cases(armseparation,quaternionsign/rotation,commonhorizon). Actual acceptedofficial-seedpilot002 extraction terminal0 and separateindependentrecount terminal0:452rawframes,113commonframes, peaknativeleft0.397426mm/right0.909798mm; BIboth0. Fixedidenticalinputnative1call normalizedRMS0.000119412/max0.000741422; BI2calls0. Report lingbot_retained_pilot_dual_arm_stability.json and independent_audit.json. Explicitoneadjust_bottle start only, notnew50taskcampaign/cohort acceptance; notaddedto155start3modelsummary. NoadditionalNNused.

PI B64 fullpolicyperformancecapacity supplement launchedsession10724/PID3142704 onGPU7, separatefromqueuedGPU1heldout. Existingacceptedfresh-v3diagnosticmanifest only; deterministic64diagnosticrowselection retained, sameexactadapter/config/sourcebytes,12native/BI duplicate/diverse first/middle/last arrangements. CUDAOOM retained, BI mismatch failafterrecording. Newqualify_policy_performance_capacity.py sourcepinnedwhilelive; no numericalpathchanges. This capacitysupplement isnot isolatedtiming, no fullperformanceclaim; independentauditpending.

PI B64capacity producer10724terminal0; independentfullmatrixaudit passed12arrangements,64explicitdistinctdiagnosticrows, source/config/noise/targetposition/driverhash andmetrics. Native+BIallran, BIoutputsall exact; maxincrementalCUDAallocation9,788,196,352bytes. Notisolatedlatency. Acceptedproofcopiedcurrent_evidence. CorrespondingCogB64capacity launchedGPU7 usingcompleteaccepteddiag/config; noheldoutinputs, independentauditpending. Qualifier sourceagainpinnedwhilelive.

LingBot expandedcorpus independentaudit2692terminal0: actualsnapshot67nativeepisodes (one additionalcommittedbetweenlaunchandmanifestread),22,715rawframes,486officialinput/noise/actioncalls and332selectedinputs rebuilt. CPUtransforms only, noNN. Snapshot/lock/auditorbytesbound, proofcopiedcurrent_evidence/lingbot_corpus_67_episode_independent_progress_audit.json. Full200episodegate remainsFalse; numericalmanifest/diagnostic/heldoutqualification notaccepted. Collectorcontinues, post-native pipeline waitsforits actualexit.

CogACT B64performancecapacity4093terminal0 andindependentaudit passedall12arrangements, sameacceptedfixed64token/config/diagnosticcorpus. Native+BIallran, BIalloutputs exact; maxincrementalCUDAallocation20,531,193,344bytes. Proofcopiedcurrent_evidence/cogact_B64_performance_capacity_independent_audit.json. This supplements B16/B32capacity, noisolatedlatency/serving/queue performanceclaim.

CogACT expandedpaired audit3024terminal0:39starts/468episodes,37,440policycalls reconstructed and37,908rawframes checked; allA/A+BItrajectoryguards pass. Canonicalpairedprogressaudit promoted to39withold18auditarchivedbySHA; existing18-start reportpointer updated toarchive toremainvalid.39-start driftmagnitude extraction launched92891, independentmagnitudeaudit remainsrequired beforeconsolidated155startsummarychanges.

RDT B64policycapacity recording launchedGPU7 usingexistingfull-diagnosticmanifest andsameexactconfig/sourcepath; noheldoutinputs or numericaltuning. Twelveprescribednative/BIduplicate/diversepositions. Full independentcapacityacceptance waitsforcompleteRDTdiagnosticproof (current311/600), unlikeacceptedPI/Cog; capacity timingsnotqualified. Sourcequalifier remainspinnedwhilelive.

CogACT39start quantitativepost-audit queuedsession39829, discovered andverifiedexactliveextractionprocess/startticks, pins reporting+independentmetric auditor sources. Afterextractoractualexit, requireatomiccomplete39report thenindependentrawframe/action/pose/numericalmagnituderecount. No automaticpublishedsummarypromotionbeforeverification; sourcesnowpinnedwhilewaiting. RDTB64capacitystilllive,nativeduplicate3positionsran.

RDT B64capacity47694terminal0,12recordedarrangements allran andproducerBIexactguardpassed. Acceptedcapacity audit remains gatedoncompleteRDTdiagnosticproof, stillpending; retainrawruntime/driver/records anddonotclaimisolatedperformance orfullnumericalacceptance.

CogACT39start magnitudeextraction92891terminal0; completeatomic39-startreport verifiedpresent. Queuedpostjob39829 correctlyadvancedtoindependentrawmagnitudeaudit, live firststartverified. No acceptance/promotion ofmagnitudes beforeterminalaudit; consolidated155-start summaryunchanged. OtherPI/RDTauditandmodelproducers remainlive.

Full heldout-validator syntheticintegration test added without changing sourcepinnedbyqueuedPIheldoutjob. Fivepassedcases:complete24request/1104arrangement/24fidelityfixture accepted; changedfrozenmodelsource, corruptedindexednoise, missingdenoisingstage andmissingfidelity rejected. SynthetictestmodelusesLingallowedvariablecohort solelyforvalidatorcoverage; noactualfreeze/model/simulator/heldoutacceptanceclaim.

LingBotfullcohortstoragepreflight: liveframesdirectory observed19,038,976,148bytes with1,186,840,170,496bytes scratchfree. Four-frame in-memoryPNGpacking probe ratio1.0304(noimprovement); fouractualadjacentRGBXORpairs ratio0.81627 andfullcontent_hash roundtrip exact. Probesrecordedcurrent_evidence; no codecadopted, no existingrawfile/livecollector/pairedformat/inferencepath changed. Fullretention/scoperemainsrequired. Cog39magnitudeaudit nowthroughstart18, stillpending.

CogACT39start independentmagnitudeaudit39829terminal0:15,561rawframefiles/2,404fixed-inputcomparisons independentlyrecounted. Native39/39action+physics+observationdivergent; BI0/39. Native median-startpeakEEF7.491869mm,worst288.527750mm; normalizedfixed-inputRMS0.0025290056,max0.0882707238. Controlledfixed64paddingprimary; released-native39/39divergent supplementretainedseparately. Publishedsummarypromotedto176starts(91PI/46RDT/39Cog), old155summaryarchivedbySHA. Allsameconditionrepeats exact. Initialmetadata promotion usedlegacyfirst-divergencefield; correctedtoCogattribution.native.first_action_divergence beforeprogresswrite. Summarysourcehashes and176native-startcountverified. Fullcampaignnotcomplete.

RDT expandedinput/rawstatepaired audit17882terminal0:199starts/1592episodes overPegInsertionSide,PickCube,PlugCharger,StackCube;27,214policycalls independentlyreconstructed, allrepeat+BItrajectorychecks pass. Canonicalrawpairedaudit promoted199with46-startpriorproof archivedbySHA; existing46-start progresspointer updatedtoimmutablearchive. Full250startauditpostpipeline35254 nowstarted, actualsourceauditorlive.199-start magnitudeextraction launched73875; independentmagnituderecount requiredbeforepublished46-startRDTmetrics change. Fullcampaign andheldout/freeze/performance remainincomplete.

RDT199start independentmagnitudeaudit queuedsession92691 behindverifiedliveextractor, preservingdriver/auditorSHAand/procstartidentity; rawmagnitudeoutputs remainunaccepted untilseparateauditcompletes.

### 2026-11-09T10:08:01.263462+00:00 — stability continuation / verified live waits

Primary endpoint remains singleton-versus-batched numerical and trajectory stability; success is secondary. Revalidated live PID 1989063 (complete π0.5 paired audit), 1208987 (complete RDT paired audit), 2152926 (RDT diagnostic, 365/600 pairs), 1243919 (LingBot corpus, 151/200 episodes), 2770000 (CogACT paired, 72 completed blocks), and 1234717 (199-start RDT magnitude extraction). Recorded paired blocks: π0.5 500, RDT 250; complete acceptance remains pending audits.

Queued independent RDT B64 capacity audit after live diagnostic audit pipeline PID 2270897, with captured process start identity, predecessor terminal-status gate, and auditor SHA guard. Job `/scratch/hadjiyiannis_3_gmail_com/pi0-libero/four-model-campaign/logs/rdt-post-diagnostic-B64-audit-001.py`, session 85565. No capacity acceptance or isolated timing claim yet. Existing pinned sources untouched.

### 2026-11-09T10:08:59.702979+00:00 — CogACT complete-cohort verification queued

Revalidated live full paired producer PID 2770000 and both complete π0.5/RDT independent auditors. Queued `logs/cogact-post-paired-audit-001.py` (session 93075), bound to actual producer process start ticks and SHA-pinned independent auditor/input-identity sources. After producer exit, requires all 200 prescribed starts via independent input/raw-frame reconstruction and exact-repeat/invariant trajectory checks, then final effective-input disjointness against complete paired exposure and the original numerical manifest. Failures halt the chain; no numerical freeze or heldout acceptance inferred from scheduling. LingBot collection now 152/200; RDT diagnostic remains live at 365/600.

### 2026-11-09T10:12:39.180760+00:00 — exact LingBot operator-trace continuation queued

Inspected legacy `build_lingbot_vla2_operator_config.py`: it binds the old 396-observation adapter/configuration and workspace paths, so it cannot qualify the exact RoboTwin adapter. Queued fresh mask-preserving native/invariant B1/B8 actual-policy traces after live complete-corpus audit/builder PID 2756307 (captured start identity). Job `logs/lingbot-post-corpus-operator-traces-001.py`, session 71644; source/adapter/harness SHA guards, actual official QWEN3VL_PATH, dedicated GPU5 UUID, diagnostic inputs/indexed noise only. Four initial traces select source layouts and do not establish full operator coverage, numerical invariance or performance. Existing live sources unchanged.

### 2026-11-09T10:14:09.414957+00:00 — π0.5 full paired cohort independently reconstructed

Complete auditor finished: 500/500 varied starts, 4,000 episodes, 25,444 reconstructed policy calls and 1,177,264 verified raw frames. All same-condition repeats and invariant singleton/dynamic trajectories exact; native first-action divergence in 500/500 starts. Published action-disagreement progress now binds this complete proof. Full magnitude extraction session 12886 and source-pinned independent follow-up session 17290 started; combined magnitude report remains the previous 91-start π0.5 subset until both finish. Final effective-input exposure auditor PID 1850283 now runs; numerical freeze/heldout remain unaccepted until actual gate completion. Old canonical 91-start audit archived by SHA before promotion.

### 2026-11-09T10:14:55.719870+00:00 — π0.5 final exposure gate passed; frozen heldout live

Full-cohort final exposure auditor completed: all800 unique heldout prepared observations disjoint from11951 exposed identities (including prior/rejected-renderer exposure). Final seal explicitly independently verified. Numerical freeze actually written and binds final seal SHA; complete traced operator and invariant diagnostic gates true. Both proofs copied to current_evidence. Full2400-request heldout replay launched PID1942407 (wrapper1942348), B1/2/4/8/16/32, actual frozen checkpoint/inference path, invariant violation fail-closed. Heldout results and final independent heldout audit not yet complete.

### 2026-11-09T10:16:01.162023+00:00 — full isolated π0.5 performance continuation queued

Bound a queued continuation to live heldout-plus-independent-audit pipeline PID2204086/start ticks; session63754. Requires predecessor completed status and independently accepted B64 capacity before full5-session/20-warmup/200-timed-call policy B1/2/8/64, then full serving logical4/padded8 and queue250requests per load/mode. Same frozen adapter/checkpoint, explicit frozen diagnostic inputs, GPU1UUID, source SHA guards, sampled compute/graphics isolation wrapper; any busy GPU/foreign activity or failed stage stops chain with raw evidence retained. Performance remains unmeasured and independently unaudited until actual execution. LingBot corpus156/200; RDT diagnostic369/600 confirmed live.

### 2026-11-09T10:21:53.473058+00:00 — complete RDT trajectory-magnitude follow-up queued

Queued full250-start magnitude extraction plus independent raw-data reconstruction after live complete paired audit PID1208987 and current199-start independent magnitude pipeline PID1263565 terminate successfully. Captured process start identities; reporting/audit sourceSHA pinned. Requires actual complete250 audit and candidate identity, archives old canonical audit bySHA before atomic promotion, preserves existing reports. Job `logs/rdt-full-250-stability-001.py`, session72087. Full250 magnitudes not yet measured/accepted; previously published46-start magnitudes remain current until independent follow-ups pass.

### 2026-11-09T10:25:53.761404+00:00 — exact LingBot B16/B32 capacity continuation queued

Queued unchanged full-policy capacity runner after live initial exact-adapter trace pipeline PID1851345, with process-start and predecessor terminal-status gates plus source/adapter/harnessSHA guards. Session74240, `logs/lingbot-post-traces-capacity-001.py`. Measures24 native/invariant B16/B32 duplicate/diverse first/middle/last arrangements using diagnostic-only target/companions and explicit indexed noise. Records OOM evidence; invariant nonexactness fails after retention. Capacity independent audit and applicable diagnostic batch matrix remain pending actual output. No full-policy capacity or invariance acceptance inferred from scheduling.

### 2026-11-09T10:30:10.417181+00:00 — CogACT expanded two-task audit live

Observed77 complete paired starts (50pick_can,27move_near), materially extending current39-start single-task accepted magnitudes. Started unchanged independent full-input/raw-frame reconstruction auditor PID2439198, session95455, output independent_progress_audit_expanded_v4.json. Accept only actual reconstructed completed cohort after terminal success; no interim native/invariant or second-task stability claim. Full200-start audit remains queued separately. Current accepted quantitative magnitude summary unchanged.

### 2026-11-09T10:54:35.198864+00:00 — RDT expanded magnitude extraction complete; independent audit live

Extraction session73875 terminated0, report paired_stability_rdt_199.json contains199 starts bound to immutable source audit snapshot. Queued post-pipeline launched independent raw-frame/action-error reconstruction PID3148632. Larger physical drift magnitudes remain unaccepted until actual independent audit success; consolidated published summary still uses46 RDT starts. Full250 extraction/audit continuation remains queued behind both full paired audit and this independent magnitude audit.

### 2026-11-09T11:11:42.863644+00:00 — full RDT freeze/heldout continuation queued

Queued source-pinned three-step numerical freeze, full2400-pair B1/2/4/8/16/32 heldout+singletonfidelity, and independent heldout audit after live diagnostic audit pipeline PID2270897 and completepaired/finalexposure pipeline PID2244027. Captured actual process-start identities, require both predecessor terminal statuses, original GPU2UUID and exact frozen native/invariant inference configuration, checkpoint/SigLIP acquisition roots and model/operator sources. Manifestpoints to fresh003 observation-disjoint corpus; strict final freshness and diagnostic/operator gates unchanged. Session20192; command manifest retained. No RDT freeze or heldout result accepted yet.

### 2026-11-09T11:46:48.297175+00:00 — CogACT full frozen heldout continuation queued

Queued strict numerical freeze, full2400-pair heldout replay+singletonfidelity B1/2/4/8/16/32, and independent heldout audit after live fullpaired/finalexposure pipeline PID1736131 (actual process start bound). Uses fresh003 corpus and original diagnosticGPU4UUID, exact fixed64 inference contract, CogACT/LLAMA assets, CogACT/OpenVLA/operator sources. Concrete source-pinned command manifest verified against existing diagnostic/operator/corpus proofs. Session89103. Final exposure seal remains required and unaccepted until actual completed full200 paired cohort audit; no CogACT freeze or heldout result exists yet.

### 2026-11-09T11:54:52.776680+00:00 — full LingBot native collection recorded; corpus reconstruction live

Collector PID1243919 exited after recording200/200 episodes and1000 selected observations. Independently checked collection-manifest/candidate-lock identity and complete50-task×diagnostic/heldout×ordinals0/1 roster. Full independent corpus auditor actually launched PID708991 with --require-complete; status of raw collection remains partial_pending_independent_corpus_audit until reconstruction succeeds. Subsequent manifest builder, exact-path traces and B16/B32 capacity continuations remain gated on that audit/builder pipeline. No full LingBot numerical or paired result accepted from collection completion.

### 2026-11-09T12:08:11.887509+00:00 — complete π0.5 magnitude report finalized; independent reconstruction live

Extraction session12886 terminated0; report paired_stability_pi05_500.json verified500 starts,50 per task across10 tasks, bound to immutable independent fullpaired audit snapshot. Independent raw-frame/action-error auditor actually launched PID1107620; queued pipeline session17290. Fullcohort magnitudes remain unpublished until this independent audit passes; accepted combined magnitude summary still contains prior91-start π0.5 subset. Fullcohort action/trajectory exactness proof500 native divergences/0 invariant divergences remains separately accepted. Heldout169/2400,RDTdiagnostic438/600 live at check.

### 2026-11-09T12:10:30.121307+00:00 — complete RDT paired cohort independently reconstructed

Full250-start audit finished:50 varied starts on each of5 ManiSkill tasks,2000 episodes,29344 reconstructed policy calls. Reset/A/A/invariant trajectory checks all pass; controlled fixed30 native singleton/dynamic actions diverged250/250, invariant0/250. Accepted action-disagreement progress now binds complete proof; magnitudes still limited to previously independently accepted46-start subset pending199/full250 magnitude audits. Final effective-input heldout exposure auditor actually launched PID1164781; freeze/heldout pipeline remains gated on final exposure and complete diagnostic acceptance. Raw proof copied to current_evidence; original199 canonical audit retained for queued magnitude chain.

### 2026-11-09T12:15:25.983072+00:00 — CogACT77-start two-task paired audit accepted; magnitudes live

Expanded audit session95455 terminated0:77 starts (50pick_can,27move_near),924 episodes,73920 reconstructed calls,74844 raw frames. Samecondition repeats and invariant singleton/dynamic actions/trajectories exact; controlled native singleton/dynamic diverged77/77. Released-padding supplement remains separate. Action-disagreement progress now binds this proof. Old canonical39 proof archived bySHA before promotion; immutable39 magnitude reports remain accepted. Launched77-start magnitude extraction session26603 and source-pinned independent followup session73310; combined accepted magnitude summary unchanged until both pass. Full200 paired/heldout acceptance still pending.

### 2026-11-09T12:19:57.635538+00:00 — LingBot full-cohort storage preparation

Measured complete native corpus49398 unique retained frames/33007602966 bytes, plus1816362852 preparedrequest bytes. Inspected real filesystem capacity: scratch~1.1TBfree, additional existing/vastdisk~1.9TBfree. Created only user-owned task directory `/vastdisk/hadjiyiannis_3_gmail_com/four-model-campaign` (no formatting/deletion/data migration). Existing pairedrunner already stores complete frames content-addressed; launched CPU retained904-frame/eight-episode pilot dedup+losslessgzip roundtrip measurement, session41898, before adopting a full2500-start storage projection. Probe firstattempt failed onlymoduleimport; traceback retained, retry explicitPYTHONPATH. No source/helper/livejobs/rawdata modified, no cohort reduction, no format change. Final capacity sufficiency remains unproven until measurements and later actual growth checks.

### 2026-11-09T12:21:00.767134+00:00 — LingBot exact content-addressing storage probe complete

Probe session41898 terminated0:904 retained pilot frame references map to287 complete-value identities (31.7478%), all287 serialization/gzip roundtrips content-exact, original904 fileSHA values unchanged. Existing runner already uses this content-addressing contract; no format change adopted. Native200-episode frame-size mean×planned20000episodes×pilotunique fraction projects1047918368500 frame bytes (~1.05TBdecimal), excluding requests/logs/caches and possible different horizons/compressibility. Existing/vastdisk task directory had2061476683776 available bytes at measurement; intended full paired destination can use it without shrinking2500starts/eightconditions. Projection is planning evidence only, not reservation or proof of final capacity sufficiency. No raw files moved/deleted/rewritten.

### 2026-11-09T12:23:57.820919+00:00 — RDT final effective-input exposure seal accepted

Actual final audit completed:800 distinct heldout prepared observations,11361 prior exposed identities,0 collisions. Strict final status/prepared-input identity kind verified and seal binds complete250 paired auditSHA; pipeline terminated completed_full_cohort_audit_and_final_exposure_check. Proof copied to current_evidence. RDT numerical freeze and fullheldout continuation remains waiting on complete600-pair diagnostic/independent audit; diagnostics447/600 live at check. No RDT heldout neural run started or numericalfreeze accepted yet.

### 2026-11-09T12:27:32.235266+00:00 — independently verified RDT199 magnitudes published

Independent magnitude pipeline session92691 terminated0:199 starts,158539 rawframefiles,2603 fixed-input nonsingleton comparisons independently reconstructed. Full task-specific accepted magnitude summary now329starts (91π0.5,199RDT,39CogACT); previous176 summary archived bySHA and allreport/audit hashes validated. RDTnative median/worst perstartpeak EEFdrift mm: Peg50starts76.223/327.002; Pick50starts20.539/139.377; Plug49starts34.713/186.288; Stack50starts21.137/477.973. Invariant physicaldrift/actionerrors0 on all199; fixed-input/prefix scope preserved. Complete250 action-disagreement proof remains accepted separately and was not downgraded. Full250 magnitude continuation nowactuallylive PID1640058, independently auditable report stillpending.

### 2026-11-09T12:34:22.522669+00:00 — complete RDT250 magnitude report finalized; independent audit live

Queued fullcohort extraction finished and report verified250 starts/50 on each of5 tasks, bound to immutable complete paired audit snapshot. Samepipeline launched independent raw-frame/normalized-error reconstruction PID1846113. Published magnitudes remain independently verified199-start subset; full250 report not accepted yet. No producer/model/helper sources changed.

### 2026-11-09T12:39:54.712270+00:00 — CogACT77 magnitude report finalized; independent audit live

Extraction session26603 terminated0; report independently rosterchecked77starts/50pick_can+27move_near, bound to immutable accepted77 paired audit snapshot. Source-pinned followup launched independent rawframe/action-error auditor PID2046264. Published CogACT magnitudes remain prior39start subset until actual independent verification completes; existing77 action-disagreement proof remains separately accepted. Full200campaign stillpending.

### 2026-11-09T12:46:16.802705+00:00 — LingBot complete native corpus independently accepted; first exact policy trace complete

Full independent audit accepted200/200episodes,49398 rawframes,1077officially preprocessed input/physical-action calls,1000selectedinputs. Builder actually completed exact manifest200diagnostic/800heldout and3000unique indexed request/noise pairs; audit/snapshot/noiseSHA/cardinality verified, proof retained. Exact-adapter nativeB1 actual-policy trace completed33 recorded operation patterns/6399calls with losslessmask extension; outputbinds actualmanifest. Trace pipeline PID1851345 now running nextB1invariant stage; B8mode traces/capacity remain queued. Initial record-update check raced firsttrace successfulexit; current accepted corpus proof verified and state uses live pipeline identity. No numerical/fulloperator/heldout/fullpaired acceptance inferred.

### 2026-11-09T13:02:48.062091+00:00 — LingBot exact operator sweeps running and queued

All four exact-policy traces completed; actual B1/B8 native/invariant source-pattern unions agree. Prepared cases map all33 patterns exactly once at both batch sizes:24linear,6request-scaled reductions,1sharedpooling+2rotarymatmul. Linear40800-measurement sweep live with independently audited8039-record prefix (7539batched comparisons:6939native nonexact,0invariant nonexact; synthetic operands only). Reduction10200 and remaining5100 sweeps plus independent audits queued serially onGPU5. Full600-pair model diagnostic and independent diagnostic audit queued after successful complete operator coverage. Capacity24 measurements finished; no fulloperator/fullmodel diagnostic acceptance yet. No live/frozen sources changed.

### 2026-11-09T13:11:20.499298+00:00 — LingBot heldout collision repair candidates independently reconstructed

Prepared-input signature screen against1077 native corpus calls plus200diagnostic observations found13of800held rows exposed. Original live manifest retained unchanged. Deterministic forward nonreset/noncall/nonselected replacement frames yielded13new prepared identities; official preprocessing ran without neural inference. Initial CPU-only import failed on upstream CUDA capability metadata query, before candidate creation; retry on pinned GPU5 succeeded. Independent second process rebuilt all13stored request tensors byte/signature exact from content-addressed retained frames and verified episode SHA/pre-action/noncall alignment. Evidence copy retained. Earlier pilot/legacy and futurepaired exposures still require complete-universe audit; no final seal/freeze/heldout acceptance inferred.

### 2026-11-09T13:21:46.938630+00:00 — refreshed LingBot replay metadata built and integration-checked

Composite lineage audit preserves200native episodes and1000selected inputs, changing only13independently reconstructed heldout requests. Builder rehashed all corpus raw files and generated fresh candidate replay manifest9628d1d70f4b4572a550c5d3ebf0c323f9aa06aa5f129ca5a6955e1af4fb9e98 with3000indexed noises. Unchanged exact adapter loaded metadata; all3000noise fingerprints verified;200diagnostic rows identical. Tensor-byte exposure screen reread1172native requests; priorhistory1185unique inputs and pilot companions yield0collisions with repaired800held candidates. Final exposure auditor prepared and correctly rejects pilot instead of2500start completepairedproof. No finalseal/freeze/heldout acceptance. B64capacity+indepaudit queued afterfullmodeldiagnostics. Rehashed unauthorized diagnostic/heldout snapshot mutations rejected by lineage auditor.

### 2026-11-09T13:27:45.264942+00:00 — CogACT77 magnitude audit accepted and published

Independent auditor completed77starts (50pick_can/27move_near),30723rawframes and4756fixed-input comparisons, with report/source/snapshot/roster hashes checked. All same-condition repeats exact. Controllednative action/physics/RGB divergence77/77;invariant0/77,0numerical error and0physical drift. Native perstartpeak EEF median/worst mm:pick_can6.495/377.419;move_near64.658/259.295. Native identical-input normalized RMS pick_can0.00237502,move_near0.00201795. Accepted magnitude summary now367starts (91PI/199RDT/77Cog); previous329snapshot archived bySHA. Fullcampaignpending.

### 2026-11-09T13:30:23.576002+00:00 — full LingBot paired and final heldout continuations queued

Live source-pinned dependency chain now extends through B64capacity/independent audit →2500starts/20000paired episodes on user-owned /vastdisk campaign volume →fullindependent paired audit →finalprepared-input exposure audit →explicit freeze-schema validation →numerical freeze →2400heldoutpairs →independentheldoutaudit. Fullpaired requires accepted600pair diagnostic proof and all50qualifiedtaskworkers;1.5TBfree prelaunch gate retained. No rollout/freeze/heldout acceptance inferred. Exposure schema validator verifies actual fingerprint ignoresnoise/requestIDs and allsixneuralfields affectidentity, bindsunderlyingproof/source, rereadsevidencehashes. Fourisolatedsynthetic gate tests passed (validschema/incompletecohort/changedsource/changedauditor); no actualseal generated. Pinned scripts/manifests untouched.

### 2026-11-09T13:40:18.133693+00:00 — LingBot full linear operator sweep independently accepted

All24actual source linear patterns completed408arrangements/40800seed measurements; independent auditor accepted complete coverage, config/auditor/rawSHA verified. Batched comparisons{'all_batched_comparisons': 38400, 'native_nonexact': 32400, 'invariant_nonexact': 0, 'request_scaled_batched_comparisons': 36800, 'request_scaled_native_nonexact': 32400, 'shared_internal_shape_batched_comparisons': 1600}; native maxabsoluteerror0.003173828125. Invariant0nonexact. Source-shape synthetic operator scope only, no fullmodel/trajectory/performance acceptance inferred. Reduction sweep nowrunning onGPU5.

### 2026-11-09T13:50:17.597063+00:00 — complete LingBot actual-source operator qualification accepted

All33B1/B8 patterns independently reconciled against56100completed measurements:40800linear,10200request-scaledmean/softmax,5100sharedpooling/rotarymatmul. Complete coverage auditor/source/config/raw hashes checked through actual freeze operator gate. Native mean/softmax0nonexact across9600batched comparisons; linear native32400nonexact vs invariant0. Fullmodel600pair diagnostic pipeline actually started loading strict checkpoint; no diagnostic/heldout/fullpaired/performance completion inferred.

### 2026-11-09T13:56:01.722734+00:00 — all four isolated performance/audit continuations tracked

Independent performance audit layer prepared over existing independent5session/20warmup/200timing and queue calibration/cardinality validators; frozen bytes/configuration/source archives and exact benchmark GPU PID inventories checked, nonfinite measurements rejected. Six focused isolation tests passed; no actualtiming acceptance inferred. Policy/serving/queue pipelines queued afteracceptedheldout on each modelGPU, with independent auditcontinuations. LingBot initial sortedfirst64selected32of50tasks andfirst8only4tasks; only its waitingperformance launcher superseded withv2task/episode-round-robin64inputscovering50tasks/64episodes andfirst8covering8tasks/8episodes, independently checked. No model/simulation/livefrozen sources changed. Eight actuallivewaitingprocess identities recorded. LingBot fullmodeldiagnosticPID4055960 active.

### 2026-11-09T14:12:18.096623+00:00 — complete RDT250 paired magnitudes independently accepted

Independent audit reconstructed250starts/5tasks×50,169290rawframefiles and2822fixed-input comparisons. Report/auditor/reportingsource/candidate/pairedproof snapshots and exact rosters verified; allAArepeatcontrols exact. Controlledfixed30native actiondivergence250/250,invariant0/250; invariantfixedinputerror/EEFdrift0alltasks. Native perstartpeak EEF median/worst mm:Peg76.223/327.002,Pick20.539/139.377,Plug36.044/186.288,Push5.740/34.908,Stack21.137/477.973. Accepted magnitude summary418starts (91PI/250RDT/77Cog); previous367snapshot archived bySHA. Fullnumerical/heldout/performance/fullfourmodel goal notcomplete.

### 2026-11-09T14:21:43.471533+00:00 — complete PI05 500-start paired magnitudes independently accepted

Independent audit reconstructed500starts/10tasks×50,416178rawframefiles and2620fixed-input comparisons. Report/auditor/reportingsource/candidate/pairedproof snapshots and exactrosters verified;allAArepeatcontrols exact. Native actiondivergence500/500,invariant0/500;invariantfixedinputerror/EEFdrift0alltasks on pinnednoMSAArenderer. WorstnativeEEFdrift630.275mm(task9);task-specificnativeRMS0.00254673–0.02544642,BI0. Accepted magnitude summary827starts (500PI/250RDT/77Cog); previous418snapshot archived bySHA. Fullheldout/performance/Cog200/Lingfullgoalnotcomplete.

### 2026-11-09T14:26:01.015597+00:00 — CogACT expanded raw paired audit live; full200 magnitudes queued

Independent v5reconstruction audit launched against125completed paired blocks (50pick_can/50move_near/25open_drawer) while200producer continues. Canonical77proof and accepted77magnitudes retained unchanged. Separatefull200magnitude extraction+indepaudit waiter uses full200pairedproof as gate, independent of finalheldoutfreshness; no fullcohortacceptance before actualproof. Initialwaitinglauncher exactstatuspredicate corrected in002before any extraction; superseded001state retained. Currentproof and waiter PID/start identities verified.

### Causal stability preflight: exact-adapter intervention inventory

Added a stdlib-only, no-inference trace applicability inspector and retained `current_evidence/causal_ablation_applicability_001.json`. It binds each actual B8 native trace to its exact manifest SHA. PI05, RDT and CogACT observe conv2d and SDPA; the inherited RDT convolution-inapplicability declaration belongs to an older boundary and must not be reused. LingBot observes linear/matmul/mean/softmax; absence alone does not establish source-level inapplicability. Positive conv2d and four rejection controls passed. No pinned model, adapter, runner or numerical freeze was changed. Causal measurements and propagation summaries remain outstanding; this inventory is not an ablation acceptance proof. Existing freezes must not be retrospectively described as following these missing measurements.

### Independently recounted denoising stability distributions

Added `summarize_flow_step_stability.py` and an independent raw-record recount (`audit_flow_step_stability.py`). Accepted full diagnostic evidence yields 2,000 PI05 task/mode/batch/stage groups and 400 CogACT groups, each spanning all 600 input–noise pairs at diverse-companion B2/4/8/16/32 with target first. Reports retain exact rates, median/p95/max per-comparison RMSE, element-weighted paired mean squared deviations and maximum absolute errors. All invariant recorded stages are exact. Quantiles use linear interpolation; distributions are descriptive, not episode-independent confidence intervals. Corrupt-p95 and dropped-stage-group negative controls were rejected. Exported source/audit-bound B8 PNG/PDF figures; visually inspected CogACT artifact. Intermediate causal modes, episode-bootstrap uncertainty, and RDT/LingBot propagation remain outstanding. Clarification: RDT exact adapter uses composition, not inheritance; the stale convolution exception resides on its inner legacy adapter and is not directly exposed as an outer exact-adapter contract. No live/frozen numerical sources were modified.

### RDT and LingBot propagation continuations queued

Started and verified live source-pinned CPU continuations `rdt-post-diagnostic-flow-stability-001` (PID2216571) and `lingbot-post-diagnostic-flow-stability-001` (PID2216572). Each waits its existing identified diagnostic audit pipeline, requires independently accepted complete600pair proof, then extracts all required task/mode/batch/stage distributions, independently recounts raw records and exports B8 figures. They do not reuse partial diagnostic prefixes or claim acceptance while waiting. Current producer processes and audit pipeline identities were verified live; no model inference or frozen source was changed.

### Exact-adapter six-mode causal campaigns queued

Added a source-pinned experiment wrapper around the existing six-mode causal runner. All four complete200-observation/600-input-noise preflights passed. Six negative controls reject missing source pins, changed manifest/trace, foreign adapter, held-out companions and same-episode companions. Each model has a verified live continuation waiting its existing independent performance audit, serializing GPU work after timings. These are queued measurements, not completed causal evidence. LingBot replaces stale experiment-metadata paths with its pinned eager-einsum source and actual trace; numerical adapter/operator code is unchanged. PI05 trace is an operation-applicability reference from an older manifest with matching execution configuration, not an assertion that old and current replay inputs coincide; runtime intervention coverage is still required. Independent full ablation auditing and honest freeze-addendum reconciliation remain outstanding.

Causal launch correction before any inference: PI05 execution configuration reads the loaded model config, so the first wrapper would have queried it too early. Verified and stopped only the four new waiting causal launchers, preserving their states as superseded. Added version2 wrapper/plans and fresh `causal_002` outputs, validating the configuration recorded after model loading instead. All four v2 preflights passed and all four replacement waiting processes were verified live. Existing heldout/performance/paired jobs, pinned numerical sources and freezes are unchanged.

### Independent causal measurement-contract verification prepared

Added `audit_exact_model_causal_ablation.py`, which requires the complete3600-record six-mode factorial and accepted600-pair diagnostic, pins source/manifest/plan/provenance, validates normalized and physical output errors and fingerprints, checks required stages and first recorded divergence, and rejects skipped convolution/attention interventions. Nine adversarial synthetic tests passed (complete fixture, missing/duplicate request, missing stage, skipped attention, altered noise, contradictory byte exactness, wrong boundary and signed-zero distinction). Synthetic fixtures are not model evidence. Four verified live source-pinned CPU audit continuations wait their full causal producers. This independently checks retained measurement contracts; missing tensor arrays are not reconstructed and operator-local reexecution is not claimed. Actual causal measurement/audit acceptance remains outstanding.

### Paired-start stability uncertainty independently recounted

Computed task-specific paired-start percentile95% intervals with20,000 resamples/task and retained task-keyed PCG64 seed20261109 indices for accepted500PI05,250RDT and77CogACT starts (17task groups;204 estimates). Each resampling draw preserves the native/invariant pairing and repeats an entire restored start; frames/calls/noise replicas are not independent trials. Metrics: mean per-start identical-input MSE, mean and median peak common-horizon EEF drift inmm, and fraction of starts with action divergence, plus native-minus-invariant contrasts. No model/task pooling. Equal-start MSE differs deliberately from element-weighted pooled diagnostic RMS. Independent recount verified every draw against seed, exact audited rosters, estimates and interval endpoints. Four negative controls rejected altered intervals, missing tasks, duplicate estimates and rehashed altered draws. JSON+CSV summary retained. Intervals are conditional on the retained cohort, unadjusted for multiplicity; zero/one degenerate intervals do not establish population certainty. CogACT77 is partial; LingBot absent pending actual paired proof. Full campaign remains incomplete.

### CogACT expanded uncertainty continuations queued

Verified134 produced paired blocks and live independent125-start audit; acceptance remains77 starts. Started and verified live source-pinned CPU bootstrap/recount continuations for v5 (requires at least125 independently audited magnitude starts) and full200 (requires exactly200). Each waits its existing identified magnitude-audit producer, then computes20,000 task-specific paired-start resamples and independently verifies indices, estimates and percentile endpoints. Current827-start accepted tables remain unchanged until new evidence passes. LingBot full paired magnitude extraction remains a separate outstanding requirement; the existing generic3-model reporter does not support its dual-arm path.

### Full LingBot dual-arm magnitude extractor prepared

Added `summarize_lingbot_full_paired_stability.py` with a strict complete50-task/2500-start/20000-episode independent-proof gate. It reads actual content-addressed raw observations, reports left/right EEF translation/rotation separately, compares14 physical action components without mixing units, retains first action divergence and numerical comparisons only for equal prepared inputs and indexed noise, and uses the common trajectory horizon. B1-only episodes have explicitly unobserved numerical batching error, not an invented zero. Ten focused controls passed, including full/duplicate roster gates, arm separation, known quaternion rotation/sign equivalence, first physical action divergence, noise mismatch rejection, post-feedback exclusion and unequal horizon tails. No full-campaign extraction has executed. Independent magnitude verification must bind episode bytes as well as raw-frame proof; full LingBot task summaries/uncertainty and final acceptance remain outstanding. Existing pinned inference/paired/audit pipelines were not modified.

### LingBot episode-bound audit and full magnitude continuation

Added a separate `audit_lingbot_paired_study_bound.py`, retaining the original full reconstruction checks while binding each episode to the exact compressed bytes decoded and rechecking those bytes at completion. Full magnitude extraction now requires all20,000 independently bound episode files. Five byte-binding adversarial controls and eleven measurement/protocol controls passed (16total); changed/recompressed bytes and missing bindings are rejected. Queued and verified a live source-pinned continuation after independent performance acceptance: full episode-bound reconstruction, then full dual-arm magnitude extraction. This is an additional independent CPU reconstruction, not neural inference; resulting magnitudes still require an independent recount.

Corrected two newly queued LingBot leaf analyses before execution: the actual diagnostic producer emits `independent_complete_audit.json`, unlike the other models’ filename. Only the two identified waiting leaf processes were stopped; original scripts/states retained as superseded, version2 replacements verified live. Main paired, diagnostic, heldout, performance and causal producers were unchanged.

### Full LingBot magnitude recount prepared and queued

Added `audit_lingbot_full_stability.py` without importing magnitude-reporter helpers. It requires the independently reconstructed complete2500-start/20000-episode byte-bound proof, verifies exact rosters and episode/frame inventories, independently rederives normalized output SSE/RMS/ULP and signed-zero bit differences, physical14-component action errors/first divergence, and both arms’ common-horizon translation and rotation. Rotation recount uses relative-quaternion products, independent of the reporter’s chord formula. Seven focused arithmetic controls passed. A verified live CPU continuation waits full bound extraction, then runs the independent raw-frame magnitude recount with CUDA hidden. Actual full-cohort execution remains outstanding; task-level LingBot summaries/uncertainty still need implementation. Five main producer/audit process identities and kernel activity counters were reverified live.

### LingBot per-task tables prepared and queued

Added a stdlib-only descriptive reporter requiring full2500-start/20000-episode independent magnitude acceptance. It produces200 task/mode/arm rows (JSON+CSV), preserving separate left/right peak translation and rotation, first-action divergence conditional on divergence, counts of nonsingleton exposure and fixed-identical-input comparisons, and pooled normalized numerical error. Missing numerical exposure remains null rather than zero; the shared normalized action metric is explicitly not arm-specific. Three focused controls passed for arm units/pooled errors, no-exposure handling and nonfinite rejection. Started and verified live continuation after the independent full magnitude audit. Actual full tables, independent table recount and task-level paired uncertainty remain outstanding.

### First-leaf attribution projection preflight

Reviewed the existing generic leaf tracer before scheduling operator-level attribution. Its leading first-slice projection is insufficient for CogACT diffusion CFG: source concatenates `[all conditional requests, all unconditional requests]`, so singleton target rows0/1 map to B2 rows0/2, not rows0/1. A synthetic CPU control executing the actual tracer projection reproduced a false mismatch while explicit source-backed target indices are exact. Retained source hashes and counterexample. This is not model inference or new measured instability. Do not claim generic flattened/CFG leaf scans establish operator-level first divergence without semantic request mapping. Existing exact-adapter denoising diagnostics and paired magnitude audits do not use this generic leaf projection and remain unchanged. Local operator capture/reexecution remains outstanding.

### Explicit CogACT request projection integrated into separate leaf tracer

Added `leaf_request_projection.py` and a separate `trace_exact_model_leaf_divergence.py` without changing the pinned generic tracer or numerical sources. The new tracer declares logical batch size, selects target rows inside conditional/unconditional blocks for CogACT action_model.net, binds the actual concatenation/split source contract, rejects unresolved shapes, retains both leaf scan hash streams, and explicitly sets operator attribution incomplete. Six focused controls and integrated CPU CFG row-selection/unresolved-shape controls passed. No model trace was executed. Actual traced inference, repeat checks and independent isolated local-operator input/weight/output reexecution remain outstanding; first comparable leaf cannot be presented as globally first causal operator.

### Layout-preserving operand snapshots prepared

Added `captured_tensor_layout.py` for later independent isolated-operator replay. It captures the physical operand span plus logical-byte fingerprints, dtype, shape, strides, original storage offset and pointer alignment; reconstruction preserves zero-stride broadcasts without copying into overlapping views. GPU replay defaults to requiring identical pointer alignment, rejecting failure to reproduce rather than treating a changed layout as numerical evidence. Nine CPU controls passed for transpose/nonzero offset, broadcast, gapped storage, signed-zero/int64 bits, scalar/empty tensors, corrupt payload/stride, declared capacity and alignment mismatch. These are layout controls, not GPU operator qualification. Actual model operand capture, precision/context retention and independent isolated local-operator reexecution remain outstanding. Pinned inference/model sources and current experimental jobs were unchanged.

### Supported leaf capture and isolated replay prepared

Added `captured_leaf_operator.py` for exact eager Linear/zero-padding Conv2d modules. It retains input/weights/bias/output physical snapshots and source/provenance, Torch/CUDA/GPU UUID, autocast, TF32/reduced-precision flags, cuDNN/determinism settings, numerical environment and invariant context. Isolated replay restores runtime in a finally block, verifies operand alignment on GPU, and must reproduce captured output bytes/layout twice. Integrated optional artifact capture into the separate projection-aware leaf tracer; artifact hashes retained, unsupported leaves explicit. Twenty-three combined CPU controls and a synthetic hook→file→isolated-replay control passed. A corruption test initially attempted to mutate an inference tensor; corrected the test to clone before corruption, then all controls passed. These are CPU synthetic controls, not checkpoint-derived GPU attribution. Independent artifact/projection audit CLI, actual GPU capture, repeat controls and local causal contrast remain outstanding. Pinned numerical sources and live experiment pipelines unchanged.

### Captured leaf artifact replay verifier prepared

Added `audit_captured_leaf_replay.py` to bind four native/invariant singleton/batched artifacts to the diagnostic manifest, exact indexed noise, unrelated diagnostic companions, loaded checkpoint configuration, source pins, scan hashes and capture-file bytes. It independently projects target blocks using reshape/indexing, checks static weights, replays supported local operators twice, and reports same-input/native-dependence/invariant-stability/same-operand-repair flags explicitly. The separate tracer now hashes the full leaf scan and pins relevant adapter/loader/backend/operator sources. Seven new adversarial metadata/projection controls passed; combined capture/layout/projection suite30/30. No checkpoint-derived GPU capture/audit has executed. Local supported-leaf evidence must not be expanded into a globally first operator claim; actual GPU experiment and unsupported-boundary review remain outstanding. Main campaigns unchanged.

### Predeclared local capture/replay experiments queued for all four models

Started and verified live source-pinned continuations after each complete causal measurement-contract audit. Each revalidates the actual numerical freeze (including both diagnostic and heldout manifests where applicable), then traces one fixed diagnostic input at B2/noise0, captures supported first-comparable-leaf operands for native/invariant B1/B2, and independently replays the four artifacts. Selection is deterministic first task/episode/observation/index, not outcome/error-driven: PI05 index0; RDT index0; CogACT index750; LingBot index200. Source files for projection, layout capture, supported leaf replay, tracer and artifact audit are now pinned; future corrections must use versioned sources and waiting-launcher supersession. Unsupported/unresolved leaves or failed replay remain explicit failures; no replacement input is selected. These are additional local mechanism cases, not whole-cohort or globally first-operator claims. Actual GPU traces/replays remain pending behind existing stages.

### Independent LingBot task-table verification queued

Added a separate stdlib task-table auditor that rebuilds all200 task/mode/arm statistics from independently accepted2500-start magnitude records, rejects missing/duplicate task rows and wrong units, and checks CSV values/schema against the JSON table. Five focused controls passed, including hand-calculated percentiles, unit corruption, null-versus-zero, nonfinite errors and corrupt CSV. Verified live source-pinned continuation waits full task extraction. Actual task tables are not yet accepted; task-specific uncertainty remains outstanding. Main inference/simulator pipelines unchanged.

### LingBot paired-start arm uncertainty prepared and queued

Added full-cohort dual-arm bootstrap reporter and independent recount. Actual protocol is20,000 task-keyed PCG64 paired-start resamples/task, seed20261109, 95% linear percentile intervals, no multiplicity correction. Reports separate arm position/rotation means/medians, start action-divergence fraction and native-minus-invariant contrasts (1350estimates), retaining all50 starts/task and explicit exposure counts. Frames/calls are not resampling units; conditional cohort intervals do not establish population certainty, and numerical-MSE confidence intervals are not claimed. Four focused controls passed. A clearly labelled temporary synthetic50-task/2500-start fixture at10,000resamples passed full extraction/recount; three corrupted interval/task/draw controls were rejected. Synthetic artifacts were removed with the temporary directory; no synthetic model results entered accepted summaries. Source-pinned live continuation waits full task-table acceptance, then runs actual20,000-resample extraction and independent recount. Actual LingBot intervals remain pending.

### RDT complete diagnostic accepted; heldout started

RDT diagnostic producer completed all600 request–noise pairs. Independent audit verified200 observations,1200 implementation groups and46,800 arrangement records:22,200 native nonexact arrangements, invariant outputs and all required denoising stages exact. The independent60-group output recount gives normalized native singleton/batch RMS at B2=.001223555773, B4=.002334229713, B8=.002069060858, B16=.002244519933, B32=.001472224170; invariant RMS0 throughout. Singleton fidelity is separately nonexact600/600 versus native, not a batching failure. B64 capacity audit accepted12 arrangements with native/invariant feasible and invariant exact; this is not isolated timing acceptance. Numerical freeze was created after primary diagnostic and final exposure proof; verified one actual RDT heldout model process live. Complete diagnostic summary now includes PI05/CogACT/RDT, with prior summary archived by SHA. Full paired250-start evidence is unchanged. Later causal/local diagnostic chronology remains explicit; heldout/performance/full final acceptance pending.

### Singleton fidelity aggregates independently recounted for three models

Summarized the accepted600-record singleton-fidelity source for each of PI05, RDT and CogACT, and independently recounted pooled/task aggregate arithmetic (19task groups). This compares BI B1 to controlled-native B1, not batching stability; all600 normalized/physical returned outputs per model are nonexact. Pooled normalized RMS differences are PI05 .0124746106262, RDT .0120958921296, CogACT .00826373695057; maximum normalized absolute differences1.97572010756,1.9921875,.7172460109 respectively. These scales/tasks are not comparable across models. Per-task and median/p95 values retained; no epsilon or performance objective inferred. Physical components have mixed units and are not numerically pooled. Four corrupt-report controls rejected; three synthetic weighted-MSE/mixed-unit/zero-error-bit-mismatch controls passed. Arithmetic recount does not rederive tensor-array errors from absent arrays. Existing batching-stability acceptance remains unchanged. LingBot aggregates wait its complete diagnostic proof.

Queued and verified a live source-pinned LingBot singleton-fidelity extraction/recount continuation after its complete600-pair diagnostic proof. It uses the producer’s actual `independent_complete_audit.json` filename and preserves controlled-native/atomic-MoE scope. Current three-model fidelity summary remains a fixed accepted snapshot; no LingBot values are implied by queued analysis.

### Cross-manifest episode split verified

Read the exact diagnostic and heldout manifests selected by all four actual freeze commands, including separate fresh heldout manifests. Each has200 diagnostic and800 heldout observations with zero task/episode intersection: PI05 20/80episodes, RDT20/80, CogACT20/80, LingBot100/100. Both manifests were additionally verified against actual PI05/RDT numerical-freeze content hashes. CogACT/LingBot freeze creation remains pending; this split check does not replace their final prepared-neural-input exposure audits or numerical gates. Retained command-source/manifests hashes and split evidence.

### Primary CogACT flow projection directly checked

Inspected the frozen exact adapter’s callback: DDIM `[conditional B, unconditional B]` tensors are stacked into `[logical request, branch, ...]` before `flow.step_*` comparison. A synthetic CPU parent-forward fixture executed that actual callback and verified target branches0/2 and companion1/3 are retained together. No checkpoint/model inference occurred, no frozen source was modified, and the callback was restored normally. This corroborates the distinction already recorded: the generic leaf tracer’s first-slice CFG counterexample does not affect accepted exact-adapter denoising diagnostics. Updated live primary-job identities to track RDT heldout rather than its completed diagnostic producer.


### 2026-11-09: paired stability figures from accepted magnitudes

Generated per-model PNG/PDF figures for PI05 500 starts, RDT 250 and CogACT 77, directly from reports bound to matching independent magnitude audits. Plots show identical-input normalized action RMS, median per-start peak closed-loop EEF drift in mm, and start action-divergence frequency. All34 plotted task/mode rows match the accepted summary; source/audit/plotter/artifact hashes retained. Final figure version003 was visually inspected for all three models; versions001/002 are superseded formatting drafts. Exact zeros are labelled, point summaries are not confidence intervals, and models are not ranked. This adds presentation artifacts, not new accepted starts or full-campaign completion. Expanded CogACT125-start extraction remains live.


### 2026-11-09: versioned CogACT drawer magnitude schema recovery

The125-start raw audit completed successfully, but the older magnitude extractor failed on the first drawer episode because those official observations have empty extra dictionaries and retain EEF pose in agent.eef_pos instead of extra.tcp_pose. Its partial v5 report has no independent magnitude acceptance. Added versioned reporter/auditor without changing pinned original numerical sources or simulator jobs. Drawer drift uses recorded robot-base xyz/wxyz (first7 of8 proprioception values); pick-can/move-near retain world TCP. Coordinate-frame metadata is explicit and tasks/frames are not pooled. Initial/terminal raw frames from all4 tasks passed decoder crosschecks; malformed/nonfinite/zero-quaternion controls rejected. Versioned reporter takes an explicit immutable raw-audit proof to avoid mutable canonical-roster races, and uses a separate pose-cache namespace. Replaced only verified waiting full200 magnitude/bootstrap processes, retaining original states and failed evidence. New125-start pipeline PID2720050 is extracting; full200 PID2720051 waits the original full raw audit. Both run independent magnitude recount then20k paired-start bootstrap/recount. New sources are pinned. Accepted magnitude cohort remains827 until actual recount passes.


Complete native/invariant comparison controls on retained open_drawer/start0 and close_drawer/start0 episodes passed the new decoder path. Invariant physical/observation/action disagreement counts and EEF position/rotation drift were zero for both tasks. These targeted controls are not independent full-cohort magnitude acceptance. Evidence: results/four_model/current_evidence/cogact_drawer_complete_comparison_v2_controls_001.json. Expanded125 extraction and full200 waiting continuation remain live.


### 2026-11-09: complete causal-path figure export queued

Added a six-mode B2 causal flow figure exporter and separate NumPy table recount, binding the complete3600-record/600-request contract proof and raw bytes. This preserves the reference native/existing-ops/patch-projection/full-invariant curves and also shows native-deterministic and explicit-per-matrix modes. Per-task medians/p95 and full-invariant exact counts are explicit; plots paginate ten tasks, distinguish denoising quantities, and make no confidence-interval or tensor-array reconstruction claim. Temporary synthetic3600-record fixture passed full export/recount; changed median, missing/duplicate group, wrong exact count and altered figure controls were rejected. Fixture deleted; no model evidence accepted. Source-pinned live continuations wait all four independent causal contract audits, then export PNG/PDF/table and recount. Actual causal measurements, actual figures and visual inspection remain pending.


### 2026-11-09: final diagnostic chronology acceptance remains open

Direct inspection confirms PI05 phase freeze at2026-11-09T10:14:19Z and RDT phase freeze at16:35:53Z precede completion of the later causal contract audits; those proofs are absent and not bound into the phase freezes. CogACT/LingBot causal/local proofs also remain pending. Under acceptance4/6, the final campaign cannot claim all diagnostic gates passed before these freezes. Preserve existing freezes, heldout runs and audited subsets as phase evidence. Final acceptance requires explicit chronological closure after actual causal/local acceptance: an appropriately ordered final numerical freeze and heldout evaluation with numerical identity and heldout exposure qualification. Any changed numerical implementation requires requalification and fresh heldout inputs. No retroactive completion claim or silent exemption is permitted. This is an open full-goal gate, not a reason to interrupt live jobs. Evidence: results/four_model/current_evidence/final_diagnostic_chronology_gap_001.json.


### 2026-11-09: isolated operator performance gap and source-bound plans

Acceptance9/reference PLAN23 require operator kernel timing independently of tracing; current isolated wrappers cover only policy/serving/queue, while qualification sweeps explicitly mark their shared-GPU timings unqualified. This remains an open campaign stage. Added a stdlib plan builder binding complete qualified native B1/B8 trace cases: PI05 58, RDT70, CogACT64, LingBot66 (258total). Source dtype/shapes/strides and lossless masks are retained; attention cases prescribe native, persistent invariant and explicit invariant per-matrix reference. RDT old B1 mask derivation is restricted to identical request0 of accepted B8 traces with matching execution configuration/shape and agreement of all matching target masks. All258 case/source rosters matched byte-bound traces; bad mask encoding/hash, missing/foreign request derivation and conflicting matching target masks were rejected. No GPU timing performed or accepted. Isolated execution, timing/capacity audit and queue integration remain to implement. Existing model jobs/source pins unchanged. Evidence: results/four_model/current_evidence/operator_performance_plan_preflight_001.json.


### 2026-11-09: isolated operator timing stage implemented and queued

Added exact source-descriptor operand regeneration, lossless mask restoration, separate per-matrix invariant references, CUDA-event plus synchronized wall timing, sampled compute/graphics isolation and a separate raw-record auditor. Plans002 retain258 qualified B1/B8 source cases and add32 standalone QK/PV components (290variants;3270 case/variant/mode/session slots). Thus fused-SDPA models also have native BMM versus persistent invariant BMM versus per-matrix invariant MM component comparisons. Five sessions,20warmups and200retained timed calls per slot; compilation, allocations, layout setup and mode registration are outside measured regions. Generated P values for PV are source-shaped random operands, not captured probabilities. Original pointer alignment/storage offset are not reproduced and are not claimed. Source masks were reconstructed bitwise on CPU for all4 captured/derived cases;32component layouts passed metadata-only crosschecks. Twenty-one CPU controls passed, including synthetic timing/roster/layout/mask/source/isolation corruptions, capacity record retention, standalone components and mocked CUDA-event setup.

A prelaunch review caught lazy CUDA-event initialization in the first worker: all four001 launchers were verified waiting with no output directories/children, stopped, and retained as superseded before GPU timing. Versioned worker/wrapper/auditor v2 prime all400events before measured calls; the mock control verifies all event initialization occurs after warmup/reference and before the200timed operator calls. Four source-pinned002 continuations are live, waiting only for existing local GPU diagnostic processes to end plus accepted causal contracts and observed idle GPUs. Local diagnostic terminal statuses are retained separately; local acceptance remains a final goal gate. Each operator run then performs its independent raw timing audit. No actual operator GPU timing or full GPU integration is accepted yet. Existing simulator, diagnostic and heldout jobs unchanged. Sources now pinned; fixes require versioned replacements and verified waiting-launcher supersession. Evidence: results/four_model/current_evidence/isolated_operator_performance_queue_002.json.


### 2026-11-09: reference chronology interpretation corrected

Re-read reference PLAN13/14/16/17/33. The reference explicitly freezes the numerical implementation in14, before causal ablation16, and requires freeze-before-heldout plus complete ablations at final acceptance. The earlier inference that merely performing causal analysis after a freeze invalidates that freeze was overly strict; preserve the historical001 note but use reconciliation002. Current causal work is scheduled after the initial heldout phase, so retain that scheduling difference explicitly. It cannot retroactively be described as completed before heldout. Final acceptance still requires all actual causal/local audits and unchanged numerical identity across the frozen/heldout path. If late diagnostics expose a real implementation bug or cause any numerical-path change, affected heldout acceptance must be invalidated, requalification/new freeze completed, and the full properly exposure-qualified heldout campaign rerun. No additional complete rerun is justified solely by freeze-before-causal ordering. All original scientific deliverables remain required; no final acceptance is claimed.

### Serving stability figure continuation — 2026-11-09

Queued all four CPU figure/recount continuations after their corresponding independent real policy/serving/queue audits. Eight synthetic controls passed; this is preparation, not model measurement acceptance. Figures will annotate exactness to the declared reference alongside throughput and p95 latency. Fixed padding uses its same-position B8 reference and does not establish singleton equivalence. Actual exports require later visual inspection. Source hashes and live waiting parent identities: `results/four_model/current_evidence/serving_stability_figure_queue_001.json`. Singleton-versus-batched action and trajectory stability remains the primary endpoint; full campaign incomplete.

### Task-level singleton/batched differences — 2026-11-09

`results/four_model/singleton_batched_stability_task_table_001.csv` exposes all 34 accepted task/mode rows (827 starts), including normalized RMS, derived mean squared difference, max absolute action difference, and closed-loop EEF drift. Source report and independent audit hashes are retained in the adjacent provenance JSON; all values were checked against the accepted reports and CSV round trip. Mean squared difference is RMS squared, not centered variance or a new raw-tensor audit. Expanded CogACT and LingBot results remain excluded pending acceptance.


### π0.5 completed held-out stability audit — 2026-11-10

The independently audited held-out replay completed 2,400 request/noise pairs and 187,200 arrangement records. Batch-invariant outputs and all recorded stages were exact across the tested arrangements; native inference had 88,800 nonexact arrangement records. All six audit-bound input artifacts and the auditor source were rehashed when promoting this record. The 2,400 BI-singleton versus native-singleton fidelity comparisons were nonexact; this is a separate comparison from batching stability. These replay measurements add no simulator starts to the accepted 950-start cohort. Full campaign completion remains unproven. Proof and hashes: `results/four_model/execution_state.json`, field `pi05_complete_heldout_stability_latest`.


### π0.5 serving, causal comparison and captured-operator replay — 2026-11-10

The primary endpoint remains singleton-versus-batched stability. The completed, independently audited π0.5 serving queue contains 1,000 requests per configuration across four offered loads and eight replay fixtures. Native dynamic batching differed from its native-singleton reference on 746 requests; invariant dynamic batching differed from its invariant-singleton reference on zero requests. Fixed native padding uses a same-position B8 reference, so its exactness does not imply equality to native B1. The [serving figure](results/four_model/figures/pi05_serving_stability_tradeoff_001/serving_tradeoff.png) and its 16-row table were independently recounted and visually reviewed. [Acceptance record](results/four_model/current_evidence/pi05_serving_stability_acceptance001.json).

The six-mode causal comparison completed 3,600 records over 600 diagnostic input/noise pairs at B2. Native and deterministic-native outputs were nonexact on all 600 pairs; each of the four invariant intervention modes was exact on all 600 pairs. The first recorded divergent native boundary was flow-step-0 velocity. The corrected version compares task and episode together when checking companions, preserving the failed version and numerical sources. Captured linear-operator replay reproduced all four conditions twice: identical target operands gave batch-dependent native outputs and exact invariant outputs. This establishes repair for the captured supported leaf. One earlier unresolved leaf remains; globally first-operator attribution is unproven. The causal table was independently recounted and both figure pages were reviewed. [Causal/local evidence](results/four_model/current_evidence/pi05_causal_and_local_replay_acceptance003.json).

Isolated operator timing completed all 710 planned slots across 58 cases, and independent recounting passed. The audit covers retained timings, layouts, source attention masks and sampled compute/graphics inventory; generated numerical operand bytes were not independently rederived. [Timing evidence](results/four_model/current_evidence/pi05_operator_performance_acceptance002.json). These milestones add no new simulation starts to the accepted 950-start cohort, and do not establish completion of the four-model campaign.

### 2026-11-11: LingBot precision rejection and versioned recovery

The original LingBot full diagnostic completed 600 request/noise pairs but **failed independent acceptance**: actual runtime provenance records `tf32_matmul=true` although the declared execution configuration requires TF32 disabled. The released model constructor calls `torch.set_float32_matmul_precision("high")` after the original adapter disables TF32. Preserve this run as rejected evidence; do not use its measurements as accepted TF32-disabled results. See `results/four_model/current_evidence/lingbot_diagnostic_precision_rejection001.json` for retained artifact hashes.

The new `lingbot_robotwin_exact_v2` adapter enforces highest FP32 precision after construction and checks TF32 before and after each inference. Actual corrected capacity runtime records both TF32 flags false. All 24 B16/B32 capacity arrangements ran, with zero invariant output mismatches; full diagnostic requalification is running in `campaign/lingbot_robotwin_exact/precision_v2_001` and remains pending independent acceptance. The original adapter and pinned drivers are unchanged.

The versioned independent diagnostic audit, B64 capacity/audit, and full 2,500-start paired simulator study/audit are queued behind explicit completion gates. The new simulator driver uses the corrected adapter and its audit checks recorded precision. Operator provenance review, numerical freeze/held-out migration, later measurements, and final reporting remain incomplete. Stability is the primary endpoint; no success-driven tuning or WAM work is authorized. The accepted 950-start three-model simulator stability evidence is unchanged.


### 2026-11-11: corrected LingBot downstream gates

The corrected source-trace chain now has a versioned operator inventory auditor queued after it. Reuse of the 56,100 standalone arithmetic records requires recorded TF32-off measurements, their unchanged independent audit hashes, identical execution configuration, and exact source-pattern coverage of the new B1/B8 actual-model traces. The old actual-model traces fail the new adapter gate; the new inventory audit has not yet run. See `results/four_model/current_evidence/lingbot_precision_operator_inventory_audit_launch002.json`.

A separate versioned numerical freeze and held-out auditor require the corrected adapter and actual TF32-off runtime/trace flags. The freeze rejects the old operator qualification. Freshness validation, freeze creation, held-out replay, and independent held-out audit are queued after the corrected operator audit. Policy/serving/request-queue measurements and their versioned independent audit are queued after the corrected held-out audit. These are prepared dependencies, not accepted results. Proofs: `results/four_model/current_evidence/lingbot_precision_freeze_heldout_launch002.json` and `lingbot_precision_performance_launch002.json` in the same directory.

Corrected LingBot causal/local replay, isolated operator performance plans and measurements, serving figures, paired magnitude summaries, and final independent reporting still need migration and execution against the corrected sources. The four-model campaign remains incomplete. All prior numerical sources and rejected artifacts remain preserved.


### CogACT completed held-out stability audit — 2026-11-11

Independent audit accepted 2,400 held-out request/noise pairs and 187,200 arrangement records. Invariant outputs and all recorded stages were exact across tested arrangements; native inference had 88,800 nonexact arrangement records. All six audit-bound input artifacts and the auditor source were rehashed at promotion. The 2,400 invariant-singleton versus native-singleton fidelity comparisons were nonexact, a separate comparison from batching stability. This replay result adds no simulation starts to the accepted 950-start cohort. Full campaign completion remains unproven. Acceptance: `results/four_model/current_evidence/cogact_complete_heldout_acceptance001.json`.


### CogACT causal stability and local replay — 2026-11-11

The six-mode B2 causal study completed 3,600 records over 600 diagnostic input/noise pairs. Native and deterministic-native outputs differed from their singleton references on all 600 pairs; each invariant intervention mode had zero nonexact outputs or recorded-stage differences. The first recorded divergent native boundary was flow-step-0 sample. The independent contract audit, retained hashes, figure table recount and visual review passed; tensor errors were not independently reconstructed. The study proceeded after accepted held-out stability while three isolated timing attempts were rejected for foreign GPU activity. Numerical sources and the predeclared causal plan were unchanged. Timing/serving/queue and isolated operator performance remain pending stable GPU access.

Independent captured-linear replay reproduced all four recorded conditions twice. Native target operands were equal across singleton/batched captures but outputs differed; invariant target operands and projected outputs were equal. Operands differed across implementations, so same-operand repair and globally first-operator attribution remain unproven. Acceptance records: `results/four_model/current_evidence/cogact_causal_contract_acceptance003.json` and `cogact_local_replay_acceptance003.json` in that directory. Full campaign completion remains unproven.


### CogACT same-native-operand local intervention — 2026-11-11

A separate counterfactual replay restored the exact retained native singleton and batched linear operands, weights, bias, layouts, pointer alignment and runtime, then ran both native and invariant implementations on those bytes. Native projected outputs differed by a maximum 0.03125; invariant projected outputs were bitwise equal. Each condition repeated exactly twice, and native outputs reproduced the original captures. All bound source and input hashes were rechecked. This establishes same-operand local repair for the captured supported linear operator. It does not change the earlier natural-path captures, whose operands differed across implementations, and does not establish globally first-operator attribution or full campaign completion. Proof: `results/four_model/current_evidence/cogact_same_native_operand_intervention_acceptance001.json`.

### Stability analysis scheduling correction (precision003)

Singleton-versus-batch stability remains the primary endpoint. The corrected LingBot CPU episode-byte audit, action/dual-arm magnitudes, task tables, and paired-start arm bootstrap now depend directly on the completed corrected paired-study audit and final exposure check, rather than isolated performance acceptance. Four verified sleeping CPU launchers were replaced with new -003 launchers; original source/state files were preserved. Measurement scripts, source hash guards, full-cohort requirements, independent recounts, and bootstrap seed/resample count are unchanged. New analysis outputs use precision003 and the episode audit uses independent_complete_episode_bound_audit_stability003.json. GPU inference/performance queues are unchanged. This is scheduling preparation, not an accepted numerical result.

Evidence: results/four_model/current_evidence/lingbot_stability_analysis_priority_launch003.json and lingbot_stability_analysis_priority_verification003.json.

### Corrected precision serving report queue (precision002)

The CPU-only LingBot serving figure and independent table recount launcher now waits on the live corrected-precision performance audit (PID 3624985, start ticks 438580367). It reads only `precision_v2_001/isolated_performance_002` and writes `figures/lingbot_serving_stability_tradeoff_precision002`. The existing plotter and independent recount sources are unchanged. This is a queued report, not an accepted measurement or a completed visual inspection. Launch provenance: `results/four_model/current_evidence/lingbot_precision_serving_figure_launch002.json`. The rejected precision artifacts remain preserved.

### Corrected precision diagnostic report queues

The CPU-only LingBot flow-step distributions and singleton-fidelity aggregates now have versioned launchers waiting on the corrected diagnostic independent audit (PID 3216922, start ticks 438501475). They consume only `precision_v2_001/diagnostic`, retain the existing 600-pair full-cohort checks and independent recount sources, and write `lingbot_flow_step_stability_precision003*` and `lingbot_singleton_fidelity_precision002*`. The flow launcher also requires the corrected audit producer terminal status. These are queued reports, with no accepted new measurements or visually inspected figures yet. Provenance: `results/four_model/current_evidence/lingbot_precision_diagnostic_reports_launch003.json`.

### Corrected precision operator timing plan preparation

`build_operator_performance_plans_v2.py` adds a LingBot-only qualified operator directory override and single-model selection. The override requires the corrected precision coverage audit, v2 adapter identity, and actual TF32-off B1/B8 traces. Existing case/mask generation and timing protocol remain unchanged; plans for π0.5, RDT and CogACT compare exactly with the original builder. The old LingBot qualification is rejected by the new override. No corrected LingBot timing plan or timing measurements have been created: actual corrected traces remain pending. Preparation and validation evidence are in `results/four_model/current_evidence/lingbot_precision_operator_plan_builder_{preparation,validation}002.json`.

The corrected LingBot operator timing plan builder is queued as a CPU-only launcher behind the live corrected operator qualification auditor (PID 3550876, start ticks 438564511). It requires the accepted terminal status and corrected precision coverage flags before creating `results/four_model/operator_performance_plans_precision002/lingbot.json`. This queues plan generation only; operator timing execution remains pending. Provenance: `results/four_model/current_evidence/lingbot_precision_operator_timing_plan_launch002.json`.


### RDT heldout and diagnostic attribution acceptance — 2026-11-12

Stability remains the primary endpoint. The completed fresh heldout audit covers 2,400 request/noise pairs and 187,200 arrangements: native has 88,800 nonexact arrangements; invariant output and all recorded stages are exact. Invariant versus native singleton fidelity is a separate comparison: all 2,400 fidelity records are nonexact. Raw-file and auditor hashes were checked in `results/four_model/current_evidence/rdt_complete_heldout_acceptance001.json`.

The unchanged six-mode diagnostic plan completed after heldout acceptance, independently of timing availability. Native and native deterministic each have 600/600 nonexact outputs, maximum absolute output error 0.048828125, first recorded divergence at `flow.step_00.prediction`. Four invariant variants each have zero nonexact outputs or recorded-boundary divergence. `rdt_causal_contract_acceptance003.json` verifies raw, provenance, plan, diagnostic acceptance, and auditor bindings. The contract auditor does not independently reconstruct tensor errors. Both prediction and sample plots were visually inspected; the independent table recount covers 300 groups. See `rdt_causal_figure_visual_acceptance003.json`.

The predeclared four-condition local linear replay reproduced every captured output twice with pointer alignment checked. Native target inputs match while outputs differ; invariant target inputs and outputs match. Same operands across implementations and local repair are demonstrated. One earlier unresolved leaf remains, so globally first operator attribution is not established. Captured artifact, trace, manifest, and auditor hashes were verified in `rdt_local_replay_acceptance003.json`.

The first policy timing launch was rejected before measurement because GPU 2 had a graphics process using 617 MiB. No timing samples were created or accepted. `rdt_performance_isolation_rejection001.json` preserves the rejection. RDT policy/serving/operator timing still requires isolated execution and independent acceptance. LingBot precision requalification and its dependent campaign stages remain unfinished; the full four-model campaign is incomplete.

### Remaining timing queues — 2026-11-12

RDT's second policy/serving timing launcher waits for 300 consecutive idle seconds before running the unchanged measurement protocol in `full/isolated_performance_002`. Its independent audit waits for successful measurement completion; operator timing waits for that audit and the accepted local replay. These three launcher processes were confirmed live on November 12. GPU 2 still has a foreign graphics allocation, so no new timing result is accepted. Launch evidence: `rdt_idle_gated_performance_launch002.json`, `rdt_idle_gated_performance_audit_launch002.json`, and `rdt_operator_timing_launch004.json` in `results/four_model/current_evidence`.

Corrected LingBot operator timing is queued behind corrected local replay and the corrected operator plan, using the versioned v3 timing worker/wrapper/auditor and v2 plan builder. This preserves the corrected precision qualification rather than consuming the rejected original traces. Launch evidence: `results/four_model/current_evidence/lingbot_precision_operator_timing_launch003.json`. These timing queues do not replace or gate the primary paired-trajectory stability analysis. Queued execution is not evidence of completed measurements.

### RDT serving stability figure dependency repair — 2026-11-12

The new `rdt-idle-gated-serving-stability-figures-002.py` waits for the live independently audited timing attempt 002 (PID 1791898, start ticks 450444333). It reads only attempt 002 queue measurements and their accepted audit, then exports the singleton-versus-dynamic stability/latency table and figures and independently recounts the table. The launcher source guards passed; its live process is PID 2782251, start ticks 452729791. Launch evidence: `results/four_model/current_evidence/rdt_serving_stability_figure_launch002.json`. This is pending work, not accepted measurement evidence; visual inspection remains required after export. The failed attempt 001 and its downstream artifacts are preserved.

### User priority: singleton versus batched stability — 2026-11-13

The user explicitly reaffirmed that the objective is stability, not maximizing success. Primary reporting compares native singleton with native batched inference, and invariant singleton with invariant batched inference, under matched initial simulator state, prepared input, indexed noise, and reset policy state. Report exact agreement, RMS/MSE and maximum action deviation, first divergence, and closed-loop trajectory drift with exact same-condition repeat controls. Cover varied task/start pairs while keeping each comparison matched. Success is a secondary recorded outcome; do not tune checkpoints, tasks, starts, or inference settings to improve success. Timing and throughput are ancillary and must not gate paired stability measurements or their reporting. The accepted summary already labels its primary endpoint singleton_vs_batched_stability; this clarification changes no measurements or numerical configuration.

### LingBot corrected diagnostic accepted; paired storage stop — 2026-11-13

Verified all five diagnostic input hashes, auditor source, runtime numerical source bindings, and disabled TF32. The independent audit covers 600 request/noise pairs and 46,800 arrangements: 22,200 native arrangements are nonexact; invariant output and every recorded denoising stage are exact. Native and invariant share deterministic token-choice MoE; the released atomic baseline is separate. Flow descriptive recount covers 10,000 stage groups; singleton fidelity recount covers 600 comparisons across 50 tasks and is separate from batching stability. Evidence: `results/four_model/current_evidence/lingbot_corrected_precision_diagnostic_acceptance001.json`. Tensor errors have not been independently rederived; figure visual acceptance remains pending.

B64 capacity completed its audit. The paired launcher 002 exited before any simulator execution because available retained-data space was below its 1.5 TB prelaunch reserve. Its missing process and logged AssertionError establish a terminal stop despite its stale waiting status file. Preserve the storage guard and failed attempt; downstream waiting states do not establish running measurements. Resolve retained-data storage before a versioned relaunch and downstream dependency repair. No LingBot paired trajectory result is accepted.

### LingBot descriptive flow figure visual review — 2026-11-13

Both corrected B8 figures (prediction and state) were viewed: 50 tasks, ten denoising steps, native nonzero deviations, invariant zero median/p95, with a descriptive-distribution footer. Verified report, audit, plotter and all PNG/PDF hashes. Evidence: `results/four_model/current_evidence/lingbot_corrected_precision_flow_visual_acceptance003.json`. This does not establish paired closed-loop stability or causal attribution. Storage inventory found only about 26 GB of our files on vastdisk versus approximately 177 GB needed to restore the paired prelaunch reserve; moving those files alone is insufficient. No retained data or other users' files were removed.

### Corrected LingBot operator trace scheduling — 2026-11-13

Versioned trace launcher 003 uses the unchanged B1/B8 commands, corrected adapter, source hashes and output paths from launcher 002, with completed corrected capacity as its prerequisite. The paired simulator failed before launch for storage; source tracing uses already-exposed diagnostic inputs and does not depend on simulated trajectories. Trace PID 1722887/start ticks 455494270 and dependent independent inventory audit PID 1723397/start ticks 455494294 were confirmed live. Old failed launchers are preserved. Evidence: `results/four_model/current_evidence/lingbot_precision_operator_trace_independent_launch003.json`. This is execution in progress, not accepted operator coverage. Held-out freeze still requires final paired exposure qualification; no held-out gate was bypassed.

### LingBot corrected traced operator qualification accepted — 2026-11-13

B1 and B8 traces each completed 6,399 calls and 33 source patterns. The independent inventory auditor reconciled every pattern, including shape/stride/mask metadata, against 56,100 completed independently audited component measurements (40,800 linear, 10,200 reduction, 5,100 remaining). Verified trace hashes and TF32-off fields, all component configurations, raw measurement hashes, component audit hashes, and inventory auditor source. No patterns are missing. Evidence: `results/four_model/current_evidence/lingbot_corrected_precision_operator_acceptance003.json`. This qualifies the observed high-level source inventory; numerical held-out qualification, simulator trajectories, and timing remain separate and incomplete.

### Corrected LingBot operator timing plan built — 2026-11-13

Built 66 timing cases with the pinned v2 builder against the accepted corrected B1/B8 inventory, into `results/four_model/operator_performance_plans_precision003/lingbot.json`. Evidence: `results/four_model/current_evidence/lingbot_corrected_operator_timing_plan003.json`. This is a plan, not timed measurements. The old precision002 plan queue failed with the old dependency and is preserved. Future operator timing must explicitly bind to this generated plan. Asked the user asynchronously for a writable location with at least 1.5 TB free or added vastdisk capacity; existing largest-volume availability is about 1.32 TB and retained data remains intact.

### CogACT timing attempt 004 scheduled — 2026-11-13

GPU4 was observed without the foreign compute allocation that rejected prior attempts. The versioned fourth single-attempt launcher (PID 1830913/start ticks 455519838) retains the 300-consecutive-idle-second gate, the unchanged measurement workload, pinned sources, and strict whole-run GPU-isolation observer. It writes only `full/isolated_performance_004`. Independent audit is queued behind its successful completion; no causal or local diagnostics are rerun. Evidence: `results/four_model/current_evidence/cogact_idle_gated_performance_launch004.json` and `cogact_performance_audit_launch004.json`. No timing acceptance is claimed, and no automatic retry is configured. Primary paired stability results remain unchanged.

### CogACT timing attempt 004 downstream repair — 2026-11-13

Queued isolated operator timing and serving-stability figure export behind independently accepted timing attempt 004 (audit PID 1840208/start ticks 455522270). The operator launcher additionally binds the accepted local replay proof and existing causal audit; it does not rerun those diagnostics. Figure paths consume only attempt004 and write a new tradeoff004 directory. Both launchers were confirmed live. Evidence: `results/four_model/current_evidence/cogact_timing_downstream_launch004.json`. No measurements, performance acceptance, or figure visual review is claimed from queue creation.

### LingBot fixed-input batch error magnitudes — 2026-11-13

Extracted `results/four_model/lingbot_singleton_batch_magnitudes_precision001.json` and separately recounted the aggregate arithmetic with streaming Decimal sums in `lingbot_singleton_batch_magnitudes_precision001_arithmetic_recount.json`. The 1,020 rows cover each task and an ALL_TASKS descriptive aggregate, two implementations, five batch sizes, and normalized/physical domains. Selection is diverse companions, target first, batch-position transformation; each model mode is compared against its own singleton with equal input and indexed noise. Each aggregate has 600 comparisons. Native normalized RMSE at B2/4/8/16/32 is 0.0002252291793 / 0.00006708833117 / 0.00006713652867 / 0.00006646108331 / 0.00006710771992; all 600 native pairs at each size differ. Invariant RMSE, MSE and max absolute error are zero. Arithmetic recount is not independent tensor-error rederivation. These fixed-input diagnostics are separate from pending paired trajectory drift.

### CogACT timing attempt 004 rejected — 2026-11-13

After clearing the 300-second idle gate and completing native B1/B2 sessions, the strict isolation observer detected foreign GPU activity at 01:47:52 UTC. It stopped only the campaign child (exit -15) and marked policy isolation `foreign_gpu_activity`, accepted false. Preserved 2,394 partial records; none are accepted timing evidence. Verified all six attempt004 measurement/audit/downstream processes terminal and no independent performance acceptance exists. Evidence: `results/four_model/current_evidence/cogact_performance_isolation_rejection004.json`. No fifth automatic attempt is configured; stable GPU access is needed before another timing run. Accepted held-out, causal/local, and paired singleton-versus-batched stability results are unchanged.

### LingBot recovery on new host — 2026-11-26

The focused LingBot goal is active. Current scratch has about 12 TB available; the historical storage blocker no longer applies there. Hardware is the same GPU model but different UUIDs. Recovered clean pinned LingBot, RoboTwin and LeRobot source checkouts in `/scratch/hadjiyiannis_3_gmail_com/lingbot-recovery-20261126`. Twenty old diagnostic-bound source files match; four LeRobot dataset/utility files differ from the old modified checkout and require fresh numerical equivalence checks.

Recovered 31 public asset files (about 40.5 GB): all checkpoint and RoboTwin ZIP files match archived sizes/hashes; Qwen metadata is downloaded at its pinned revision and hashed. ZIP members were path-checked, extracted with CRC validation, and linked under the simulator source. Archived Python dependency versions were restored in separate Python 3.12 inference and Python 3.10 simulator environments. The inference archive requires no-dependency installation, consistent with the official separate LeRobot install, because its historical DeepDiff pin conflicts with LeRobot package metadata. Runtime inventories are retained.

The released server imports successfully with PyTorch 2.8.0+cu128, FlashAttention 2.8.3 and Transformers 4.57.3. Official constructor/load-sequence verification passed for all 1,708 tensors / 6,375,907,511 elements. Thirty-one CPU protocol/analysis tests passed. These do not qualify GPU numerical inference or simulator reset/replay on the new host. No new paired trajectory or success result is accepted. Evidence uses the `results/four_model/current_evidence/lingbot_recovery_20261126_*` prefix; runtime inventories are in `results/four_model/environments/*-recovery-20261126.txt`.

### New-host RoboTwin reset qualification — 2026-11-26

Verified SAPIEN and CUDA select GPU3 at PCI 0000:0b:00.0. Built official Curobo v0.7.8 (`d64c4b005459db10c5dd867d8b30a87d5bda9bdb`) for SM120 using hash-verified NVIDIA CUDA 12.8.1 compiler/development components in user scratch. Restored the pinned RoboTwin install script's URDF UTF-8 and MPLIB setup changes and generated six embodiment configurations using the official asset-root placeholder substitution. Recorded before/after hashes; no planner numeric parameters were tuned. Earlier probes that stopped before measurements are retained.

Two seed0 reconstructions and 20-step fixed-action trajectories matched exactly; the initial full observation and physics hashes also equal the archived LingBot pilot's hashes. SAPIEN logs unsupported OIDN CUDA denoiser errors; these are retained and must not be interpreted as denoiser availability. Captured camera pixels are nonconstant and initial observations match the historical baseline.

The independent complete replay audit then verified adjust_bottle at the first expert-eligible seed100001 (seed100000 rejected as physically unstable): three complete 400-step trajectories, 1,203 raw frames, exact observations/physics/controller/source-defined task state, and CPU/CUDA RNG coverage. Controls are the 112 retained policy actions followed by the prescribed final-command hold. This is model-free reset/replay qualification, not policy outcome evidence. Acceptance: `results/four_model/current_evidence/lingbot_recovery_20261126_adjust_bottle_acceptance001.json`.

Launched the complete 50-task model-free reset roster with frozen helper sources, three full replays and an independent audit per task. Driver PID2460614/start ticks50431580; launch evidence `lingbot_recovery_20261126_task_roster_launch001.json`. All task failures remain explicit. The one-task success does not establish the remaining roster or paired campaign.

### Corrected source-corpus precision and incremental collection — 2026-11-26

The live reset roster independently qualified lift_pot and hanging_mug and advanced to stack_bowls_three. This is partial reset qualification, not completed model trials. A repository snapshot is `lingbot_recovery_20261126_reset_roster_progress001.json`; the live scratch union remains authoritative.

The legacy native source collector set TF32 flags before a released constructor that can re-enable TF32. Its existing files and historical evidence remain unchanged. New collector v3 sets highest precision after construction and checks matmul TF32, cuDNN TF32 and cuDNN benchmarking before and after every inference. It records flags in the candidate lock and each model call; corpus auditor v2 rejects missing or incorrect flags. Fourteen relevant tests passed, including mocked constructor re-enable and mid-call drift. All official LingBot and LeRobot Python sources are now bound by the corpus source lock. This is an explicit corrected source-generation path, without changing checkpoints, sampling, or the primary native/invariant comparison.

Started incremental source collection with currently qualified tasks, fixed diagnostic/heldout seed phases and the unchanged 200-episode target. PID2620456/start ticks50476233, GPU3. Subsequent collection must explicitly resume only after this handle is terminal and retained state is checked. Collector execution/partial episodes are not complete-corpus or numerical-invariance acceptance. Launch and guard-test evidence use `lingbot_recovery_20261126_native_collection_launch001.json` and `lingbot_recovery_20261126_collector_precision_validation001.json`.

### First corrected source episodes independently rebuilt; collection resumed — 2026-11-26

The first incremental collector became terminal after sealing four lift_pot episodes and 20 selected observations. Corpus auditor v2 independently rebuilt all four episodes: 418 frames, 11 recorded model calls, official input/physical-action transforms and per-call corrected precision metadata. This is partial source reconstruction, not batching-invariance, held-out freshness or full-corpus acceptance. Proof: `lingbot_recovery_20261126_native_reconstruction_acceptance001.json`; both immutable progress-audit snapshots are retained.

Verified the old collector handle is terminal, all source pins still match, every episode directory is sealed, and the current manifest equals the independent audit snapshot. Explicitly resumed the unchanged corpus with an immutable qualification snapshot admitting lift_pot, hanging_mug, stack_bowls_three and scan_object. New producer PID2882319/start ticks50546176. No failing/unfinished episode was silently retried; the 200-episode target and source lock are unchanged. Evidence: `lingbot_recovery_20261126_native_collection_resume002.json`.

Historical exposure recovery remains incomplete. The archived full native-corpus audit preserves 1,000 selected-request content addresses (noise-free) and 1,077 noisy-request addresses. A direct screen of the first 20 newly selected inputs found two historical selected-request matches, and current noisy requests include four historical exact matches. Noise cannot be stripped from a hash to infer a prepared-input identity. Full original request bytes/prior exposure index remain absent; final all-history held-out freshness must not be certified from this metadata-only screen. Asked the user asynchronously about the previous host or a backup while current collection and qualification continue. Evidence: `lingbot_recovery_20261126_historical_exposure_inventory001.json`.

### Task callback coverage and cabinet cold initialization restored — 2026-11-26

The main reset roster retained click_bell's failure: its check_arm_function is a Python task-bound method, which the baseline recorder correctly rejected as uncovered. Added a separate v2 recorder that captures owner/selected function identity, source hashes and defaults; foreign-owner or closure-backed methods remain rejected. Auditor v2 checks the selected function against the task's referenced attributes and source AST, archives callback source, and verifies the raw repeated trajectories. Neither live baseline helpers nor model/task behavior were changed.

The isolated click_bell retry passed three 400-step complete replays, 1,203 raw frames and 1,203 independently checked callback records at seed100000. Thirteen callback/state tests passed; with the cabinet initialization tests the combined count is 15. The verified merged qualification snapshot contains seven tasks (five baseline, prior adjust_bottle, and repaired click_bell), retaining the original failed-attempt record. Evidence: `lingbot_recovery_20261126_qualified_union002.json` and `qualification/recovery-20261126-click_bell002/`. The full 50-task roster remains incomplete.

put_object_cabinet subsequently reached its first expert-eligible seed100026, then failed because a cold task instance lacked origin_z. Restored arm_tag and origin_z exactly as the pinned upstream play_once initializes them before motion, using a separate worker version and frozen source directory. Started three-repeat requalification in a new output directory, preserving the failed attempt. No candidate range, outcome filter, dynamics or planner parameters changed. Cabinet PID3278581/start ticks50653022 remains unaccepted until independent replay audit. Launch evidence: `lingbot_recovery_20261126_cabinet_qualification_launch003.json`. Source collection continued to 11/200 episodes at the latest observation; only the first four have completed independent corpus reconstruction.
