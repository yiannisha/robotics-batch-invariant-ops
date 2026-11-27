# Goal: finish the LingBot-VLA2 experiment campaign

Requested November 26, 2026. The active research priority is LingBot-VLA2;
existing pi05/RDT/CogACT evidence is preserved. No WAM experiments are in scope.
The focused LingBot goal is now active in the application. The previous broader
four-model campaign remains incomplete; creating this goal does not mark it complete.

## Objective and fixed comparison

Finish the released fine-tuned LingBot-VLA2 RoboTwin campaign across all 50 tasks
and 50 qualified starts per task (2,500 paired starts, eight episodes per start).
Primary results compare native singleton with native dynamic batching and
invariant singleton with invariant dynamic batching. Report bitwise exactness,
action RMS/MSE/max deviation, first divergence, and physical trajectory drift.
Record paired success outcomes as secondary metrics; do not maximize success,
change the prescribed sample after observing errors, or filter on policy success.

Within each comparison, restore the same canonical physics/controller/task/RNG
state and initial observations, reset all policy state, and use explicit indexed
per-request noise. Retain the prescribed dynamic batch cycle, companions and
target positions. Require identical fixed-action replay and complete
same-condition repeats before accepting simulator evidence.

## Accepted starting evidence

- Released checkpoint revision `0451855729ec904f970600e0aec8b84661423afe`;
  corrected adapter `scripts.model_invariance.adapters.lingbot_robotwin_exact_v2`.
- Corrected TF32-off diagnostics: 600 request/noise pairs, 46,800 arrangements;
  22,200 native arrangements nonexact; invariant outputs and recorded flow stages exact.
- Corrected B1/B8 source inventory: 33 patterns each, reconciled to 56,100 audited
  component records. This does not establish held-out or performance completion.
- Native and invariant primary paths share deterministic token-choice MoE.
  The released atomic-MoE baseline is a separate nonrepeatable supplement.
- Reset qualification and prior accepted numerical evidence have recorded source
  hashes. Their raw files must be recovered or regenerated before renewed acceptance.

## Required completion evidence

- [ ] Recover pinned checkpoint, source, simulator/runtime and raw evidence. Inventory
  current mounts and GPUs; the November 13 storage and GPU observations are historical.
  The original scratch campaign directory is absent on the November 26 host.
- [x] Resolve retained-data storage without discarding provenance. Reassess and
  document capacity against the full prescribed study; preserve the earlier 1.5 TB
  prelaunch reserve until an evidence-backed replacement is qualified.
- [ ] Restore or requalify exact reset, fixed-action and repeat controls on the pinned
  runtime. Requalify any changed checkpoint, inference, preprocessing or simulator path.
- [ ] Run all 2,500 paired starts, retaining eight conditions/repeats per start,
  inputs, outputs, snapshots, trajectories, source hashes and runtime provenance.
- [ ] Independently audit full episode coverage and bindings; compute per-task
  action errors, dual-arm trajectory drift, first divergence, paired bootstrap
  intervals and secondary success differences. Review exported figures.
- [ ] Perform the final prepared-neural-input exposure audit, excluding noise and
  request IDs from observation identity; seal a valid numerical freeze.
- [ ] Run the prescribed fresh held-out replay and singleton-fidelity measurements;
  independently verify exactness, coverage and all freeze/input/source bindings.
- [ ] Complete six-mode causal comparisons and captured local operator replay;
  distinguish local repair evidence from any unproven globally earliest operator.
- [ ] Complete isolated policy, serving/queue and operator measurements using the
  corrected operator plan and independently accepted isolation evidence. Reject
  interference; preserve interrupted attempts without accepting their timings.
- [ ] Publish accepted results, reviewed figures, research notes and provenance;
  perform requirement-by-requirement completion audit. No completion claim from
  scripts, queued jobs, partial records or unavailable raw evidence.

The paired stability study is the priority. Ancillary timing must not gate its
execution or reporting once simulator and retention requirements are satisfied.

## Recovery progress: November 26, 2026

The new host has roughly 12 TB available on scratch; the historical 1.5 TB
capacity blocker no longer applies there. GPU UUIDs differ, so old numerical and
simulator runtime acceptance cannot simply be transferred. Recovery root:
`/scratch/hadjiyiannis_3_gmail_com/lingbot-recovery-20261126`.

Recovered clean source checkouts: LingBot `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`,
RoboTwin `13c3c47ff4312dd62484bcd51be034af55c062d1`, and LeRobot v0.4.2
`58f70b6bd370864139a3795ac3497a9eae8c42d5`. Twenty source files bound by the old
diagnostic acceptance match; four LeRobot dataset/utility files differ from the
old modified checkout. Those differences are recorded, not silently accepted.

The released checkpoint and RoboTwin asset archives have been recovered against
their archived SHA-256/size records, together with metadata-only Qwen assets at
the pinned revision. Separate Python 3.12 inference and Python 3.10 simulator
environments are restored. Acquisition and checkpoint loading do not establish
numerical equivalence on this host; renewed numerical checks remain required.

New-host simulator progress: Curobo v0.7.8 is built for SM120, official
post-install package changes and embodiment path expansion are recorded, and
`adjust_bottle` passed independent three-repeat complete replay qualification
(400 controls/repeat, 1,203 frames). Initial RGB and physics hashes match the
archived pilot. The complete 50-task reset roster is running with frozen helper
sources; full-roster coverage and paired inference are not yet accepted.

The versioned merged record
`results/four_model/current_evidence/lingbot_recovery_20261126_qualified_union003.json`
binds eight independently audited tasks: `adjust_bottle`, `lift_pot`,
`hanging_mug`, `stack_bowls_three`, `scan_object`, `handover_block`, `click_bell`,
and `open_microwave`. The microwave qualification includes three complete
1,500-control replays and 4,503 exact raw frames. The click-bell recorder covers
its task-owned callback explicitly. Failed attempts remain retained, and the
cabinet initialization repair is still awaiting replay acceptance.

The subsequent `qualified_union007` record binds 12 accepted tasks, adding
`stack_blocks_three`, `put_object_cabinet`, `place_shoe`, and `beat_block_hammer`.
The cabinet initialization repair passed three complete 700-control replays
and independent verification of 2,103 raw frames; the earlier failed attempt
remains retained.

Corrected, per-call TF32-off native source collection has recorded 16 of the
prescribed 200 episodes at this checkpoint. All 16 have independently verified
reconstruction evidence: 6,010 raw frames, 127 policy calls and 80 selected
inputs, with corrected precision records verified (`progress_audit004`).
Collection resumed with the sealed 11-task qualification snapshot after source
and raw-file hashes were rechecked; the later hammer task awaits the next resume.
These
source episodes supply future numerical inputs and are not paired stability
trials. The 2,500-start paired campaign has not begun. Missing historical raw
exposure identities remain unresolved; prospective collection alone does not
prove held-out freshness against the previous campaign.

### Rigid-body coverage correction

Inspection of the dump-bin task exposed direct SAPIEN Entity references. It also
showed that the legacy physics helper's optional `get_velocity` calls do not
capture rigid-body velocities on Entity objects in the installed SAPIEN version.
The retained legacy replay reports establish exactness for their recorded fields;
they do **not** qualify the stronger rigid-body state contract required before
primary paired acceptance. Prior task counts must be read with this limitation.

`probe_robotwin_exact_initialization_v3.py` explicitly captures dynamic-component
linear/angular velocities, sleeping state, mass/inertia, damping, gravity,
kinematic state/target, locked axes and solver settings. Task recorder v4 binds
Entity references to scene actor indices and poses; auditor v4 requires the new
physics schema and verifies these bindings. Twenty-seven focused/regression tests
and an actual CPU SAPIEN API smoke passed. Full replay requalification remains
pending; no stronger task acceptance is claimed from tests or the smoke alone.
The first v4 helper variant used the wrong sleeping-property API and its launched
attempt was stopped and retained. The corrected immutable variant is
`rigid_source_v4_002`, currently running `dump_bin_bigbin_rigid004`.

The full 50-task stronger requalification is running separately in
`task_roster_rigid_v4_001`; legacy results are not reused. The paired runner and
bound episode auditor now require the stronger qualification report and check
the rigid-body schema and Entity bindings on every retained frame. Forty-seven
relevant tests pass, including rejection of legacy and partial attestations.
At this checkpoint no task has completed the stronger independent replay audit.
The superseded legacy roster job was stopped with all outputs retained; its
terminal receipt and 15 completed legacy reports are archived under
`qualification/recovery-20261126-legacy_roster_stopped001`. The source collector
and both stronger replay jobs continue. No old physics acceptance is promoted
to the stronger contract.

The first stronger merged union (`rigid_qualified_union001`) now accepts two
tasks: `lift_pot` and `dump_bin_bigbin`. The dump-bin report verifies 1,803 full
rigid-state frames across three 600-control replays, 3,606 Action records and
9,015 Entity-to-physics bindings. The stronger all-task roster continues; these
two task reports do not establish full-roster or paired model completion.

`rigid_qualified_union003` adds `turn_switch` and `hanging_mug`, for four accepted
tasks. The stamp-seal attempt exposed a render-only Entity target outside the
rigid-actor list. Recorder/auditor v5 captures its scene index, name, pose and
explicit render-only component type, rejecting unknown components and conflicting
aliases. Rigid-body capture is unchanged; earlier nonvisual task paths remain
frozen. Fifty-one relevant tests and a SAPIEN check with a render system passed.
The CPU-only visual smoke lacked a render system and was rejected. The failed
stamp attempt remains retained; `stamp_seal_visual002` is running and is not yet
accepted. The paired runner/auditor also validates this visual state extension.

`rigid_qualified_union004` accepts six tasks after adding `stack_bowls_three`
and the repaired `stamp_seal`. The stamp report independently verifies all
1,203 rigid-state frames and 1,203 render-only Entity records across three
400-control replays. This is reset/replay evidence, not a paired model result.

`rigid_qualified_union006` adds `scan_object` and `stack_bowls_two`, bringing the
stronger accepted count to eight tasks. The corrected source-corpus snapshot
audit 005 verifies 29 completed episodes, 10,001 raw frames, 213 policy calls and
142 selected inputs. Collection continues toward 200 episodes; paired stability
has not been measured on this recovered runtime.

Model-free reconstruction recovered six distinct historical pilot requests,
independently matching every tensor's dtype, shape and byte SHA-256 to the
archived evidence. These cover all 24 calls in `lingbot-closed-loop-pilot-001`;
raw request files and independent audits are retained in
`historical_recovery/20261126-pilot001`. Other historical categories, including
the 1,172 source-corpus request records, remain unresolved. This partial history
recovery is not a final freshness seal and does not gate the primary paired
launch. Numerical qualification, full reset coverage and the full input corpus
still precede that launch; ancillary timing does not.

`rigid_qualified_union009` accepts 12 tasks after adding `handover_block`,
`click_bell`, `stack_blocks_two`, and the cabinet rigid-state replay. After that
cabinet acceptance, the v4 roster was stopped to upgrade capture; its partial
next-task attempt remains retained. The visual-aware v2 roster driver is now
running in `task_roster_visual_v5_002`, reusing those 12 byte-bound reports without
rewriting their audits. Fifty-nine relevant tests pass. Source collection's
current segment finished normally at 44/200 episodes, and independent snapshot
audit 006 is running. Those 44 episodes are not all accepted merely by collection.

Append-only source collection resumed as `native_source_v3_resume004` while the
independent auditor reads its immutable 44-episode snapshot. The old collector
was terminal; sealed episode hashes, producer/source pins and the 12-task proof
snapshot were rechecked before launch. All 44 sealed episode payloads remained
unchanged after launch. This removes an intermediate scheduling wait, not an
acceptance check: the complete 200-episode corpus must still pass independent
audit before numerical qualification/primary paired acceptance.

## Recovery checkpoint: November 27, 2026

`lingbot_recovery_20261126_rigid_qualified_union014.json` binds 24 independently
audited tasks under the corrected rigid-component capture contract. These are
model-free fixed-action replay qualifications, not paired inference results.
The append-only native source collector completed 64 of 200 source episodes.
Independent corpus audit 007 is rebuilding that sealed 64-episode snapshot;
44 episodes remain the latest independently accepted corpus scope until that
audit finishes. Resume 005 runs concurrently with this intermediate audit using
an immutable 23-task qualification snapshot. The complete 200-episode corpus
must pass independent reconstruction before numerical input sealing.

The paired runner and independent episode auditor now explicitly reject native
simulator drift, horizon differences or outcome differences when singleton and
dynamic inference issue identical commands. This closes an acceptance gap; it
is not a measured batching result. The relevant 72-test suite passed. A separate
source and recorded-frame check confirms dual-arm drift uses actual simulated
end-effector link poses. Benchmark `joint_action.vector` contains drive targets;
actual articulation qpos is retained separately in the physics snapshots.

The first supplemental `put_bottles_dustbin` reset attempt failed at seed 100000
inside the released expert's `play_once`, before policy evaluation. Its raw
attempt is retained. The released evaluator rejects such expert candidates,
whereas the earlier reset qualifier stopped on non-`UnStableError` exceptions.
Versioned qualifier v2 now records the exact stderr byte range and hash for
released expert failures; independent replay auditor v6 verifies those bytes
and distinguishes expert exceptions from instrumentation/reset errors. Four
new rejection tests and 31 existing simulator tests passed. The retry starts
again at seed 100000 and retains every candidate; no policy-outcome filtering
is introduced. This retry is not accepted until three complete exact replays
and independent audit finish. Existing live frozen helpers are unchanged.

The primary 2,500-start paired stability study has not started. No numerical
variance or paired success result is inferred from these source episodes or
reset qualifications. Historical evidence recovery and ancillary timing do not
gate its launch once its simulator, numerical and retention requirements pass.

The subsequent `rigid_qualified_union021` checkpoint binds 32 independently
verified reset tasks. Accepted supplemental proofs include `press_stapler`,
`put_bottles_dustbin`, `place_can_basket` and `place_burger_fries`; the failed
first dustbin attempt remains retained. The accepted dustbin retry covers
three 1,700-control replays and 5,103 exact raw frames.

Corpus audit 007 has independently rebuilt all 64 episodes in its immutable
snapshot: 20,371 raw frames, 436 model-call input/action reconstructions, and
320 selected neural inputs. All 21,563 retained file hashes were subsequently
rechecked, as were the producer source pins and the preserved 44-episode
prefix. The accepted scope is 64 of 200 source episodes; neither full numerical
qualification nor paired stability is established by this audit. Append-only
source collection continues beyond that snapshot.

## Native collection exposure correction

A comparison of independently rebuilt signatures from corpus audit 007 found
that 7 of its 256 selected held-out observations had already been consumed by
native source-policy calls. The earlier final freshness checker included old
native request banks, diagnostics and paired requests, but omitted the new
source collection's own model-call request bank. These selections therefore
cannot establish fresh held-out coverage. The accepted 64-episode input/action
reconstruction remains valid; its preliminary held-out selection is not a
freshness seal.

Versioned freshness auditor
`scripts/audit_lingbot_heldout_observation_disjointness_v2.py` now includes the
complete independently audited native collection request bank, verifies every
container SHA and the exact bank roster, and excludes noise and request IDs
from input identity. Four new tests cover noise-independent identity, changed
and unrecorded files, and rejection of partial source audits; eight relevant
tests passed. The full historical and 2,500-start paired audit requirements are
unchanged.

Before numerical qualification, derive a separate versioned prospective
held-out selection from retained source frames that excludes source-inference
inputs, independently reconstruct that selection, and preserve the existing
producer, source episodes and preliminary selections. Keep the prescribed
per-episode input counts and diagnostic inputs. This correction requires no
additional inference or policy-outcome filtering. Final freshness against
history, diagnostics and the completed paired study remains required afterward;
missing historical evidence still does not gate the primary paired launch.

The retained-frame selector `scripts/reselect_lingbot_source_holdout.py` now
produces a separate derived corpus with the original episodes and diagnostics
unchanged. It keeps eight held-out anchors per episode and chooses the nearest
unexposed prepared input, with lower-index ties, recording every rejected
candidate. Source native calls, all diagnostic inputs and earlier held-out
selections are excluded by the complete six-key neural-input identity. No
neural inference or policy-outcome selection is used. The original producer and
its preliminary selections remain retained.

A 64-episode producer smoke retained all 320 selected inputs and changed exactly
seven held-out rows; metadata screening found zero source/diagnostic collisions
and 256 distinct held-out identities. This is producer evidence, not accepted
freshness. Independent auditor v3 is reconstructing the source and selections,
including each rejected nearest candidate, from actual retained tensors and
official CPU preprocessing. Eleven relevant tests passed. The full 200-episode
source corpus must still be collected, reconstructed, reselected and audited
before numerical sealing; final historical/paired freshness remains separate.

`rigid_qualified_union030` binds 44 independently audited reset tasks, including
the accepted cold laptop initialization repair, stapler-pad and empty-cup tasks.

## Full reset qualification recovered

`rigid_qualified_union033` binds all 50 prescribed tasks to independently audited
three-repeat cold-reset fixed-action replays: 150 complete repetitions and
93,150 checked frames. Initial observations, controller/task/RNG state, complete
rigid-body state and replay trajectories match exactly within each task's
repeats. This is model-free simulator qualification, not paired policy results.
The scale task required a versioned cold pre-motion arm selection restoration;
its failed first attempt remains retained. The final scale and basket replay
audits passed. The superseded roster driver was stopped after its unique work
was accepted, with its partial duplicate basket attempt retained.

The corrected native collector completed its 92-episode segment and resumed
with the immutable 48-task snapshot that existed when its predecessor gate
opened. Its separate 92-episode reconstruction audit is running. The full
two-task reset proofs are now available for the subsequent explicit resume;
all 200 source episodes remain required. Latest independently accepted source
coverage is still 64 episodes, and the separate 64-episode held-out reselection
audit remains pending. No new numerical or paired stability completion is
inferred from full reset qualification.
