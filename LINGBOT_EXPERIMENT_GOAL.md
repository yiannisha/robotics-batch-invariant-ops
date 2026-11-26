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
