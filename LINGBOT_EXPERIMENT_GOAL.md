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
- [ ] Resolve retained-data storage without discarding provenance. Reassess and
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

`scripts/recover_lingbot_pinned_assets.py` is recovering the released checkpoint
and RoboTwin asset archives against their archived SHA-256/size records, plus
metadata-only Qwen assets at the pinned revision. Acquisition does not qualify
inference or simulation. Separate Python 3.12 inference and Python 3.10 simulator
environments are being rebuilt. Reset, runtime and numerical checks must precede
new paired result acceptance. No new trajectory or success result is claimed.
