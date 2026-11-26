# Four-model stability experiment archive

The primary endpoint is **singleton versus batched stability**, separately for
native and batch-invariant inference. Success outcomes are secondary; the campaign
does not optimize success or select starts based on model outcomes.

## Read these first

- [Audited paired stability summary](stability_audited_summary.json): 950 accepted
  paired starts, comprising 500 pi05, 250 RDT and 200 CogACT starts across 19 tasks.
  Native actions diverged in every start; invariant actions and measured physical
  trajectories matched exactly in these cohorts. Same-condition repeats were exact.
- [Task-level paired bootstrap results](paired_stability_bootstrap_verified_summary_003.csv)
  and [JSON](paired_stability_bootstrap_verified_summary_003.json): conditional
  paired-start uncertainty, not population success estimates or model rankings.
- [LingBot batch-error magnitudes](lingbot_singleton_batch_magnitudes_precision001.json)
  and [arithmetic recount](lingbot_singleton_batch_magnitudes_precision001_arithmetic_recount.json):
  fixed-input results, separate from closed-loop trajectories.
- [Remaining evidence inventory](current_evidence/four_model_remaining_evidence_inventory002.json)
  and [resource audit](current_evidence/four_model_resource_blocked_audit001.json).
- [Campaign research log](../../FOUR_MODEL_CAMPAIGN.md) and
  [LingBot completion goal](../../LINGBOT_EXPERIMENT_GOAL.md).

## Acceptance and historical evidence

`current_evidence/` records hash-bound acceptance, rejection, launch and progress
reports. A filename in that directory does not itself establish acceptance: read
the status and scope. `figures/` contains exported figures and their table audits.
Historical reports, snapshots, partial output, and rejected attempts are retained
for provenance; they must not be pooled into accepted results. In particular,
superseded LingBot TF32 configurations and interrupted performance attempts are
not accepted corrected-precision or isolated-timing evidence. Plans and launch
records are not measurements. Independent arithmetic/contract checks do not imply
independent rederivation of the underlying tensor errors.

## Recovery limitations, checked 2026-11-26

This Git archive retains the files present in the worktree. The original
`/scratch/hadjiyiannis_3_gmail_com/pi0-libero/four-model-campaign` directory is absent
on the current host. References to that directory or `/vastdisk` are historical
provenance, not portable asset locations. Checkpoints, simulator installations,
old virtual environments, and the complete original raw simulator corpus are
**not** restored by cloning this repository. Their absence must not be interpreted
as evidence that they were deleted everywhere, nor may raw-data reproducibility
be claimed from aggregate reports alone. Recover the pinned assets and retained
raw evidence, or rerun and requalify the affected stages in versioned directories.

Last accepted campaign measurements recorded here date to November 13, 2026.
The full four-model campaign is **incomplete**. No WAM simulator campaign was run
under this four-model scope; any separate historical WAM files are not evidence
of its completion.
