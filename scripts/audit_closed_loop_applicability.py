"""Create an evidence-backed closed-loop applicability audit for all paths."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from audit_cross_model_campaign import CONFIGURATIONS


# Public evaluation surface for the exact checkpoint used by each audited path.
# Evidence points to pinned local upstream/research files, so this audit remains
# reproducible without relying on mutable web pages.
SURFACES = {
    "pi05": (None, "The executed pi0.5 DROID checkpoint has no released, checkpoint-matched public simulator evaluator.", "research/pi05_openpi_architecture/investigation.md"),
    "pi_fast": ("LIBERO", "Official LIBERO policy checkpoint.", "research/pi_fast/investigation.md"),
    "internvla_a15_standard": ("LIBERO", "Official LIBERO checkpoint and evaluator client.", "/workspace/models/internvla-a-series/evaluation/LIBERO/README.md"),
    "internvla_a15_optimized": ("LIBERO", "Same official policy and simulator surface; alternate released action backend.", "/workspace/models/internvla-a-series/evaluation/LIBERO/README.md"),
    "molmoact2": ("LIBERO", "Official LIBERO policy checkpoint.", "research/molmoact2/investigation.md"),
    "groot_n17": ("LIBERO", "Official GR00T N1.7 LIBERO-10 checkpoint.", "research/groot_n17/investigation.md"),
    "cosmos3_nano": ("RoboLab", "Official DROID policy server documents RoboLab as its simulation client.", "/workspace/models/cosmos-framework/docs/action_policy_droid_server.md"),
    "cosmos3_nano_video": ("RoboLab", "Decoded video is a joint output of the same policy/controller evaluation surface.", "/workspace/models/cosmos-framework/docs/action_policy_droid_server.md"),
    "cosmos3_edge": ("RoboLab", "Official DROID policy server documents RoboLab as its simulation client.", "/workspace/models/cosmos-framework/docs/action_policy_droid_server.md"),
    "cosmos3_edge_video": ("RoboLab", "Decoded video is a joint output of the same policy/controller evaluation surface.", "/workspace/models/cosmos-framework/docs/action_policy_droid_server.md"),
    "xvla": ("LIBERO", "Official LIBERO policy checkpoint.", "research/xvla/investigation.md"),
    "uva": ("LIBERO", "Official LIBERO-10 checkpoint and eval_sim entrypoint.", "/workspace/models/uva/README.md"),
    "uva_joint": ("LIBERO", "Joint video/action path of the same LIBERO-10 controller.", "/workspace/models/uva/README.md"),
    "smolvla": ("LIBERO", "Official SmolVLA LIBERO checkpoint.", "research/smolvla/investigation.md"),
    "openvla_oft": ("LIBERO", "Official LIBERO-Spatial checkpoint and evaluation script.", "/workspace/models/openvla-oft/LIBERO.md"),
    "dexvla": (None, "The complete executed checkpoint is a community reconstruction; the released official evaluator targets a physical AgileX robot, so no checkpoint-matched public simulator claim is possible.", "research/dexvla/investigation.md"),
    "lingbot_vla2": ("RoboTwin", "Official RoboTwin checkpoint and pinned evaluation launcher.", "/workspace/models/lingbot-vla-v2/README.md"),
    "univla_baai": ("LIBERO", "Official LIBERO image-policy checkpoint.", "research/univla_baai/investigation.md"),
    "spatialvla": ("SimplerEnv", "Upstream releases a SimplerEnv evaluation integration.", "/workspace/models/spatialvla/README.md"),
    "rdt_1b": ("ManiSkill", "Exact official ManiSkill checkpoint and evaluator.", "/workspace/models/rdt/README.md"),
    "cogact_small": ("SimplerEnv", "Upstream documents the CogACT SimplerEnv policy integration.", "/workspace/models/cogact/README.md"),
    "openhelix": ("CALVIN", "Official CALVIN checkpoint and online evaluator.", "/workspace/models/openhelix/README.md"),
    "wsa_base": ("LIBERO", "Official WSA Base LIBERO checkpoint.", "research/wsa_base/investigation.md"),
    "vitra_vla_3b": (None, "The released checkpoint/evaluation path targets real robot inference; no checkpoint-matched public simulator evaluator is released.", "research/vitra_vla_3b/investigation.md"),
    "opendw_dw05": ("RoboTwin", "Official DW05 RoboTwin action policy checkpoint.", "research/opendw_dw05/sources.md"),
    "opendw_dw05_joint": ("RoboTwin", "Joint video/action path of the same RoboTwin controller.", "research/opendw_dw05/sources.md"),
    "openwam_alpha_robotwin": ("RoboTwin", "Official OpenWAM Alpha RoboTwin checkpoint and benchmark client.", "/workspace/models/openwam/README.md"),
    "openwam_alpha_video": ("RoboTwin", "Decoded video is a joint output of the same RoboTwin controller.", "/workspace/models/openwam/README.md"),
    "xiaomi_robotics_1": ("RoboCasa", "Official RoboCasa checkpoint and evaluator.", "/workspace/models/xiaomi-robotics-1/eval_robocasa/README.md"),
}


def resolve_evidence(repository: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--results", type=Path, default=Path("results/cross_model"))
    args = parser.parse_args()
    shared_probe_path = args.results / "closed_loop" / "rdt_maniskill_probe.json"
    shared_probe = json.loads(shared_probe_path.read_text()) if shared_probe_path.is_file() else None
    simpler_probe_path = args.results / "closed_loop" / "simplerenv_runtime_probe.json"
    simpler_probe = json.loads(simpler_probe_path.read_text()) if simpler_probe_path.is_file() else None
    records = []
    for identifier, name in CONFIGURATIONS:
        simulator, reason, evidence_value = SURFACES[identifier]
        evidence = resolve_evidence(args.repository, evidence_value)
        attempt_path = args.results / identifier / "full" / "closed_loop" / "evaluator_attempt.json"
        attempt = json.loads(attempt_path.read_text()) if attempt_path.is_file() else None
        if simulator is None:
            status = "NOT APPLICABLE: no checkpoint-matched public simulator evaluator"
            blocker = None
        elif attempt is None:
            status = "NOT PROBED"
            blocker = None
        elif attempt.get("status") == "probe_succeeded_requires_paired_campaign":
            status = "PROBE SUCCEEDED: paired campaign pending"
            blocker = None
        elif attempt.get("status") in {
            "ready_pending_model_campaign",
            "checkpoint_stream_ready_pending_model_probe",
            "exact_checkpoint_ready_pending_model_probe",
        }:
            status = "PREFLIGHT SUCCEEDED: paired campaign pending"
            blocker = None
        elif attempt.get("status") == "completed":
            status = "PAIRED CAMPAIGN COMPLETED"
            blocker = None
        elif (attempt.get("status") == "blocked"
              and attempt.get("probe_episodes_completed", 0) == 0
              and bool(attempt.get("failure_class"))):
            status = "BLOCKED BEFORE EPISODE 1"
            blocker = attempt["failure_class"]
        else:
            status = "MODEL-SPECIFIC PROBE FAILED: investigate"
            blocker = attempt.get("failure_class")
        records.append({
            "id": identifier,
            "name": name,
            "public_simulator": simulator,
            "applicability_reason": reason,
            "evidence_path": str(evidence),
            "evidence_exists": evidence.is_file() or evidence.is_dir(),
            "status": status,
            "blocker": blocker,
            "probe": str(attempt_path) if attempt is not None else None,
            "probe_episodes_completed": attempt.get("probe_episodes_completed", 0) if attempt else 0,
            "paired_episodes_completed": attempt.get("paired_episodes_completed", 0) if attempt else 0,
        })
    missing = [record["id"] for record in records if not record["evidence_exists"]]
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_count": len(records),
        "applicable_public_simulator": sum(record["public_simulator"] is not None for record in records),
        "not_applicable": sum(record["public_simulator"] is None for record in records),
        "blocked_before_episode_1": sum(
            record["status"] == "BLOCKED BEFORE EPISODE 1" for record in records
        ),
        "model_specific_probe_failures": sum(record["status"] == "MODEL-SPECIFIC PROBE FAILED: investigate" for record in records),
        "paired_episodes_completed": sum(record["paired_episodes_completed"] for record in records),
        "missing_evidence": missing,
        "shared_runtime_probe": str(shared_probe_path) if shared_probe is not None else None,
        "simplerenv_runtime_probe": (
            str(simpler_probe_path) if simpler_probe is not None else None
        ),
        "configurations": records,
    }
    output_dir = args.results / "closed_loop"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "applicability.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Closed-loop applicability and execution gate",
        "",
        "The exact campaign checkpoints were checked against their pinned upstream evaluation surfaces. A public, checkpoint-matched simulator exists for "
        f"{report['applicable_public_simulator']}/{report['configuration_count']} audited paths; the other {report['not_applicable']} paths are not assigned a substitute benchmark.",
        "",
        "No paired success-rate campaign is claimed by a feasibility probe. Dedicated runtime probes now establish working MuJoCo/EGL, ManiSkill3/Vulkan, and SimplerEnv/SAPIEN2 execution on this host; these shared simulator checks are startup evidence only and cannot substitute for a checkpoint-matched policy episode. Each applicable checkpoint therefore requires its own evaluator probe and paired campaign, or precise model-specific evidence of a blocker.",
        "",
        "| Configuration | Public evaluator | Status | Evidence |",
        "|---|---|---|---|",
    ]
    for record in records:
        evidence_text = record["evidence_path"]
        lines.append(f"| {record['name']} | {record['public_simulator'] or 'none matched'} | {record['status']} | `{evidence_text}` |")
    lines += [
        "",
        "Retained numerical fixtures are not counted as closed-loop episodes. A precise preflight blocker is separated from an attempted evaluator failure; neither is reported as a completed campaign.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "configurations"}, indent=2))
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
