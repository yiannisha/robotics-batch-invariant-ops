"""Run full UVA joint video/action coverage after the UVA action campaign."""

import os

import run_uva_action_full_campaign as campaign


campaign.MODEL_ID = "uva_joint"
campaign.MODEL_ROOT = campaign.ROOT / "results/cross_model/uva_joint"
campaign.FULL = campaign.MODEL_ROOT / "full"
campaign.MANIFEST = campaign.FULL / "replay_manifest.json"
campaign.CONTEXT_INVENTORY = (
    campaign.ROOT / "results/cross_model/uva/full/context_source_inventory.json"
)
campaign.ADAPTER = "scripts.model_invariance.adapters.uva_libero10_joint_full"
campaign.PILOT = campaign.MODEL_ROOT / "pilots/non_success_pre_full_coverage"
campaign.START_MARKER = campaign.FULL / "non_success_full_campaign_started.json"
campaign.PREDECESSOR_SCRIPT = "run_uva_action_full_campaign.py"
campaign.PREDECESSOR_LABEL = "UVA action"
campaign.ADAPTER_FILES = {
    "base_adapter": campaign.ROOT / "scripts/model_invariance/adapters/uva_libero10.py",
    "action_full_adapter": (
        campaign.ROOT / "scripts/model_invariance/adapters/uva_libero10_full.py"
    ),
    "joint_full_adapter": (
        campaign.ROOT / "scripts/model_invariance/adapters/uva_libero10_joint_full.py"
    ),
}
campaign.OPERATOR_BUILDER_EXTRA_ARGS = ["--joint"]
campaign.FLOW_SUMMARY_ARGS = [
    "--flow-steps", "100",
    "--flow-series", "action=action.diffusion.step_{step:03d}",
    "--flow-series", "video=video.diffusion.step_{step:03d}",
]
os.environ["UVA_JOINT_REPLAY_MANIFEST"] = str(campaign.MANIFEST)


if __name__ == "__main__":
    campaign.main()
