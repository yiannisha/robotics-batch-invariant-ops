"""Version 3: capture rigid state, including SAPIEN's sleeping property."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml

if __package__:
    from .qualify_simpler_fixed_action_replay import digest
else:
    from qualify_simpler_fixed_action_replay import digest


def capture_state(env):
    state={"schema":"robotwin_physics_rigid_components_v2","actors":[],"articulations":[]}
    for actor in env.scene.get_all_actors():
        pose=actor.get_pose()
        value={"name":actor.get_name(),"position":pose.p,"quaternion":pose.q}
        components = actor.get_components()
        value['component_types'] = [type(c).__module__ + '.' + type(c).__name__ for c in components]
        bodies = [c for c in components if type(c).__name__ in
                  ('PhysxRigidDynamicComponent', 'PhysxRigidStaticComponent')]
        if len(bodies) != 1:
            raise ValueError('Rigid actor must have exactly one explicitly covered body')
        body = bodies[0]
        value['body_type'] = type(body).__name__
        value['body_state'] = {}
        if value['body_type'] == 'PhysxRigidDynamicComponent':
            for method in ('get_linear_velocity', 'get_angular_velocity', 'get_mass',
                           'get_inertia', 'get_linear_damping', 'get_angular_damping',
                           'get_disable_gravity', 'get_kinematic', 'get_locked_motion_axes',
                           'get_sleep_threshold', 'is_sleeping', 'get_max_contact_impulse',
                           'get_max_depenetration_velocity', 'get_solver_position_iterations',
                           'get_solver_velocity_iterations', 'get_auto_compute_mass'):
                attribute = getattr(body, method)
                if method == 'is_sleeping':
                    value['body_state'][method] = attribute() if callable(attribute) else bool(attribute)
                else:
                    value['body_state'][method] = attribute()
            local = body.get_cmass_local_pose()
            value['body_state']['cmass_local_pose'] = {'position':local.p,'quaternion':local.q}
            target = body.get_kinematic_target() if body.get_kinematic() else None
            value['body_state']['kinematic_target'] = (None if target is None else
                {'position':target.p,'quaternion':target.q})
        state["actors"].append(value)
    for articulation in env.scene.get_all_articulations():
        pose=articulation.get_pose()
        state["articulations"].append({"name":articulation.get_name(),"position":pose.p,"quaternion":pose.q,
            "qpos":articulation.get_qpos(),"qvel":articulation.get_qvel(),
            "qacc":articulation.get_qacc(),"qf":articulation.get_qf(),
            "drive_targets":[joint.get_drive_target() for joint in articulation.get_active_joints()],
            "drive_velocity_targets":[joint.get_drive_velocity_target() for joint in articulation.get_active_joints()]})
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--task",default="adjust_bottle")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--repeats",type=int,default=2)
    parser.add_argument("--steps",type=int,default=0)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.repeats<2:parser.error("at least two repeats required")
    args.output.mkdir(parents=True)
    os.chdir(args.source)
    sys.path.insert(0,str(args.source))
    task_class=getattr(importlib.import_module("envs."+args.task),args.task)
    config=yaml.safe_load((args.source/"task_config/demo_clean.yml").read_text())
    embodiments=yaml.safe_load((args.source/"task_config/_embodiment_config.yml").read_text())
    robot_file=embodiments[config["embodiment"][0]]["file_path"]
    robot_config=yaml.safe_load((args.source/robot_file/"config.yml").read_text())
    cameras=yaml.safe_load((args.source/"task_config/_camera_config.yml").read_text())
    camera=cameras[config["camera"]["head_camera_type"]]
    config.update({"task_name":args.task,"task_config":"demo_clean","head_camera_h":camera["h"],
                   "head_camera_w":camera["w"],"left_robot_file":robot_file,"right_robot_file":robot_file,
                   "left_embodiment_config":robot_config,"right_embodiment_config":robot_config,
                   "dual_arm_embodied":True,"eval_mode":True,"render_freq":0,"save_data":False,
                   "eval_video_log":False,"collect_data":False})
    runtime={"scope":"fixed_action_replay_pilot" if args.steps else "initialization_and_render_probe_only","model_evaluated":False,
             "task":args.task,"seed":args.seed,"repeats":args.repeats,"config":config,
             "fixed_action_steps":args.steps,
             "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
             "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/"runtime.json").write_text(json.dumps(runtime,indent=2)+"\n")
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    records=[]
    for repeat in range(args.repeats):
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        env=task_class()
        try:
            env.setup_demo(now_ep_num=0,seed=args.seed,is_test=True,**config)
            observation=env.get_obs()
            state=capture_state(env)
            record={"repeat":repeat,"observation_sha256":digest(observation),"state_sha256":digest(state),
                    "camera_rgb_sha256":{name:digest(value["rgb"]) for name,value in observation["observation"].items() if "rgb" in value},
                    "camera_rgb_statistics":{name:{"shape":list(value["rgb"].shape),"min":int(value["rgb"].min()),
                        "max":int(value["rgb"].max()),"mean":float(value["rgb"].mean())} for name,value in observation["observation"].items() if "rgb" in value}}
            torch.save({"observation":observation,"state":state},args.output/f"initialization-{repeat}.pt")
            trajectory=[{"observation_sha256":record["observation_sha256"],"state_sha256":record["state_sha256"]}]
            initial_action=np.asarray(observation["joint_action"]["vector"],dtype=np.float32)
            for step in range(args.steps):
                action=initial_action.copy()
                action+=np.float32(0.002*np.sin(step/7))
                action[6]=action[13]=np.float32(1. if (step//5)%2 else 0.)
                env.take_action(action,action_type="qpos")
                next_observation=env.get_obs()
                next_state=capture_state(env)
                trajectory.append({"step":step,"action":action.tolist(),"observation_sha256":digest(next_observation),
                                   "state_sha256":digest(next_state),"success":bool(env.eval_success),
                                   "take_action_count":env.take_action_cnt})
                torch.save({"observation":next_observation,"state":next_state},args.output/f"repeat-{repeat}-step-{step:04d}.pt")
                if env.eval_success or env.take_action_cnt>=env.step_lim:break
            record["trajectory"]=trajectory
            with (args.output/"initializations.jsonl").open("a") as stream:stream.write(json.dumps(record)+"\n")
            records.append(record)
            print(f"CAPTURED {args.task}/{args.seed}/{repeat}: {record['observation_sha256']}",flush=True)
        finally:
            if hasattr(env,"scene"):env.close_env(clear_cache=True)
    exact=len({row["observation_sha256"] for row in records})==1 and len({row["state_sha256"] for row in records})==1
    replay_exact=all(row["trajectory"]==records[0]["trajectory"] for row in records)
    (args.output/"summary.json").write_text(json.dumps({"scope":runtime["scope"],"initial_state_and_observation_exact":exact,
        "records":records,"fixed_action_replay_pilot_exact":bool(args.steps) and replay_exact,"closed_loop_complete":False},indent=2)+"\n")
    if not exact or not replay_exact:raise RuntimeError("RoboTwin initial state or replay is not exactly repeatable; retain and investigate")


if __name__=="__main__":main()
