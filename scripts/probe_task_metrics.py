"""Exercise the configured success metric against live Isaac scene tensors.

This is an integration probe, not a scripted task policy: it injects controlled
states solely to verify that the formal terminal predicate accepts and rejects
the rubric's edge cases.
"""

from __future__ import annotations

import argparse
import os
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Bin-PickPlace-Franka-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    parser.error("this environment's observation contract requires --enable_cameras")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import bin_pickplace_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from bin_pickplace_tasks.tasks.bin_pick_place import mdp  # noqa: E402


def _write_asset(env, name: str, position_w, *, linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0)):
    asset = env.scene[name]
    pose = torch.tensor([[*position_w, 1.0, 0.0, 0.0, 0.0]], device=env.device)
    velocity = torch.tensor([[*linear, *angular]], device=env.device)
    asset.write_root_pose_to_sim(pose)
    asset.write_root_velocity_to_sim(velocity)


def _set_gripper(env, opening_per_finger: float) -> None:
    robot = env.scene["robot"]
    finger_ids = robot.find_joints("panda_finger_joint.*")[0]
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos[:, finger_ids] = opening_per_finger
    robot.write_joint_state_to_sim(joint_pos, joint_vel)


def _predicate(env, *, consecutive_steps: int = 10) -> bool:
    env.scene.update(0.0)
    return bool(mdp.target_placed(env, consecutive_steps=consecutive_steps)[0].item())


def _clear_streak(env) -> None:
    env.success_streak.zero_()


def main() -> None:
    env = None
    try:
        cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        env = gym.make(args.task, cfg=cfg).unwrapped
        env.reset()
        env.target_tool_index.zero_()  # screwdriver is the instructed target

        blue_position = env.scene["blue_bin"].data.root_pos_w[0].clone()
        source_position = torch.tensor((0.70, -0.225, 0.02), device=env.device)
        valid_position = blue_position + torch.tensor((0.02, 0.0, 0.05), device=env.device)
        _set_gripper(env, 0.04)

        # Wrong-tool probe: wrench is in the target bin; instructed screwdriver remains at source.
        _write_asset(env, "screwdriver", source_position.tolist())
        _write_asset(env, "wrench", valid_position.tolist())
        _clear_streak(env)
        wrong_tool = _predicate(env, consecutive_steps=1)
        assert not wrong_tool, "wrong-tool placement must not satisfy the target-aware metric"

        # Still-held probe: target is geometrically valid, but fingers are closed.
        _write_asset(env, "screwdriver", valid_position.tolist())
        _set_gripper(env, 0.0)
        _clear_streak(env)
        held = _predicate(env, consecutive_steps=1)
        assert not held, "a still-held target must be rejected"

        # Moving/bouncing probe: target is inside and released but exceeds the speed threshold.
        _set_gripper(env, 0.04)
        _write_asset(env, "screwdriver", valid_position.tolist(), linear=(0.10, 0.0, 0.0))
        _clear_streak(env)
        moving = _predicate(env, consecutive_steps=1)
        assert not moving, "a moving target must be rejected"

        # Rim probe: root is near the rim and part of its measured OBB lies outside.
        rim_position = blue_position + torch.tensor((0.19, 0.0, 0.05), device=env.device)
        _write_asset(env, "screwdriver", rim_position.tolist())
        _clear_streak(env)
        rim = _predicate(env, consecutive_steps=1)
        assert not rim, "a tool resting across the rim must be rejected"

        # Persistence probe: exactly nine valid frames remain false; the tenth becomes true.
        _write_asset(env, "screwdriver", valid_position.tolist())
        _clear_streak(env)
        first_nine = [_predicate(env) for _ in range(9)]
        tenth = _predicate(env)
        assert not any(first_nine), "success fired before ten consecutive valid frames"
        assert tenth, "success did not fire on the tenth consecutive valid frame"

        print("Metric integration probe passed", flush=True)
        print("  wrong tool in blue bin: rejected", flush=True)
        print("  instructed tool still held: rejected", flush=True)
        print("  instructed tool moving at 0.10 m/s: rejected", flush=True)
        print("  instructed tool straddling rim: rejected", flush=True)
        print("  valid + released + settled: accepted only on frame 10", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException as error:
        print(f"Metric integration probe failed: {error!r}", flush=True)
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    if exit_code:
        os._exit(exit_code)
