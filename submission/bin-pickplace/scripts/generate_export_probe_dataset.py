"""Generate a TEST-ONLY four-episode dataset for the video-export pipeline.

This probe pre-positions each target in the blue bin and records idle actions.
It verifies Isaac Lab HDF5 state restoration, action replay, camera rendering,
and the persistent success metric.  It is not a manipulation policy and the
result must never be submitted as a human keyboard demonstration.
"""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Bin-PickPlace-Franka-Play-v0")
parser.add_argument("--output", type=Path, default=Path("/tmp/bin_pickplace_export_probe.hdf5"))
parser.add_argument("--settle-steps", type=int, default=240, help="Maximum frames allowed for physical settling")
parser.add_argument("--replay-steps", type=int, default=30)
parser.add_argument("--overwrite", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    parser.error("the Bin-PickPlace observation contract requires --enable_cameras")
if args.output.suffix != ".hdf5":
    parser.error("--output must end in .hdf5")
if args.settle_steps < 1 or args.replay_steps < 10:
    parser.error("--settle-steps must be positive and --replay-steps must be at least 10")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import torch

import bin_pickplace_tasks  # noqa: F401, E402
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from bin_pickplace_tasks.tasks.bin_pick_place import mdp  # noqa: E402
from bin_pickplace_tasks.tasks.bin_pick_place.mdp.events import (  # noqa: E402
    CANONICAL_INSTRUCTION,
    TOOL_DISPLAY_NAMES,
    TOOL_NAMES,
)
from bin_pickplace_tasks.tasks.bin_pick_place.mdp.geometry import (  # noqa: E402
    TOOL_BOUNDS_MAX,
    TOOL_BOUNDS_MIN,
    box_corners,
    quat_apply,
    quat_apply_inverse,
)


def _add_env_state(episode: EpisodeData, key: str, state: dict, env_index: int = 0) -> None:
    """Mirror RecorderManager.add_to_episodes for one environment."""
    for sub_key, value in state.items():
        nested_key = f"{key}/{sub_key}"
        if isinstance(value, dict):
            _add_env_state(episode, nested_key, value, env_index)
        else:
            episode.add(nested_key, value[env_index])


def _place_target_for_probe(env, target_index: int) -> None:
    """Place one tool flat, centered, and above the blue-bin floor."""
    tool = env.scene[TOOL_NAMES[target_index]]
    blue = env.scene["blue_bin"]
    bounds_min = torch.tensor(TOOL_BOUNDS_MIN[target_index], device=env.device)
    bounds_max = torch.tensor(TOOL_BOUNDS_MAX[target_index], device=env.device)

    # Cancel the tool-local XY AABB center.  The 60 mm starting height lets the
    # rigid body settle naturally onto the packaged bin collider before capture.
    local_position = torch.stack(
        (
            -0.5 * (bounds_min[0] + bounds_max[0]),
            -0.5 * (bounds_min[1] + bounds_max[1]),
            torch.tensor(0.060, device=env.device),
        )
    ).unsqueeze(0)
    world_position = blue.data.root_pos_w + quat_apply(blue.data.root_quat_w, local_position)
    pose = torch.cat((world_position, blue.data.root_quat_w), dim=-1)
    tool.write_root_pose_to_sim(pose)
    tool.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
    env.scene.update(0.0)


def _placement_diagnostics(env, target_index: int) -> str:
    tool = env.scene[TOOL_NAMES[target_index]]
    blue = env.scene["blue_bin"]
    bounds_min = torch.tensor([TOOL_BOUNDS_MIN[target_index]], device=env.device)
    bounds_max = torch.tensor([TOOL_BOUNDS_MAX[target_index]], device=env.device)
    corners = box_corners(bounds_min, bounds_max)
    corners_world = tool.data.root_pos_w[:, None, :] + quat_apply(
        tool.data.root_quat_w[:, None, :].expand(-1, 8, -1), corners
    )
    corners_blue = quat_apply_inverse(
        blue.data.root_quat_w[:, None, :].expand(-1, 8, -1),
        corners_world - blue.data.root_pos_w[:, None, :],
    )
    robot = env.scene["robot"]
    finger_ids = robot.find_joints("panda_finger_joint.*")[0]
    opening = robot.data.joint_pos[:, finger_ids].sum(dim=-1)
    tcp = env.scene["ee_frame"].data.target_pos_w[:, 0]
    separation = torch.linalg.vector_norm(tool.data.root_pos_w - tcp, dim=-1)
    return (
        f"pos={tool.data.root_pos_w[0].tolist()}, quat={tool.data.root_quat_w[0].tolist()}, "
        f"corners_blue_min={corners_blue.amin(dim=1)[0].tolist()}, "
        f"corners_blue_max={corners_blue.amax(dim=1)[0].tolist()}, "
        f"lin={tool.data.root_lin_vel_w[0].tolist()}, ang={tool.data.root_ang_vel_w[0].tolist()}, "
        f"opening={opening[0].item():.6f}, tcp_separation={separation[0].item():.6f}"
    )


def _annotate_probe_dataset(path: Path) -> None:
    with h5py.File(path, "r+") as stream:
        data = stream["data"]
        data.attrs["target_order"] = ",".join(TOOL_DISPLAY_NAMES)
        data.attrs["probe_only"] = True
        data.attrs["probe_description"] = (
            "Automated exporter integration probe with pre-positioned targets; not a human demonstration."
        )
        for target_index, tool_name in enumerate(TOOL_DISPLAY_NAMES):
            episode = data[f"demo_{target_index}"]
            episode.attrs["target_tool_index"] = target_index
            episode.attrs["target_tool_name"] = tool_name
            episode.attrs["language_instruction"] = CANONICAL_INSTRUCTION.format(tool_name=tool_name)
            episode.attrs["probe_only"] = True
        stream.flush()


def main() -> None:
    env = None
    dataset = None
    output = args.output.expanduser().resolve()
    try:
        if output.exists():
            if not args.overwrite:
                raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
            output.unlink()

        cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        cfg.recorders = {}
        cfg.terminations = {}
        env = gym.make(args.task, cfg=cfg).unwrapped

        dataset = HDF5DatasetFileHandler()
        dataset.create(str(output), env_name=args.task)
        dataset.add_env_args(
            {
                "probe_only": True,
                "probe_description": "pre-positioned target; idle-action exporter integration test",
            }
        )

        idle_action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
        if idle_action.shape[1] != 7:
            raise AssertionError(f"expected a 7D action, got {tuple(idle_action.shape)}")

        for target_index, tool_name in enumerate(TOOL_DISPLAY_NAMES):
            env.reset()
            env.target_tool_index[:] = target_index
            env.success_streak.zero_()
            env.language_instructions[0] = CANONICAL_INSTRUCTION.format(tool_name=tool_name)
            _place_target_for_probe(env, target_index)

            stable_frames = 0
            for _ in range(args.settle_steps):
                env.step(idle_action)
                env.success_streak.zero_()
                valid_now = bool(mdp.target_placed(env, consecutive_steps=1)[0].item())
                env.success_streak.zero_()
                stable_frames = stable_frames + 1 if valid_now else 0
                if stable_frames >= 30:
                    break

            if stable_frames < 30:
                raise RuntimeError(
                    f"{tool_name} did not remain metric-valid for 30 frames within "
                    f"{args.settle_steps} steps: "
                    f"{_placement_diagnostics(env, target_index)}"
                )
            env.success_streak.zero_()

            episode = EpisodeData()
            _add_env_state(episode, "initial_state", env.scene.get_state(is_relative=True))
            detected_success = False
            for _ in range(args.replay_steps):
                episode.add("actions", idle_action[0])
                env.step(idle_action)
                _add_env_state(episode, "states", env.scene.get_state(is_relative=True))
                detected_success = detected_success or bool(mdp.target_placed(env)[0].item())

            if not detected_success:
                raise RuntimeError(f"{tool_name} never reached the persistent success metric")
            episode.success = True
            episode.pre_export()
            dataset.write_episode(episode)
            tool = env.scene[TOOL_NAMES[target_index]]
            print(
                f"probe episode demo_{target_index}: target={tool_name}, frames={args.replay_steps}, "
                f"final_pos={tool.data.root_pos_w[0].tolist()}",
                flush=True,
            )

        dataset.flush()
        dataset.close()
        dataset = None
        _annotate_probe_dataset(output)
        print(f"TEST-ONLY exporter probe dataset written to {output}", flush=True)
        print("This file is not a human demonstration and must not be submitted as one.", flush=True)
    finally:
        if dataset is not None:
            dataset.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(f"Exporter probe generation failed: {error!r}", flush=True)
        traceback.print_exc()
        # Isaac Kit may terminate with status 0 from close() even after a
        # Python exception.  The environment has already closed in main().
        os._exit(1)
    simulation_app.close()
