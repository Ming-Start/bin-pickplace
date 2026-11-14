"""TEST-ONLY integration probe for the Bin-PickPlace low-memory recorder profile."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Bin-PickPlace-Franka-Play-v0")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument(
    "--output", type=Path, default=Path("/tmp/bin_pickplace_low_memory_recorder_probe.hdf5")
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    parser.error("the probe requires --enable_cameras to prove RGB is filtered while cameras run")
if args.steps < 1 or args.output.suffix != ".hdf5":
    parser.error("--steps must be positive and --output must end in .hdf5")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import torch

import bin_pickplace_tasks  # noqa: F401, E402
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from recording_profile import install_low_memory_recording_profile  # noqa: E402


EXPECTED_LOW_DIM_OBSERVATIONS = {
    "joint_positions",
    "joint_velocities",
    "end_effector_pose",
    "gripper_state",
    "tool_poses",
    "bin_bounds",
    "target_relative_ee",
    "target_relative_blue",
    "target_tool_one_hot",
    "previous_action",
}


def main() -> None:
    env = None
    output = args.output.expanduser().resolve()
    try:
        install_low_memory_recording_profile()
        cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        cfg.terminations = {}
        recorder_cfg = ActionStateRecorderManagerCfg()
        recorder_cfg.dataset_export_dir_path = str(output.parent)
        recorder_cfg.dataset_filename = output.stem
        recorder_cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        cfg.recorders = recorder_cfg

        env = gym.make(args.task, cfg=cfg).unwrapped
        env.reset()
        action = torch.zeros((1, 7), device=env.device)
        for _ in range(args.steps):
            env.step(action)
        env.recorder_manager.set_success_to_episodes(
            [0], torch.ones((1,), dtype=torch.bool, device=env.device)
        )
        env.recorder_manager.export_episodes([0])
        env.close()
        env = None

        with h5py.File(output, "r+") as stream:
            episode = stream["data/demo_0"]
            observation_names = set(episode["obs"].keys())
            if observation_names != EXPECTED_LOW_DIM_OBSERVATIONS:
                raise AssertionError(
                    f"unexpected recorded observation keys: {sorted(observation_names)}"
                )
            if tuple(episode["actions"].shape) != (args.steps, 7):
                raise AssertionError(f"unexpected action shape: {episode['actions'].shape}")
            if "initial_state" not in episode or "states" not in episode:
                raise AssertionError("initial_state/states were not retained")
            stream["data"].attrs["probe_only"] = True
            stream["data"].attrs["recording_profile"] = (
                "actions_states_low_dim_observations"
            )
            stream.flush()

        print("Low-memory recorder probe passed", flush=True)
        print(f"  steps: {args.steps}", flush=True)
        print(f"  observation keys: {sorted(EXPECTED_LOW_DIM_OBSERVATIONS)}", flush=True)
        print(f"  RGB keys retained: none", flush=True)
        print(f"  output: {output} ({output.stat().st_size} bytes, TEST-ONLY)", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(f"Low-memory recorder probe failed: {error!r}", flush=True)
        traceback.print_exc()
        os._exit(1)
    simulation_app.close()
