"""Verify canonical target ordering used by the stock demonstration recorder."""

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

os.environ["BIN_PICKPLACE_RECORDING"] = "1"
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym

import bin_pickplace_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from bin_pickplace_tasks.tasks.bin_pick_place.mdp.events import TOOL_DISPLAY_NAMES  # noqa: E402


def main() -> None:
    env = None
    try:
        cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        env = gym.make(args.task, cfg=cfg).unwrapped
        observed = []
        for successful_count, expected_name in enumerate(TOOL_DISPLAY_NAMES):
            env.recorder_manager._exported_successful_episode_count = {0: successful_count}
            env.reset()
            actual_index = int(env.target_tool_index[0].item())
            assert actual_index == successful_count, (
                f"recorder count {successful_count} selected target index {actual_index}"
            )
            instruction = env.language_instructions[0]
            assert expected_name in instruction, f"instruction/target mismatch: {instruction}"
            observed.append(instruction)

        print("Recorder target-order probe passed", flush=True)
        for index, instruction in enumerate(observed):
            print(f"  demo_{index}: {instruction}", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException as error:
        print(f"Recorder target-order probe failed: {error!r}", flush=True)
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    if exit_code:
        os._exit(exit_code)
