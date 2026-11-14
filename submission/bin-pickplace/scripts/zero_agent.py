"""Launch the deterministic environment with zero actions."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Bin-PickPlace-Franka-Play-v0")
parser.add_argument("--steps", type=int, default=300, help="Number of control steps; <=0 runs until the app closes")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch
import bin_pickplace_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env = gym.make(args.task, cfg=env_cfg)
env.reset()
step = 0
while simulation_app.is_running() and (args.steps <= 0 or step < args.steps):
    with torch.inference_mode():
        env.step(torch.zeros((env.unwrapped.num_envs, 7), device=env.unwrapped.device))
    step += 1
env.close()
simulation_app.close()
