"""Cheap package/registration/config validation; does not launch the simulator."""

import gymnasium as gym
import bin_pickplace_tasks  # noqa: F401

expected = {"Bin-PickPlace-Franka-v0", "Bin-PickPlace-Franka-Play-v0"}
registered = set(gym.registry)
missing = expected - registered
if missing:
    raise SystemExit(f"Missing environments: {sorted(missing)}")
print("Registered:", *sorted(expected), sep="\n  ")

