import gymnasium as gym
import bin_pickplace_tasks  # noqa: F401

for spec in gym.registry.values():
    if spec.id.startswith("Bin-PickPlace-"):
        print(spec.id)

