"""Reset and episode-conditioning events for bin pick-and-place."""

from __future__ import annotations

import os

import torch


TOOL_NAMES = ("screwdriver", "wrench", "hammer", "pallet_knife")
TOOL_DISPLAY_NAMES = ("screwdriver", "wrench", "hammer", "pallet knife")
CANONICAL_INSTRUCTION = "Pick the {tool_name} from the white bin and place it in the blue bin."

# All packaged USD roots are bottom references. The source-bin mesh is scaled 2.4x in XY by the
# scene config. The tool poses below were authored interactively in the full simulator scene and
# are copied into the scene defaults as well as this reset event so every episode starts identically.
BIN_ROOT_POSITIONS = {
    "white_bin": (0.55, -0.30, 0.005),
    "blue_bin": (0.55, 0.32, 0.005),
}
TOOL_RESET_POSES = {
    "hammer": ((0.399235, -0.350000, 0.020), (1.0, 0.0, 0.0, 0.0)),
    "screwdriver": ((0.640262, -0.310392, 0.020), (-0.6720588, 0.0, 0.0, 0.7404978)),
    "wrench": ((0.469094, -0.243457, 0.020), (0.9429091, 0.0, 0.0, -0.3330501)),
    "pallet_knife": ((0.759419, -0.320000, 0.020), (0.9225082, 0.0, 0.0, -0.3859775)),
}


def ensure_episode_state(env) -> None:
    """Create target/streak state early enough for observation-manager shape discovery."""
    if not hasattr(env, "target_tool_index"):
        env.target_tool_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "success_streak"):
        env.success_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "target_cycle_count"):
        env.target_cycle_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "language_instructions"):
        env.language_instructions = [CANONICAL_INSTRUCTION.format(tool_name=TOOL_DISPLAY_NAMES[0])] * env.num_envs


def recording_target_index(env) -> int:
    """Map stock-recorder progress to a stable canonical target index."""
    recorder = getattr(env, "recorder_manager", None)
    counter = getattr(recorder, "exported_successful_episode_count", 0)
    count = counter() if callable(counter) else counter
    return int(count or 0) % len(TOOL_NAMES)


def reset_bin_pick_place(env, env_ids: torch.Tensor | None, deterministic: bool = False) -> None:
    """Restore a valid scene and select the language-conditioned target.

    PLAY deterministically cycles screwdriver -> wrench -> hammer -> pallet knife on resets, so a
    single environment can cover all four demonstrations while remaining fully reproducible.
    """
    ensure_episode_state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)

    robot = env.scene["robot"]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    for name, position in BIN_ROOT_POSITIONS.items():
        asset = env.scene[name]
        state = asset.data.default_root_state[env_ids].clone()
        state[:, :3] = torch.tensor(position, device=env.device) + env.scene.env_origins[env_ids]
        state[:, 7:] = 0.0
        asset.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)

    for name, (position, quaternion) in TOOL_RESET_POSES.items():
        tool = env.scene[name]
        state = tool.data.default_root_state[env_ids].clone()
        state[:, :3] = torch.tensor(position, device=env.device) + env.scene.env_origins[env_ids]
        state[:, 3:7] = torch.tensor(quaternion, device=env.device)
        state[:, 7:] = 0.0
        tool.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        tool.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)

    if deterministic and os.environ.get("BIN_PICKPLACE_RECORDING") == "1":
        # The stock recorder performs internal resets before the operator sees
        # the scene. Bind target identity to the number of exported successes,
        # so demo_0..demo_3 are canonically ordered and a manual retry keeps the
        # same instruction.
        env.target_tool_index[env_ids] = recording_target_index(env)
    elif deterministic:
        env.target_tool_index[env_ids] = env.target_cycle_count[env_ids] % len(TOOL_NAMES)
        env.target_cycle_count[env_ids] += 1
    else:
        env.target_tool_index[env_ids] = torch.randint(0, len(TOOL_NAMES), (env_ids.numel(),), device=env.device)

    env.success_streak[env_ids] = 0
    for env_id in env_ids.detach().cpu().tolist():
        tool_name = TOOL_DISPLAY_NAMES[int(env.target_tool_index[env_id].item())]
        env.language_instructions[env_id] = CANONICAL_INSTRUCTION.format(tool_name=tool_name)
        if deterministic:
            print(f"[Bin-PickPlace][env_{env_id}] {env.language_instructions[env_id]}", flush=True)
