"""Target-aware, probe-resistant termination terms."""

from __future__ import annotations

import torch

from .events import TOOL_NAMES, ensure_episode_state
from .geometry import TOOL_BOUNDS_MAX, TOOL_BOUNDS_MIN, placement_components


def _selected(env, attribute: str) -> torch.Tensor:
    ensure_episode_state(env)
    values = torch.stack([getattr(env.scene[name].data, attribute) for name in TOOL_NAMES], dim=1)
    width = values.shape[-1]
    index = env.target_tool_index[:, None, None].expand(-1, 1, width)
    return values.gather(1, index).squeeze(1)


def target_placed(env, consecutive_steps: int = 10) -> torch.Tensor:
    """Require the instructed tool to be fully inside, settled and released for N steps."""
    ensure_episode_state(env)
    tool_pos = _selected(env, "root_pos_w")
    bounds_min_all = torch.tensor(TOOL_BOUNDS_MIN, device=env.device, dtype=tool_pos.dtype)
    bounds_max_all = torch.tensor(TOOL_BOUNDS_MAX, device=env.device, dtype=tool_pos.dtype)
    bounds_min = bounds_min_all[env.target_tool_index]
    bounds_max = bounds_max_all[env.target_tool_index]

    robot = env.scene["robot"]
    finger_ids = robot.find_joints("panda_finger_joint.*")[0]
    opening = robot.data.joint_pos[:, finger_ids].sum(dim=-1)
    blue = env.scene["blue_bin"]
    inside, settled, released = placement_components(
        tool_pos,
        _selected(env, "root_quat_w"),
        bounds_min,
        bounds_max,
        blue.data.root_pos_w,
        blue.data.root_quat_w,
        _selected(env, "root_lin_vel_w"),
        _selected(env, "root_ang_vel_w"),
        opening,
        env.scene["ee_frame"].data.target_pos_w[:, 0],
    )
    valid = inside & settled & released
    env.success_streak = torch.where(valid, env.success_streak + 1, torch.zeros_like(env.success_streak))
    return env.success_streak >= consecutive_steps


def target_dropped(env, minimum_z: float = -0.10, workspace_radius: float = 1.25) -> torch.Tensor:
    """Unrecoverable target loss; timeout remains a separate truncation term."""
    pos_env = _selected(env, "root_pos_w") - env.scene.env_origins
    return (pos_env[:, 2] < minimum_z) | (torch.linalg.vector_norm(pos_env[:, :2], dim=-1) > workspace_radius)

