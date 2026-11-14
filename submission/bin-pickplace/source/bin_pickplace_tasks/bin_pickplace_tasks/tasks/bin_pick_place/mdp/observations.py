"""Dictionary observation terms with explicit, consistent frames."""

from __future__ import annotations

import torch

from .events import TOOL_NAMES, ensure_episode_state
from .geometry import BLUE_INTERIOR_MAX, BLUE_INTERIOR_MIN, WHITE_OUTER_MAX, WHITE_OUTER_MIN


def _selected(env, attribute: str) -> torch.Tensor:
    ensure_episode_state(env)
    values = torch.stack([getattr(env.scene[name].data, attribute) for name in TOOL_NAMES], dim=1)
    width = values.shape[-1]
    index = env.target_tool_index[:, None, None].expand(-1, 1, width)
    return values.gather(1, index).squeeze(1)


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    xyz = -quat[:, 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + quat[:, :1] * t + torch.cross(xyz, t, dim=-1)


def end_effector_pose(env) -> torch.Tensor:
    sensor = env.scene["ee_frame"]
    pos_env = sensor.data.target_pos_w[:, 0] - env.scene.env_origins
    return torch.cat((pos_env, sensor.data.target_quat_w[:, 0]), dim=-1)


def gripper_state(env) -> torch.Tensor:
    robot = env.scene["robot"]
    ids = robot.find_joints("panda_finger_joint.*")[0]
    return robot.data.joint_pos[:, ids]


def tool_poses(env) -> torch.Tensor:
    poses = []
    for name in TOOL_NAMES:
        asset = env.scene[name]
        poses.append(torch.cat((asset.data.root_pos_w - env.scene.env_origins, asset.data.root_quat_w), dim=-1))
    return torch.cat(poses, dim=-1)


def bin_bounds(env) -> torch.Tensor:
    """Both bin root poses plus documented local bounds (26 values total)."""
    white = env.scene["white_bin"]
    blue = env.scene["blue_bin"]
    white_pose = torch.cat((white.data.root_pos_w - env.scene.env_origins, white.data.root_quat_w), dim=-1)
    blue_pose = torch.cat((blue.data.root_pos_w - env.scene.env_origins, blue.data.root_quat_w), dim=-1)
    white_bounds = torch.tensor(WHITE_OUTER_MIN + WHITE_OUTER_MAX, device=env.device).expand(env.num_envs, -1)
    blue_bounds = torch.tensor(BLUE_INTERIOR_MIN + BLUE_INTERIOR_MAX, device=env.device).expand(env.num_envs, -1)
    return torch.cat((white_pose, white_bounds, blue_pose, blue_bounds), dim=-1)


def target_relative_ee(env) -> torch.Tensor:
    target_pos = _selected(env, "root_pos_w")
    return target_pos - env.scene["ee_frame"].data.target_pos_w[:, 0]


def target_relative_blue(env) -> torch.Tensor:
    """Target root position expressed in the blue-bin local frame."""
    target_pos = _selected(env, "root_pos_w")
    blue = env.scene["blue_bin"]
    return _quat_apply_inverse(blue.data.root_quat_w, target_pos - blue.data.root_pos_w)


def instruction_token(env) -> torch.Tensor:
    ensure_episode_state(env)
    return torch.nn.functional.one_hot(env.target_tool_index, num_classes=len(TOOL_NAMES)).float()


def camera_rgb(env, sensor_name: str) -> torch.Tensor:
    return env.scene[sensor_name].data.output["rgb"]
