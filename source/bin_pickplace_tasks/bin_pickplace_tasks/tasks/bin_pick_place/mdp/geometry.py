"""Pure-torch placement geometry used by termination terms and unit probes."""

from __future__ import annotations

import torch


TOOL_BOUNDS_MIN = (
    (-0.141665, -0.015668, 0.000002),
    (-0.106018, -0.023178, 0.0),
    (-0.114628, -0.069106, 0.000065),
    (-0.011789, -0.087425, 0.000016),
)
TOOL_BOUNDS_MAX = (
    (0.057663, 0.015700, 0.031423),
    (0.093941, 0.023144, 0.012857),
    (0.193994, 0.075486, 0.032770),
    (0.011789, 0.087914, 0.015193),
)

# The source-bin mesh is 0.2976 x 0.141108 x 0.129489 m. The scene uses a
# documented 2.4x XY scale. These conservative interior limits inset its walls.
WHITE_OUTER_MIN = (-0.357120, -0.169330, 0.0)
WHITE_OUTER_MAX = (0.357120, 0.169330, 0.129489)
# The bin root is 5 mm above the tabletop. A 10 mm lower tolerance covers the
# thin packaged floor/contact skin while still rejecting any tool that falls.
WHITE_INTERIOR_MIN = (-0.330, -0.142, -0.010)
WHITE_INTERIOR_MAX = (0.330, 0.142, 0.120)

# Conservative recommendations shown by the interactive layout editor. They
# are warnings rather than formal rejection thresholds: rotated AABBs include
# large empty regions for these non-rectangular tools. The final authority for
# collision-free reset is exact PhysX-collider settling and horizontal drift.
RECOMMENDED_WALL_CLEARANCE_XY = 0.020
RECOMMENDED_PAIR_AISLE_XY = 0.040

# Conservative inset of the measured blue-bin outer bounds
# x[-0.2458,0.2541], y[-0.1794,0.1767], z[0,0.2776].  A CUDA contact probe
# places a settled screwdriver's lowest rotated OBB corner at local z ~=
# 0.00285 m.  The strict z > 0 test uses the measured shell bottom and accepts
# physical floor contact without permitting any corner below the bin.
BLUE_INTERIOR_MIN = (-0.205, -0.135, 0.0)
BLUE_INTERIOR_MAX = (0.205, 0.135, 0.235)


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by wxyz quaternions with arbitrary leading dimensions."""
    xyz = quat[..., 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + quat[..., :1] * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    conjugate = torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)
    return quat_apply(conjugate, vec)


def box_corners(bounds_min: torch.Tensor, bounds_max: torch.Tensor) -> torch.Tensor:
    """Return eight corners for batched AABBs, shape (N, 8, 3)."""
    bits = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        device=bounds_min.device,
        dtype=bounds_min.dtype,
    )
    return bounds_min[:, None, :] + bits[None, :, :] * (bounds_max - bounds_min)[:, None, :]


def placement_components(
    tool_pos_w: torch.Tensor,
    tool_quat_w: torch.Tensor,
    tool_bounds_min: torch.Tensor,
    tool_bounds_max: torch.Tensor,
    blue_pos_w: torch.Tensor,
    blue_quat_w: torch.Tensor,
    linear_velocity_w: torch.Tensor,
    angular_velocity_w: torch.Tensor,
    gripper_opening: torch.Tensor,
    tcp_pos_w: torch.Tensor,
    *,
    max_linear_speed: float = 0.05,
    max_angular_speed: float = 0.50,
    min_gripper_opening: float = 0.055,
    min_tcp_separation: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute inside, settled and released booleans without simulator dependencies."""
    corners_tool = box_corners(tool_bounds_min, tool_bounds_max)
    q_tool = tool_quat_w[:, None, :].expand(-1, corners_tool.shape[1], -1)
    corners_world = tool_pos_w[:, None, :] + quat_apply(q_tool, corners_tool)
    q_blue = blue_quat_w[:, None, :].expand(-1, corners_tool.shape[1], -1)
    corners_blue = quat_apply_inverse(q_blue, corners_world - blue_pos_w[:, None, :])

    interior_min = torch.tensor(BLUE_INTERIOR_MIN, device=tool_pos_w.device, dtype=tool_pos_w.dtype)
    interior_max = torch.tensor(BLUE_INTERIOR_MAX, device=tool_pos_w.device, dtype=tool_pos_w.dtype)
    inside = torch.all((corners_blue > interior_min) & (corners_blue < interior_max), dim=(1, 2))
    settled = (torch.linalg.vector_norm(linear_velocity_w, dim=-1) < max_linear_speed) & (
        torch.linalg.vector_norm(angular_velocity_w, dim=-1) < max_angular_speed
    )
    released = (gripper_opening > min_gripper_opening) & (
        torch.linalg.vector_norm(tool_pos_w - tcp_pos_w, dim=-1) > min_tcp_separation
    )
    return inside, settled, released
