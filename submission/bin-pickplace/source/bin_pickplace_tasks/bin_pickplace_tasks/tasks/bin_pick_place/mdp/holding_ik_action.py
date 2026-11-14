"""End-effector pose-holding action for manual teleoperation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction


ZERO_COMMAND_DEADBAND = 1.0e-7


class HoldLastPoseDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Apply standard relative motion while holding the last target on zero input.

    The standard Isaac Lab relative IK action uses the measured end-effector pose
    as the reference at every control step. Under load, a small compliant drop can
    therefore become the next target when the command is zero, producing gradual
    target drift. This action changes only the zero-command behavior:

    * nonzero input still uses ``current pose + relative increment``;
    * zero input retains the previous target so the joint controller resists load;
    * after an environment reset, the first target is initialized from the actual pose.

    The public action dimension and scale remain the same: 6D relative SE(3).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._target_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        previous_position_target = self._ik_controller.ee_pos_des.clone()
        previous_orientation_target = self._ik_controller.ee_quat_des.clone()

        # Preserve upstream scaling, clipping, and current-pose-plus-delta behavior.
        super().process_actions(actions)

        stationary = self._target_initialized & torch.all(
            torch.abs(self._processed_actions) <= ZERO_COMMAND_DEADBAND, dim=1
        )
        self._ik_controller.ee_pos_des[stationary] = previous_position_target[stationary]
        self._ik_controller.ee_quat_des[stationary] = previous_orientation_target[stationary]
        self._target_initialized[:] = True

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._processed_actions[env_ids] = 0.0
        self._target_initialized[env_ids] = False
        self._ik_controller.reset(env_ids)
