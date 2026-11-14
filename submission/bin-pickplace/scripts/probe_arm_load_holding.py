"""Test Franka pose holding under an equivalent hammer load and zero teleop input.

The test applies the weight of the heaviest tool (the 0.65 kg hammer) directly to
``panda_hand``. This isolates arm joint holding from gripper friction and tool contact.

Example:

    python scripts/probe_arm_load_holding.py --headless --enable_cameras
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Bin-PickPlace-Franka-Play-v0")
parser.add_argument("--steps", type=int, default=180, help="30 Hz control steps after applying the load")
parser.add_argument("--load-mass", type=float, default=0.65, help="equivalent end-effector payload in kg")
parser.add_argument("--lever-arm", type=float, default=0.15, help="horizontal payload lever arm in meters")
parser.add_argument("--max-drop", type=float, default=None, help="optional maximum permitted TCP drop in meters")
parser.add_argument("--max-angle", type=float, default=None, help="optional maximum orientation error in radians")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    parser.error("This environment has camera observations; add --enable_cameras")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import bin_pickplace_tasks  # noqa: F401, E402  # register Gym environments
from isaaclab.utils.math import quat_error_magnitude  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


GRAVITY_M_S2 = 9.81


def main() -> None:
    env = None
    try:
        cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        # Isolate controller holding by disabling task success/failure resets.
        cfg.terminations = None
        env = gym.make(args.task, cfg=cfg).unwrapped
        env.reset()

        robot = env.scene["robot"]
        arm_action = env.action_manager.get_term("arm_action")
        hand_ids, hand_names = robot.find_bodies("panda_hand")
        if len(hand_ids) != 1:
            raise RuntimeError(f"Expected one panda_hand body, found {hand_names}")

        action = torch.zeros((1, 7), device=env.device)
        action[:, -1] = 1.0  # Keep the gripper open to exclude contact effects.
        for _ in range(15):
            env.step(action)

        initial_pos, initial_quat = arm_action._compute_frame_pose()
        initial_joint_pos = robot.data.joint_pos[0, :7].clone()
        initial_z = float(initial_pos[0, 2].item())

        load_force_n = args.load_mass * GRAVITY_M_S2
        forces = torch.tensor([[[0.0, 0.0, -load_force_n]]], device=env.device)
        load_torque_nm = load_force_n * args.lever_arm
        # Approximate a lateral hammer grasp with a +X center-of-mass offset; r x F gives +Y torque.
        torques = torch.tensor([[[0.0, load_torque_nm, 0.0]]], device=env.device)
        robot.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            body_ids=hand_ids,
            is_global=True,
        )

        z_samples = []
        angle_samples = []
        desired_z_samples = []
        for _ in range(args.steps):
            env.step(action)
            current_pos, current_quat = arm_action._compute_frame_pose()
            z_samples.append(float(current_pos[0, 2].item()))
            angle_samples.append(float(quat_error_magnitude(initial_quat, current_quat)[0].item()))
            desired_z_samples.append(float(arm_action._ik_controller.ee_pos_des[0, 2].item()))

        final_joint_pos = robot.data.joint_pos[0, :7]
        min_z = min(z_samples)
        final_z = z_samples[-1]
        max_drop = initial_z - min_z
        final_drop = initial_z - final_z
        target_drop = initial_z - desired_z_samples[-1]
        max_angle = max(angle_samples)
        max_joint_drift = float(torch.max(torch.abs(final_joint_pos - initial_joint_pos)).item())
        tail_count = min(30, len(z_samples))
        tail_creep = max(z_samples[-tail_count:]) - min(z_samples[-tail_count:])

        print("\nFranka joint-holding load test")
        print(f"Equivalent payload: {args.load_mass:.3f} kg ({load_force_n:.3f} N downward)")
        print(f"Equivalent offset torque: {load_torque_nm:.3f} N m (lever arm {args.lever_arm:.3f} m)")
        print(f"Test duration: {args.steps / 30.0:.2f} s ({args.steps} control steps at 30 Hz)")
        print(f"Initial TCP height: {initial_z:.6f} m")
        print(f"Minimum TCP height: {min_z:.6f} m")
        print(f"Maximum TCP drop: {max_drop * 1000.0:.3f} mm")
        print(f"Final TCP drop: {final_drop * 1000.0:.3f} mm")
        print(f"Controller target drop: {target_drop * 1000.0:.3f} mm")
        print(f"Maximum end-effector orientation error: {max_angle:.6f} rad")
        print(f"Height range during final second: {tail_creep * 1000.0:.3f} mm")
        print(f"Maximum joint-position drift: {max_joint_drift:.6f} rad")

        if args.max_drop is not None and max_drop > args.max_drop:
            raise AssertionError(
                f"TCP drop {max_drop:.6f} m exceeds the limit {args.max_drop:.6f} m"
            )
        if args.max_angle is not None and max_angle > args.max_angle:
            raise AssertionError(
                f"End-effector orientation error {max_angle:.6f} rad exceeds "
                f"the limit {args.max_angle:.6f} rad"
            )
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
