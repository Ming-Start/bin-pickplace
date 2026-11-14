"""Launch one deterministic environment and verify its public contract.

Example::

    python scripts/smoke_test.py --headless --enable_cameras --steps 4
"""

from __future__ import annotations

import argparse
import math
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Bin-PickPlace-Franka-Play-v0")
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--capture-dir", type=Path, help="Optionally save one PNG from each camera for visual QA")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    parser.error("camera contract validation requires --enable_cameras")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from pxr import PhysxSchema, Usd, UsdPhysics

import bin_pickplace_tasks  # noqa: F401, E402  # register Gym environments
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from bin_pickplace_tasks.tasks.bin_pick_place.mdp.events import (  # noqa: E402
    CANONICAL_INSTRUCTION,
    TOOL_DISPLAY_NAMES,
    TOOL_NAMES,
    TOOL_RESET_POSES,
)
from bin_pickplace_tasks.tasks.bin_pick_place.mdp.geometry import (  # noqa: E402
    TOOL_BOUNDS_MAX,
    TOOL_BOUNDS_MIN,
    WHITE_INTERIOR_MAX,
    WHITE_INTERIOR_MIN,
    box_corners,
    quat_apply,
    quat_apply_inverse,
)
from bin_pickplace_tasks.tasks.bin_pick_place.config.franka.bin_pick_place_env_cfg import TOOL_MASSES_KG  # noqa: E402


CAMERA_NAMES = ("external_camera_1", "external_camera_2", "wrist_camera")
EXPECTED_RGB_SHAPE = (1, 480, 640, 3)
EXPECTED_POLICY_SHAPES = {
    "joint_positions": (1, 9),
    "joint_velocities": (1, 9),
    "end_effector_pose": (1, 7),
    "gripper_state": (1, 2),
    "tool_poses": (1, 28),
    "bin_bounds": (1, 26),
    "target_relative_ee": (1, 3),
    "target_relative_blue": (1, 3),
    "target_tool_one_hot": (1, 4),
    "previous_action": (1, 7),
    "external_camera_1_rgb": EXPECTED_RGB_SHAPE,
    "external_camera_2_rgb": EXPECTED_RGB_SHAPE,
    "wrist_camera_rgb": EXPECTED_RGB_SHAPE,
}


def validate_collision_assets(cfg) -> None:
    """Verify that the environment points at the intended hybrid colliders."""
    screwdriver_path = Path(cfg.scene.screwdriver.spawn.usd_path)
    wrench_path = Path(cfg.scene.wrench.spawn.usd_path)
    assert screwdriver_path.name == "screwdriver_hybrid_physics.usd"
    assert wrench_path.name == "wrench_sdf_physics.usd"
    assert screwdriver_path.is_file(), f"missing hybrid screwdriver asset: {screwdriver_path}"
    assert wrench_path.is_file(), f"missing SDF wrench asset: {wrench_path}"

    screwdriver_stage = Usd.Stage.Open(str(screwdriver_path))
    wrench_stage = Usd.Stage.Open(str(wrench_path))
    assert screwdriver_stage is not None and wrench_stage is not None
    screwdriver_visual = screwdriver_stage.GetPrimAtPath("/screwdriver/geometry/mesh")
    shaft = screwdriver_stage.GetPrimAtPath("/screwdriver/hybrid_colliders/shaft_capsule")
    handle = screwdriver_stage.GetPrimAtPath("/screwdriver/hybrid_colliders/handle_convex")
    assert not screwdriver_visual.HasAPI(UsdPhysics.CollisionAPI)
    assert shaft.HasAPI(UsdPhysics.CollisionAPI)
    assert handle.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.MeshCollisionAPI(handle).GetApproximationAttr().Get() == UsdPhysics.Tokens.convexHull

    wrench_mesh = wrench_stage.GetPrimAtPath("/wrench/geometry/mesh")
    assert wrench_mesh.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.MeshCollisionAPI(wrench_mesh).GetApproximationAttr().Get() == PhysxSchema.Tokens.sdf
    sdf = PhysxSchema.PhysxSDFMeshCollisionAPI(wrench_mesh)
    assert sdf.GetSdfResolutionAttr().Get() == 256
    assert sdf.GetSdfSubgridResolutionAttr().Get() == 6


def main() -> None:
    env = None
    try:
        print("[smoke] parsing configuration", flush=True)
        cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        assert cfg.scene.num_envs == 1, f"PLAY smoke test requires one environment, got {cfg.scene.num_envs}"
        assert cfg.events.reset_task.params["deterministic"] is True, "PLAY reset must be deterministic"
        assert math.isclose(cfg.decimation * cfg.sim.dt, 1.0 / 30.0), "control rate must be 30 Hz"
        assert cfg.sim.render_interval == cfg.decimation, "camera render must align with each control step"
        assert math.isclose(cfg.sim.dt, 1.0 / 120.0), "contact-robust physics must run at 120 Hz"
        assert cfg.decimation == 4, "30 Hz control must use four 120 Hz physics substeps"
        expected_ccd = not str(cfg.sim.device).startswith("cuda")
        assert cfg.sim.physx.enable_ccd is expected_ccd, (
            "CCD must be enabled on supported CPU PhysX and disabled on the unsupported GPU pipeline"
        )
        robot_spawn = cfg.scene.robot.spawn
        assert robot_spawn.articulation_props.solver_position_iteration_count == 32
        assert robot_spawn.articulation_props.solver_velocity_iteration_count == 4
        assert robot_spawn.rigid_props.solver_position_iteration_count == 32
        assert robot_spawn.rigid_props.solver_velocity_iteration_count == 4
        assert cfg.scene.robot.actuators["panda_shoulder"].stiffness == 800.0
        assert cfg.scene.robot.actuators["panda_forearm"].stiffness == 800.0
        assert cfg.scene.robot.actuators["panda_shoulder"].damping == 80.0
        assert cfg.scene.robot.actuators["panda_forearm"].damping == 80.0
        # Keep realistic Franka torque limits. Load holding comes from target retention
        # and controller stiffness, not from unrealistically increasing joint effort.
        assert cfg.scene.robot.actuators["panda_shoulder"].effort_limit_sim == 87.0
        assert cfg.scene.robot.actuators["panda_forearm"].effort_limit_sim == 12.0
        assert cfg.actions.arm_action.class_type.__name__ == "HoldLastPoseDifferentialIKAction"
        for tool_name in TOOL_NAMES:
            tool_spawn = getattr(cfg.scene, tool_name).spawn
            assert math.isclose(tool_spawn.mass_props.mass, TOOL_MASSES_KG[tool_name])
            assert tool_spawn.rigid_props.solver_position_iteration_count == 16
            assert tool_spawn.rigid_props.solver_velocity_iteration_count == 4
            assert math.isclose(tool_spawn.collision_props.contact_offset, 0.003)
            assert math.isclose(tool_spawn.collision_props.rest_offset, 0.0)
        validate_collision_assets(cfg)
        teleop_multiplier = float(os.environ.get("BIN_PICKPLACE_TELEOP_SENSITIVITY", "1.0"))
        keyboard_cfg = cfg.teleop_devices.devices["keyboard"]
        assert math.isclose(keyboard_cfg.pos_sensitivity, 0.30 * teleop_multiplier), (
            f"keyboard translation sensitivity did not receive multiplier x{teleop_multiplier}: "
            f"got {keyboard_cfg.pos_sensitivity}"
        )
        assert math.isclose(keyboard_cfg.rot_sensitivity, 0.25 * teleop_multiplier), (
            f"keyboard rotation sensitivity did not receive multiplier x{teleop_multiplier}: "
            f"got {keyboard_cfg.rot_sensitivity}"
        )

        print("[smoke] constructing environment", flush=True)
        env = gym.make(args.task, cfg=cfg).unwrapped
        print("[smoke] resetting environment", flush=True)
        observations, _ = env.reset()
        reset_instructions = [env.language_instructions[0]]
        for _ in range(3):
            observations, _ = env.reset()
            reset_instructions.append(env.language_instructions[0])
        expected_instructions = {
            CANONICAL_INSTRUCTION.format(tool_name=tool_name) for tool_name in TOOL_DISPLAY_NAMES
        }
        assert set(reset_instructions) == expected_instructions, (
            f"four deterministic PLAY resets must cover every target exactly once: {reset_instructions}"
        )
        assert env.action_manager.total_action_dim == 7, (
            f"expected [relative SE(3), gripper] = 7 actions, got {env.action_manager.total_action_dim}"
        )

        action = torch.zeros((1, 7), device=env.device)
        print(f"[smoke] stepping {args.steps} control frames", flush=True)
        for step in range(args.steps):
            observations, _, terminated, truncated, _ = env.step(action)
            assert not bool(terminated.any().item()), f"unexpected terminal reset at step {step + 1}"
            assert not bool(truncated.any().item()), f"unexpected timeout at step {step + 1}"

        print("[smoke] checking reset geometry and stability", flush=True)
        # A simulation-level reset invariant: after settling, every transformed
        # tool OBB remains inside the white bin's conservative local interior.
        white = env.scene["white_bin"]
        interior_min = torch.tensor(WHITE_INTERIOR_MIN, device=env.device)
        interior_max = torch.tensor(WHITE_INTERIOR_MAX, device=env.device)
        tool_states = []
        for index, tool_name in enumerate(TOOL_NAMES):
            tool = env.scene[tool_name]
            bounds_min = torch.tensor([TOOL_BOUNDS_MIN[index]], device=env.device)
            bounds_max = torch.tensor([TOOL_BOUNDS_MAX[index]], device=env.device)
            corners = box_corners(bounds_min, bounds_max)
            corners_world = tool.data.root_pos_w[:, None, :] + quat_apply(
                tool.data.root_quat_w[:, None, :].expand(-1, 8, -1), corners
            )
            corners_white = quat_apply_inverse(
                white.data.root_quat_w[:, None, :].expand(-1, 8, -1),
                corners_world - white.data.root_pos_w[:, None, :],
            )
            assert bool(torch.all((corners_white > interior_min) & (corners_white < interior_max)).item()), (
                f"{tool_name} escaped the white-bin interior after {args.steps} steps: "
                f"min={corners_white.amin(dim=1).tolist()}, max={corners_white.amax(dim=1).tolist()}"
            )
            linear_speed = torch.linalg.vector_norm(tool.data.root_lin_vel_w, dim=-1)
            angular_speed = torch.linalg.vector_norm(tool.data.root_ang_vel_w, dim=-1)
            assert bool(torch.all(linear_speed < 0.05).item()), f"{tool_name} is not settled: {linear_speed.tolist()}"
            assert bool(torch.all(angular_speed < 0.50).item()), f"{tool_name} is rotating: {angular_speed.tolist()}"
            expected_xy = torch.tensor(TOOL_RESET_POSES[tool_name][0][:2], device=env.device)
            horizontal_drift = torch.linalg.vector_norm(tool.data.root_pos_w[0, :2] - expected_xy)
            assert horizontal_drift.item() < 0.01, (
                f"{tool_name} drifted {horizontal_drift.item():.4f} m from its authored XY pose; "
                "the reset layout may contain a real collider overlap"
            )
            tool_states.append(
                f"{tool_name}: pos={tool.data.root_pos_w[0].tolist()}, "
                f"speed=({linear_speed[0].item():.4f}, {angular_speed[0].item():.4f}), "
                f"xy_drift={horizontal_drift.item():.5f}"
            )

        policy = observations["policy"]
        assert set(policy) == set(EXPECTED_POLICY_SHAPES), (
            f"unexpected policy observation keys: {sorted(policy)}"
        )
        for term_name, expected_shape in EXPECTED_POLICY_SHAPES.items():
            assert tuple(policy[term_name].shape) == expected_shape, (
                f"{term_name}: expected {expected_shape}, got {tuple(policy[term_name].shape)}"
            )
        for camera_name in CAMERA_NAMES:
            image = policy[f"{camera_name}_rgb"]
            assert tuple(image.shape) == EXPECTED_RGB_SHAPE, (
                f"{camera_name}: expected {EXPECTED_RGB_SHAPE}, got {tuple(image.shape)}"
            )
            assert image.dtype == torch.uint8, f"{camera_name}: expected uint8 RGB, got {image.dtype}"
            assert image.float().std().item() > 1.0, f"{camera_name}: image is effectively uniform"

            if args.capture_dir is not None:
                from PIL import Image

                args.capture_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image[0].cpu().numpy()).save(args.capture_dir / f"{camera_name}.png")

        frames = {name: int(env.scene[name].frame[0].item()) for name in CAMERA_NAMES}
        assert len(set(frames.values())) == 1, f"camera frame counters are not synchronized: {frames}"

        print("Smoke test passed", flush=True)
        print(f"  task: {args.task}", flush=True)
        print(f"  action dimension: {env.action_manager.total_action_dim}", flush=True)
        print(
            f"  PhysX contacts: 120 Hz, CCD={'enabled' if expected_ccd else 'GPU-unsupported (substeps active)'}, "
            "Franka solver=32/4, tool contact offset=0.003 m",
            flush=True,
        )
        print("  collision assets: screwdriver=capsule+convex hull, wrench=sparse SDF 256/6", flush=True)
        print(
            f"  keyboard sensitivity: translation={keyboard_cfg.pos_sensitivity:.3f}, "
            f"rotation={keyboard_cfg.rot_sensitivity:.3f} (multiplier x{teleop_multiplier:g})",
            flush=True,
        )
        print(f"  RGB shape: {EXPECTED_RGB_SHAPE}", flush=True)
        print(f"  synchronized camera frames: {frames}", flush=True)
        print(f"  deterministic target cycle: {reset_instructions}", flush=True)
        print("  settled reset tools:", flush=True)
        for state in tool_states:
            print(f"    {state}", flush=True)
        if args.capture_dir is not None:
            print(f"  camera captures: {args.capture_dir.resolve()}", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException as error:
        print(f"Smoke test failed: {error!r}", flush=True)
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    if exit_code:
        # Kit may replace a pending SystemExit while shutting down. Force a
        # machine-readable non-zero result after it has released the GPU.
        os._exit(exit_code)
