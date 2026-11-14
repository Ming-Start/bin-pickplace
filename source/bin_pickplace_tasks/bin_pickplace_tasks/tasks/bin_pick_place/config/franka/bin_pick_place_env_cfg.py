"""Franka configuration for the Bin-PickPlace bin pick-and-place environment."""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices import DevicesCfg, Se3KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from ... import mdp

ASSET_ROOT = Path(__file__).resolve().parents[4] / "assets"
CAMERA_RATE_HZ = 30.0
CAMERA_UPDATE_PERIOD_S = 1.0 / CAMERA_RATE_HZ
TELEOP_BASE_POS_SENSITIVITY = 0.30
TELEOP_BASE_ROT_SENSITIVITY = 0.25
ASSET_CONTACT_OFFSET_M = 0.003
ROBOT_SOLVER_POSITION_ITERATIONS = 32
ROBOT_SOLVER_VELOCITY_ITERATIONS = 4
ARM_HOLD_STIFFNESS = 800.0
ARM_HOLD_DAMPING = 80.0
TOOL_MASSES_KG = {
    "screwdriver": 0.12,
    "wrench": 0.25,
    "hammer": 0.65,
    "pallet_knife": 0.10,
}


def asset_path(folder: str, filename: str) -> str:
    return str(ASSET_ROOT / folder / filename)


def franka_robot_cfg():
    """Return a contact-robust Franka configuration without modifying Isaac Lab's stock asset."""
    cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # The heaviest configured tool is the 0.65 kg hammer.  Keep Franka's
    # physical 87/12 N·m effort limits, but double the task-space arm joint
    # stiffness so the load produces only a small static deflection.  Damping
    # stays high to avoid oscillation during manual stop/start motion.
    for actuator_name in ("panda_shoulder", "panda_forearm"):
        cfg.actuators[actuator_name].stiffness = ARM_HOLD_STIFFNESS
        cfg.actuators[actuator_name].damping = ARM_HOLD_DAMPING
    # The stock high-PD Franka uses articulation solver iterations 8/0.  A
    # non-zero velocity pass and additional position passes make the stiff
    # fingertips much less likely to push through thin tool colliders.
    cfg.spawn.articulation_props.solver_position_iteration_count = ROBOT_SOLVER_POSITION_ITERATIONS
    cfg.spawn.articulation_props.solver_velocity_iteration_count = ROBOT_SOLVER_VELOCITY_ITERATIONS
    cfg.spawn.rigid_props.solver_position_iteration_count = ROBOT_SOLVER_POSITION_ITERATIONS
    cfg.spawn.rigid_props.solver_velocity_iteration_count = ROBOT_SOLVER_VELOCITY_ITERATIONS
    cfg.spawn.rigid_props.max_depenetration_velocity = 1.0
    # The stock Franka's collision meshes are instanceable, so collider-level
    # offsets cannot be safely overwritten from the external project.  The
    # pair contact envelope is enlarged on the supplied tool side instead.
    return cfg


def rigid_asset(
    name: str,
    folder: str,
    filename: str,
    pos,
    mass: float,
    kinematic: bool = False,
    visual_color: tuple[float, float, float] | None = None,
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    scale: tuple[float, float, float] | None = None,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=UsdFileCfg(
            usd_path=asset_path(folder, filename),
            scale=scale,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=4,
                max_depenetration_velocity=1.0, disable_gravity=kinematic,
                kinematic_enabled=kinematic, enable_gyroscopic_forces=True,
            ),
            collision_props=CollisionPropertiesCfg(contact_offset=ASSET_CONTACT_OFFSET_M, rest_offset=0.0),
            mass_props=MassPropertiesCfg(mass=mass),
            visual_material=(
                sim_utils.PreviewSurfaceCfg(diffuse_color=visual_color, roughness=0.8)
                if visual_color is not None
                else None
            ),
        ),
    )


@configclass
class BinPickPlaceSceneCfg(InteractiveSceneCfg):
    robot = franka_robot_cfg()
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.7071068, 0.0, 0.0, 0.7071068)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    plane = AssetBaseCfg(prim_path="/World/Ground", init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05)), spawn=GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)))

    # All packaged rigid assets use a z-at-bottom root. The Seattle tabletop
    # is z=0, so 5 mm keeps the bin bottoms just above the support surface.
    # The supplied source bin is only 0.298 x 0.141 m. Scaling its horizontal
    # footprint by 2.4 (while preserving wall height) gives the four supplied
    # tools both disjoint reset slots and clearance for a vertical finger
    # approach, without replacing the required asset.
    white_bin = rigid_asset(
        "WhiteBin",
        "N01705 Bin-IsaacSim",
        "white_bin_physics.usd",
        (0.55, -0.30, 0.005),
        0.4,
        True,
        visual_color=(0.90, 0.90, 0.90),
        scale=(2.4, 2.4, 1.0),
    )
    blue_bin = rigid_asset(
        "BlueBin", "A03071 Blue Stackable Plastic Bin 40L", "blue_bin_physics.usd", (0.55, 0.32, 0.005), 2.0, True
    )

    # Single-layer poses authored interactively in the full simulator scene.
    # Keep these defaults exactly synchronized with mdp.events.TOOL_RESET_POSES.
    hammer = rigid_asset(
        "Hammer",
        "A00040 Hammer-IsaacSim",
        "hammer_physics.usd",
        (0.399235, -0.350000, 0.020),
        TOOL_MASSES_KG["hammer"],
    )
    screwdriver = rigid_asset(
        "Screwdriver",
        "A00038 Screwdriver-IsaacSim",
        # A thin shaft needs an analytic collider to avoid tunnelling while
        # the wider handle remains a cheap convex hull.  The visual mesh is
        # unchanged and does not participate in collision.
        "screwdriver_hybrid_physics.usd",
        (0.640262, -0.310392, 0.020),
        TOOL_MASSES_KG["screwdriver"],
        rot=(-0.6720588, 0.0, 0.0, 0.7404978),
    )
    wrench = rigid_asset(
        "Wrench",
        "A00039 Wrench-IsaacSim",
        # The open jaw is concave and important for finger contact, so use a
        # sparse signed-distance field instead of filling it with convex hulls.
        "wrench_sdf_physics.usd",
        (0.469094, -0.243457, 0.020),
        TOOL_MASSES_KG["wrench"],
        rot=(0.9429091, 0.0, 0.0, -0.3330501),
    )
    pallet_knife = rigid_asset(
        "PalletKnife",
        "N00838 Sm Paint Supplies Palletknife Simready-IsaacSim",
        "pallet_knife_physics.usd",
        (0.759419, -0.320000, 0.020),
        TOOL_MASSES_KG["pallet_knife"],
        rot=(0.9225082, 0.0, 0.0, -0.3859775),
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_hand", name="tcp", offset=OffsetCfg(pos=(0.0, 0.0, 0.107)))],
    )
    external_camera_1 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ExternalCamera1",
        update_period=CAMERA_UPDATE_PERIOD_S,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.05, 5.0)),
        # ROS camera +Z looks at the centre of the two-bin transfer workspace.
        offset=CameraCfg.OffsetCfg(
            pos=(1.65, -1.55, 1.20),
            rot=(0.503149, -0.797526, -0.281514, 0.177604),
            convention="ros",
        ),
    )
    external_camera_2 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ExternalCamera2",
        update_period=CAMERA_UPDATE_PERIOD_S,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.05, 5.0)),
        # Opposite/elevated view; ROS camera +Z targets the same workspace centre.
        offset=CameraCfg.OffsetCfg(
            pos=(-0.75, 1.55, 1.25),
            rot=(0.168949, -0.275852, 0.806920, -0.494209),
            convention="ros",
        ),
    )
    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
        update_period=CAMERA_UPDATE_PERIOD_S,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.03, 2.0)),
        # Isaac Lab's validated Franka wrist-camera mount: it looks along the
        # hand/TCP +Z direction with a small inward pitch toward the fingers.
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15),
            rot=(-0.70614, 0.03701, 0.03701, -0.70614),
            convention="ros",
        ),
    )


@configclass
class ActionsCfg:
    arm_action = DifferentialInverseKinematicsActionCfg(
        class_type=mdp.HoldLastPoseDifferentialIKAction,
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=(0.04, 0.04, 0.04, 0.15, 0.15, 0.15),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04}, close_command_expr={"panda_finger_joint.*": 0.0},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_positions = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_velocities = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        end_effector_pose = ObsTerm(func=mdp.end_effector_pose)
        gripper_state = ObsTerm(func=mdp.gripper_state)
        tool_poses = ObsTerm(func=mdp.tool_poses)
        bin_bounds = ObsTerm(func=mdp.bin_bounds)
        target_relative_ee = ObsTerm(func=mdp.target_relative_ee)
        target_relative_blue = ObsTerm(func=mdp.target_relative_blue)
        target_tool_one_hot = ObsTerm(func=mdp.instruction_token)
        previous_action = ObsTerm(func=mdp.last_action)
        external_camera_1_rgb = ObsTerm(func=mdp.camera_rgb, params={"sensor_name": "external_camera_1"})
        external_camera_2_rgb = ObsTerm(func=mdp.camera_rgb, params={"sensor_name": "external_camera_2"})
        wrist_camera_rgb = ObsTerm(func=mdp.camera_rgb, params={"sensor_name": "wrist_camera"})

        def __post_init__(self):
            self.concatenate_terms = False
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_task = EventTerm(func=mdp.reset_bin_pick_place, mode="reset", params={"deterministic": False})


@configclass
class TerminationsCfg:
    success = DoneTerm(func=mdp.target_placed)
    failure = DoneTerm(func=mdp.target_dropped)
    timeout = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RewardsCfg:
    """Intentionally empty: this homework does not claim RL readiness."""


@configclass
class FrankaBinPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: BinPickPlaceSceneCfg = BinPickPlaceSceneCfg(num_envs=4, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        self.seed = 42
        # Four 120 Hz PhysX substeps per 30 Hz action reduce tunneling when the
        # high-PD gripper approaches thin handles.  Camera/action timing stays
        # at the assignment's documented 30 Hz rate.
        self.decimation = 4
        self.episode_length_s = 120.0
        self.sim.dt = 1 / 120
        # Isaac Sim 5.1 does not support CCD in its GPU dynamics pipeline.
        # Enable it for CPU PhysX; CUDA uses the four 120 Hz substeps above as
        # the supported anti-tunnelling mechanism.
        self.sim.physx.enable_ccd = not str(self.sim.device).startswith("cuda")
        # One render and one camera sample per 30 Hz control step. All three
        # cameras share this schedule, so policy images and actions are aligned.
        self.sim.render_interval = self.decimation
        self.num_rerenders_on_reset = 2
        self.viewer.eye = (1.6, -1.5, 1.2)
        self.viewer.lookat = (0.5, 0.0, 0.25)
        # The named keyboard device bypasses Isaac Lab's fallback sensitivity.
        # scripts/teleop.py therefore passes its multiplier through this
        # environment variable.  At x1, 0.30 * 0.04 gives 0.012 m per 30 Hz
        # control step (about 0.36 m/s); x2 gives about 0.72 m/s.  Users can
        # lower the multiplier for final grasp alignment without changing the
        # environment's 7D action contract.
        sensitivity_multiplier = float(os.environ.get("BIN_PICKPLACE_TELEOP_SENSITIVITY", "1.0"))
        if not 0.0 < sensitivity_multiplier <= 5.0:
            raise ValueError("BIN_PICKPLACE_TELEOP_SENSITIVITY must be in (0, 5]")
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=TELEOP_BASE_POS_SENSITIVITY * sensitivity_multiplier,
                    rot_sensitivity=TELEOP_BASE_ROT_SENSITIVITY * sensitivity_multiplier,
                    sim_device=self.sim.device,
                )
            }
        )


@configclass
class FrankaBinPickPlaceEnvCfg_PLAY(FrankaBinPickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.events.reset_task.params["deterministic"] = True
