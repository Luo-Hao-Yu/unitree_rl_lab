from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG


# Fixed deployment-compatible policy order. Do not replace this with a wildcard.
G1_23DOF_POLICY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)

G1_AMP_KEY_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
)


@configclass
class G1AmpEnvCfg(DirectRLEnvCfg):
    """Skeleton configuration for a G1-23DoF Direct AMP environment.

    Reference-motion training is intentionally disabled until a motion file retargeted to the G1 skeleton is
    available. The environment therefore defaults to a normal standing reset and can be instantiated and stepped
    without pretending that the official 28-DoF Humanoid motion is compatible with G1.
    """

    # environment
    episode_length_s = 20.0
    decimation = 4
    action_space = 23
    observation_space = 78
    state_space = 0

    # AMP interface: 71 features per frame, two consecutive frames for the discriminator.
    num_amp_observations = 2
    amp_observation_space = 71

    # Keep the current PPO/deployment action semantics: q_target = q_default + 0.35 * action.
    action_scale = 0.35
    policy_joint_names = G1_23DOF_POLICY_JOINT_NAMES

    # Minimal fixed forward command used by the 78-D policy observation and task reward.
    velocity_command = (0.5, 0.0, 0.0)

    # AMP body mapping, verified from the current G1 articulation.
    reference_body = "torso_link"
    motion_root_body = "pelvis"
    key_body_names = G1_AMP_KEY_BODY_NAMES

    # Placeholder for a future G1-retargeted NPZ. The official Humanoid motion must not be used here.
    motion_file: str | None = None
    reset_strategy = "default"  # supported: default, random, random-start

    early_termination = True
    termination_height = 0.30

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # Reuse the validated G1 asset unchanged; joint selection is performed explicitly by G1AmpEnv.
    robot: ArticulationCfg = UNITREE_G1_23DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    ground_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
