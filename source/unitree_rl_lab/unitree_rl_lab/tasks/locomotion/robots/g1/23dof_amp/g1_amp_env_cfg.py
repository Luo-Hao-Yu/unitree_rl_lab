from __future__ import annotations

from pathlib import Path

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

# Per-joint median of ``dof_positions`` over every frame of all 30 strictly
# validated G1-23DoF AMP clips, stored in ``G1_23DOF_POLICY_JOINT_NAMES``
# order. This is one static residual-action center, never a time-varying
# reference target.
G1_23DOF_AMP_REFERENCE_CENTER = (
    -0.053957726806402206,
    0.021534139290452003,
    0.11950450390577316,
    0.33395761251449585,
    -0.29004335403442383,
    0.0040415311232209206,
    -0.10638310760259628,
    -0.06431636214256287,
    -0.07071777433156967,
    0.3901515007019043,
    -0.20892232656478882,
    0.10049572587013245,
    -0.05345692113041878,
    0.003766000736504793,
    0.25450289249420166,
    -0.2478148639202118,
    1.1357628107070923,
    -0.010694418102502823,
    0.027017267420887947,
    -0.27165064215660095,
    0.2043786346912384,
    1.1201633214950562,
    0.19554904103279114,
)

# Validated legged_lab G1-29DoF -> current G1-23DoF conversion. Keep this
# explicit so AMP training cannot silently fall back to the official Humanoid
# reference motion or to an unvalidated file.
G1_23DOF_AMP_MOTION_FILE = str(
    Path(__file__).resolve().parents[8] / "data/motions/g1_23dof_amp/walk_b13_turn_right_45.npz"
)
G1_23DOF_AMP_MULTIMOTION_MANIFEST = str(
    Path(__file__).resolve().parents[8]
    / "data/motions/g1_23dof_amp/all30/g1_23dof_amp_multimotion_manifest.json"
)


@configclass
class G1AmpEnvCfg(DirectRLEnvCfg):
    """G1-23DoF Direct AMP environment using the validated G1 reference motion."""

    # environment
    episode_length_s = 20.0
    decimation = 4
    action_space = 23
    observation_space = 78
    state_space = 0

    # AMP interface: 71 features per frame, two consecutive frames for the discriminator.
    num_amp_observations = 2
    amp_observation_space = 71

    # Preserve 23-D residual position actions: q_target = q_reference_center + 0.35 * action.
    action_scale = 0.35
    action_offset = G1_23DOF_AMP_REFERENCE_CENTER
    policy_joint_names = G1_23DOF_POLICY_JOINT_NAMES

    # Fallback command for the single-clip compatibility path. With the all-30
    # manifest, RSI replaces it by the selected reference state's body-frame
    # [vx, vy, yaw_rate], while policy observation size remains 78.
    velocity_command = (0.5, 0.0, 0.0)
    command_from_reference_state = True
    linear_velocity_tracking_sigma = 0.25
    yaw_rate_tracking_sigma = 0.25
    yaw_rate_tracking_weight = 0.5

    # AMP body mapping, verified from the current G1 articulation.
    reference_body = "torso_link"
    motion_root_body = "pelvis"
    key_body_names = G1_AMP_KEY_BODY_NAMES

    # The official 28-DoF Humanoid motion must never be used for this task.
    # The validated all-30 manifest samples category -> independent clip ->
    # valid time from update 0. No curriculum or concatenated trajectory is used.
    # Set motion_manifest=None to preserve the original single-file behavior.
    motion_manifest: str | None = G1_23DOF_AMP_MULTIMOTION_MANIFEST
    motion_file: str | None = G1_23DOF_AMP_MOTION_FILE
    # Reference State Initialization (RSI): sample one valid walking state at
    # reset only. This does not turn the residual controller into a tracker.
    reset_strategy = "rsi"  # supported: default, rsi

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
