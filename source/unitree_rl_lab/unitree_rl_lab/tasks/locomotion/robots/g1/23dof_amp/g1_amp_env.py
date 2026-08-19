from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab_tasks.direct.humanoid_amp.motions import MotionLoader

from .g1_amp_env_cfg import G1AmpEnvCfg
from .multi_motion_loader import MultiMotionLoader, MultiMotionSampleContext


def reference_state_command(
    reference_rotation_w: torch.Tensor,
    reference_linear_velocity_w: torch.Tensor,
    reference_angular_velocity_w: torch.Tensor,
) -> torch.Tensor:
    """Return the body-frame [vx, vy, yaw_rate] command of a reference state."""

    linear_velocity_b = quat_apply_inverse(reference_rotation_w, reference_linear_velocity_w)
    angular_velocity_b = quat_apply_inverse(reference_rotation_w, reference_angular_velocity_w)
    return torch.cat((linear_velocity_b[:, :2], angular_velocity_b[:, 2:3]), dim=-1)


class G1AmpEnv(DirectRLEnv):
    """Unitree G1-23DoF Direct environment exposing the standard skrl AMP interface."""

    cfg: G1AmpEnvCfg

    def __init__(self, cfg: G1AmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.policy_joint_ids, resolved_joint_names = self.robot.find_joints(
            list(self.cfg.policy_joint_names), preserve_order=True
        )
        self.policy_joint_names = tuple(resolved_joint_names)
        if self.policy_joint_names != tuple(self.cfg.policy_joint_names):
            raise RuntimeError(
                "G1 policy joint mapping mismatch. "
                f"Expected {tuple(self.cfg.policy_joint_names)}, resolved {self.policy_joint_names}."
            )
        if len(self.policy_joint_ids) != self.cfg.action_space:
            raise RuntimeError(
                f"Expected {self.cfg.action_space} policy joints, resolved {len(self.policy_joint_ids)}."
            )

        reference_body_ids, reference_body_names = self.robot.find_bodies([self.cfg.reference_body])
        key_body_ids, key_body_names = self.robot.find_bodies(list(self.cfg.key_body_names), preserve_order=True)
        if reference_body_names != [self.cfg.reference_body]:
            raise RuntimeError(
                f"G1 reference body mismatch: expected {self.cfg.reference_body}, resolved {reference_body_names}."
            )
        if tuple(key_body_names) != tuple(self.cfg.key_body_names):
            raise RuntimeError(
                f"G1 AMP key-body mapping mismatch. Expected {self.cfg.key_body_names}, resolved {key_body_names}."
            )
        self.reference_body_id = reference_body_ids[0]
        self.key_body_ids = key_body_ids
        self.key_body_names = tuple(key_body_names)

        self._default_policy_joint_pos = self.robot.data.default_joint_pos[:, self.policy_joint_ids].clone()
        action_offset = torch.as_tensor(self.cfg.action_offset, dtype=torch.float32, device=self.device)
        if action_offset.shape != (self.cfg.action_space,):
            raise ValueError(
                f"Expected static AMP action offset shape {(self.cfg.action_space,)}, got {tuple(action_offset.shape)}."
            )
        if not torch.all(torch.isfinite(action_offset)):
            raise ValueError("Static AMP action offset contains NaN/Inf.")
        joint_limits = self.robot.data.joint_pos_limits[0, self.policy_joint_ids]
        limit_violation = torch.maximum(joint_limits[:, 0] - action_offset, action_offset - joint_limits[:, 1])
        maximum_violation, violating_index = torch.max(limit_violation, dim=0)
        if maximum_violation.item() > 0.0:
            raise ValueError(
                "Static AMP action offset violates an articulation joint limit: "
                f"joint={self.policy_joint_names[violating_index.item()]}, violation={maximum_violation.item():.9g} rad."
            )
        self._action_offset = action_offset.unsqueeze(0).repeat(self.num_envs, 1)
        self._velocity_command = torch.tensor(
            self.cfg.velocity_command, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self._fallback_velocity_command = self._velocity_command.clone()

        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self._motion_loader: MotionLoader | MultiMotionLoader | None = None
        self.motion_dof_indices: list[int] = []
        self.motion_root_body_index: int | None = None
        self.motion_reference_body_index: int | None = None
        self.motion_key_body_indices: list[int] = []
        self._rsi_valid_frame_indices: np.ndarray | list[np.ndarray] | None = None
        self._last_reference_sampling_context: MultiMotionSampleContext | None = None
        if self.cfg.motion_manifest is not None:
            self._configure_multi_motion_loader(self.cfg.motion_manifest)
        elif self.cfg.motion_file is not None:
            self._configure_motion_loader(self.cfg.motion_file)

        self.amp_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.amp_observation_size,), dtype=np.float32
        )
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space),
            dtype=torch.float32,
            device=self.device,
        )
        # A reset observation is requested immediately after ``_reset_idx``.
        # Mark RSI environments so that this first observation replaces the
        # preloaded current reference frame without shifting history twice.
        self._rsi_history_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_rsi_times = torch.full((self.num_envs,), torch.nan, dtype=torch.float64, device=self.device)
        self._last_rsi_frame_indices = torch.full(
            (self.num_envs,), -1, dtype=torch.int64, device=self.device
        )
        self._last_rsi_motion_indices = torch.full(
            (self.num_envs,), -1, dtype=torch.int64, device=self.device
        )
        reference_probe = self.collect_reference_motions(num_samples=4)
        expected_shape = (4, self.amp_observation_size)
        if tuple(reference_probe.shape) != expected_shape:
            raise RuntimeError(
                f"AMP reference probe shape mismatch: expected {expected_shape}, got {tuple(reference_probe.shape)}."
            )
        if not torch.all(torch.isfinite(reference_probe)):
            raise RuntimeError("AMP reference probe contains NaN/Inf.")

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(physics_material=self.cfg.ground_material),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _configure_motion_loader(self, motion_file: str):
        """Configure the original official single-clip MotionLoader path."""
        self._motion_loader = MotionLoader(motion_file=motion_file, device=self.device)
        self._configure_motion_mappings(self._motion_loader)
        self._rsi_valid_frame_indices = self._build_rsi_valid_frame_indices(self._motion_loader)

    def _configure_multi_motion_loader(self, motion_manifest: str):
        """Configure independent clips and manifest-driven category balancing."""
        self._motion_loader = MultiMotionLoader(
            motion_manifest, device=self.device, rng=np.random.default_rng(self.cfg.seed)
        )
        self._configure_motion_mappings(self._motion_loader)
        self._rsi_valid_frame_indices = [
            self._build_rsi_valid_frame_indices(loader) for loader in self._motion_loader.loaders
        ]

    def _configure_motion_mappings(self, loader: MotionLoader | MultiMotionLoader):
        """Resolve the common G1 joint/body schema shared by all reference clips."""
        self.motion_dof_indices = loader.get_dof_index(list(self.cfg.policy_joint_names))
        self.motion_root_body_index = loader.get_body_index([self.cfg.motion_root_body])[0]
        self.motion_reference_body_index = loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indices = loader.get_body_index(list(self.cfg.key_body_names))

    def _build_rsi_valid_frame_indices(self, loader: MotionLoader) -> np.ndarray:
        """Return discrete joint-limit-safe frames with valid AMP history."""
        # Build the admissible discrete RSI set once. Rejecting invalid source
        # frames avoids silently clamping or modifying the validated NPZ.
        motion_joint_pos = loader.dof_positions[:, self.motion_dof_indices]
        joint_limits = self.robot.data.joint_pos_limits[0, self.policy_joint_ids]
        within_limits = torch.all(
            (motion_joint_pos >= joint_limits[:, 0]) & (motion_joint_pos <= joint_limits[:, 1]), dim=-1
        )
        frame_indices = torch.arange(loader.num_frames, device=self.device)
        history_frames = self.cfg.num_amp_observations - 1
        valid = within_limits & (frame_indices >= history_frames) & (frame_indices < loader.num_frames - 1)
        valid_indices = frame_indices[valid].cpu().numpy()
        if valid_indices.size == 0:
            raise RuntimeError("Reference motion contains no joint-limit-safe frame usable for RSI.")
        return valid_indices

    def _pre_physics_step(self, actions: torch.Tensor):
        if actions.shape != (self.num_envs, self.cfg.action_space):
            raise ValueError(
                f"Expected actions with shape {(self.num_envs, self.cfg.action_space)}, got {tuple(actions.shape)}."
            )
        self.actions = actions.clone()

    def _apply_action(self):
        targets = self._action_offset + self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target(targets, joint_ids=self.policy_joint_ids)

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos[:, self.policy_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.policy_joint_ids]

        policy_obs = torch.cat(
            (
                self.robot.data.root_ang_vel_b * 0.2,
                self.robot.data.projected_gravity_b,
                self._velocity_command,
                joint_pos - self._default_policy_joint_pos,
                joint_vel * 0.05,
                self.actions,
            ),
            dim=-1,
        )

        amp_obs = compute_amp_observation(
            joint_pos,
            joint_vel,
            self.robot.data.body_pos_w[:, self.reference_body_id],
            self.robot.data.body_quat_w[:, self.reference_body_id],
            self.robot.data.body_lin_vel_w[:, self.reference_body_id],
            self.robot.data.body_ang_vel_w[:, self.reference_body_id],
            self.robot.data.body_pos_w[:, self.key_body_ids],
        )
        # A reset preloads the complete same-clip reference history. Keep it
        # intact for this first policy/discriminator observation: immediately
        # replacing frame 0 with articulation-derived link velocities would
        # make an otherwise exact RSI history inconsistent at its current
        # frame. From the next simulation step onward every environment is
        # regular and receives policy-generated AMP observations as usual.
        regular_envs = ~self._rsi_history_pending
        for index in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[regular_envs, index + 1] = self.amp_observation_buffer[
                regular_envs, index
            ]
        self.amp_observation_buffer[regular_envs, 0] = amp_obs[regular_envs]
        self._rsi_history_pending[:] = False
        self.extras = {
            "amp_obs": self.amp_observation_buffer.view(self.num_envs, self.amp_observation_size),
        }
        if hasattr(self, "_task_metrics"):
            self.extras.update(self._task_metrics)

        return {"policy": policy_obs}

    def _get_rewards(self) -> torch.Tensor:
        velocity_error = torch.sum(
            torch.square(self.robot.data.root_lin_vel_b[:, :2] - self._velocity_command[:, :2]), dim=-1
        )
        yaw_rate_error = torch.square(self.robot.data.root_ang_vel_b[:, 2] - self._velocity_command[:, 2])
        velocity_tracking = torch.exp(-velocity_error / self.cfg.linear_velocity_tracking_sigma)
        yaw_rate_tracking = torch.exp(-yaw_rate_error / self.cfg.yaw_rate_tracking_sigma)
        upright = torch.exp(-torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=-1) / 0.25)
        angular_stability = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=-1)
        task_reward = (
            velocity_tracking
            + self.cfg.yaw_rate_tracking_weight * yaw_rate_tracking
            + 0.25 * upright
            + 0.05
            - 0.02 * angular_stability
        )
        self._task_metrics = {
            "task_reward": task_reward,
            "velocity_tracking_error": velocity_error,
            "velocity_tracking_reward": velocity_tracking,
            "yaw_rate_tracking_error": yaw_rate_error,
            "yaw_rate_tracking_reward": yaw_rate_tracking,
            "upright_reward": upright,
            "angular_stability_cost": angular_stability,
            "episode_length": self.episode_length_buf.to(dtype=torch.float32),
            "root_height": self.robot.data.root_pos_w[:, 2],
            "command_vx": self._velocity_command[:, 0],
            "command_vy": self._velocity_command[:, 1],
            "command_yaw_rate": self._velocity_command[:, 2],
        }
        return task_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.root_pos_w[:, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Clear policy/action history before selecting the reset source. A
        # reference-motion reset repopulates the AMP history below, so this
        # must happen before (rather than after) the reset strategy runs.
        self.actions[env_ids] = 0.0
        self.amp_observation_buffer[env_ids] = 0.0
        self._rsi_history_pending[env_ids] = False
        self._last_rsi_times[env_ids] = torch.nan
        self._last_rsi_frame_indices[env_ids] = -1
        self._last_rsi_motion_indices[env_ids] = -1
        self._velocity_command[env_ids] = self._fallback_velocity_command[env_ids]

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy == "rsi":
            root_state, joint_pos, joint_vel = self._reset_strategy_reference_motion(env_ids)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        if self.cfg.reset_strategy == "rsi":
            # MotionLoader stores the root body's link velocity. Writing it as
            # COM velocity would introduce a lever-arm offset immediately at
            # reset and create an avoidable AMP velocity mismatch.
            self.robot.write_root_link_velocity_to_sim(root_state[:, 7:], env_ids)
        else:
            self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_strategy_default(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_reference_motion(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._motion_loader is None:
            raise RuntimeError(
                "Reference-motion reset requires a G1-retargeted NPZ. "
                "Configure motion_manifest or motion_file only after G1 retargeting is complete."
            )

        num_samples = env_ids.shape[0]
        # Uniformly sample discrete, joint-limit-safe reference frames. Frame
        # zero and the terminal frame were excluded when these sets were
        # built, leaving valid prior frames for AMP history.
        if isinstance(self._motion_loader, MultiMotionLoader):
            assert isinstance(self._rsi_valid_frame_indices, list)
            sampled_motion_indices = self._motion_loader.sample_motion_indices(num_samples)
            sampled_frame_indices = np.empty(num_samples, dtype=np.int64)
            times = np.empty(num_samples, dtype=np.float64)
            for motion_index in np.unique(sampled_motion_indices):
                rows = np.flatnonzero(sampled_motion_indices == motion_index)
                valid_frames = self._rsi_valid_frame_indices[int(motion_index)]
                sampled_frame_indices[rows] = self._motion_loader.rng.choice(
                    valid_frames, size=rows.size, replace=True
                )
                times[rows] = (
                    sampled_frame_indices[rows].astype(np.float64)
                    * float(self._motion_loader.loaders[int(motion_index)].dt)
                )
            sampled_state = self._motion_loader.sample(sampled_motion_indices, times)
        else:
            assert isinstance(self._motion_loader, MotionLoader)
            assert isinstance(self._rsi_valid_frame_indices, np.ndarray)
            sampled_motion_indices = None
            sampled_frame_indices = np.random.choice(
                self._rsi_valid_frame_indices, size=num_samples, replace=True
            )
            times = sampled_frame_indices.astype(np.float64) * float(self._motion_loader.dt)
            sampled_state = self._motion_loader.sample(num_samples=num_samples, times=times)
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = sampled_state

        sampled_tensors = (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        )
        if not all(torch.all(torch.isfinite(value)) for value in sampled_tensors):
            raise RuntimeError("RSI sampled state contains NaN/Inf.")

        assert self.motion_root_body_index is not None
        if isinstance(self._motion_loader, MultiMotionLoader) and self.cfg.command_from_reference_state:
            self._velocity_command[env_ids] = reference_state_command(
                body_rotations[:, self.motion_root_body_index],
                body_linear_velocities[:, self.motion_root_body_index],
                body_angular_velocities[:, self.motion_root_body_index],
            )
        root_state = self.robot.data.default_root_state[env_ids].clone()
        sampled_root_position = body_positions[:, self.motion_root_body_index]
        if isinstance(self._motion_loader, MultiMotionLoader):
            assert sampled_motion_indices is not None
            motion_start_position = self._motion_loader.initial_body_positions(
                sampled_motion_indices, self.motion_root_body_index
            )
        else:
            motion_start_position = self._motion_loader.body_positions[0, self.motion_root_body_index]
        # Keep each environment anchored to its own origin. XY is the motion's
        # displacement from frame zero; Z remains the validated reference
        # height. Preserve the source world heading and world-frame velocities
        # because AMP encodes those same world-frame conventions.
        root_state[:, :2] = (
            sampled_root_position[:, :2]
            - motion_start_position[..., :2]
            + self.scene.env_origins[env_ids, :2]
        )
        root_state[:, 2] = sampled_root_position[:, 2] + self.scene.env_origins[env_ids, 2]
        root_state[:, 3:7] = body_rotations[:, self.motion_root_body_index]
        root_state[:, 7:10] = body_linear_velocities[:, self.motion_root_body_index]
        root_state[:, 10:13] = body_angular_velocities[:, self.motion_root_body_index]

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_pos[:, self.policy_joint_ids] = dof_positions[:, self.motion_dof_indices]
        joint_vel[:, self.policy_joint_ids] = dof_velocities[:, self.motion_dof_indices]

        policy_joint_pos = joint_pos[:, self.policy_joint_ids]
        joint_limits = self.robot.data.joint_pos_limits[env_ids][:, self.policy_joint_ids]
        limit_violation = torch.maximum(
            joint_limits[..., 0] - policy_joint_pos, policy_joint_pos - joint_limits[..., 1]
        )
        maximum_violation = torch.max(limit_violation)
        if maximum_violation.item() > 0.0:
            flat_index = int(torch.argmax(limit_violation).item())
            joint_index = flat_index % len(self.policy_joint_ids)
            raise RuntimeError(
                "RSI sampled joint position violates an articulation limit: "
                f"joint={self.policy_joint_names[joint_index]}, violation={maximum_violation.item():.9g} rad."
            )

        reference_observations = self.collect_reference_motions(
            num_samples, times, motion_indices=sampled_motion_indices
        )
        if not torch.all(torch.isfinite(reference_observations)):
            raise RuntimeError("RSI AMP history contains NaN/Inf.")
        self.amp_observation_buffer[env_ids] = reference_observations.view(
            num_samples, self.cfg.num_amp_observations, self.cfg.amp_observation_space
        )
        self._rsi_history_pending[env_ids] = True
        self._last_rsi_times[env_ids] = torch.as_tensor(times, dtype=torch.float64, device=self.device)
        self._last_rsi_frame_indices[env_ids] = torch.as_tensor(
            sampled_frame_indices, dtype=torch.int64, device=self.device
        )
        if sampled_motion_indices is not None:
            self._last_rsi_motion_indices[env_ids] = torch.as_tensor(
                sampled_motion_indices, dtype=torch.int64, device=self.device
            )
        return root_state, joint_pos, joint_vel

    def collect_reference_motions(
        self,
        num_samples: int,
        current_times: np.ndarray | None = None,
        motion_indices: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Collect same-clip AMP histories from one or more validated motions."""
        if self._motion_loader is None:
            raise RuntimeError(
                "No G1 reference motion is configured. Set G1AmpEnvCfg.motion_manifest or motion_file first."
            )
        if isinstance(self._motion_loader, MultiMotionLoader):
            sampled, context = self._motion_loader.sample_history(
                num_samples=num_samples,
                num_history_steps=self.cfg.num_amp_observations,
                motion_indices=motion_indices,
                current_times=current_times,
            )
            self._last_reference_sampling_context = context
            (
                dof_positions,
                dof_velocities,
                body_positions,
                body_rotations,
                body_linear_velocities,
                body_angular_velocities,
            ) = sampled
        else:
            if motion_indices is not None:
                raise ValueError("motion_indices is only valid with MultiMotionLoader.")
            if current_times is None:
                # Keep the pre-existing single-motion sampling behavior for
                # backward compatibility; MotionLoader clips boundary times.
                current_times = self._motion_loader.sample_times(num_samples)
            times = (
                np.expand_dims(current_times, axis=-1)
                - float(self._motion_loader.dt) * np.arange(self.cfg.num_amp_observations)
            ).flatten()
            (
                dof_positions,
                dof_velocities,
                body_positions,
                body_rotations,
                body_linear_velocities,
                body_angular_velocities,
            ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        assert self.motion_reference_body_index is not None
        amp_obs = compute_amp_observation(
            dof_positions[:, self.motion_dof_indices],
            dof_velocities[:, self.motion_dof_indices],
            body_positions[:, self.motion_reference_body_index],
            body_rotations[:, self.motion_reference_body_index],
            body_linear_velocities[:, self.motion_reference_body_index],
            body_angular_velocities[:, self.motion_reference_body_index],
            body_positions[:, self.motion_key_body_indices],
        )
        return amp_obs.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(quaternion: torch.Tensor) -> torch.Tensor:
    reference_tangent = torch.zeros_like(quaternion[..., :3])
    reference_normal = torch.zeros_like(quaternion[..., :3])
    reference_tangent[..., 0] = 1.0
    reference_normal[..., 2] = 1.0
    tangent = quat_apply(quaternion, reference_tangent)
    normal = quat_apply(quaternion, reference_normal)
    return torch.cat((tangent, normal), dim=-1)


@torch.jit.script
def compute_amp_observation(
    joint_positions: torch.Tensor,
    joint_velocities: torch.Tensor,
    reference_positions: torch.Tensor,
    reference_rotations: torch.Tensor,
    reference_linear_velocities: torch.Tensor,
    reference_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    """Build one 71-D AMP frame, separate from the 78-D policy observation."""
    return torch.cat(
        (
            joint_positions,
            joint_velocities,
            reference_positions[:, 2:3],
            quaternion_to_tangent_and_normal(reference_rotations),
            reference_linear_velocities,
            reference_angular_velocities,
            (key_body_positions - reference_positions.unsqueeze(1)).reshape(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
