from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.direct.humanoid_amp.motions import MotionLoader

from .g1_amp_env_cfg import G1AmpEnvCfg


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
        self._velocity_command = torch.tensor(
            self.cfg.velocity_command, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)

        self._motion_loader: MotionLoader | None = None
        self.motion_dof_indices: list[int] = []
        self.motion_root_body_index: int | None = None
        self.motion_reference_body_index: int | None = None
        self.motion_key_body_indices: list[int] = []
        if self.cfg.motion_file is not None:
            self._configure_motion_loader(self.cfg.motion_file)

        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.amp_observation_size,), dtype=np.float32
        )
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space),
            dtype=torch.float32,
            device=self.device,
        )

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
        """Configure official MotionLoader mappings for a future G1-retargeted motion file."""
        self._motion_loader = MotionLoader(motion_file=motion_file, device=self.device)
        self.motion_dof_indices = self._motion_loader.get_dof_index(list(self.cfg.policy_joint_names))
        self.motion_root_body_index = self._motion_loader.get_body_index([self.cfg.motion_root_body])[0]
        self.motion_reference_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indices = self._motion_loader.get_body_index(list(self.cfg.key_body_names))

    def _pre_physics_step(self, actions: torch.Tensor):
        if actions.shape != (self.num_envs, self.cfg.action_space):
            raise ValueError(
                f"Expected actions with shape {(self.num_envs, self.cfg.action_space)}, got {tuple(actions.shape)}."
            )
        self.actions = actions.clone()

    def _apply_action(self):
        targets = self._default_policy_joint_pos + self.cfg.action_scale * self.actions
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
        for index in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, index + 1] = self.amp_observation_buffer[:, index]
        self.amp_observation_buffer[:, 0] = amp_obs
        self.extras = {"amp_obs": self.amp_observation_buffer.view(self.num_envs, self.amp_observation_size)}

        return {"policy": policy_obs}

    def _get_rewards(self) -> torch.Tensor:
        velocity_error = torch.sum(
            torch.square(self.robot.data.root_lin_vel_b[:, :2] - self._velocity_command[:, :2]), dim=-1
        )
        velocity_tracking = torch.exp(-velocity_error / 0.25)
        upright = torch.exp(-torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=-1) / 0.25)
        angular_stability = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=-1)
        return velocity_tracking + 0.25 * upright + 0.05 - 0.02 * angular_stability

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

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_reference_motion(env_ids, start=start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
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
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._motion_loader is None:
            raise RuntimeError(
                "Reference-motion reset requires a G1-retargeted NPZ. "
                "Set motion_file only after Human-to-G1 retargeting is complete."
            )

        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        assert self.motion_root_body_index is not None
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = body_positions[:, self.motion_root_body_index] + self.scene.env_origins[env_ids]
        root_state[:, 3:7] = body_rotations[:, self.motion_root_body_index]
        root_state[:, 7:10] = body_linear_velocities[:, self.motion_root_body_index]
        root_state[:, 10:13] = body_angular_velocities[:, self.motion_root_body_index]

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_pos[:, self.policy_joint_ids] = dof_positions[:, self.motion_dof_indices]
        joint_vel[:, self.policy_joint_ids] = dof_velocities[:, self.motion_dof_indices]

        reference_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = reference_observations.view(
            num_samples, self.cfg.num_amp_observations, self.cfg.amp_observation_space
        )
        return root_state, joint_pos, joint_vel

    def collect_reference_motions(
        self, num_samples: int, current_times: np.ndarray | None = None
    ) -> torch.Tensor:
        """Collect two-frame AMP states from a future G1-retargeted motion file."""
        if self._motion_loader is None:
            raise RuntimeError(
                "No G1 reference motion is configured. Retarget a motion and set G1AmpEnvCfg.motion_file first."
            )
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(self.cfg.num_amp_observations)
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
