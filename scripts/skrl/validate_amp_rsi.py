"""Validate G1 AMP Reference State Initialization without training."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Unitree-G1-23DoF-AMP-Walk-Direct-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--single-motion",
    action="store_true",
    help="Disable motion_manifest and validate the backward-compatible motion_file path",
)
parser.add_argument(
    "--fast-shutdown",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Exit the standalone validator without waiting for full Kit teardown",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.math import quat_apply_inverse


def _maximum_quaternion_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    direct = torch.linalg.vector_norm(actual - expected, dim=-1)
    negated = torch.linalg.vector_norm(actual + expected, dim=-1)
    return float(torch.minimum(direct, negated).max().item())


@hydra_task_config(args_cli.task, "skrl_amp_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    if args_cli.single_motion:
        env_cfg.motion_manifest = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped

    observations, _ = env.reset(seed=args_cli.seed)
    times = raw_env._last_rsi_times.detach().cpu().numpy()
    loader = raw_env._motion_loader
    is_multi_motion = hasattr(loader, "loaders")
    motion_indices = (
        raw_env._last_rsi_motion_indices.detach().cpu().numpy() if is_multi_motion else None
    )
    reference = raw_env.collect_reference_motions(
        args_cli.num_envs, times, motion_indices=motion_indices
    ).view(
        args_cli.num_envs, raw_env.cfg.num_amp_observations, raw_env.cfg.amp_observation_space
    )
    policy_history = raw_env.amp_observation_buffer
    current_error = torch.abs(policy_history[:, 0] - reference[:, 0])
    frame_slices = {
        "joint_positions": slice(0, 23),
        "joint_velocities": slice(23, 46),
        "root_height": slice(46, 47),
        "root_orientation": slice(47, 53),
        "root_linear_velocity": slice(53, 56),
        "root_angular_velocity": slice(56, 59),
        "key_body_positions": slice(59, 71),
    }
    if is_multi_motion:
        sampled = loader.sample(motion_indices, times)
        start_root_position = loader.initial_body_positions(
            motion_indices, raw_env.motion_root_body_index
        )
        sampled_dt = np.asarray(
            [float(loader.loaders[index].dt) for index in motion_indices], dtype=np.float64
        )
        sampled_categories = [loader.motion_categories[index] for index in motion_indices]
        sampled_motion_ids = [loader.motion_ids[index] for index in motion_indices]
        valid_frame_counts = [int(indices.size) for indices in raw_env._rsi_valid_frame_indices]
        total_motion_frames = [int(motion.num_frames) for motion in loader.loaders]
    else:
        sampled = loader.sample(num_samples=args_cli.num_envs, times=times)
        start_root_position = loader.body_positions[0, raw_env.motion_root_body_index]
        sampled_dt = np.full(args_cli.num_envs, float(loader.dt), dtype=np.float64)
        sampled_categories = []
        sampled_motion_ids = []
        valid_frame_counts = [int(raw_env._rsi_valid_frame_indices.size)]
        total_motion_frames = [int(loader.num_frames)]
    dof_pos, dof_vel, body_pos, body_quat, body_lin_vel, body_ang_vel = sampled
    root_index = raw_env.motion_root_body_index
    command_reference_error = None
    command_summary = {
        "source": "fixed fallback velocity_command",
        "vx_minimum": float(raw_env._velocity_command[:, 0].min().item()),
        "vx_maximum": float(raw_env._velocity_command[:, 0].max().item()),
        "vy_minimum": float(raw_env._velocity_command[:, 1].min().item()),
        "vy_maximum": float(raw_env._velocity_command[:, 1].max().item()),
        "yaw_rate_minimum": float(raw_env._velocity_command[:, 2].min().item()),
        "yaw_rate_maximum": float(raw_env._velocity_command[:, 2].max().item()),
    }
    if is_multi_motion and raw_env.cfg.command_from_reference_state:
        expected_command = torch.cat(
            (
                quat_apply_inverse(body_quat[:, root_index], body_lin_vel[:, root_index])[:, :2],
                quat_apply_inverse(body_quat[:, root_index], body_ang_vel[:, root_index])[:, 2:3],
            ),
            dim=-1,
        )
        command_reference_error = float(torch.abs(raw_env._velocity_command - expected_command).max().item())
        command_summary["source"] = "selected reference root state in body frame"
        command_summary["reference_state_max_abs_error"] = command_reference_error
    expected_root_position = body_pos[:, root_index].clone()
    expected_root_position[:, :2] = (
        expected_root_position[:, :2]
        - start_root_position[..., :2]
        + raw_env.scene.env_origins[:, :2]
    )
    expected_root_position[:, 2] += raw_env.scene.env_origins[:, 2]

    actual_joint_pos = raw_env.robot.data.joint_pos[:, raw_env.policy_joint_ids]
    actual_joint_vel = raw_env.robot.data.joint_vel[:, raw_env.policy_joint_ids]
    expected_joint_pos = dof_pos[:, raw_env.motion_dof_indices]
    expected_joint_vel = dof_vel[:, raw_env.motion_dof_indices]
    joint_limits = raw_env.robot.data.joint_pos_limits[:, raw_env.policy_joint_ids]
    limit_violation = torch.maximum(
        joint_limits[..., 0] - actual_joint_pos, actual_joint_pos - joint_limits[..., 1]
    ).clamp_min(0.0)

    report = {
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "reset_strategy": raw_env.cfg.reset_strategy,
        "motion_loader": "MultiMotionLoader" if is_multi_motion else "MotionLoader",
        "observation_shapes": {
            "policy": list(observations["policy"].shape),
            "amp_history": list(raw_env.extras["amp_obs"].shape),
        },
        "sampled_motion_ids": dict(
            sorted((name, sampled_motion_ids.count(name)) for name in set(sampled_motion_ids))
        ),
        "sampled_categories": dict(
            sorted((name, sampled_categories.count(name)) for name in set(sampled_categories))
        ),
        "sample_time_s": {
            "minimum": float(np.min(times)),
            "maximum": float(np.max(times)),
            "valid_discrete_frames_per_motion": valid_frame_counts,
            "total_frames_per_motion": total_motion_frames,
            "sampled_frame_index_minimum": int(raw_env._last_rsi_frame_indices.min().item()),
            "sampled_frame_index_maximum": int(raw_env._last_rsi_frame_indices.max().item()),
        },
        "history": {
            "frames": int(raw_env.cfg.num_amp_observations),
            "reference_interval_s_minimum": float(np.min(sampled_dt)),
            "reference_interval_s_maximum": float(np.max(sampled_dt)),
            "policy_interval_s": float(raw_env.step_dt),
            "policy_frequency_hz": float(1.0 / raw_env.step_dt),
            "pending_flags_after_reset_observation": int(raw_env._rsi_history_pending.sum().item()),
            "current_frame_max_abs_error": float(torch.abs(policy_history[:, 0] - reference[:, 0]).max().item()),
            "previous_frame_max_abs_error": float(torch.abs(policy_history[:, 1] - reference[:, 1]).max().item()),
            "current_frame_max_abs_error_by_group": {
                name: float(current_error[:, indices].max().item()) for name, indices in frame_slices.items()
            },
        },
        "physical_state": {
            "joint_position_max_abs_error_rad": float(torch.abs(actual_joint_pos - expected_joint_pos).max().item()),
            "joint_velocity_max_abs_error_rad_s": float(torch.abs(actual_joint_vel - expected_joint_vel).max().item()),
            "root_position_max_abs_error_m": float(
                torch.abs(raw_env.robot.data.root_link_pos_w - expected_root_position).max().item()
            ),
            "root_quaternion_max_l2_error": _maximum_quaternion_error(
                raw_env.robot.data.root_link_quat_w, body_quat[:, root_index]
            ),
            "root_linear_velocity_max_abs_error_m_s": float(
                torch.abs(raw_env.robot.data.root_link_lin_vel_w - body_lin_vel[:, root_index]).max().item()
            ),
            "root_angular_velocity_max_abs_error_rad_s": float(
                torch.abs(raw_env.robot.data.root_link_ang_vel_w - body_ang_vel[:, root_index]).max().item()
            ),
            "maximum_joint_limit_violation_rad": float(limit_violation.max().item()),
        },
        "root_handling": {
            "xy": "reference root displacement from motion frame 0, added to each Isaac environment origin",
            "z": "validated reference root height, added to environment-origin z",
            "heading": "reference world quaternion preserved unchanged (wxyz)",
            "velocities": "reference world-frame root linear/angular velocities preserved unchanged",
        },
        "commands": command_summary,
        "all_finite": bool(
            torch.all(torch.isfinite(policy_history))
            and torch.all(torch.isfinite(actual_joint_pos))
            and torch.all(torch.isfinite(actual_joint_vel))
            and torch.all(torch.isfinite(raw_env.robot.data.root_link_state_w))
        ),
    }
    if command_reference_error is not None and command_reference_error > 1.0e-5:
        raise RuntimeError(
            "RSI command/reference mismatch: "
            f"maximum absolute error is {command_reference_error:.6g}."
        )
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    if args_cli.fast_shutdown:
        exit_code = 0
        try:
            main()
        except BaseException:
            traceback.print_exc()
            exit_code = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    else:
        try:
            main()
        finally:
            simulation_app.close()
