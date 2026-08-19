"""Measure why the trained G1 AMP discriminator separates policy and reference samples."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Unitree-G1-23DoF-AMP-Walk-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--reference_samples", type=int, default=16384)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--use-default-action-offset",
    action="store_true",
    help="Diagnostic-only replay of the legacy q_default-centered controller.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch
from skrl.utils.runner.torch import Runner

from isaaclab.utils.math import quat_apply_inverse
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

compute_amp_observation = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.23dof_amp.g1_amp_env"
).compute_amp_observation


FRAME_SIZE = 71


def _feature_layout(joint_names: list[str], key_body_names: list[str]) -> tuple[list[str], dict[str, list[int]]]:
    frame_labels: list[str] = []
    frame_groups: dict[str, list[int]] = {}

    def add(group: str, labels: list[str]) -> None:
        start = len(frame_labels)
        frame_labels.extend(labels)
        frame_groups[group] = list(range(start, start + len(labels)))

    add("joint_positions", [f"joint_position/{name}" for name in joint_names])
    add("joint_velocities", [f"joint_velocity/{name}" for name in joint_names])
    add("root_height", ["reference_body_height"])
    add("root_orientation", [
        "reference_tangent_world/x", "reference_tangent_world/y", "reference_tangent_world/z",
        "reference_normal_world/x", "reference_normal_world/y", "reference_normal_world/z",
    ])
    add("root_linear_velocity", [f"reference_linear_velocity_world/{axis}" for axis in "xyz"])
    add("root_angular_velocity", [f"reference_angular_velocity_world/{axis}" for axis in "xyz"])
    for body_name in key_body_names:
        add(f"key_body_position/{body_name}", [f"{body_name}_minus_reference_world/{axis}" for axis in "xyz"])
    if len(frame_labels) != FRAME_SIZE:
        raise RuntimeError(f"Expected {FRAME_SIZE} features per frame, built {len(frame_labels)}")

    labels: list[str] = []
    groups: dict[str, list[int]] = {}
    for frame_index, frame_name in enumerate(("current", "previous")):
        offset = frame_index * FRAME_SIZE
        labels.extend([f"{frame_name}/{label}" for label in frame_labels])
        for group, indices in frame_groups.items():
            groups.setdefault(group, []).extend([offset + index for index in indices])
    return labels, groups


def _stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().reshape(-1)
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "median": float(values.median().item()),
    }


def _group_stats(samples: torch.Tensor, groups: dict[str, list[int]]) -> dict[str, dict[str, float]]:
    return {name: _stats(samples[:, indices]) for name, indices in groups.items()}


def _collect_reference(raw_env, current_times: np.ndarray, interval: float) -> torch.Tensor:
    loader = raw_env._motion_loader
    times = (np.expand_dims(current_times, axis=-1) - interval * np.arange(2)).reshape(-1)
    dof_pos, dof_vel, body_pos, body_quat, body_lin_vel, body_ang_vel = loader.sample(
        num_samples=times.shape[0], times=times
    )
    obs = compute_amp_observation(
        dof_pos[:, raw_env.motion_dof_indices],
        dof_vel[:, raw_env.motion_dof_indices],
        body_pos[:, raw_env.motion_reference_body_index],
        body_quat[:, raw_env.motion_reference_body_index],
        body_lin_vel[:, raw_env.motion_reference_body_index],
        body_ang_vel[:, raw_env.motion_reference_body_index],
        body_pos[:, raw_env.motion_key_body_indices],
    )
    return obs.view(-1, 2 * FRAME_SIZE)


def _discriminator(agent, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = agent._amp_state_preprocessor(samples)
    with torch.no_grad():
        logits, _, _ = agent.discriminator.act({"states": normalized}, role="discriminator")
        probability = torch.sigmoid(logits).reshape(-1)
        style_reward = -torch.log(torch.clamp(1.0 - probability, min=0.0001)) * agent._discriminator_reward_scale
    return normalized, probability, style_reward


def _subset_scores(probability: torch.Tensor, style: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int | None]:
    mask = mask.bool()
    if not torch.any(mask):
        return {
            "count": 0,
            "probability_mean": None,
            "probability_std": None,
            "style_reward_mean": None,
            "classified_as_policy_fraction": None,
        }
    return {
        "count": int(mask.sum().item()),
        "probability_mean": float(probability[mask].mean().item()),
        "probability_std": float(probability[mask].std(unbiased=False).item()),
        "style_reward_mean": float(style[mask].mean().item()),
        "classified_as_policy_fraction": float((probability[mask] < 0.5).float().mean().item()),
    }


@hydra_task_config(args_cli.task, "skrl_amp_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    agent_cfg["seed"] = args_cli.seed
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    if args_cli.use_default_action_offset:
        raw_env._action_offset.copy_(raw_env._default_policy_joint_pos)
    wrapped_env = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(wrapped_env, agent_cfg)
    checkpoint = Path(args_cli.checkpoint).resolve()
    runner.agent.load(str(checkpoint))
    runner.agent.set_running_mode("eval")

    labels, groups = _feature_layout(list(raw_env.policy_joint_names), list(raw_env.key_body_names))
    loader = raw_env._motion_loader
    current_times = loader.sample_times(args_cli.reference_samples)
    reference_30hz = _collect_reference(raw_env, current_times, float(loader.dt))
    reference_50hz = _collect_reference(raw_env, current_times, float(raw_env.step_dt))

    policy_batches: list[torch.Tensor] = []
    age_batches: list[torch.Tensor] = []
    height_batches: list[torch.Tensor] = []
    tilt_batches: list[torch.Tensor] = []
    obs, _ = wrapped_env.reset()
    for _ in range(args_cli.steps):
        with torch.no_grad():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # outputs[0] is the stochastic action used by training; mean_actions is playback-only.
            actions = outputs[0]
            obs, _, _, _, _ = wrapped_env.step(actions)
            policy_batches.append(raw_env.amp_observation_buffer.view(args_cli.num_envs, -1).clone())
            age_batches.append(raw_env.episode_length_buf.clone())
            height_batches.append(raw_env.robot.data.root_pos_w[:, 2].clone())
            tilt_batches.append(torch.linalg.vector_norm(raw_env.robot.data.projected_gravity_b[:, :2], dim=-1))

    policy = torch.cat(policy_batches, dim=0)
    ages = torch.cat(age_batches, dim=0)
    heights = torch.cat(height_batches, dim=0)
    tilts = torch.cat(tilt_batches, dim=0)
    sample_count = min(policy.shape[0], args_cli.reference_samples)
    selection = torch.randperm(policy.shape[0], device=policy.device)[:sample_count]
    policy = policy[selection]
    ages = ages[selection]
    heights = heights[selection]
    tilts = tilts[selection]
    reference_30hz = reference_30hz[:sample_count]
    reference_50hz = reference_50hz[:sample_count]

    normalized_policy, policy_probability, policy_style = _discriminator(runner.agent, policy)
    normalized_reference, reference_probability, reference_style = _discriminator(runner.agent, reference_30hz)
    normalized_reference_50hz, reference_50_probability, reference_50_style = _discriminator(
        runner.agent, reference_50hz
    )

    mean_ref = reference_30hz.mean(dim=0)
    mean_policy = policy.mean(dim=0)
    std_ref = reference_30hz.std(dim=0, unbiased=False)
    std_policy = policy.std(dim=0, unbiased=False)
    separation = torch.abs(mean_ref - mean_policy) / (std_ref + std_policy + 1.0e-8)
    root_group_names = ("root_height", "root_orientation", "root_linear_velocity", "root_angular_velocity")
    key_body_group_names = tuple(name for name in groups if name.startswith("key_body_position/"))
    foot_group_names = tuple(name for name in key_body_group_names if "ankle" in name)
    separation_groups = {
        "joint_positions": groups["joint_positions"],
        "joint_velocities": groups["joint_velocities"],
        "root_state": [index for name in root_group_names for index in groups[name]],
        **{name: groups[name] for name in root_group_names},
        "foot_positions": [index for name in foot_group_names for index in groups[name]],
        "all_key_body_positions": [index for name in key_body_group_names for index in groups[name]],
        "shoulder_yaw_roll_positions": [
            index
            for index, label in enumerate(labels)
            if "joint_position/" in label and ("shoulder_yaw_joint" in label or "shoulder_roll_joint" in label)
        ],
    }
    feature_separation_by_group = {
        name: _stats(separation[indices]) for name, indices in separation_groups.items()
    }
    top_indices = torch.topk(separation, k=20).indices.tolist()
    top_features = []
    for rank, index in enumerate(top_indices, start=1):
        top_features.append({
            "rank": rank,
            "index": index,
            "label": labels[index],
            "separation_score": float(separation[index].item()),
            "reference": _stats(reference_30hz[:, index]),
            "policy": _stats(policy[:, index]),
            "normalized_reference": _stats(normalized_reference[:, index]),
            "normalized_policy": _stats(normalized_policy[:, index]),
        })

    temporal_groups = {}
    for group_name, group_indices in groups.items():
        current_indices = [index for index in group_indices if index < FRAME_SIZE]
        previous_indices = [index + FRAME_SIZE for index in current_indices]
        temporal_groups[group_name] = {
            "reference_30hz_abs_delta_mean": float(
                torch.abs(reference_30hz[:, current_indices] - reference_30hz[:, previous_indices]).mean().item()
            ),
            "reference_50hz_abs_delta_mean": float(
                torch.abs(reference_50hz[:, current_indices] - reference_50hz[:, previous_indices]).mean().item()
            ),
            "policy_50hz_abs_delta_mean": float(
                torch.abs(policy[:, current_indices] - policy[:, previous_indices]).mean().item()
            ),
        }

    early_upright = (ages <= 15) & (heights >= 0.65) & (tilts <= 0.35)
    all_upright = (heights >= 0.65) & (tilts <= 0.35)
    late_episode = ages >= 40
    near_fall = (heights <= 0.50) | (tilts >= 0.65)

    gradient_samples = normalized_reference[:4096].detach().clone().requires_grad_(True)
    gradient_logits, _, _ = runner.agent.discriminator.act(
        {"states": gradient_samples}, role="discriminator"
    )
    gradient = torch.autograd.grad(
        gradient_logits,
        gradient_samples,
        grad_outputs=torch.ones_like(gradient_logits),
        create_graph=False,
        retain_graph=False,
        only_inputs=True,
    )[0]
    gradient_penalty = torch.sum(torch.square(gradient), dim=-1).mean()

    reference_body = raw_env.motion_reference_body_index
    reference_world_velocity = loader.body_linear_velocities[:, reference_body]
    reference_local_velocity = quat_apply_inverse(
        loader.body_rotations[:, reference_body], reference_world_velocity
    )
    scaler = runner.agent._amp_state_preprocessor
    scaler_std = torch.sqrt(scaler.running_variance.float())
    initial_action_std = float(np.exp(agent_cfg["models"]["policy"]["initial_log_std"]))
    initial_joint_target_std = float(raw_env.cfg.action_scale * initial_action_std)
    default_joint_pos = raw_env._default_policy_joint_pos[0]
    action_offset = raw_env._action_offset[0]
    joint_limits = raw_env.robot.data.joint_pos_limits[0, raw_env.policy_joint_ids]
    action_offset_limit_margin = torch.minimum(
        action_offset - joint_limits[:, 0], joint_limits[:, 1] - action_offset
    )
    reference_joint_mean = reference_30hz[:, : len(raw_env.policy_joint_names)].mean(dim=0)
    control_prior_offsets = []
    for joint_index, joint_name in enumerate(raw_env.policy_joint_names):
        gap = float((reference_joint_mean[joint_index] - action_offset[joint_index]).item())
        control_prior_offsets.append({
            "joint": joint_name,
            "default_position": float(default_joint_pos[joint_index].item()),
            "action_offset": float(action_offset[joint_index].item()),
            "reference_mean_position": float(reference_joint_mean[joint_index].item()),
            "reference_mean_minus_action_offset": gap,
            "required_mean_action": gap / float(raw_env.cfg.action_scale),
            "gap_in_initial_target_noise_std": abs(gap) / initial_joint_target_std,
        })
    control_prior_offsets.sort(key=lambda item: item["gap_in_initial_target_noise_std"], reverse=True)

    report = {
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "sample_counts": {
            "reference": sample_count,
            "policy": sample_count,
            "policy_rollout_steps": args_cli.steps,
            "policy_environments": args_cli.num_envs,
        },
        "timing": {
            "simulation_dt_s": float(raw_env.cfg.sim.dt),
            "decimation": int(raw_env.cfg.decimation),
            "policy_amp_timestep_s": float(raw_env.step_dt),
            "reference_amp_timestep_s": float(loader.dt),
            "policy_frequency_hz": float(1.0 / raw_env.step_dt),
            "reference_frequency_hz": float(1.0 / loader.dt),
            "motion_loader_interpolation": "linear positions/velocities and quaternion SLERP at arbitrary times",
            "two_frame_interval_match": bool(np.isclose(raw_env.step_dt, loader.dt)),
        },
        "feature_layout": {"frame_size": FRAME_SIZE, "history_frames": 2, "total_size": len(labels), "labels": labels},
        "coordinate_conventions": {
            "quaternion": "wxyz",
            "root_orientation": "full world orientation encoded as rotated world tangent + normal; no heading removal",
            "reference_body": raw_env.cfg.reference_body,
            "root_height": "world z of reference body",
            "root_linear_velocity": "world-frame reference-body linear velocity",
            "root_angular_velocity": "world-frame reference-body angular velocity",
            "key_body_positions": "world-axis displacement: key_body_position_w - reference_body_position_w; not heading-rotated",
            "joint_order": list(raw_env.policy_joint_names),
            "key_body_order": list(raw_env.key_body_names),
            "units": "SI: radians, rad/s, meters, m/s",
        },
        "raw_group_statistics": {
            "reference_30hz": _group_stats(reference_30hz, groups),
            "policy_50hz": _group_stats(policy, groups),
            "reference_resampled_50hz": _group_stats(reference_50hz, groups),
        },
        "normalized_group_statistics": {
            "reference_30hz": _group_stats(normalized_reference, groups),
            "policy_50hz": _group_stats(normalized_policy, groups),
            "reference_resampled_50hz": _group_stats(normalized_reference_50hz, groups),
        },
        "normalization": {
            "class": type(scaler).__name__,
            "shared_for_reference_policy_replay": True,
            "clip_threshold": float(scaler.clip_threshold),
            "running_count": float(scaler.current_count.item()),
            "running_std_min": float(scaler_std.min().item()),
            "running_std_max": float(scaler_std.max().item()),
            "running_std_smallest_dimension": int(torch.argmin(scaler_std).item()),
            "running_std_largest_dimension": int(torch.argmax(scaler_std).item()),
        },
        "control_prior": {
            "action_scale": float(raw_env.cfg.action_scale),
            "fixed_policy_log_std": float(agent_cfg["models"]["policy"]["initial_log_std"]),
            "action_std": initial_action_std,
            "joint_target_noise_std_rad": initial_joint_target_std,
            "default_joint_positions": default_joint_pos.tolist(),
            "action_offset_positions": action_offset.tolist(),
            "action_offset_finite": bool(torch.all(torch.isfinite(action_offset)).item()),
            "action_offset_maximum_joint_limit_violation_rad": float(
                torch.clamp(-action_offset_limit_margin.min(), min=0.0).item()
            ),
            "action_offset_minimum_joint_limit_margin_rad": float(action_offset_limit_margin.min().item()),
            "action_offset_closest_limit_joint": raw_env.policy_joint_names[
                int(torch.argmin(action_offset_limit_margin).item())
            ],
            "reference_mean_offset_ranked": control_prior_offsets,
        },
        "feature_separation_by_group": feature_separation_by_group,
        "top_20_separable_features": top_features,
        "temporal_abs_delta_by_group": temporal_groups,
        "discriminator": {
            "reference_30hz_probability": _stats(reference_probability),
            "policy_50hz_probability": _stats(policy_probability),
            "reference_resampled_50hz_probability": _stats(reference_50_probability),
            "reference_30hz_style_reward": _stats(reference_style),
            "policy_50hz_style_reward": _stats(policy_style),
            "reference_resampled_50hz_style_reward": _stats(reference_50_style),
            "accuracy_balanced_reference_vs_policy": float(
                (0.5 * ((reference_probability > 0.5).float().mean() + (policy_probability < 0.5).float().mean())).item()
            ),
            "gradient_penalty_raw": float(gradient_penalty.item()),
            "gradient_penalty_scale": float(runner.agent._discriminator_gradient_penalty_scale),
            "gradient_penalty_contribution_before_total_loss_scale": float(
                gradient_penalty.item() * runner.agent._discriminator_gradient_penalty_scale
            ),
            "learning_rate": float(agent_cfg["agent"]["learning_rate"]),
            "learning_epochs": int(agent_cfg["agent"]["learning_epochs"]),
            "mini_batches": int(agent_cfg["agent"]["mini_batches"]),
            "discriminator_optimizer_steps_per_ppo_update": int(
                agent_cfg["agent"]["learning_epochs"] * agent_cfg["agent"]["mini_batches"]
            ),
            "configured_discriminator_batch_size": int(agent_cfg["agent"]["discriminator_batch_size"]),
            "amp_reference_additions_per_update": int(agent_cfg["agent"]["amp_batch_size"]),
            "motion_dataset_capacity": int(agent_cfg["motion_dataset"]["memory_size"]),
            "replay_buffer_capacity": int(agent_cfg["reply_buffer"]["memory_size"]),
        },
        "policy_state_categories": {
            "early_upright": _subset_scores(policy_probability, policy_style, early_upright),
            "all_upright": _subset_scores(policy_probability, policy_style, all_upright),
            "late_episode": _subset_scores(policy_probability, policy_style, late_episode),
            "near_fall": _subset_scores(policy_probability, policy_style, near_fall),
            "all_policy": _subset_scores(policy_probability, policy_style, torch.ones_like(ages, dtype=torch.bool)),
            "episode_age_bins": {
                "0_to_2": _subset_scores(policy_probability, policy_style, ages <= 2),
                "3_to_5": _subset_scores(policy_probability, policy_style, (ages >= 3) & (ages <= 5)),
                "6_to_10": _subset_scores(policy_probability, policy_style, (ages >= 6) & (ages <= 10)),
                "11_to_15": _subset_scores(policy_probability, policy_style, (ages >= 11) & (ages <= 15)),
                "16_to_25": _subset_scores(policy_probability, policy_style, (ages >= 16) & (ages <= 25)),
                "26_to_40": _subset_scores(policy_probability, policy_style, (ages >= 26) & (ages <= 40)),
                "over_40": _subset_scores(policy_probability, policy_style, ages > 40),
            },
            "exact_episode_age": {
                str(age): _subset_scores(policy_probability, policy_style, ages == age)
                for age in (1, 3, 5, 10)
            },
            "criteria": {
                "early_upright": "episode_age <= 15, root height >= 0.65 m, tilt norm <= 0.35",
                "late_episode": "episode_age >= 40",
                "near_fall": "root height <= 0.50 m OR projected-gravity xy norm >= 0.65",
            },
        },
        "velocity_compatibility": {
            "reference_forward_velocity_local_mean": float(reference_local_velocity[:, 0].mean().item()),
            "reference_forward_velocity_local_min": float(reference_local_velocity[:, 0].min().item()),
            "reference_forward_velocity_local_max": float(reference_local_velocity[:, 0].max().item()),
            "reference_world_x_velocity_mean": float(reference_world_velocity[:, 0].mean().item()),
            "reference_world_x_velocity_min": float(reference_world_velocity[:, 0].min().item()),
            "reference_world_x_velocity_max": float(reference_world_velocity[:, 0].max().item()),
            "training_command_vx_range": [float(raw_env.cfg.velocity_command[0]), float(raw_env.cfg.velocity_command[0])],
            "training_command_vy_range": [float(raw_env.cfg.velocity_command[1]), float(raw_env.cfg.velocity_command[1])],
            "training_command_yaw_range": [float(raw_env.cfg.velocity_command[2]), float(raw_env.cfg.velocity_command[2])],
        },
    }

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args_cli.output.resolve()),
        "timing": report["timing"],
        "discriminator": report["discriminator"],
        "policy_state_categories": report["policy_state_categories"],
        "velocity_compatibility": report["velocity_compatibility"],
        "top_20_separable_features": report["top_20_separable_features"],
    }, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
