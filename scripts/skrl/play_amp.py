"""Replay a trained G1 Direct AMP policy and report gait-coordination diagnostics."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Unitree-G1-23DoF-AMP-Walk-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.num_envs <= 0 or args_cli.steps <= 0:
    parser.error("--num_envs and --steps must be positive")
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from skrl.utils.runner.torch import Runner

from isaaclab.utils.math import quat_apply_inverse
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or np.std(a) < 1.0e-8 or np.std(b) < 1.0e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


@hydra_task_config(args_cli.task, "skrl_amp_cfg_entry_point")
def main(env_cfg, experiment_cfg: dict) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    # Follow env 0's robot so resets and forward motion remain visible.
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (3.0, 3.0, 1.8)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.65)
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    video_length = args_cli.video_length or args_cli.steps
    run_dir = checkpoint.parent.parent
    env_cfg.log_dir = str(run_dir)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    raw_env = env.unwrapped

    if args_cli.video:
        video_folder = run_dir / "videos" / "play_amp"
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            disable_logger=True,
        )
        print(f"[INFO] Recording {video_length} steps to {video_folder}")

    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)
    runner.agent.load(str(checkpoint))
    runner.agent.set_running_mode("eval")

    body_names = raw_env.robot.body_names
    joint_names = raw_env.robot.joint_names
    body_ids = {
        name: body_names.index(name)
        for name in (
            "torso_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_roll_rubber_hand",
            "right_wrist_roll_rubber_hand",
        )
    }
    shoulder_ids = {
        name: joint_names.index(name)
        for name in ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint")
    }

    trajectories = {name: [] for name in (
        "left_foot_x", "right_foot_x", "left_hand_x", "right_hand_x",
        "left_shoulder_pitch", "right_shoulder_pitch", "root_height", "velocity_tracking_error",
    )}
    episode_lengths: list[int] = []
    current_episode_length = 0
    nonfinite = False
    maximum_joint_target_limit_violation = 0.0
    obs, _ = env.reset()

    for timestep in range(args_cli.steps):
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            if not torch.all(torch.isfinite(actions)):
                nonfinite = True
                raise FloatingPointError(f"Non-finite action during replay at step {timestep}.")
            action_targets = raw_env._action_offset + raw_env.cfg.action_scale * actions
            joint_limits = raw_env.robot.data.joint_pos_limits[:, raw_env.policy_joint_ids]
            target_violation = torch.maximum(
                joint_limits[..., 0] - action_targets, action_targets - joint_limits[..., 1]
            ).clamp_min(0.0)
            maximum_joint_target_limit_violation = max(
                maximum_joint_target_limit_violation, float(target_violation.max().item())
            )
            obs, _, terminated, truncated, _ = env.step(actions)

            torso_id = body_ids["torso_link"]
            torso_pos = raw_env.robot.data.body_pos_w[0, torso_id]
            torso_quat = raw_env.robot.data.body_quat_w[0, torso_id]
            local_positions = {}
            for key, body_name in (
                ("left_foot_x", "left_ankle_roll_link"),
                ("right_foot_x", "right_ankle_roll_link"),
                ("left_hand_x", "left_wrist_roll_rubber_hand"),
                ("right_hand_x", "right_wrist_roll_rubber_hand"),
            ):
                offset = raw_env.robot.data.body_pos_w[0, body_ids[body_name]] - torso_pos
                local_positions[key] = float(quat_apply_inverse(torso_quat, offset)[0].item())
            for key, value in local_positions.items():
                trajectories[key].append(value)
            trajectories["left_shoulder_pitch"].append(
                float(raw_env.robot.data.joint_pos[0, shoulder_ids["left_shoulder_pitch_joint"]].item())
            )
            trajectories["right_shoulder_pitch"].append(
                float(raw_env.robot.data.joint_pos[0, shoulder_ids["right_shoulder_pitch_joint"]].item())
            )
            trajectories["root_height"].append(float(raw_env.robot.data.root_pos_w[0, 2].item()))
            velocity_error = torch.sum(
                torch.square(raw_env.robot.data.root_lin_vel_b[0, :2] - raw_env._velocity_command[0, :2])
            )
            trajectories["velocity_tracking_error"].append(float(velocity_error.item()))

        current_episode_length += 1
        if bool((terminated[0] | truncated[0]).item()):
            episode_lengths.append(current_episode_length)
            current_episode_length = 0

    if current_episode_length:
        episode_lengths.append(current_episode_length)

    data = {key: np.asarray(value, dtype=np.float64) for key, value in trajectories.items()}
    longest_episode_index = int(np.argmax(episode_lengths))
    longest_start = int(sum(episode_lengths[:longest_episode_index]))
    longest_stop = longest_start + episode_lengths[longest_episode_index]
    segment = {key: value[longest_start:longest_stop] for key, value in data.items()}
    foot_difference = segment["left_foot_x"] - segment["right_foot_x"]
    hand_difference = segment["right_hand_x"] - segment["left_hand_x"]
    report = {
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "nonfinite_action": nonfinite,
        "episodes_observed": len(episode_lengths),
        "episode_length_mean": float(np.mean(episode_lengths)),
        "episode_length_max": int(max(episode_lengths)),
        "coordination_window": "longest continuous episode",
        "root_height_min": float(np.min(data["root_height"])),
        "root_height_max": float(np.max(data["root_height"])),
        "velocity_tracking_error_mean": float(np.mean(data["velocity_tracking_error"])),
        "velocity_tracking_error_longest_episode_mean": float(np.mean(segment["velocity_tracking_error"])),
        "maximum_joint_target_limit_violation": maximum_joint_target_limit_violation,
        "left_shoulder_pitch_range": float(np.ptp(data["left_shoulder_pitch"])),
        "right_shoulder_pitch_range": float(np.ptp(data["right_shoulder_pitch"])),
        "shoulder_antiphase_correlation": _correlation(
            segment["left_shoulder_pitch"], segment["right_shoulder_pitch"]
        ),
        "contralateral_coordination": _correlation(foot_difference, hand_difference),
        "left_leg_right_arm": _correlation(segment["left_foot_x"], segment["right_hand_x"]),
        "right_leg_left_arm": _correlation(segment["right_foot_x"], segment["left_hand_x"]),
    }
    if not all(math.isfinite(value) for values in trajectories.values() for value in values):
        raise FloatingPointError("Non-finite kinematic state during replay.")
    output_path = run_dir / f"{checkpoint.stem}_play_diagnostics.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== G1 AMP policy replay diagnostics ===")
    print(json.dumps(report, indent=2))
    print(f"Diagnostics: {output_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
