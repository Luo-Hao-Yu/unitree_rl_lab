"""Train the G1-23DoF Direct AMP task with strict numerical diagnostics.

This entry point uses skrl's AMP implementation unchanged for optimization and
style-reward generation. The subclass below only observes tensors and adds
logging / fail-fast checks around the upstream implementation.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Unitree-G1-23DoF-AMP-Walk-Direct-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--run_name", type=str, default="smoke")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.max_iterations <= 0:
    parser.error("--max_iterations must be positive")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

import omni
from skrl.agents.torch.amp import AMP
from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


class InstrumentedAMP(AMP):
    """Upstream skrl AMP with logging and fail-fast checks only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._diagnostic_update = 0

    @staticmethod
    def _require_finite(name: str, value: torch.Tensor, *, update: int, timestep: int) -> None:
        if not torch.all(torch.isfinite(value)):
            bad_count = int(torch.count_nonzero(~torch.isfinite(value)).item())
            raise FloatingPointError(
                f"Non-finite {name} at AMP update {update}, timestep {timestep}; bad_values={bad_count}."
            )

    def _append_jsonl(self, payload: dict[str, Any]) -> None:
        path = Path(self.experiment_dir) / "amp_diagnostics.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        update = self._diagnostic_update + 1
        self._require_finite("policy states", states, update=update, timestep=timestep)
        self._require_finite("policy actions", actions, update=update, timestep=timestep)
        self._require_finite("task reward", rewards, update=update, timestep=timestep)
        self._require_finite("next policy states", next_states, update=update, timestep=timestep)
        if "amp_obs" not in infos:
            raise KeyError(f"Environment infos is missing amp_obs at timestep {timestep}.")
        amp_states = infos["amp_obs"]
        self._require_finite("policy AMP samples", amp_states, update=update, timestep=timestep)
        expected_amp_dim = int(self.amp_observation_space.shape[0])
        if amp_states.ndim != 2 or amp_states.shape[1] != expected_amp_dim:
            raise RuntimeError(
                f"Policy AMP feature mismatch at timestep {timestep}: "
                f"expected [N, {expected_amp_dim}], got {tuple(amp_states.shape)}."
            )

        super().record_transition(
            states,
            actions,
            rewards,
            next_states,
            terminated,
            truncated,
            infos,
            timestep,
            timesteps,
        )

        self.track_data("Reward / Task reward (mean)", rewards.mean().item())
        self.track_data("Diagnostics / Action mean", actions.mean().item())
        self.track_data("Diagnostics / Action std", actions.std().item())
        self.track_data("Diagnostics / Action abs max", actions.abs().max().item())
        self.track_data("Diagnostics / Termination rate", terminated.float().mean().item())
        self.track_data("Diagnostics / Timeout rate", truncated.float().mean().item())
        info_tags = {
            "velocity_tracking_error": "Task / Velocity tracking error",
            "velocity_tracking_reward": "Task / Velocity tracking reward",
            "upright_reward": "Safety / Upright reward",
            "angular_stability_cost": "Safety / Angular stability cost",
            "episode_length": "Episode / Current length mean",
            "root_height": "Safety / Root height mean",
        }
        for info_name, tag in info_tags.items():
            if info_name in infos:
                value = infos[info_name]
                if torch.is_tensor(value):
                    self._require_finite(info_name, value, update=update, timestep=timestep)
                    self.track_data(tag, value.float().mean().item())

    def _update(self, timestep: int, timesteps: int) -> None:
        self._diagnostic_update += 1
        update = self._diagnostic_update
        try:
            super()._update(timestep, timesteps)

            task_rewards = self.memory.get_tensor_by_name("rewards")
            policy_amp_states = self.memory.get_tensor_by_name("amp_states").view(
                -1, int(self.amp_observation_space.shape[0])
            )
            sample_count = min(4096, policy_amp_states.shape[0])
            policy_amp_states = policy_amp_states[:sample_count]
            reference_amp_states = self.collect_reference_motions(sample_count)
            if reference_amp_states.shape != policy_amp_states.shape:
                raise RuntimeError(
                    "Reference/policy AMP feature mismatch: "
                    f"reference={tuple(reference_amp_states.shape)}, policy={tuple(policy_amp_states.shape)}."
                )
            self._require_finite("reference AMP samples", reference_amp_states, update=update, timestep=timestep)

            with torch.no_grad():
                policy_logits, _, _ = self.discriminator.act(
                    {"states": self._amp_state_preprocessor(policy_amp_states)}, role="discriminator"
                )
                reference_logits, _, _ = self.discriminator.act(
                    {"states": self._amp_state_preprocessor(reference_amp_states)}, role="discriminator"
                )
                policy_probability = torch.sigmoid(policy_logits)
                reference_probability = torch.sigmoid(reference_logits)
                style_reward = -torch.log(
                    torch.maximum(
                        1.0 - policy_probability,
                        torch.tensor(0.0001, device=self.device),
                    )
                )
                style_reward *= self._discriminator_reward_scale
                task_flat = task_rewards.view(-1)[:sample_count]
                combined_reward = (
                    self._task_reward_weight * task_flat
                    + self._style_reward_weight * style_reward.view(-1)
                )
                entropy = self.policy.get_entropy(role="policy").mean()

            diagnostics = {
                "update": update,
                "timestep": int(timestep),
                "task_reward_mean": float(task_flat.mean().item()),
                "style_reward_mean": float(style_reward.mean().item()),
                "style_reward_min": float(style_reward.min().item()),
                "style_reward_max": float(style_reward.max().item()),
                "combined_reward_mean": float(combined_reward.mean().item()),
                "policy_discriminator_probability": float(policy_probability.mean().item()),
                "reference_discriminator_probability": float(reference_probability.mean().item()),
                "policy_discriminator_accuracy": float((policy_probability < 0.5).float().mean().item()),
                "reference_discriminator_accuracy": float((reference_probability > 0.5).float().mean().item()),
                "discriminator_accuracy": float(
                    0.5
                    * (
                        (policy_probability < 0.5).float().mean()
                        + (reference_probability > 0.5).float().mean()
                    ).item()
                ),
                "policy_saturation_fraction": float(
                    ((policy_probability < 0.01) | (policy_probability > 0.99)).float().mean().item()
                ),
                "reference_saturation_fraction": float(
                    ((reference_probability < 0.01) | (reference_probability > 0.99)).float().mean().item()
                ),
                "entropy": float(entropy.item()),
                "amp_feature_dim": int(policy_amp_states.shape[1]),
                "reference_samples": int(reference_amp_states.shape[0]),
                "policy_samples": int(policy_amp_states.shape[0]),
            }

            loss_tags = {
                "policy_loss": "Loss / Policy loss",
                "value_loss": "Loss / Value loss",
                "discriminator_loss": "Loss / Discriminator loss",
            }
            for name, tag in loss_tags.items():
                values = self.tracking_data.get(tag)
                if not values:
                    raise RuntimeError(f"Upstream skrl AMP did not emit required metric: {tag}")
                diagnostics[name] = float(values[-1])

            for name, value in diagnostics.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite AMP diagnostic {name} at update {update}: {value}")
            for role, model in self.models.items():
                if model is not None:
                    for parameter_name, parameter in model.named_parameters():
                        self._require_finite(
                            f"{role} parameter {parameter_name}", parameter, update=update, timestep=timestep
                        )

            tracked = {
                "Reward / AMP style reward (mean)": diagnostics["style_reward_mean"],
                "Reward / Combined reward (mean)": diagnostics["combined_reward_mean"],
                "AMP / Policy discriminator probability": diagnostics["policy_discriminator_probability"],
                "AMP / Reference discriminator probability": diagnostics["reference_discriminator_probability"],
                "AMP / Discriminator accuracy": diagnostics["discriminator_accuracy"],
                "AMP / Policy saturation fraction": diagnostics["policy_saturation_fraction"],
                "AMP / Reference saturation fraction": diagnostics["reference_saturation_fraction"],
                "Policy / Entropy": diagnostics["entropy"],
            }
            for tag, value in tracked.items():
                self.track_data(tag, value)
            self._append_jsonl(diagnostics)
            print(f"[AMP_DIAGNOSTICS] {json.dumps(diagnostics, sort_keys=True)}", flush=True)
        except Exception as error:
            self._append_jsonl(
                {
                    "update": update,
                    "timestep": int(timestep),
                    "status": "FAILED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise


class InstrumentedRunner(Runner):
    """Select InstrumentedAMP while reusing the upstream skrl Runner."""

    def _component(self, name: str):
        if name.lower() == "amp":
            return InstrumentedAMP
        return super()._component(name)


@hydra_task_config(args_cli.task, "skrl_amp_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed if args_cli.seed >= 0 else random.randint(0, 10000)
    agent_cfg["seed"] = env_cfg.seed
    rollouts = int(agent_cfg["agent"]["rollouts"])
    agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * rollouts
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_amp_torch_{args_cli.run_name}"
    log_root = Path("logs/skrl") / agent_cfg["agent"]["experiment"]["directory"]
    log_root = log_root.resolve()
    log_dir = log_root / run_name
    agent_cfg["agent"]["experiment"]["directory"] = str(log_root)
    agent_cfg["agent"]["experiment"]["experiment_name"] = run_name
    env_cfg.log_dir = str(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(str(log_dir / "params/env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params/agent.yaml"), agent_cfg)

    print(f"[INFO] Exact task: {args_cli.task}")
    print(f"[INFO] Log directory: {log_dir}")
    print(f"[INFO] Environments: {env_cfg.scene.num_envs}")
    print(f"[INFO] AMP updates requested: {args_cli.max_iterations}")
    print(f"[INFO] Task/style weights: {agent_cfg['agent']['task_reward_weight']} / "
          f"{agent_cfg['agent']['style_reward_weight']}")
    print(f"[INFO] Reference motion: {env_cfg.motion_file}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    reference_probe = env.unwrapped.collect_reference_motions(8)
    env.reset()
    policy_probe = env.unwrapped.extras["amp_obs"]
    if reference_probe.shape[1:] != policy_probe.shape[1:]:
        raise RuntimeError(
            f"AMP probe mismatch: reference={tuple(reference_probe.shape)}, policy={tuple(policy_probe.shape)}"
        )
    if not torch.all(torch.isfinite(reference_probe)) or not torch.all(torch.isfinite(policy_probe)):
        raise FloatingPointError("Initial AMP reference/policy probes contain NaN/Inf.")
    print(
        f"[INFO] AMP feature validation: reference={tuple(reference_probe.shape)}, "
        f"policy={tuple(policy_probe.shape)}"
    )

    wrapped_env = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = InstrumentedRunner(wrapped_env, agent_cfg)
    if args_cli.checkpoint:
        checkpoint = retrieve_file_path(args_cli.checkpoint)
        print(f"[INFO] Loading checkpoint: {checkpoint}")
        runner.agent.load(checkpoint)

    failure_path = log_dir / "training_failure.json"
    try:
        runner.run()
        final_checkpoint = log_dir / "checkpoints" / "final_agent.pt"
        final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        runner.agent.save(str(final_checkpoint))
        print(f"[INFO] Final diagnostic checkpoint: {final_checkpoint}")
    except Exception as error:
        with failure_path.open("w", encoding="utf-8") as stream:
            json.dump(
                {"error_type": type(error).__name__, "error": str(error)},
                stream,
                indent=2,
            )
            stream.write("\n")
        omni.log.error(f"AMP training stopped: {type(error).__name__}: {error}")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
