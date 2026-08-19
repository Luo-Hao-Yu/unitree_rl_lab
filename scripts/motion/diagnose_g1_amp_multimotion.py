"""Numerically validate category-balanced G1 AMP multi-motion sampling."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--num-amp-observations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--category-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--fast-shutdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit this standalone diagnostic without waiting for full Kit teardown",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    package = "unitree_rl_lab.tasks.locomotion.robots.g1.23dof_amp"
    env_module = importlib.import_module(f"{package}.g1_amp_env")
    cfg_module = importlib.import_module(f"{package}.g1_amp_env_cfg")
    loader_module = importlib.import_module(f"{package}.multi_motion_loader")
    compute_amp_observation = env_module.compute_amp_observation
    key_body_names = cfg_module.G1_AMP_KEY_BODY_NAMES
    policy_joint_names = cfg_module.G1_23DOF_POLICY_JOINT_NAMES
    multi_motion_loader = loader_module.MultiMotionLoader

    if args.num_samples <= 0 or args.num_amp_observations <= 0:
        raise ValueError("Sample counts must be positive.")
    loader = multi_motion_loader(
        args.manifest,
        device=args.device,
        rng=np.random.default_rng(args.seed),
    )
    samples, context = loader.sample_history(args.num_samples, args.num_amp_observations)
    if context.history_motion_indices is None or context.history_times is None:
        raise RuntimeError("MultiMotionLoader did not return history identity diagnostics.")
    if not np.all(context.history_motion_indices == context.motion_indices[:, None]):
        raise RuntimeError("Cross-clip AMP history detected.")

    for row, motion_index in enumerate(context.motion_indices):
        history_times = context.history_times[row]
        motion = loader.loaders[int(motion_index)]
        expected_delta = float(motion.dt)
        if np.any(history_times < -1.0e-9) or np.any(history_times > float(motion.duration) + 1.0e-9):
            raise RuntimeError(f"History time left clip boundaries at row {row}.")
        if history_times.size > 1 and not np.allclose(
            history_times[:-1] - history_times[1:], expected_delta, atol=1.0e-9
        ):
            raise RuntimeError(f"History timing is inconsistent at row {row}.")

    dof_indices = loader.get_dof_index(list(policy_joint_names))
    reference_body_index = loader.get_body_index(["torso_link"])[0]
    key_body_indices = loader.get_body_index(list(key_body_names))
    amp_frames = compute_amp_observation(
        samples[0][:, dof_indices],
        samples[1][:, dof_indices],
        samples[2][:, reference_body_index],
        samples[3][:, reference_body_index],
        samples[4][:, reference_body_index],
        samples[5][:, reference_body_index],
        samples[2][:, key_body_indices],
    )
    amp_observations = amp_frames.view(args.num_samples, -1)
    expected_shape = (args.num_samples, args.num_amp_observations * 71)
    if tuple(amp_observations.shape) != expected_shape:
        raise RuntimeError(f"AMP observation shape mismatch: {amp_observations.shape} != {expected_shape}")
    if not torch.all(torch.isfinite(amp_observations)):
        raise RuntimeError("Multi-motion AMP observations contain NaN/Inf.")

    category_counts = Counter(loader.motion_categories[index] for index in context.motion_indices)
    clip_counts = Counter(loader.motion_ids[index] for index in context.motion_indices)
    observed_categories = {
        category: category_counts[category] / args.num_samples for category in loader.categories
    }
    expected_categories = {
        category: float(probability)
        for category, probability in zip(loader.categories, loader.category_probabilities)
    }
    category_errors = {
        category: observed_categories[category] - expected_categories[category]
        for category in loader.categories
    }
    maximum_category_error = max(abs(value) for value in category_errors.values())
    if maximum_category_error > args.category_tolerance:
        raise RuntimeError(
            f"Observed category distribution exceeds tolerance: max error={maximum_category_error:.6f}."
        )
    if any(category_counts[category] == 0 for category in loader.categories):
        raise RuntimeError("At least one active category was never sampled.")
    if any(clip_counts[motion_id] == 0 for motion_id in loader.motion_ids):
        raise RuntimeError("At least one active clip was never sampled.")

    observed_clips = {
        motion_id: clip_counts[motion_id] / args.num_samples for motion_id in loader.motion_ids
    }
    report = {
        "status": "PASS",
        "manifest": str(args.manifest.expanduser().resolve()),
        "seed": args.seed,
        "num_samples": args.num_samples,
        "num_active_categories": len(loader.categories),
        "num_active_clips": loader.num_motions,
        "amp_observation_shape": list(amp_observations.shape),
        "amp_observation_finite": True,
        "cross_clip_histories": 0,
        "history_steps": args.num_amp_observations,
        "expected_category_probabilities": expected_categories,
        "observed_category_probabilities": observed_categories,
        "category_probability_errors": category_errors,
        "maximum_absolute_category_error": maximum_category_error,
        "observed_clip_probabilities": observed_clips,
        "clip_categories": dict(zip(loader.motion_ids, loader.motion_categories)),
    }
    output_path = args.output_json
    if output_path is None:
        output_path = args.manifest.with_name("g1_23dof_amp_multimotion_sampling_diagnostics.json")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Multi-motion diagnostic: PASS ({args.num_samples} histories)")
    print(f"AMP observation shape: {tuple(amp_observations.shape)}")
    print(f"Cross-clip histories: 0")
    for category in loader.categories:
        print(
            f"Category {category}: expected={expected_categories[category]:.6f}, "
            f"observed={observed_categories[category]:.6f}"
        )
    for motion_id in loader.motion_ids:
        print(f"Clip {motion_id}: observed={observed_clips[motion_id]:.6f}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    args_cli = _parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    if args_cli.fast_shutdown:
        exit_code = 0
        try:
            main(args_cli)
        except BaseException:
            traceback.print_exc()
            exit_code = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    else:
        try:
            main(args_cli)
        finally:
            simulation_app.close()
