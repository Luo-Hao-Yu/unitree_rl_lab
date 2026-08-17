"""Convert one legged_lab G1-29DoF pickle to the current G1-23DoF AMP NPZ.

The 29->23 conversion is name based. Target body states are reconstructed by
the current G1-23DoF Isaac articulation; source key-body states are never used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-pkl", type=Path, required=True, help="legged_lab G1-29DoF motion pickle")
parser.add_argument(
    "--source-config",
    type=Path,
    required=True,
    help="Inspected legged_lab g1_29dof.yaml containing lab_dof_names",
)
parser.add_argument("--output-npz", type=Path, required=True, help="Target G1-23DoF AMP NPZ")
parser.add_argument(
    "--small-violation-tolerance",
    type=float,
    default=1.0e-4,
    help="Maximum numerical joint-limit violation (rad) that may be clipped and reported",
)
parser.add_argument(
    "--fast-shutdown",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use Isaac Sim fast shutdown (recommended for standalone conversion)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_tasks.direct.humanoid_amp.motions import MotionLoader

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG

from g1_amp_motion_utils import (
    KEY_BODY_NAMES,
    MOTION_ROOT_BODY,
    NPZ_FIELDS,
    REFERENCE_BODY,
    TARGET_DOF_NAMES,
    build_name_mapping,
    compute_motion_diagnostics,
    finite_difference,
    load_leggedlab_source,
    make_quaternions_continuous,
    quaternion_angular_velocity_world,
    save_json,
    validate_npz_arrays,
)


def _joint_limit_report(
    positions: np.ndarray, limits: np.ndarray, joint_names: tuple[str, ...]
) -> tuple[dict[str, object], np.ndarray]:
    lower_violation = np.maximum(limits[None, :, 0] - positions, 0.0)
    upper_violation = np.maximum(positions - limits[None, :, 1], 0.0)
    violation = np.maximum(lower_violation, upper_violation)
    flat_index = int(np.argmax(violation))
    frame, joint = np.unravel_index(flat_index, violation.shape)
    maximum = float(violation[frame, joint])
    report = {
        "maximum_violation": maximum,
        "joint_name": joint_names[joint],
        "frame_index": int(frame),
        "position": float(positions[frame, joint]),
        "lower_limit": float(limits[joint, 0]),
        "upper_limit": float(limits[joint, 1]),
        "clipped": False,
    }
    return report, violation


def _reconstruct_body_states(
    robot: Articulation,
    sim: sim_utils.SimulationContext,
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    policy_joint_ids: list[int],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    num_frames = dof_pos.shape[0]
    body_names = list(robot.data.body_names)
    body_positions = np.empty((num_frames, len(body_names), 3), dtype=np.float64)
    body_rotations = np.empty((num_frames, len(body_names), 4), dtype=np.float64)
    full_joint_pos = robot.data.default_joint_pos.clone()
    full_joint_vel = robot.data.default_joint_vel.clone()

    for frame in range(num_frames):
        root_pose = torch.as_tensor(
            np.concatenate((root_pos[frame], root_rot[frame]))[None], dtype=torch.float32, device=sim.device
        )
        full_joint_pos[:, policy_joint_ids] = torch.as_tensor(dof_pos[frame], dtype=torch.float32, device=sim.device)
        full_joint_vel[:, policy_joint_ids] = torch.as_tensor(dof_vel[frame], dtype=torch.float32, device=sim.device)
        robot.write_root_link_pose_to_sim(root_pose)
        robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel)
        sim.forward()
        robot.update(0.0)
        body_positions[frame] = robot.data.body_pos_w[0].detach().cpu().numpy()
        body_rotations[frame] = robot.data.body_quat_w[0].detach().cpu().numpy()

    return body_names, body_positions, make_quaternions_continuous(body_rotations)


def _verify_official_loader(output_path: Path, expected_frames: int, expected_bodies: int) -> dict[str, object]:
    loader = MotionLoader(motion_file=str(output_path), device="cpu")
    if loader.num_frames != expected_frames or loader.num_dofs != 23 or loader.num_bodies != expected_bodies:
        raise RuntimeError(
            "Official MotionLoader shape verification failed: "
            f"frames={loader.num_frames}, dofs={loader.num_dofs}, bodies={loader.num_bodies}."
        )
    motion_dof_indices = loader.get_dof_index(list(TARGET_DOF_NAMES))
    if motion_dof_indices != list(range(23)):
        raise RuntimeError("Official MotionLoader did not preserve the target policy joint order.")
    required_bodies = [MOTION_ROOT_BODY, REFERENCE_BODY, *KEY_BODY_NAMES]
    loader.get_body_index(required_bodies)
    sample_times = np.asarray([0.0, 0.5 * loader.duration, loader.duration], dtype=np.float64)
    samples = loader.sample(num_samples=3, times=sample_times)
    if any(not torch.all(torch.isfinite(value)) for value in samples):
        raise RuntimeError("Official MotionLoader returned NaN/Inf during endpoint/interior sampling.")
    single_frame_amp_dim = 23 + 23 + 1 + 6 + 3 + 3 + 4 * 3
    if single_frame_amp_dim != 71:
        raise RuntimeError(f"Unexpected G1 AMP frame dimension: {single_frame_amp_dim}")

    # Exercise the exact function used by both collect_reference_motions() and
    # policy extras["amp_obs"], not just the generic MotionLoader schema.
    amp_env_module = importlib.import_module(
        "unitree_rl_lab.tasks.locomotion.robots.g1.23dof_amp.g1_amp_env"
    )
    history_base_times = np.asarray(
        [loader.dt, max(loader.dt, 0.5 * loader.duration), loader.duration], dtype=np.float64
    )
    history_times = (
        np.expand_dims(history_base_times, axis=-1) - loader.dt * np.arange(2, dtype=np.float64)
    ).reshape(-1)
    history_samples = loader.sample(num_samples=history_times.shape[0], times=history_times)
    reference_body_index = loader.get_body_index([REFERENCE_BODY])[0]
    key_body_indices = loader.get_body_index(list(KEY_BODY_NAMES))
    amp_observations = amp_env_module.compute_amp_observation(
        history_samples[0][:, motion_dof_indices],
        history_samples[1][:, motion_dof_indices],
        history_samples[2][:, reference_body_index],
        history_samples[3][:, reference_body_index],
        history_samples[4][:, reference_body_index],
        history_samples[5][:, reference_body_index],
        history_samples[2][:, key_body_indices],
    ).view(3, 2 * single_frame_amp_dim)
    if tuple(amp_observations.shape) != (3, 142) or not torch.all(torch.isfinite(amp_observations)):
        raise RuntimeError(
            f"Current G1 collect_reference_motions-compatible AMP observation failed: {amp_observations.shape}."
        )
    return {
        "loader": "isaaclab_tasks.direct.humanoid_amp.motions.MotionLoader",
        "frames": loader.num_frames,
        "dofs": loader.num_dofs,
        "bodies": loader.num_bodies,
        "duration_s": float(loader.duration),
        "sampled_times_s": sample_times.tolist(),
        "single_frame_amp_observation_dim": single_frame_amp_dim,
        "two_frame_amp_observation_dim": 2 * single_frame_amp_dim,
        "current_g1_compute_amp_observation": "PASS",
        "collect_reference_motions_compatible_shape": list(amp_observations.shape),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    amp_cfg_module = importlib.import_module(
        "unitree_rl_lab.tasks.locomotion.robots.g1.23dof_amp.g1_amp_env_cfg"
    )
    if tuple(amp_cfg_module.G1_23DOF_POLICY_JOINT_NAMES) != TARGET_DOF_NAMES:
        raise RuntimeError("Converter target order has drifted from G1AmpEnvCfg.policy_joint_names.")
    if tuple(amp_cfg_module.G1_AMP_KEY_BODY_NAMES) != (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_roll_rubber_hand",
        "right_wrist_roll_rubber_hand",
    ):
        raise RuntimeError("Converter key-body definition has drifted from G1AmpEnvCfg.key_body_names.")

    source_motion, source_names = load_leggedlab_source(args_cli.input_pkl, args_cli.source_config)
    mapping, removed_names = build_name_mapping(source_names)
    fps = float(source_motion["fps"])
    root_pos = source_motion["root_pos"]
    root_rot = source_motion["root_rot"]
    dof_pos = source_motion["dof_pos"][:, mapping].copy()
    if dof_pos.shape[1] != 23:
        raise RuntimeError(f"Target DoF count is {dof_pos.shape[1]}, expected 23.")

    source_velocity_key = next(
        (key for key in ("dof_vel", "dof_velocity", "dof_velocities") if key in source_motion), None
    )
    velocity_source = "central finite difference from converted 23DoF positions"
    if source_velocity_key is not None:
        source_dof_vel = np.asarray(source_motion[source_velocity_key], dtype=np.float64)
        if source_dof_vel.shape != source_motion["dof_pos"].shape or not np.all(np.isfinite(source_dof_vel)):
            raise ValueError(
                f"Source field {source_velocity_key!r} is not a finite array with shape "
                f"{source_motion['dof_pos'].shape}: got {source_dof_vel.shape}."
            )
        dof_vel = source_dof_vel[:, mapping]
        velocity_source = f"name-mapped source field {source_velocity_key}"
    else:
        dof_vel = finite_difference(dof_pos, fps)

    # FK uses direct state writes and does not integrate physics. Keep the
    # normal Isaac physics step independent from the preserved motion FPS.
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, gravity=(0.0, 0.0, 0.0))
    )
    robot = Articulation(UNITREE_G1_23DOF_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    robot.reset()

    policy_joint_ids, resolved_names = robot.find_joints(list(TARGET_DOF_NAMES), preserve_order=True)
    if tuple(resolved_names) != TARGET_DOF_NAMES or len(policy_joint_ids) != 23:
        raise RuntimeError(
            f"Current articulation joint mismatch; expected {TARGET_DOF_NAMES}, resolved {tuple(resolved_names)}."
        )
    limits = robot.data.joint_pos_limits[0, policy_joint_ids].detach().cpu().numpy().astype(np.float64)
    limit_report, violation = _joint_limit_report(dof_pos, limits, TARGET_DOF_NAMES)
    max_violation = float(limit_report["maximum_violation"])
    if max_violation > args_cli.small_violation_tolerance:
        raise ValueError(
            "Meaningful target joint-limit violation; refusing to write training-ready NPZ. "
            f"joint={limit_report['joint_name']}, frame={limit_report['frame_index']}, "
            f"magnitude={max_violation:.9f} rad, tolerance={args_cli.small_violation_tolerance:.9f} rad."
        )
    if max_violation > 0.0:
        violating = violation > 0.0
        dof_pos = np.clip(dof_pos, limits[:, 0], limits[:, 1])
        dof_vel = finite_difference(dof_pos, fps)
        limit_report["clipped"] = True
        limit_report["clipped_values"] = int(np.count_nonzero(violating))
        velocity_source = "central finite difference after explicitly reported numerical limit clipping"

    body_names, body_pos, body_rot = _reconstruct_body_states(
        robot, sim, root_pos, root_rot, dof_pos, dof_vel, policy_joint_ids
    )
    if body_names.count(MOTION_ROOT_BODY) != 1 or body_names.count(REFERENCE_BODY) != 1:
        raise RuntimeError(f"Current articulation is missing required pelvis/torso bodies: {body_names}")
    body_lin_vel = finite_difference(body_pos, fps)
    body_ang_vel = quaternion_angular_velocity_world(body_rot, fps)

    root_body_index = body_names.index(MOTION_ROOT_BODY)
    root_position_fk_error = float(np.max(np.abs(body_pos[:, root_body_index] - root_pos)))
    root_quat_alignment = np.abs(np.sum(body_rot[:, root_body_index] * root_rot, axis=-1))
    root_rotation_fk_error = float(np.max(1.0 - root_quat_alignment))
    if root_position_fk_error > 1.0e-4 or root_rotation_fk_error > 1.0e-5:
        raise RuntimeError(
            "Isaac root reconstruction does not preserve the source root trajectory: "
            f"position_error={root_position_fk_error:.6g}, quaternion_alignment_error={root_rotation_fk_error:.6g}."
        )

    output_data = {
        "fps": np.asarray(fps, dtype=np.float64),
        "dof_names": np.asarray(TARGET_DOF_NAMES, dtype=np.str_),
        "body_names": np.asarray(body_names, dtype=np.str_),
        "dof_positions": dof_pos.astype(np.float32),
        "dof_velocities": dof_vel.astype(np.float32),
        "body_positions": body_pos.astype(np.float32),
        "body_rotations": body_rot.astype(np.float32),
        "body_linear_velocities": body_lin_vel.astype(np.float32),
        "body_angular_velocities": body_ang_vel.astype(np.float32),
    }
    validate_npz_arrays(output_data)
    args_cli.output_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{args_cli.output_npz.stem}.", suffix=".npz", dir=args_cli.output_npz.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        np.savez(temporary_path, **output_data)
        with np.load(temporary_path) as saved:
            saved_data = {key: saved[key] for key in saved.files}
        if tuple(saved_data) != NPZ_FIELDS:
            raise RuntimeError(f"Written NPZ field order/schema mismatch: {tuple(saved_data)}")
        validate_npz_arrays(saved_data)
        loader_report = _verify_official_loader(temporary_path, dof_pos.shape[0], len(body_names))
        temporary_path.chmod(0o644)
        os.replace(temporary_path, args_cli.output_npz)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    report, trajectories = compute_motion_diagnostics(saved_data, removed_names)
    report.update(
        {
            "source_motion": str(args_cli.input_pkl.resolve()),
            "source_joint_config": str(args_cli.source_config.resolve()),
            "output_npz": str(args_cli.output_npz.resolve()),
            "source_joint_order": source_names,
            "target_body_order": body_names,
            "mapping_target_to_source_index": {
                name: int(source_index) for name, source_index in zip(TARGET_DOF_NAMES, mapping)
            },
            "joint_velocity_source": velocity_source,
            "joint_limits": limit_report,
            "root_fk_preservation": {
                "maximum_position_error_m": root_position_fk_error,
                "maximum_quaternion_alignment_error": root_rotation_fk_error,
            },
            "npz_fields": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in saved_data.items()
            },
            "conventions": {
                "joint_position_unit": "radian",
                "joint_velocity_unit": "radian/second",
                "position_unit": "meter",
                "linear_velocity_unit": "meter/second",
                "angular_velocity_unit": "radian/second",
                "body_position_frame": "world",
                "body_rotation_frame": "world",
                "quaternion_order": "wxyz",
                "body_velocity_frame": "world",
                "root_trajectory_modified": False,
                "resampled": False,
            },
            "sha256": {
                "source_motion": _sha256(args_cli.input_pkl),
                "source_joint_config": _sha256(args_cli.source_config),
                "output_npz": _sha256(args_cli.output_npz),
            },
            "official_motion_loader_validation": loader_report,
            "source_key_body_positions_copied": False,
            "target_body_states_reconstructed_with_current_g1_23dof_articulation": True,
            "training_ready": True,
        }
    )
    report_path = args_cli.output_npz.with_suffix(".diagnostics.json")
    trajectories_path = args_cli.output_npz.with_suffix(".trajectories.npz")
    save_json(report_path, report)
    np.savez(trajectories_path, **{name: value.astype(np.float32) for name, value in trajectories.items()})
    trajectories_path.chmod(0o644)

    print("\n=== G1-29DoF -> G1-23DoF AMP conversion complete ===")
    print(f"Source: {args_cli.input_pkl.resolve()}")
    print(f"Output: {args_cli.output_npz.resolve()}")
    print(f"Frames: {dof_pos.shape[0]} -> {dof_pos.shape[0]}")
    print(f"FPS: {fps:.12g} -> {fps:.12g} (preserved; no resampling)")
    print(f"DoFs: 29 -> 23")
    print(f"Removed: {removed_names}")
    print(f"Target order: {list(TARGET_DOF_NAMES)}")
    print(f"Maximum joint-limit violation: {max_violation:.9g} rad")
    print(f"Root height: {report['root_height']['min']:.6f} .. {report['root_height']['max']:.6f} m")
    print(f"Contralateral coordination score: {report['contralateral_coordination']['score']}")
    print(f"Diagnostics: {report_path.resolve()}")
    print(f"Trajectories: {trajectories_path.resolve()}")
    print("Official MotionLoader validation: PASS")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
