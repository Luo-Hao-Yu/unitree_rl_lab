"""Shared validation and numerical utilities for G1-23DoF AMP motions.

This module deliberately has no Isaac Sim imports so that the converter and
replay tools share one definition of the motion schema and diagnostics.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml


TARGET_DOF_NAMES = (
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

MOTION_ROOT_BODY = "pelvis"
REFERENCE_BODY = "torso_link"
KEY_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
)

NPZ_FIELDS = (
    "fps",
    "dof_names",
    "body_names",
    "dof_positions",
    "dof_velocities",
    "body_positions",
    "body_rotations",
    "body_linear_velocities",
    "body_angular_velocities",
)


def _require_finite(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        bad = np.argwhere(~np.isfinite(value))[0].tolist()
        raise ValueError(f"{name} contains NaN/Inf at index {bad}.")


def _duplicates(names: list[str] | tuple[str, ...]) -> list[str]:
    return sorted(name for name, count in Counter(names).items() if count > 1)


def load_leggedlab_source(
    motion_path: str | Path, source_config_path: str | Path
) -> tuple[dict[str, Any], list[str]]:
    """Load one trusted legged_lab pickle and its authoritative Lab DoF order."""

    motion_path = Path(motion_path).expanduser().resolve()
    source_config_path = Path(source_config_path).expanduser().resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(f"Source motion does not exist: {motion_path}")
    if not source_config_path.is_file():
        raise FileNotFoundError(f"Source joint configuration does not exist: {source_config_path}")

    # Pickle is unsafe for untrusted inputs. This converter only accepts the
    # inspected legged_lab dataset checked out by the user.
    with motion_path.open("rb") as stream:
        motion = pickle.load(stream)
    with source_config_path.open("r", encoding="utf-8") as stream:
        source_cfg = yaml.safe_load(stream)

    if not isinstance(motion, dict):
        raise TypeError(f"Expected a dict in {motion_path}, got {type(motion).__name__}.")
    required = {"fps", "root_pos", "root_rot", "dof_pos"}
    missing = sorted(required.difference(motion))
    if missing:
        raise KeyError(f"Source motion is missing required fields: {missing}")
    if not isinstance(source_cfg, dict) or "lab_dof_names" not in source_cfg:
        raise KeyError("Source YAML must contain the authoritative 'lab_dof_names' list.")

    source_names = [str(name) for name in source_cfg["lab_dof_names"]]
    duplicate_names = _duplicates(source_names)
    if duplicate_names:
        raise ValueError(f"Source joint configuration contains duplicate names: {duplicate_names}")
    if len(source_names) != 29:
        raise ValueError(f"Expected exactly 29 source DoFs, found {len(source_names)}.")

    fps = float(np.asarray(motion["fps"]).item())
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Invalid source FPS: {fps}")
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Expected root_pos shape [N, 3], got {root_pos.shape}.")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"Expected root_rot shape [N, 4] (wxyz), got {root_rot.shape}.")
    if dof_pos.ndim != 2 or dof_pos.shape[1] != len(source_names):
        raise ValueError(
            f"Expected dof_pos shape [N, {len(source_names)}] from lab_dof_names, got {dof_pos.shape}."
        )
    if root_pos.shape[0] != dof_pos.shape[0] or root_rot.shape[0] != dof_pos.shape[0]:
        raise ValueError(
            "Source frame counts differ: "
            f"root_pos={root_pos.shape[0]}, root_rot={root_rot.shape[0]}, dof_pos={dof_pos.shape[0]}."
        )
    if dof_pos.shape[0] < 2:
        raise ValueError("At least two source frames are required.")
    _require_finite("root_pos", root_pos)
    _require_finite("root_rot", root_rot)
    _require_finite("dof_pos", dof_pos)

    quat_norm = np.linalg.norm(root_rot, axis=-1)
    max_quat_error = float(np.max(np.abs(quat_norm - 1.0)))
    if max_quat_error > 1.0e-3:
        raise ValueError(f"Source root_rot is not normalized; maximum norm error is {max_quat_error:.6g}.")

    # Keep the source trajectory and convention unchanged, correcting only
    # floating-point normalization noise before sending quaternions to PhysX.
    root_rot = root_rot / quat_norm[:, None]
    motion = dict(motion)
    motion["fps"] = fps
    motion["root_pos"] = root_pos
    motion["root_rot"] = make_quaternions_continuous(root_rot)
    motion["dof_pos"] = dof_pos
    return motion, source_names


def build_name_mapping(source_names: list[str]) -> tuple[list[int], list[str]]:
    """Build and validate the deterministic source-to-target name mapping."""

    duplicate_source = _duplicates(source_names)
    duplicate_target = _duplicates(TARGET_DOF_NAMES)
    if duplicate_source:
        raise ValueError(f"Source joint names are duplicated: {duplicate_source}")
    if duplicate_target:
        raise RuntimeError(f"Target joint names are duplicated: {duplicate_target}")

    counts = Counter(source_names)
    invalid = [name for name in TARGET_DOF_NAMES if counts[name] != 1]
    if invalid:
        details = {name: counts[name] for name in invalid}
        raise ValueError(f"Every target joint must occur exactly once in the source: {details}")

    mapping = [source_names.index(name) for name in TARGET_DOF_NAMES]
    if len(mapping) != 23 or len(set(mapping)) != 23:
        raise RuntimeError(f"Invalid 29->23 mapping: {mapping}")
    removed = [name for name in source_names if name not in TARGET_DOF_NAMES]
    if len(removed) != 6:
        raise ValueError(f"Expected exactly six removed source DoFs, found {len(removed)}: {removed}")
    return mapping, removed


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    """Differentiate a uniformly sampled trajectory with central differences."""

    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 2:
        raise ValueError("Finite differences require at least two frames.")
    edge_order = 2 if values.shape[0] >= 3 else 1
    return np.gradient(values, 1.0 / fps, axis=0, edge_order=edge_order)


def make_quaternions_continuous(quaternions: np.ndarray) -> np.ndarray:
    """Normalize wxyz quaternions and choose temporally continuous signs."""

    result = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-12):
        raise ValueError("Encountered a zero-norm quaternion.")
    result /= norms
    if result.ndim == 2:
        result = result[:, None, :]
        squeeze = True
    elif result.ndim == 3:
        squeeze = False
    else:
        raise ValueError(f"Expected quaternion shape [N, 4] or [N, B, 4], got {result.shape}.")
    for frame in range(1, result.shape[0]):
        flip = np.sum(result[frame - 1] * result[frame], axis=-1) < 0.0
        result[frame, flip] *= -1.0
    return result[:, 0] if squeeze else result


def _quat_multiply(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = np.moveaxis(q0, -1, 0)
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    return np.stack(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ),
        axis=-1,
    )


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    result = q.copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_angular_velocity_world(quaternions: np.ndarray, fps: float) -> np.ndarray:
    """Compute world-frame angular velocity from wxyz body orientations."""

    q = make_quaternions_continuous(quaternions)
    if q.ndim == 2:
        q = q[:, None, :]
        squeeze = True
    else:
        squeeze = False
    delta = _quat_multiply(q[1:], _quat_conjugate(q[:-1]))
    delta = make_quaternions_continuous(delta)
    vector = delta[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(delta[..., 0], 0.0, None))
    axis = np.divide(vector, vector_norm[..., None], out=np.zeros_like(vector), where=vector_norm[..., None] > 1e-12)
    interval_velocity = axis * angle[..., None] * fps

    velocity = np.empty((q.shape[0], q.shape[1], 3), dtype=np.float64)
    velocity[0] = interval_velocity[0]
    velocity[-1] = interval_velocity[-1]
    if q.shape[0] > 2:
        velocity[1:-1] = 0.5 * (interval_velocity[:-1] + interval_velocity[1:])
    return velocity[:, 0] if squeeze else velocity


def rotate_inverse_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate world vectors into the local frame represented by wxyz quaternions."""

    q_vec = quaternion[..., 1:]
    qw = quaternion[..., :1]
    a = vector * (2.0 * qw * qw - 1.0)
    b = 2.0 * np.cross(q_vec, vector) * qw
    c = 2.0 * q_vec * np.sum(q_vec * vector, axis=-1, keepdims=True)
    return a - b + c


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) < 1.0e-8 or np.std(b) < 1.0e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def compute_motion_diagnostics(
    data: dict[str, np.ndarray], removed_names: list[str]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute gait, arm-leg coordination and trajectory diagnostics."""

    dof_names = data["dof_names"].tolist()
    body_names = data["body_names"].tolist()
    q = np.asarray(data["dof_positions"], dtype=np.float64)
    body_pos = np.asarray(data["body_positions"], dtype=np.float64)
    body_quat = np.asarray(data["body_rotations"], dtype=np.float64)
    fps = float(np.asarray(data["fps"]).item())
    num_frames = q.shape[0]

    body_index = {name: body_names.index(name) for name in (REFERENCE_BODY, *KEY_BODY_NAMES)}
    joint_index = {
        name: dof_names.index(name)
        for name in (
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_knee_joint",
            "right_knee_joint",
        )
    }
    torso_pos = body_pos[:, body_index[REFERENCE_BODY]]
    torso_quat = body_quat[:, body_index[REFERENCE_BODY]]
    local_key_positions: dict[str, np.ndarray] = {}
    for name in KEY_BODY_NAMES:
        local_key_positions[name] = rotate_inverse_wxyz(
            torso_quat, body_pos[:, body_index[name]] - torso_pos
        )

    left_foot = local_key_positions[KEY_BODY_NAMES[0]]
    right_foot = local_key_positions[KEY_BODY_NAMES[1]]
    left_hand = local_key_positions[KEY_BODY_NAMES[2]]
    right_hand = local_key_positions[KEY_BODY_NAMES[3]]
    foot_forward_difference = left_foot[:, 0] - right_foot[:, 0]
    contralateral_hand_difference = right_hand[:, 0] - left_hand[:, 0]

    left_foot_height = body_pos[:, body_index[KEY_BODY_NAMES[0]], 2]
    right_foot_height = body_pos[:, body_index[KEY_BODY_NAMES[1]], 2]
    left_shoulder = q[:, joint_index["left_shoulder_pitch_joint"]]
    right_shoulder = q[:, joint_index["right_shoulder_pitch_joint"]]
    time = np.arange(num_frames, dtype=np.float64) / fps

    trajectory = {
        "time": time,
        "left_foot_height": left_foot_height,
        "right_foot_height": right_foot_height,
        "left_shoulder_pitch": left_shoulder,
        "right_shoulder_pitch": right_shoulder,
        "left_foot_forward_local": left_foot[:, 0],
        "right_foot_forward_local": right_foot[:, 0],
        "left_hand_forward_local": left_hand[:, 0],
        "right_hand_forward_local": right_hand[:, 0],
    }
    report: dict[str, Any] = {
        "source_frames": num_frames,
        "target_frames": num_frames,
        "source_fps": fps,
        "target_fps": fps,
        "source_dof": 29,
        "target_dof": len(dof_names),
        "removed_joint_names": removed_names,
        "target_joint_order": dof_names,
        "root_height": {
            "min": float(np.min(body_pos[:, body_names.index(MOTION_ROOT_BODY), 2])),
            "max": float(np.max(body_pos[:, body_names.index(MOTION_ROOT_BODY), 2])),
        },
        "foot_height": {
            "left_min": float(np.min(left_foot_height)),
            "left_max": float(np.max(left_foot_height)),
            "right_min": float(np.min(right_foot_height)),
            "right_max": float(np.max(right_foot_height)),
            "left_right_correlation": safe_correlation(left_foot_height, right_foot_height),
            "height_difference_sign_changes": int(
                np.count_nonzero(np.diff(np.signbit(left_foot_height - right_foot_height)))
            ),
        },
        "shoulder_pitch": {
            "left_min": float(np.min(left_shoulder)),
            "left_max": float(np.max(left_shoulder)),
            "right_min": float(np.min(right_shoulder)),
            "right_max": float(np.max(right_shoulder)),
            "left_right_correlation": safe_correlation(left_shoulder, right_shoulder),
        },
        "knee_position": {
            "left_min": float(np.min(q[:, joint_index["left_knee_joint"]])),
            "left_max": float(np.max(q[:, joint_index["left_knee_joint"]])),
            "right_min": float(np.min(q[:, joint_index["right_knee_joint"]])),
            "right_max": float(np.max(q[:, joint_index["right_knee_joint"]])),
        },
        "contralateral_coordination": {
            "metric": "corr(left_foot_x-right_foot_x, right_hand_x-left_hand_x) in torso frame",
            "score": safe_correlation(foot_forward_difference, contralateral_hand_difference),
            "left_leg_right_arm": safe_correlation(left_foot[:, 0], right_hand[:, 0]),
            "right_leg_left_arm": safe_correlation(right_foot[:, 0], left_hand[:, 0]),
            "left_leg_left_arm_ipsilateral": safe_correlation(left_foot[:, 0], left_hand[:, 0]),
            "right_leg_right_arm_ipsilateral": safe_correlation(right_foot[:, 0], right_hand[:, 0]),
        },
    }
    return report, trajectory


def validate_npz_arrays(data: dict[str, np.ndarray]) -> None:
    missing = [name for name in NPZ_FIELDS if name not in data]
    extra = [name for name in data if name not in NPZ_FIELDS]
    if missing or extra:
        raise ValueError(f"AMP NPZ schema mismatch; missing={missing}, extra={extra}.")
    if data["dof_names"].tolist() != list(TARGET_DOF_NAMES):
        raise ValueError("NPZ DoF order does not exactly match G1-23DoF policy order.")
    if _duplicates(data["body_names"].tolist()):
        raise ValueError("NPZ contains duplicate body names.")
    for name in (MOTION_ROOT_BODY, REFERENCE_BODY, *KEY_BODY_NAMES):
        if data["body_names"].tolist().count(name) != 1:
            raise ValueError(f"Required body '{name}' must occur exactly once in NPZ body_names.")

    n = data["dof_positions"].shape[0]
    d = len(TARGET_DOF_NAMES)
    b = data["body_names"].shape[0]
    expected_shapes = {
        "dof_positions": (n, d),
        "dof_velocities": (n, d),
        "body_positions": (n, b, 3),
        "body_rotations": (n, b, 4),
        "body_linear_velocities": (n, b, 3),
        "body_angular_velocities": (n, b, 3),
    }
    for name, shape in expected_shapes.items():
        if data[name].shape != shape:
            raise ValueError(f"{name} has shape {data[name].shape}; expected {shape}.")
        _require_finite(name, data[name])
    quat_error = np.max(np.abs(np.linalg.norm(data["body_rotations"], axis=-1) - 1.0))
    if quat_error > 1.0e-4:
        raise ValueError(f"NPZ body quaternion maximum norm error is {quat_error:.6g}.")


def save_json(path: str | Path, report: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
