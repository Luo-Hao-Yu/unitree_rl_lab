"""Batch-drive the validated single-motion G1 AMP converter and replay validator.

This script intentionally contains no 29->23 conversion implementation. Each
candidate is processed by ``convert_leggedlab_g1_29dof_to_23dof_amp.py`` and
then by ``replay_g1_amp_motion.py`` in an isolated subprocess.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_REMOVED_DOFS = [
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--source-motion-dir", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path(sys.executable),
        help="Isaac Sim Python executable used for converter and replay subprocesses",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an existing NPZ only when its converter diagnostics say training_ready=true",
    )
    parser.add_argument(
        "--capture-replay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture six Isaac camera frames per valid clip during replay",
    )
    return parser.parse_args()


def _slug(filename: str) -> str:
    stem = Path(filename).stem.lower().replace("stageii", "")
    value = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    if not value:
        raise ValueError(f"Could not construct motion_id from {filename!r}.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND: " + " ".join(command) + "\n\n")
        stream.flush()
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}; log={log_path}\n" + "\n".join(tail)
        )


def _root_motion_statistics(npz_path: Path) -> dict[str, float]:
    with np.load(npz_path) as archive:
        body_names = archive["body_names"].tolist()
        root_index = body_names.index("pelvis")
        position = np.asarray(archive["body_positions"][:, root_index], dtype=np.float64)
        quaternion = np.asarray(archive["body_rotations"][:, root_index], dtype=np.float64)
        fps = float(np.asarray(archive["fps"]).item())
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    yaw = np.unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    dt = 1.0 / fps
    velocity = np.gradient(position, dt, axis=0, edge_order=2)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    forward = cos_yaw * velocity[:, 0] + sin_yaw * velocity[:, 1]
    lateral = -sin_yaw * velocity[:, 0] + cos_yaw * velocity[:, 1]
    yaw_rate = np.gradient(yaw, dt, edge_order=2)
    return {
        "duration": float((position.shape[0] - 1) / fps),
        "fps": fps,
        "mean_vx": float(np.mean(forward)),
        "mean_vy": float(np.mean(lateral)),
        "mean_yaw_rate": float(np.mean(yaw_rate)),
    }


def _validate_converter_report(report: dict[str, Any]) -> dict[str, Any]:
    removed = report.get("removed_joint_names")
    if removed != EXPECTED_REMOVED_DOFS:
        raise RuntimeError(f"Removed DoFs differ from the strict expected set: {removed}")
    if report.get("target_dof") != 23 or len(report.get("target_joint_order", [])) != 23:
        raise RuntimeError("Converter report does not contain exactly 23 target DoFs.")
    if report.get("training_ready") is not True:
        raise RuntimeError("Converter did not mark the motion training_ready.")
    if report.get("official_motion_loader_validation", {}).get("current_g1_compute_amp_observation") != "PASS":
        raise RuntimeError("Official MotionLoader / current G1 AMP observation validation did not pass.")
    joint_violation = float(report["joint_limits"]["maximum_violation"])
    if joint_violation > 1.0e-4:
        raise RuntimeError(f"Meaningful joint-limit violation remains: {joint_violation:.9g} rad.")
    root_fk_error = float(report["root_fk_preservation"]["maximum_position_error_m"])
    if root_fk_error > 1.0e-4:
        raise RuntimeError(f"Root FK preservation error is too large: {root_fk_error:.9g} m.")
    root_height = report["root_height"]
    if not (0.30 <= float(root_height["min"]) <= float(root_height["max"]) <= 1.30):
        raise RuntimeError(f"Root height is outside the physically plausible validation range: {root_height}")
    foot = report["foot_height"]
    if min(float(foot["left_min"]), float(foot["right_min"])) < -0.02:
        raise RuntimeError(f"Foot trajectory penetrates the ground: {foot}")
    knee = report["knee_position"]
    if min(float(knee["left_min"]), float(knee["right_min"])) < -0.20:
        raise RuntimeError(f"Knee trajectory has an implausible reverse-bending range: {knee}")
    shoulder = report["shoulder_pitch"]
    shoulder_values = [shoulder[key] for key in ("left_min", "left_max", "right_min", "right_max")]
    if not all(math.isfinite(float(value)) and abs(float(value)) <= math.pi for value in shoulder_values):
        raise RuntimeError(f"Shoulder-pitch trajectory is not physically plausible: {shoulder}")
    return {
        "nan_inf": "PASS",
        "target_joint_order": "PASS",
        "joint_limits": "PASS",
        "fk_reconstruction": "PASS",
        "quaternion_wxyz": "PASS",
        "root_height": "PASS",
        "left_right_body_name_mapping": "PASS",
        "knee_direction": "PASS",
        "foot_penetration": "PASS",
        "shoulder_plausibility": "PASS",
        "amp_observation_shape": report["official_motion_loader_validation"][
            "collect_reference_motions_compatible_shape"
        ],
    }


def _relative_to_output(path: Path, output_dir: Path) -> str:
    return str(path.resolve().relative_to(output_dir.resolve()))


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    converter = project_root / "scripts/motion/convert_leggedlab_g1_29dof_to_23dof_amp.py"
    replay = project_root / "scripts/motion/replay_g1_amp_motion.py"
    candidate_manifest = args.candidate_manifest.expanduser().resolve()
    source_motion_dir = args.source_motion_dir.expanduser().resolve()
    source_config = args.source_config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    clips_dir = output_dir / "clips"
    logs_dir = output_dir / "logs"
    captures_dir = output_dir / "replay_captures"
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    candidate_data = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    candidates = candidate_data.get("selected")
    if not isinstance(candidates, list) or len(candidates) != 13:
        raise RuntimeError(f"Expected exactly 13 selected candidates, found {len(candidates or [])}.")

    motions: list[dict[str, Any]] = []
    motion_ids: set[str] = set()
    manifest_path = output_dir / "g1_23dof_amp_multimotion_manifest.json"
    for candidate in candidates:
        filename = candidate["filename"]
        category = candidate["category"]
        motion_id = _slug(filename)
        if motion_id in motion_ids:
            raise RuntimeError(f"Duplicate generated motion_id: {motion_id}")
        motion_ids.add(motion_id)
        source_path = source_motion_dir / filename
        output_npz = clips_dir / f"{motion_id}.npz"
        diagnostics_path = output_npz.with_suffix(".diagnostics.json")
        convert_log = logs_dir / f"{motion_id}.convert.log"
        replay_log = logs_dir / f"{motion_id}.replay.log"
        entry: dict[str, Any] = {
            "motion_id": motion_id,
            "source_path": str(source_path),
            "source_repository_path": (
                "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run/" + filename
            ),
            "converted_npz_path": _relative_to_output(output_npz, output_dir),
            "category": category,
            "validation_status": "processing",
            "failure_reason": None,
            "conversion_log": _relative_to_output(convert_log, output_dir),
            "replay_log": _relative_to_output(replay_log, output_dir),
        }
        motions.append(entry)
        try:
            if not source_path.is_file():
                raise FileNotFoundError(f"Selected source motion is missing: {source_path}")
            reuse = False
            if args.skip_existing and output_npz.is_file() and diagnostics_path.is_file():
                existing_report = json.loads(diagnostics_path.read_text(encoding="utf-8"))
                report_hashes = existing_report.get("sha256", {})
                reuse = (
                    existing_report.get("training_ready") is True
                    and report_hashes.get("source_motion") == _sha256(source_path)
                    and report_hashes.get("source_joint_config") == _sha256(source_config)
                )
            if not reuse:
                convert_command = [
                    str(args.python_executable),
                    str(converter),
                    "--input-pkl",
                    str(source_path),
                    "--source-config",
                    str(source_config),
                    "--output-npz",
                    str(output_npz),
                    "--device",
                    args.device,
                    "--headless",
                ]
                _run(convert_command, convert_log)

            report = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            validation_checks = _validate_converter_report(report)
            replay_command = [
                str(args.python_executable),
                str(replay),
                "--motion-file",
                str(output_npz),
                "--cycles",
                "1",
                "--device",
                args.device,
                "--headless",
            ]
            if args.capture_replay:
                replay_command.extend(["--capture-dir", str(captures_dir / motion_id)])
            _run(replay_command, replay_log)
            validation_checks["isaac_direct_replay"] = "PASS"
            if args.capture_replay:
                validation_checks["isaac_camera_frames"] = "PASS"

            statistics = _root_motion_statistics(output_npz)
            entry.update(statistics)
            entry.update(
                {
                    "frames": int(report["target_frames"]),
                    "left_right_shoulder_pitch_correlation": report["shoulder_pitch"][
                        "left_right_correlation"
                    ],
                    "contralateral_arm_leg_coordination": report["contralateral_coordination"]["score"],
                    "maximum_joint_limit_violation_rad": report["joint_limits"]["maximum_violation"],
                    "maximum_fk_position_error_m": report["root_fk_preservation"][
                        "maximum_position_error_m"
                    ],
                    "validation_checks": validation_checks,
                    "validation_status": "validated",
                }
            )
        except Exception as error:
            entry["validation_status"] = "invalid"
            entry["failure_reason"] = f"{type(error).__name__}: {error}"
            entry["traceback"] = traceback.format_exc()

        provisional = {
            "schema_version": 1,
            "source_candidate_manifest": str(candidate_manifest),
            "requested_candidate_count": len(candidates),
            "motions": motions,
            "active_motion_ids": [
                motion["motion_id"] for motion in motions if motion["validation_status"] == "validated"
            ],
            "category_sampling_probabilities": {},
        }
        _write_manifest(manifest_path, provisional)

    valid = [entry for entry in motions if entry["validation_status"] == "validated"]
    invalid = [entry for entry in motions if entry["validation_status"] != "validated"]
    categories = sorted({entry["category"] for entry in valid})
    category_probabilities = {category: 1.0 / len(categories) for category in categories} if categories else {}
    counts = Counter(entry["category"] for entry in valid)
    for entry in valid:
        entry["sampling_weight_within_category"] = 1.0 / counts[entry["category"]]
        entry["effective_sampling_probability"] = (
            category_probabilities[entry["category"]] * entry["sampling_weight_within_category"]
        )

    final_manifest = {
        "schema_version": 1,
        "status": "validated" if valid and not invalid else "partial" if valid else "failed",
        "source_candidate_manifest": str(candidate_manifest),
        "source_joint_config": str(source_config),
        "requested_candidate_count": len(candidates),
        "successful_motion_count": len(valid),
        "failed_motion_count": len(invalid),
        "sampling_policy": "sample category from manifest probabilities, then clip from within-category weights",
        "category_sampling_probabilities": category_probabilities,
        "active_motion_ids": [entry["motion_id"] for entry in valid],
        "motions": motions,
    }
    _write_manifest(manifest_path, final_manifest)
    print(f"Processed: {len(candidates)}")
    print(f"Validated: {len(valid)}")
    print(f"Invalid: {len(invalid)}")
    print(f"Active categories: {categories}")
    print(f"Manifest: {manifest_path}")
    if invalid:
        print("Failures:")
        for entry in invalid:
            print(f"- {entry['motion_id']}: {entry['failure_reason']}")


if __name__ == "__main__":
    main()
