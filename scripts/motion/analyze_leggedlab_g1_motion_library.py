"""Analyze and select motions from the legged_lab G1-29DoF AMP library.

The reported linear velocities are expressed in the root heading frame:
positive x is forward and positive y is left. Positive yaw rate is a left
turn under the standard right-handed z-up convention.

Pickle is unsafe for untrusted inputs. Only run this tool on the inspected
legged_lab repository checkout (or another source you explicitly trust).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from g1_amp_motion_utils import TARGET_DOF_NAMES, build_name_mapping, load_leggedlab_source


DEFAULT_REPOSITORY_URL = (
    "https://github.com/zitongbai/legged_lab/tree/main/"
    "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run"
)

CSV_FIELDS = (
    "filename",
    "frames",
    "fps",
    "duration_s",
    "mean_forward_mps",
    "mean_lateral_mps",
    "mean_yaw_rate_radps",
    "mean_abs_yaw_rate_radps",
    "net_yaw_deg",
    "mean_planar_speed_mps",
    "median_planar_speed_mps",
    "active_fraction",
    "root_height_min_m",
    "root_height_max_m",
    "source_family",
    "source_intent",
    "transition_clip",
    "kinematic_category",
    "semantic_consistency",
    "selection_score",
    "sha256",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", type=Path, required=True, help="Directory containing legged_lab .pkl motions")
    parser.add_argument(
        "--source-config",
        type=Path,
        required=True,
        help="legged_lab g1_29dof.yaml containing the authoritative lab_dof_names order",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/g1_amp_motion_library"),
        help="Directory for CSV/JSON reports",
    )
    parser.add_argument("--repository-url", default=DEFAULT_REPOSITORY_URL)
    parser.add_argument("--active-speed-threshold", type=float, default=0.20, help="Planar speed defining active frames")
    parser.add_argument("--minimum-forward-speed", type=float, default=0.15)
    parser.add_argument("--lateral-speed-threshold", type=float, default=0.12)
    parser.add_argument("--turn-rate-threshold", type=float, default=0.10)
    parser.add_argument("--minimum-duration", type=float, default=2.0)
    parser.add_argument("--max-per-category", type=int, default=2)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _yaw_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Return continuous z-yaw from normalized wxyz quaternions."""

    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.unwrap(yaw)


def _source_semantics(filename: str) -> tuple[str, bool, str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", filename.lower())
    is_transition = any(
        token in normalized
        for token in ("stand_to", "to_stand", "walk_to_run", "run_to_walk", "change_direction", "turn_change")
    )
    if "side_step" in normalized:
        family = "walk"
    elif "run" in normalized:
        family = "run"
    elif "walk" in normalized:
        family = "walk"
    else:
        family = "unknown"

    if "side_step_left" in normalized:
        intent = "side_left"
    elif "side_step_right" in normalized:
        intent = "side_right"
    elif "turn_around" in normalized or "turn_change" in normalized or "change_direction" in normalized:
        intent = "mixed_direction"
    elif "turn_left" in normalized:
        intent = "turn_left"
    elif "turn_right" in normalized:
        intent = "turn_right"
    elif "back" in normalized:
        intent = "backward"
    elif family == "run":
        intent = "straight"
    else:
        intent = "unknown"
    return family, is_transition, intent


def _classify(
    family: str,
    transition: bool,
    forward: float,
    lateral: float,
    yaw_rate: float,
    minimum_forward: float,
    lateral_threshold: float,
    turn_threshold: float,
    source_intent: str,
) -> str:
    prefix = "run" if family == "run" else "walk"
    suffix = "_transition" if transition else ""

    if forward < -minimum_forward:
        return f"{prefix}_backward{suffix}"
    if source_intent.startswith("side_") and abs(lateral) >= lateral_threshold:
        direction = "left" if lateral > 0.0 else "right"
        return f"{prefix}_side_{direction}{suffix}"
    if abs(yaw_rate) >= turn_threshold:
        direction = "left" if yaw_rate > 0.0 else "right"
        return f"{prefix}_turn_{direction}{suffix}"
    if abs(lateral) >= lateral_threshold and abs(lateral) >= 0.55 * max(abs(forward), 1.0e-6):
        direction = "left" if lateral > 0.0 else "right"
        return f"{prefix}_side_{direction}{suffix}"
    if forward >= minimum_forward:
        return f"{prefix}_straight{suffix}"
    return f"{prefix}_low_or_mixed_motion{suffix}"


def _semantic_consistency(source_intent: str, category: str) -> bool | None:
    measured_intent = category.split("_", maxsplit=1)[1] if "_" in category else category
    if source_intent in {"unknown", "mixed_direction"}:
        return None
    return source_intent == measured_intent


def _selection_score(record: dict[str, Any], minimum_duration: float) -> float:
    category = _base_category(record["kinematic_category"])
    minimum_active_fraction = 0.25 if "_side_" in category else 0.45
    if record["duration_s"] < minimum_duration or record["active_fraction"] < minimum_active_fraction:
        return -math.inf
    speed_mean = max(record["mean_planar_speed_mps"], 1.0e-6)
    speed_cv = record["std_planar_speed_mps"] / speed_mean
    score = 2.0 * record["active_fraction"] + min(record["duration_s"], 10.0) / 10.0
    score -= 0.15 * min(speed_cv, 5.0)
    if record["transition_clip"]:
        score -= 1.25
    if record["semantic_consistency"] is False:
        score -= 2.0
    return float(score)


def _base_category(category: str) -> str:
    suffix = "_transition"
    return category[: -len(suffix)] if category.endswith(suffix) else category


def _analyze_motion(
    path: Path,
    source_config: Path,
    active_speed_threshold: float,
    minimum_forward: float,
    lateral_threshold: float,
    turn_threshold: float,
    minimum_duration: float,
) -> dict[str, Any]:
    motion, source_names = load_leggedlab_source(path, source_config)
    if len(source_names) != 29 or motion["dof_pos"].shape[1] != 29:
        raise ValueError(f"{path.name}: expected source DoF count 29")

    root_pos = motion["root_pos"]
    root_rot = motion["root_rot"]
    fps = float(motion["fps"])
    frames = int(root_pos.shape[0])
    dt = 1.0 / fps
    duration = (frames - 1) * dt

    edge_order = 2 if frames >= 3 else 1
    velocity_world = np.gradient(root_pos, dt, axis=0, edge_order=edge_order)
    yaw = _yaw_from_wxyz(root_rot)
    yaw_rate = np.gradient(yaw, dt, edge_order=edge_order)

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    forward_velocity = cos_yaw * velocity_world[:, 0] + sin_yaw * velocity_world[:, 1]
    lateral_velocity = -sin_yaw * velocity_world[:, 0] + cos_yaw * velocity_world[:, 1]
    planar_speed = np.linalg.norm(velocity_world[:, :2], axis=1)

    family, transition, source_intent = _source_semantics(path.name)
    mean_forward = float(np.mean(forward_velocity))
    mean_lateral = float(np.mean(lateral_velocity))
    mean_yaw_rate = float(np.mean(yaw_rate))
    category = _classify(
        family,
        transition,
        mean_forward,
        mean_lateral,
        mean_yaw_rate,
        minimum_forward,
        lateral_threshold,
        turn_threshold,
        source_intent,
    )

    quaternion_norm_error = float(np.max(np.abs(np.linalg.norm(root_rot, axis=1) - 1.0)))
    record: dict[str, Any] = {
        "filename": path.name,
        "frames": frames,
        "fps": fps,
        "duration_s": float(duration),
        "mean_forward_mps": mean_forward,
        "mean_lateral_mps": mean_lateral,
        "mean_yaw_rate_radps": mean_yaw_rate,
        "mean_abs_yaw_rate_radps": float(np.mean(np.abs(yaw_rate))),
        "net_yaw_deg": float(np.rad2deg(yaw[-1] - yaw[0])),
        "mean_planar_speed_mps": float(np.mean(planar_speed)),
        "median_planar_speed_mps": float(np.median(planar_speed)),
        "std_planar_speed_mps": float(np.std(planar_speed)),
        "active_fraction": float(np.mean(planar_speed >= active_speed_threshold)),
        "root_height_min_m": float(np.min(root_pos[:, 2])),
        "root_height_max_m": float(np.max(root_pos[:, 2])),
        "quaternion_norm_max_error": quaternion_norm_error,
        "source_family": family,
        "source_intent": source_intent,
        "transition_clip": transition,
        "kinematic_category": category,
        "semantic_consistency": _semantic_consistency(source_intent, _base_category(category)),
        "sha256": _sha256(path),
    }
    record["selection_score"] = _selection_score(record, minimum_duration)
    return record


def _select_balanced(records: list[dict[str, Any]], max_per_category: int) -> tuple[list[dict[str, Any]], list[str]]:
    desired_categories = (
        "walk_straight",
        "walk_turn_left",
        "walk_turn_right",
        "walk_side_left",
        "walk_side_right",
        "run_straight",
        "run_turn_left",
        "run_turn_right",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_base_category(record["kinematic_category"])].append(record)

    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for category in desired_categories:
        expected_intent = category.split("_", maxsplit=1)[1]
        eligible = [
            record
            for record in grouped.get(category, [])
            if math.isfinite(record["selection_score"]) and record["source_intent"] == expected_intent
        ]
        eligible.sort(key=lambda item: (-item["selection_score"], item["filename"]))
        if not eligible:
            missing.append(category)
            continue
        # Prefer non-transition clips. A transition is used only if no steady clip exists.
        steady = [record for record in eligible if not record["transition_clip"]]
        pool = steady or eligible
        selected.extend(pool[:max_per_category])
    return selected, missing


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: _json_value(record.get(key)) for key in CSV_FIELDS})


def _write_markdown(
    path: Path,
    records: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    missing: list[str],
    removed_dof_names: list[str],
) -> None:
    lines = [
        "# legged_lab G1-29DoF motion library analysis",
        "",
        "Velocities are full-clip means in the root heading frame: +x is forward, +y is left, "
        "and positive yaw rate is a left/counter-clockwise turn.",
        "",
        "The source configuration has 29 DoFs and passes exact name-based mapping to the current 23DoF policy. "
        f"Removed DoFs: `{', '.join(removed_dof_names)}`.",
        "",
        "## All motions",
        "",
        "| Motion | Frames | FPS | Duration (s) | Forward (m/s) | Lateral (m/s) | Yaw rate (rad/s) | Measured category | Source intent |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for record in records:
        filename = record["filename"].replace("|", "\\|")
        lines.append(
            f"| `{filename}` | {record['frames']} | {record['fps']:.3f} | {record['duration_s']:.3f} "
            f"| {record['mean_forward_mps']:.3f} | {record['mean_lateral_mps']:.3f} "
            f"| {record['mean_yaw_rate_radps']:.3f} | {record['kinematic_category']} "
            f"| {record['source_intent']} |"
        )

    lines.extend(
        [
            "",
            "## Balanced pre-conversion candidates",
            "",
            "These files still require the strict 29→23 conversion, joint-limit check, FK reconstruction, "
            "and Isaac replay before they are training-ready.",
            "",
            "| Category | Motion | Duration (s) | Forward (m/s) | Lateral (m/s) | Yaw rate (rad/s) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for record in selected:
        filename = record["filename"].replace("|", "\\|")
        lines.append(
            f"| {_base_category(record['kinematic_category'])} | `{filename}` | {record['duration_s']:.3f} "
            f"| {record['mean_forward_mps']:.3f} | {record['mean_lateral_mps']:.3f} "
            f"| {record['mean_yaw_rate_radps']:.3f} |"
        )
    lines.extend(["", f"Missing desired categories: `{', '.join(missing) if missing else 'none'}`.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    motion_dir = args.motion_dir.expanduser().resolve()
    source_config = args.source_config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    paths = sorted(motion_dir.glob("*.pkl"), key=lambda item: item.name.lower())
    if not paths:
        raise FileNotFoundError(f"No .pkl files found in {motion_dir}")

    _, source_names = load_leggedlab_source(paths[0], source_config)
    mapping_indices, removed_dof_names = build_name_mapping(source_names)

    records = [
        _analyze_motion(
            path,
            source_config,
            args.active_speed_threshold,
            args.minimum_forward_speed,
            args.lateral_speed_threshold,
            args.turn_rate_threshold,
            args.minimum_duration,
        )
        for path in paths
    ]
    selected, missing = _select_balanced(records, args.max_per_category)
    selected_categories = sorted({_base_category(record["kinematic_category"]) for record in selected})
    category_probability = 1.0 / len(selected_categories) if selected_categories else 0.0
    selected_per_category = Counter(_base_category(record["kinematic_category"]) for record in selected)
    duplicate_hashes = {
        digest: filenames
        for digest, filenames in (
            (digest, [record["filename"] for record in records if record["sha256"] == digest])
            for digest in {record["sha256"] for record in records}
        )
        if len(filenames) > 1
    }
    category_counts = Counter(_base_category(record["kinematic_category"]) for record in records)

    thresholds = {
        "active_speed_mps": args.active_speed_threshold,
        "minimum_forward_speed_mps": args.minimum_forward_speed,
        "lateral_speed_mps": args.lateral_speed_threshold,
        "turn_rate_radps": args.turn_rate_threshold,
        "minimum_duration_s": args.minimum_duration,
        "max_per_category": args.max_per_category,
    }
    report = {
        "schema_version": 1,
        "source_repository": args.repository_url,
        "motion_directory": str(motion_dir),
        "source_joint_config": str(source_config),
        "joint_mapping": {
            "source_dof_count": len(source_names),
            "target_dof_count": len(TARGET_DOF_NAMES),
            "source_indices_in_target_order": mapping_indices,
            "target_dof_names": list(TARGET_DOF_NAMES),
            "removed_dof_names": removed_dof_names,
        },
        "coordinate_convention": {
            "linear_velocity_frame": "root heading frame",
            "forward": "+x",
            "lateral": "+y is left",
            "yaw_rate": "+z is left/counter-clockwise",
            "quaternion": "wxyz",
        },
        "thresholds": thresholds,
        "motion_count": len(records),
        "category_counts": dict(sorted(category_counts.items())),
        "exact_duplicate_files": duplicate_hashes,
        "records": [{key: _json_value(value) for key, value in record.items()} for record in records],
    }
    recommendation = {
        "schema_version": 1,
        "status": "pre_conversion_candidates",
        "warning": (
            "Every selected G1-29DoF source must still pass name-based 29->23 conversion, "
            "joint-limit validation, G1-23DoF FK reconstruction, and Isaac replay before AMP training."
        ),
        "joint_mapping_validated": True,
        "source_dof_count": len(source_names),
        "target_dof_count": len(TARGET_DOF_NAMES),
        "removed_dof_names": removed_dof_names,
        "selection_policy": "up to max_per_category highest-scoring steady clips per desired category",
        "sampling_policy": "uniform over action categories, then uniform over clips within each category",
        "thresholds": thresholds,
        "missing_categories": missing,
        "selected": [
            {
                "category": _base_category(record["kinematic_category"]),
                "filename": record["filename"],
                "duration_s": record["duration_s"],
                "mean_forward_mps": record["mean_forward_mps"],
                "mean_lateral_mps": record["mean_lateral_mps"],
                "mean_yaw_rate_radps": record["mean_yaw_rate_radps"],
                "selection_score": record["selection_score"],
                "suggested_sampling_probability": (
                    category_probability / selected_per_category[_base_category(record["kinematic_category"])]
                ),
                "sha256": record["sha256"],
            }
            for record in selected
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "leggedlab_g1_29dof_motion_statistics.csv"
    report_path = output_dir / "leggedlab_g1_29dof_motion_statistics.json"
    recommendation_path = output_dir / "g1_23dof_amp_candidate_manifest.json"
    markdown_path = output_dir / "README.md"
    _write_csv(csv_path, records)
    _write_markdown(markdown_path, records, selected, missing, removed_dof_names)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    recommendation_path.write_text(
        json.dumps(recommendation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Analyzed {len(records)} motions from {motion_dir}")
    print(f"Statistics CSV: {csv_path}")
    print(f"Statistics JSON: {report_path}")
    print(f"Readable report: {markdown_path}")
    print(f"Candidate manifest: {recommendation_path}")
    print(f"Selected {len(selected)} candidates; missing categories: {missing or 'none'}")


if __name__ == "__main__":
    main()
