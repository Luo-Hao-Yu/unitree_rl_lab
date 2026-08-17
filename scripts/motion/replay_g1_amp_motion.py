"""Replay and validate a converted G1-23DoF AMP NPZ without reinforcement learning."""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion-file", type=Path, required=True, help="Converted G1-23DoF AMP NPZ")
parser.add_argument("--cycles", type=int, default=1, help="Number of complete replay cycles; use 0 to loop")
parser.add_argument("--real-time", action="store_true", help="Throttle replay to the motion FPS")
parser.add_argument(
    "--capture-dir",
    type=Path,
    default=None,
    help="Optionally save six evenly spaced first-cycle RGB frames from an Isaac camera",
)
parser.add_argument(
    "--fast-shutdown",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use Isaac Sim fast shutdown when a finite replay finishes",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.cycles < 0:
    parser.error("--cycles must be >= 0")
if args_cli.capture_dir is not None:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab_tasks.direct.humanoid_amp.motions import MotionLoader

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG

from g1_amp_motion_utils import (
    KEY_BODY_NAMES,
    MOTION_ROOT_BODY,
    TARGET_DOF_NAMES,
    compute_motion_diagnostics,
    validate_npz_arrays,
)


def main() -> None:
    amp_cfg_module = importlib.import_module(
        "unitree_rl_lab.tasks.locomotion.robots.g1.23dof_amp.g1_amp_env_cfg"
    )
    if tuple(amp_cfg_module.G1_23DOF_POLICY_JOINT_NAMES) != TARGET_DOF_NAMES:
        raise RuntimeError("Replay target order has drifted from G1AmpEnvCfg.policy_joint_names.")
    if tuple(amp_cfg_module.G1_AMP_KEY_BODY_NAMES) != KEY_BODY_NAMES:
        raise RuntimeError("Replay key-body definition has drifted from G1AmpEnvCfg.key_body_names.")

    motion_path = args_cli.motion_file.expanduser().resolve()
    with np.load(motion_path) as archive:
        data = {key: archive[key] for key in archive.files}
    validate_npz_arrays(data)
    loader = MotionLoader(str(motion_path), device=args_cli.device)
    fps = float(np.asarray(data["fps"]).item())
    dt = 1.0 / fps

    # The motion timing is controlled by frame selection and --real-time;
    # direct kinematic replay does not require a 30 Hz physics step.
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    spawn_ground_plane("/World/ground", GroundPlaneCfg())
    robot = Articulation(UNITREE_G1_23DOF_CFG.replace(prim_path="/World/Robot"))
    camera = None
    if args_cli.capture_dir is not None:
        from isaaclab.sensors.camera import Camera, CameraCfg

        camera = Camera(
            CameraCfg(
                prim_path="/World/ValidationCamera",
                update_period=0.0,
                height=720,
                width=720,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=32.0,
                    focus_distance=4.0,
                    horizontal_aperture=24.0,
                    clipping_range=(0.05, 100.0),
                ),
            )
        )
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.9])
    sim.reset()
    robot.reset()

    policy_joint_ids, resolved_joint_names = robot.find_joints(list(TARGET_DOF_NAMES), preserve_order=True)
    if tuple(resolved_joint_names) != TARGET_DOF_NAMES:
        raise RuntimeError(f"Replay articulation joint order mismatch: {resolved_joint_names}")
    motion_body_names = data["body_names"].tolist()
    robot_body_names = list(robot.data.body_names)
    if motion_body_names != robot_body_names:
        raise RuntimeError("NPZ body order does not exactly match the current G1-23DoF articulation.")
    root_index = motion_body_names.index(MOTION_ROOT_BODY)
    observed_body_ids, observed_body_names = robot.find_bodies(motion_body_names, preserve_order=True)
    if observed_body_names != motion_body_names:
        raise RuntimeError("Could not resolve every stored body by name during replay.")

    positions = data["dof_positions"]
    velocities = data["dof_velocities"]
    limits = robot.data.joint_pos_limits[0, policy_joint_ids].detach().cpu().numpy()
    max_limit_violation = float(
        np.max(
            np.maximum(
                np.maximum(limits[None, :, 0] - positions, 0.0),
                np.maximum(positions - limits[None, :, 1], 0.0),
            )
        )
    )
    if max_limit_violation > 1.0e-6:
        raise RuntimeError(f"Replay motion violates current articulation limits by {max_limit_violation:.9g} rad.")

    full_joint_pos = robot.data.default_joint_pos.clone()
    full_joint_vel = robot.data.default_joint_vel.clone()
    maximum_fk_position_error = 0.0
    maximum_fk_rotation_alignment_error = 0.0
    capture_frames: set[int] = set()
    capture_paths: list[Path] = []
    if args_cli.capture_dir is not None:
        args_cli.capture_dir.mkdir(parents=True, exist_ok=True)
        capture_frames = set(np.linspace(0, loader.num_frames - 1, num=6, dtype=int).tolist())
    cycle = 0
    frame = 0
    while simulation_app.is_running() and (args_cli.cycles == 0 or cycle < args_cli.cycles):
        frame_start = time.perf_counter()
        root_pose = np.concatenate(
            (data["body_positions"][frame, root_index], data["body_rotations"][frame, root_index])
        )
        root_velocity = np.concatenate(
            (
                data["body_linear_velocities"][frame, root_index],
                data["body_angular_velocities"][frame, root_index],
            )
        )
        full_joint_pos[:, policy_joint_ids] = torch.as_tensor(
            positions[frame], dtype=torch.float32, device=sim.device
        )
        full_joint_vel[:, policy_joint_ids] = torch.as_tensor(
            velocities[frame], dtype=torch.float32, device=sim.device
        )
        robot.write_root_link_pose_to_sim(torch.as_tensor(root_pose[None], dtype=torch.float32, device=sim.device))
        robot.write_root_com_velocity_to_sim(
            torch.as_tensor(root_velocity[None], dtype=torch.float32, device=sim.device)
        )
        robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel)
        sim.forward()
        robot.update(0.0)
        sim.render()

        if camera is not None and cycle == 0 and frame in capture_frames:
            from PIL import Image

            root_position = data["body_positions"][frame, root_index]
            eye = torch.as_tensor(
                (root_position + np.asarray([2.2, 2.2, 1.3]))[None], dtype=torch.float32, device=sim.device
            )
            target = torch.as_tensor(
                (root_position + np.asarray([0.0, 0.0, 0.7]))[None], dtype=torch.float32, device=sim.device
            )
            camera.set_world_poses_from_view(eye, target)
            sim.render()
            camera.update(sim.get_physics_dt())
            rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            capture_path = args_cli.capture_dir / f"frame_{frame:04d}.png"
            Image.fromarray(rgb).save(capture_path)
            capture_paths.append(capture_path)

        observed_pos = robot.data.body_pos_w[0, observed_body_ids].detach().cpu().numpy()
        observed_rot = robot.data.body_quat_w[0, observed_body_ids].detach().cpu().numpy()
        maximum_fk_position_error = max(
            maximum_fk_position_error, float(np.max(np.abs(observed_pos - data["body_positions"][frame])))
        )
        alignment_error = 1.0 - np.abs(np.sum(observed_rot * data["body_rotations"][frame], axis=-1))
        maximum_fk_rotation_alignment_error = max(
            maximum_fk_rotation_alignment_error, float(np.max(alignment_error))
        )

        frame += 1
        if frame == loader.num_frames:
            frame = 0
            cycle += 1
        if args_cli.real_time:
            sleep_time = dt - (time.perf_counter() - frame_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    report, _ = compute_motion_diagnostics(data, removed_names=[])
    foot = report["foot_height"]
    knee = report["knee_position"]
    coordination = report["contralateral_coordination"]
    if maximum_fk_position_error > 1.0e-4 or maximum_fk_rotation_alignment_error > 1.0e-4:
        raise RuntimeError(
            "Replay FK differs from stored states: "
            f"position={maximum_fk_position_error:.6g}, rotation={maximum_fk_rotation_alignment_error:.6g}."
        )
    if min(foot["left_min"], foot["right_min"]) < -0.02:
        raise RuntimeError(f"Foot-link trajectory penetrates the ground: {foot}")
    if not all(np.isfinite(value).all() for key, value in data.items() if key not in ("dof_names", "body_names")):
        raise RuntimeError("Replay data contains NaN/Inf.")
    missing_captures = [str(path) for path in capture_paths if not path.is_file()]
    if missing_captures:
        raise RuntimeError(f"Isaac camera did not produce expected captures: {missing_captures}")

    print("\n=== G1-23DoF AMP direct replay validation ===")
    print(f"Motion: {motion_path}")
    print(f"Frames/FPS: {loader.num_frames} / {fps:.12g}")
    print(f"Cycles replayed: {cycle}")
    print(f"Left foot height: {foot['left_min']:.6f} .. {foot['left_max']:.6f} m")
    print(f"Right foot height: {foot['right_min']:.6f} .. {foot['right_max']:.6f} m")
    print(f"Foot-height correlation: {foot['left_right_correlation']}")
    print(f"Foot-height difference sign changes: {foot['height_difference_sign_changes']}")
    print(f"Left knee range: {knee['left_min']:.6f} .. {knee['left_max']:.6f} rad")
    print(f"Right knee range: {knee['right_min']:.6f} .. {knee['right_max']:.6f} rad")
    print(f"Contralateral coordination score: {coordination['score']}")
    print(f"Maximum joint-limit violation: {max_limit_violation:.9g} rad")
    print(f"Maximum replay FK position error: {maximum_fk_position_error:.9g} m")
    print(f"Maximum replay FK quaternion alignment error: {maximum_fk_rotation_alignment_error:.9g}")
    print(f"Key-body left/right order: {list(KEY_BODY_NAMES)}")
    if capture_paths:
        print(f"Captured RGB frames: {[str(path.resolve()) for path in capture_paths]}")
    print("NaN/Inf: none")
    print("Replay validation: PASS")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
