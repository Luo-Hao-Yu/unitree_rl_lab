"""Category-balanced wrapper around Isaac Lab's single-clip AMP MotionLoader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from isaaclab_tasks.direct.humanoid_amp.motions import MotionLoader


@dataclass(frozen=True)
class MultiMotionSampleContext:
    """Clip identity and times associated with one reference-history batch."""

    motion_indices: np.ndarray
    current_times: np.ndarray
    history_motion_indices: np.ndarray | None = None
    history_times: np.ndarray | None = None


class MultiMotionLoader:
    """Sample independent AMP clips with manifest-defined hierarchical probabilities.

    The wrapper delegates NPZ parsing and interpolation to one official
    :class:`MotionLoader` per clip. It never concatenates trajectories.
    """

    def __init__(
        self,
        manifest_file: str | Path,
        device: torch.device | str,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_file).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Invalid multi-motion manifest: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict):
            raise TypeError("Multi-motion manifest root must be a JSON object.")

        motions = manifest.get("motions")
        active_motion_ids = manifest.get("active_motion_ids")
        category_probabilities = manifest.get("category_sampling_probabilities")
        if not isinstance(motions, list) or not motions:
            raise ValueError("Multi-motion manifest must contain a non-empty 'motions' list.")
        if not isinstance(active_motion_ids, list) or not active_motion_ids:
            raise ValueError("Multi-motion manifest must contain non-empty 'active_motion_ids'.")
        if not isinstance(category_probabilities, dict) or not category_probabilities:
            raise ValueError("Manifest must define 'category_sampling_probabilities'.")

        by_id: dict[str, dict] = {}
        for entry in motions:
            if not isinstance(entry, dict) or not isinstance(entry.get("motion_id"), str):
                raise ValueError("Every motion entry must have a string motion_id.")
            motion_id = entry["motion_id"]
            if motion_id in by_id:
                raise ValueError(f"Duplicate motion_id in manifest: {motion_id}")
            by_id[motion_id] = entry
        if len(set(active_motion_ids)) != len(active_motion_ids):
            raise ValueError("active_motion_ids contains duplicates.")

        self.device = device
        self.rng = np.random.default_rng() if rng is None else rng
        self.entries: list[dict] = []
        self.loaders: list[MotionLoader] = []
        self.motion_ids: list[str] = []
        self.motion_categories: list[str] = []
        self.motion_paths: list[Path] = []
        for motion_id in active_motion_ids:
            if motion_id not in by_id:
                raise ValueError(f"Active motion_id is missing from motions: {motion_id}")
            entry = by_id[motion_id]
            if entry.get("validation_status") != "validated":
                raise ValueError(f"Active motion {motion_id!r} is not validated.")
            category = entry.get("category")
            converted_path = entry.get("converted_npz_path")
            if not isinstance(category, str) or not category:
                raise ValueError(f"Active motion {motion_id!r} has no category.")
            if not isinstance(converted_path, str) or not converted_path:
                raise ValueError(f"Active motion {motion_id!r} has no converted_npz_path.")
            path = Path(converted_path).expanduser()
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Converted motion for {motion_id!r} does not exist: {path}")
            self.entries.append(entry)
            self.motion_ids.append(motion_id)
            self.motion_categories.append(category)
            self.motion_paths.append(path)
            self.loaders.append(MotionLoader(str(path), device=device))

        first = self.loaders[0]
        for motion_id, loader in zip(self.motion_ids[1:], self.loaders[1:]):
            if loader.dof_names != first.dof_names:
                raise ValueError(f"DoF order differs in active motion {motion_id!r}.")
            if loader.body_names != first.body_names:
                raise ValueError(f"Body order differs in active motion {motion_id!r}.")
            if loader.num_frames < 2 or not np.isfinite(float(loader.dt)) or float(loader.dt) <= 0.0:
                raise ValueError(f"Invalid timing in active motion {motion_id!r}.")

        self._dof_names = list(first.dof_names)
        self._body_names = list(first.body_names)
        self.categories = list(category_probabilities)
        if set(self.categories) != set(self.motion_categories):
            raise ValueError(
                "category_sampling_probabilities must exactly cover active categories: "
                f"probabilities={self.categories}, active={sorted(set(self.motion_categories))}."
            )
        probabilities = np.asarray([category_probabilities[name] for name in self.categories], dtype=np.float64)
        if probabilities.shape != (len(self.categories),) or not np.all(np.isfinite(probabilities)):
            raise ValueError("Category probabilities must be finite scalars.")
        if np.any(probabilities <= 0.0):
            raise ValueError("Every active category probability must be positive.")
        if not np.isclose(np.sum(probabilities), 1.0, atol=1.0e-8):
            raise ValueError(f"Category probabilities must sum to one, got {np.sum(probabilities):.12g}.")
        self.category_probabilities = probabilities / np.sum(probabilities)

        self._category_motion_indices: list[np.ndarray] = []
        self._category_clip_probabilities: list[np.ndarray] = []
        for category in self.categories:
            indices = np.asarray(
                [index for index, value in enumerate(self.motion_categories) if value == category], dtype=np.int64
            )
            weights = np.asarray(
                [float(self.entries[index].get("sampling_weight_within_category", 1.0)) for index in indices],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
                raise ValueError(f"Clip weights for category {category!r} must be finite and positive.")
            self._category_motion_indices.append(indices)
            self._category_clip_probabilities.append(weights / np.sum(weights))

    @property
    def dof_names(self) -> list[str]:
        return list(self._dof_names)

    @property
    def body_names(self) -> list[str]:
        return list(self._body_names)

    @property
    def num_dofs(self) -> int:
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        return len(self._body_names)

    @property
    def num_motions(self) -> int:
        return len(self.loaders)

    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        return self.loaders[0].get_dof_index(dof_names)

    def get_body_index(self, body_names: list[str]) -> list[int]:
        return self.loaders[0].get_body_index(body_names)

    def sample_motion_indices(self, num_samples: int) -> np.ndarray:
        """Sample category first, then a clip within that category."""

        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        category_indices = self.rng.choice(
            len(self.categories), size=num_samples, replace=True, p=self.category_probabilities
        )
        motion_indices = np.empty(num_samples, dtype=np.int64)
        for category_index in np.unique(category_indices):
            rows = np.flatnonzero(category_indices == category_index)
            candidates = self._category_motion_indices[category_index]
            probabilities = self._category_clip_probabilities[category_index]
            motion_indices[rows] = self.rng.choice(candidates, size=rows.size, replace=True, p=probabilities)
        return motion_indices

    def sample_times(
        self,
        motion_indices: np.ndarray,
        history_steps: int = 1,
    ) -> np.ndarray:
        """Sample valid current times while reserving prior frames for AMP history."""

        motion_indices = self._validate_motion_indices(motion_indices)
        if history_steps <= 0:
            raise ValueError(f"history_steps must be positive, got {history_steps}.")
        times = np.empty(motion_indices.shape[0], dtype=np.float64)
        for motion_index in np.unique(motion_indices):
            rows = np.flatnonzero(motion_indices == motion_index)
            loader = self.loaders[int(motion_index)]
            minimum = (history_steps - 1) * float(loader.dt)
            maximum = float(loader.duration)
            if minimum > maximum:
                raise ValueError(
                    f"Motion {self.motion_ids[int(motion_index)]!r} is too short for {history_steps} history steps."
                )
            times[rows] = self.rng.uniform(minimum, maximum, size=rows.size)
        return times

    def sample(
        self,
        motion_indices: np.ndarray,
        times: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample explicitly identified clips at explicitly identified times."""

        motion_indices = self._validate_motion_indices(motion_indices)
        times = np.asarray(times, dtype=np.float64)
        if times.shape != motion_indices.shape or not np.all(np.isfinite(times)):
            raise ValueError("times must be a finite array with the same shape as motion_indices.")

        result: list[torch.Tensor] | None = None
        for motion_index in np.unique(motion_indices):
            rows = np.flatnonzero(motion_indices == motion_index)
            loader = self.loaders[int(motion_index)]
            selected_times = times[rows]
            tolerance = 1.0e-9
            if np.any(selected_times < -tolerance) or np.any(selected_times > float(loader.duration) + tolerance):
                raise ValueError(
                    f"Times for motion {self.motion_ids[int(motion_index)]!r} leave [0, duration]."
                )
            samples = loader.sample(num_samples=rows.size, times=selected_times)
            if result is None:
                result = [
                    torch.empty(
                        (motion_indices.size, *sample.shape[1:]), dtype=sample.dtype, device=sample.device
                    )
                    for sample in samples
                ]
            for output, sample in zip(result, samples):
                output[rows] = sample
        assert result is not None
        return tuple(result)  # type: ignore[return-value]

    def sample_history(
        self,
        num_samples: int,
        num_history_steps: int,
        motion_indices: np.ndarray | None = None,
        current_times: np.ndarray | None = None,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        MultiMotionSampleContext,
    ]:
        """Sample histories that are guaranteed to stay within one clip per row."""

        if motion_indices is None:
            motion_indices = self.sample_motion_indices(num_samples)
        else:
            motion_indices = self._validate_motion_indices(motion_indices)
            if motion_indices.size != num_samples:
                raise ValueError("motion_indices size does not equal num_samples.")
        if current_times is None:
            current_times = self.sample_times(motion_indices, history_steps=num_history_steps)
        else:
            current_times = np.asarray(current_times, dtype=np.float64)
            if current_times.shape != (num_samples,) or not np.all(np.isfinite(current_times)):
                raise ValueError("current_times must be finite with shape [num_samples].")

        dt = np.asarray([float(self.loaders[index].dt) for index in motion_indices], dtype=np.float64)
        offsets = np.arange(num_history_steps, dtype=np.float64)
        history_times = current_times[:, None] - dt[:, None] * offsets[None, :]
        if np.any(history_times < -1.0e-9):
            raise ValueError("A requested AMP history crosses the beginning of its clip.")
        history_motion_indices = np.repeat(motion_indices[:, None], num_history_steps, axis=1)
        samples = self.sample(history_motion_indices.reshape(-1), history_times.reshape(-1))
        context = MultiMotionSampleContext(
            motion_indices=motion_indices.copy(),
            current_times=current_times.copy(),
            history_motion_indices=history_motion_indices,
            history_times=history_times,
        )
        return samples, context

    def initial_body_positions(self, motion_indices: np.ndarray, body_index: int) -> torch.Tensor:
        """Gather frame-zero body positions for environment-relative RSI anchoring."""

        motion_indices = self._validate_motion_indices(motion_indices)
        output = torch.empty((motion_indices.size, 3), dtype=torch.float32, device=self.loaders[0].device)
        for motion_index in np.unique(motion_indices):
            rows = np.flatnonzero(motion_indices == motion_index)
            output[rows] = self.loaders[int(motion_index)].body_positions[0, body_index]
        return output

    def _validate_motion_indices(self, motion_indices: Sequence[int] | np.ndarray) -> np.ndarray:
        values = np.asarray(motion_indices, dtype=np.int64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("motion_indices must be a non-empty one-dimensional array.")
        if np.any(values < 0) or np.any(values >= self.num_motions):
            raise IndexError("motion_indices contains an out-of-range clip index.")
        return values
