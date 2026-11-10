from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class VCHDevelopmentProtocol:
    """Fixed geometry shared by all VCH development methods."""

    context_steps: int = 10
    image_size: int = 128
    total_steps: int = 101

    def validate_horizon(self, horizon: int) -> None:
        if horizon < 1 or self.context_steps + horizon > self.total_steps:
            raise ValueError(f"Horizon must be 1..{self.total_steps - self.context_steps}, got {horizon}.")


def sample_window(
    array: np.ndarray,
    trajectory_indices: np.ndarray,
    starts: np.ndarray,
    horizon: int,
    protocol: VCHDevelopmentProtocol = VCHDevelopmentProtocol(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return past phi/u, prescribed future u, and future phi target only."""

    protocol.validate_horizon(horizon)
    indices = np.asarray(trajectory_indices, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    if indices.ndim != 1 or starts.ndim != 1 or indices.shape != starts.shape:
        raise ValueError("trajectory_indices and starts must be equally sized one-dimensional arrays.")
    if np.any(indices < 0) or np.any(indices >= array.shape[0]):
        raise IndexError("A requested VCH trajectory index is outside the opened development split.")
    latest_start = protocol.total_steps - protocol.context_steps - horizon
    if np.any(starts < 0) or np.any(starts > latest_start):
        raise IndexError(f"Window start must lie in 0..{latest_start} for horizon {horizon}.")

    context_phi, context_u, future_u, target_phi = [], [], [], []
    for trajectory, start in zip(indices.tolist(), starts.tolist(), strict=True):
        clip = array[trajectory, start : start + protocol.context_steps + horizon]
        context_phi.append(np.asarray(clip[: protocol.context_steps, 0:1], dtype=np.float32))
        context_u.append(np.asarray(clip[: protocol.context_steps, 1:2], dtype=np.float32))
        future_u.append(np.asarray(clip[protocol.context_steps :, 1:2], dtype=np.float32))
        target_phi.append(np.asarray(clip[protocol.context_steps :, 0:1], dtype=np.float32))
    return (
        np.stack(context_phi, axis=0),
        np.stack(context_u, axis=0),
        np.stack(future_u, axis=0),
        np.stack(target_phi, axis=0),
    )

