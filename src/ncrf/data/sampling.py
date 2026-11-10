from __future__ import annotations
import numpy as np
import torch


def _coverage_cycle_windows(
    *,
    seed: int,
    stage_index: int,
    update: int,
    batch_size: int,
    trajectory_count: int,
    maximum_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cover trajectories before repetition with a stateless stage schedule.

    At H90 the legacy B=1 iid sampler made only 1,400 draws from 640
    trajectories, leaving about 11% expected to be unseen in the final stage.
    This sampler cycles through a fresh permutation before repetition while
    spreading valid temporal starts. Its output depends only on the supplied
    arguments, so resuming a checkpoint regenerates the same future samples.
    """

    if update < 1 or batch_size < 1 or trajectory_count < 1 or maximum_start < 0:
        raise ValueError("Invalid coverage-cycle sampler bounds.")
    serial_start = (update - 1) * batch_size
    indices = np.empty(batch_size, dtype=np.int64)
    starts = np.empty(batch_size, dtype=np.int64)
    cached_permutations: dict[int, np.ndarray] = {}
    start_count = maximum_start + 1
    for item in range(batch_size):
        serial = serial_start + item
        cycle, position = divmod(serial, trajectory_count)
        permutation = cached_permutations.get(cycle)
        if permutation is None:
            cycle_rng = np.random.default_rng(np.random.SeedSequence([seed, stage_index, cycle]))
            permutation = cycle_rng.permutation(trajectory_count).astype(np.int64, copy=False)
            cached_permutations[cycle] = permutation
        indices[item] = permutation[position]
        # This explicit arithmetic hash is stable across Python processes.
        starts[item] = (position * 17 + cycle * 31 + stage_index * 7 + seed) % start_count
    return indices, starts


def _to_device(
    batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], device: torch.device, control_mean: float, control_std: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    context_phi, context_u, future_u, target = batch
    phi = torch.from_numpy(context_phi).to(device, non_blocking=True)
    past_u = torch.from_numpy(context_u).to(device, non_blocking=True)
    known_u = torch.from_numpy(future_u).to(device, non_blocking=True)
    target_phi = torch.from_numpy(target).to(device, non_blocking=True)
    return phi, (past_u - control_mean) / control_std, (known_u - control_mean) / control_std, target_phi

