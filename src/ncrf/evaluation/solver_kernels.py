"""Reference-solver timing kernels mechanically extracted from the original benchmark."""
from __future__ import annotations
from typing import Any
import time
import numpy as np
FRAME_COUNT = 101
CONTEXT_STEPS = 10
HORIZON = 90

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def _mass_correct(*args, **kwargs):
    from solver_utils import _mass_correct as implementation
    return implementation(*args, **kwargs)

def trapezoidal_weights(forward: Any, physical: PhysicalConfig) -> tuple[Any, np.ndarray]:
    nx, ny = physical.nx_intervals, physical.ny_intervals
    hx, hy = physical.lx / nx, physical.ly / ny
    laplacian = forward.laplacian_matrix_neumann(nx, ny, hx, hy)
    trapezoidal = np.outer(forward.trapz_weights(nx + 1), forward.trapz_weights(ny + 1))
    return laplacian, hx * hy * trapezoidal


def advance_one_step(
    forward: Any,
    *,
    phi: np.ndarray,
    mu: np.ndarray,
    w: np.ndarray,
    control_now: np.ndarray,
    control_next: np.ndarray,
    target_mass: float,
    laplacian: Any,
    weights_h: np.ndarray,
    physical: PhysicalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    nx, ny = physical.nx_intervals, physical.ny_intervals
    hx, hy = physical.lx / nx, physical.ly / ny
    w_next = forward.solve_w(w, physical.dt, physical.gamma, control_now, control_next)
    phi_next, mu_next, residuals = forward.newton_raphson(
        phi, mu, w, w_next,
        physical.dt, physical.tau, physical.c1, physical.c2, physical.kappa, physical.delta_separation,
        laplacian, nx, ny, hx, hy, return_residual_history=True,
    )
    final_residual = float(residuals[-1]) if residuals else float("inf")
    require(np.isfinite(final_residual) and final_residual <= 1.0e-5, f"Newton solve failed: final residual={final_residual}.")
    phi_next = _mass_correct(phi_next, target_mass, weights_h, physical.delta_separation)
    mass_error = abs(float(np.sum(weights_h * phi_next) - target_mass))
    return phi_next, mu_next, w_next, final_residual, mass_error


def reconstruct_context_state(
    forward: Any,
    trajectory: np.ndarray,
    *,
    laplacian: Any,
    weights_h: np.ndarray,
    physical: PhysicalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Recreate phi, mu and w at frame 9 outside the timed prediction interval."""

    nx, ny = physical.nx_intervals, physical.ny_intervals
    expected_shape = (FRAME_COUNT, 2, nx + 1, ny + 1)
    require(tuple(trajectory.shape) == expected_shape, f"Unexpected VCH trajectory shape {tuple(trajectory.shape)}.")
    phi = np.asarray(trajectory[0, 0], dtype=np.float64).copy()
    control = np.asarray(trajectory[:, 1], dtype=np.float64)
    w = np.zeros_like(phi)
    mu = forward.initialize_mu(phi, w, physical.c1, physical.c2, physical.kappa, laplacian, nx, ny, physical.delta_separation)
    target_mass = float(np.sum(weights_h * phi))
    for step in range(CONTEXT_STEPS - 1):
        phi, mu, w, _, _ = advance_one_step(
            forward, phi=phi, mu=mu, w=w, control_now=control[step], control_next=control[step + 1],
            target_mass=target_mass, laplacian=laplacian, weights_h=weights_h, physical=physical,
        )
    return phi, mu, w, control, target_mass


def time_solver_rollout(
    forward: Any,
    state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
    *,
    laplacian: Any,
    weights_h: np.ndarray,
    physical: PhysicalConfig,
) -> tuple[float, float, float]:
    """Time 90 reference-solver updates from frame 9 through frame 99."""

    phi, mu, w, control, target_mass = state
    phi, mu, w = phi.copy(), mu.copy(), w.copy()
    maximum_residual, maximum_mass_error = 0.0, 0.0
    started = time.perf_counter()
    for step in range(CONTEXT_STEPS - 1, CONTEXT_STEPS - 1 + HORIZON):
        phi, mu, w, residual, mass_error = advance_one_step(
            forward, phi=phi, mu=mu, w=w, control_now=control[step], control_next=control[step + 1],
            target_mass=target_mass, laplacian=laplacian, weights_h=weights_h, physical=physical,
        )
        maximum_residual = max(maximum_residual, residual)
        maximum_mass_error = max(maximum_mass_error, mass_error)
    return time.perf_counter() - started, maximum_residual, maximum_mass_error
