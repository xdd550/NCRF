"""Generate an auditable multi-initial-condition VCH-v3.1 dataset.

V3 is a new, non-destructive dataset protocol.  It does not alter VCH-v1 or
VCH-v2.  The generator calls the pinned public 2-D VCH forward solver, creates
smooth Neumann-compatible initial fields and controls, and splits *by initial
field* before a model is ever trained.  It writes only ``phi`` and prescribed
``u`` because the forecasting task must not expose future ``mu`` or ``w``.

The required first run is ``--mode pilot --execute`` (48 trajectories).  A
full 1,200-trajectory run is intentionally refused unless that pilot manifest
is supplied and its numerical audit passed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import json
import multiprocessing as multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Iterable

import numpy as np
from scipy.fft import idctn


PROTOCOL_VERSION = "VCH-v3.1"
V3_STATUS = "prepared_vch_v3_1_multicondition_neumann_train_valid_test_frozen"
PILOT_STATUS = "vch_v3_1_multicondition_solver_pilot_not_for_model_development"
SOLVER_COMMIT = "0f46a5f23412da1dc23208d20f2414f35656e587"
FORWARD_SOLVER_SHA256 = "a885d407b2438a5bf91c981f13b9383fcf5e55b20317358fc1de3e90bc0baf42"
SOLVER_CONFIG_SHA256 = "06e4a4a57ad01fd98d594a8ab5b9767b8bac987f312622eb4e3f71164cc05f39"
PINNED_SOURCE_ATTESTATION = "VCH_PINNED_SOURCE.json"
FRAME_COUNT = 101
NODES_PER_AXIS = 128
DT = 0.01
TOTAL_TIME = 1.0
INITIAL_FAMILY_COUNT = 6
INITIAL_REALIZATIONS_PER_FAMILY = 20
CONTROLS_PER_INITIAL = 10
TRAIN_INITIAL_REALIZATIONS_PER_FAMILY = 14
VALID_INITIAL_REALIZATIONS_PER_FAMILY = 3
MINIMUM_OUTPUT_FREE_BYTES = 25 * 1024**3


@dataclass(frozen=True)
class PhysicalConfig:
    """V3 physical/discretization parameters, frozen before pilot generation."""

    nx_intervals: int = NODES_PER_AXIS - 1
    ny_intervals: int = NODES_PER_AXIS - 1
    lx: float = 1.0
    ly: float = 1.0
    total_time: float = TOTAL_TIME
    dt: float = DT
    tau: float = 0.05
    gamma: float = 10.0
    c1: float = 0.75
    c2: float = 1.0
    kappa: float = 1.0e-4
    delta_separation: float = 1.0e-2


@dataclass(frozen=True)
class TrajectorySpec:
    split: str
    initial_family: int
    initial_realization: int
    control_id: int
    initial_seed: int
    control_seed: int

    @property
    def initial_id(self) -> str:
        return f"family_{self.initial_family:02d}_realization_{self.initial_realization:02d}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_git_head(git_dir: Path) -> str | None:
    """Resolve a normal loose Git HEAD without invoking a shell command."""

    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head if len(head) == 40 else None
    ref_path = git_dir / head.removeprefix("ref: ")
    if ref_path.is_file():
        value = ref_path.read_text(encoding="utf-8").strip()
        return value if len(value) == 40 else None
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        reference = head.removeprefix("ref: ")
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, _, ref = line.partition(" ")
                if ref == reference and len(commit) == 40:
                    return commit
    return None


def validate_pinned_solver(solver_root: Path) -> dict[str, str]:
    """Verify pinned solver code whether provided as a clone or source bundle."""

    solver_dir = solver_root / "src" / "2D" / "Vch_control_2D"
    forward_path = solver_dir / "Forward2_solver.py"
    config_path = solver_dir / "config.py"
    if not forward_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Pinned public VCH solver files are missing under {solver_dir}.")
    forward_hash = sha256_file(forward_path)
    config_hash = sha256_file(config_path)
    if forward_hash != FORWARD_SOLVER_SHA256 or config_hash != SOLVER_CONFIG_SHA256:
        raise RuntimeError("Solver source SHA-256 does not match the registered VCH-v3.1 public snapshot.")
    git_dir = solver_root / ".git"
    if git_dir.is_dir():
        resolved_commit = _read_git_head(git_dir)
        if resolved_commit != SOLVER_COMMIT:
            raise RuntimeError("Solver Git HEAD does not match the registered public commit.")
    else:
        attestation_path = solver_root / PINNED_SOURCE_ATTESTATION
        if not attestation_path.is_file():
            raise FileNotFoundError(
                "Bundled solver source requires VCH_PINNED_SOURCE.json when its .git directory is unavailable."
            )
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        expected = {
            "pinned_commit": SOLVER_COMMIT,
            "forward_solver_sha256": FORWARD_SOLVER_SHA256,
            "config_sha256": SOLVER_CONFIG_SHA256,
        }
        if any(attestation.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Bundled solver attestation does not match the registered VCH-v3.1 source snapshot.")
    return {
        "repository": "https://github.com/Tejasvanvekar/Sparse-optimal-control-of-Viscous-Chan-hilliard-via-Gradient-descent--1D-2D",
        "pinned_commit": SOLVER_COMMIT,
        "forward_solver_sha256": forward_hash,
        "config_sha256": config_hash,
    }


def atomic_json_write(path: Path, value: Any) -> None:
    """Write JSON atomically without importing the PyTorch training package."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def runtime_environment() -> dict[str, str]:
    """Record versions that can influence a numerical solver reproduction."""

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": importlib_metadata.version("scipy"),
        "pydantic": importlib_metadata.version("pydantic"),
        "numba": importlib_metadata.version("numba"),
    }


def _trajectory_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _weighted_mean(value: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * value) / np.sum(weights))


def _project_weighted_mean(value: np.ndarray, target_mean: float, weights: np.ndarray, margin: float) -> np.ndarray:
    """Set the trapezoid-weighted mean without violating physical bounds."""

    lower, upper = -1.0 + margin, 1.0 - margin
    output = np.clip(np.asarray(value, dtype=np.float64), lower, upper)
    for _ in range(12):
        correction = target_mean - _weighted_mean(output, weights)
        if abs(correction) <= 1.0e-12:
            break
        interior = (output > lower + 1.0e-4) & (output < upper - 1.0e-4)
        if not np.any(interior):
            raise RuntimeError("Initial-field mean projection exhausted all admissible interior cells.")
        weighted_interior = float(weights[interior].sum())
        output[interior] += correction * float(weights.sum()) / weighted_interior
        output = np.clip(output, lower, upper)
    if abs(_weighted_mean(output, weights) - target_mean) > 2.0e-8:
        raise RuntimeError("Initial-field weighted mean projection did not converge.")
    return output


def _neumann_smooth_field(
    rng: np.random.Generator, shape: tuple[int, int], *, spectral_decay: float, zero_mean: bool = True
) -> np.ndarray:
    """Draw a smooth cosine-basis field whose normal derivative is zero at edges."""

    rows, cols = shape
    ky = np.arange(rows, dtype=np.float64)[:, None]
    kx = np.arange(cols, dtype=np.float64)[None, :]
    amplitude = (1.0 + np.square(kx) + np.square(ky)) ** (-0.5 * spectral_decay)
    coefficients = rng.standard_normal(shape) * amplitude
    if zero_mean:
        coefficients[0, 0] = 0.0
    field = idctn(coefficients, type=2, norm="ortho")
    field -= field.mean()
    standard_deviation = float(field.std())
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise RuntimeError("Smooth-field generator produced a non-finite or constant field.")
    return field / standard_deviation


def initial_field(spec: TrajectorySpec, physical: PhysicalConfig, weights: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Create one frozen initial field; all controls for it share the same phi(0)."""

    family_means = (-0.30, -0.18, -0.06, 0.06, 0.18, 0.30)
    family_amplitudes = (0.10, 0.14, 0.18, 0.22, 0.16, 0.12)
    family_decays = (2.6, 2.1, 1.7, 1.7, 2.1, 2.6)
    family = spec.initial_family
    if not 0 <= family < INITIAL_FAMILY_COUNT:
        raise ValueError("initial_family is outside the registered V3 range.")
    rng = np.random.default_rng(spec.initial_seed)
    value = family_means[family] + family_amplitudes[family] * _neumann_smooth_field(
        rng, weights.shape, spectral_decay=family_decays[family]
    )
    value = _project_weighted_mean(value, family_means[family], weights, physical.delta_separation)
    return value, {
        "initial_target_weighted_mean": family_means[family],
        "initial_amplitude": family_amplitudes[family],
        "initial_spectral_decay": family_decays[family],
    }


def control_history(spec: TrajectorySpec, physical: PhysicalConfig, shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, float]]:
    """Create a smooth prescribed control via cosine fields and cubic time splines."""

    rng = np.random.default_rng(spec.control_seed)
    amplitude = float(rng.uniform(0.16, 0.48))
    spectral_decay = float(rng.uniform(1.4, 2.2))
    knot_count = 6
    keyframes = np.stack(
        [_neumann_smooth_field(rng, shape, spectral_decay=spectral_decay) for _ in range(knot_count)], axis=0
    )
    keyframes *= amplitude
    # Piecewise cubic Hermite interpolation (Catmull--Rom slopes) is a
    # cubic spline-like, dependency-light temporal interpolant.  We keep it
    # in NumPy rather than SciPy's banded solve because the latter has caused
    # platform-dependent BLAS aborts for large [knot,H,W] right-hand sides.
    slopes = np.empty_like(keyframes)
    slopes[0] = keyframes[1] - keyframes[0]
    slopes[-1] = keyframes[-1] - keyframes[-2]
    slopes[1:-1] = 0.5 * (keyframes[2:] - keyframes[:-2])
    position = np.linspace(0.0, knot_count - 1, FRAME_COUNT)
    left = np.minimum(np.floor(position).astype(np.int64), knot_count - 2)
    fraction = position - left
    s = fraction[:, None, None]
    s2, s3 = s * s, s * s * s
    h00, h10 = 2.0 * s3 - 3.0 * s2 + 1.0, s3 - 2.0 * s2 + s
    h01, h11 = -2.0 * s3 + 3.0 * s2, s3 - s2
    controls = h00 * keyframes[left] + h10 * slopes[left] + h01 * keyframes[left + 1] + h11 * slopes[left + 1]
    controls = np.clip(controls, -1.0, 1.0)
    if not bool(np.isfinite(controls).all()):
        raise RuntimeError("Control generator produced non-finite values.")
    return controls.astype(np.float64, copy=False), {
        "control_amplitude": amplitude,
        "control_spectral_decay": spectral_decay,
        "control_knot_count": float(knot_count),
    }


def full_plan() -> dict[str, list[TrajectorySpec]]:
    """Return exactly 1,200 trajectories split by independent initial fields.

    Every physical family contributes fourteen initial realizations to train
    and three each to validation/test. Thus 120 independent initial fields,
    rather than a small number of fields with many controls, determine the
    split. No initial-field byte sequence crosses splits, while all splits
    cover the same six registered mass/correlation regimes.
    """

    plan: dict[str, list[TrajectorySpec]] = {"train": [], "valid": [], "test": []}
    for family in range(INITIAL_FAMILY_COUNT):
        for realization in range(INITIAL_REALIZATIONS_PER_FAMILY):
            split = (
                "train"
                if realization < TRAIN_INITIAL_REALIZATIONS_PER_FAMILY
                else "valid"
                if realization < TRAIN_INITIAL_REALIZATIONS_PER_FAMILY + VALID_INITIAL_REALIZATIONS_PER_FAMILY
                else "test"
            )
            initial_seed = 20261001 + family * 1009 + realization * 97
            for control_id in range(CONTROLS_PER_INITIAL):
                plan[split].append(
                    TrajectorySpec(
                        split=split,
                        initial_family=family,
                        initial_realization=realization,
                        control_id=control_id,
                        initial_seed=initial_seed,
                        control_seed=20262001 + family * 100_003 + realization * 1_009 + control_id,
                    )
                )
    expected = {"train": 840, "valid": 180, "test": 180}
    if {name: len(items) for name, items in plan.items()} != expected:
        raise AssertionError("V3 full plan counts changed unexpectedly.")
    assert_split_isolation(plan)
    return plan


def pilot_plan() -> dict[str, list[TrajectorySpec]]:
    """Return a 48-trajectory numerical pilot, never valid for model fitting."""

    items: list[TrajectorySpec] = []
    for family in range(INITIAL_FAMILY_COUNT):
        for control_id in range(8):
            items.append(
                TrajectorySpec(
                    split="pilot",
                    initial_family=family,
                    initial_realization=0,
                    control_id=control_id,
                    initial_seed=20261001 + family * 1009,
                    control_seed=20263001 + family * 100_003 + control_id,
                )
            )
    if len(items) != 48:
        raise AssertionError("V3 pilot count changed unexpectedly.")
    return {"pilot": items}


def assert_split_isolation(plan: dict[str, Iterable[TrajectorySpec]]) -> None:
    """Ensure each complete initial field belongs to exactly one logical split."""

    owners: dict[str, str] = {}
    for split, items in plan.items():
        for spec in items:
            owner = owners.setdefault(spec.initial_id, split)
            if owner != split:
                raise RuntimeError(f"Initial field {spec.initial_id} leaks between {owner} and {split}.")


def _load_forward_solver(solver_root: Path) -> Any:
    solver_dir = solver_root / "src" / "2D" / "Vch_control_2D"
    forward_path = solver_dir / "Forward2_solver.py"
    config_path = solver_dir / "config.py"
    if not forward_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Pinned public VCH solver files are missing under {solver_dir}.")
    if str(solver_dir) not in sys.path:
        sys.path.insert(0, str(solver_dir))
    # The pinned checkout contains pre-existing Numba cache artifacts created
    # under a transient module name.  Disabling JIT before its first import
    # avoids deserializing those non-portable artifacts; the expensive sparse
    # Newton solve is unaffected because it is implemented in SciPy.
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    existing = sys.modules.get("Forward2_solver")
    if existing is not None:
        existing_path = Path(str(getattr(existing, "__file__", ""))).resolve()
        if existing_path != forward_path.resolve():
            raise ImportError(f"Forward2_solver is already loaded from another source: {existing_path}")
        return existing
    # The public source decorates numerical helpers with numba.cache=True.
    # Importing it under its native module name is required for that cache to
    # resolve reliably on later runs.
    return importlib.import_module("Forward2_solver")


def _mass_correct(value: np.ndarray, target_mass: float, weights_h: np.ndarray, margin: float) -> np.ndarray:
    output = np.clip(value, -1.0 + margin, 1.0 - margin)
    error = float(np.sum(weights_h * output) - target_mass)
    if abs(error) <= 1.0e-14:
        return output
    interior = np.abs(output) < (1.0 - margin - 5.0e-3)
    interior_weight = float(weights_h[interior].sum())
    if interior_weight <= 0.0:
        raise RuntimeError("Mass correction has no admissible interior cells.")
    output[interior] -= error / interior_weight
    return np.clip(output, -1.0 + margin, 1.0 - margin)


def simulate_trajectory(
    forward: Any, phi0: np.ndarray, control: np.ndarray, physical: PhysicalConfig
) -> tuple[np.ndarray, dict[str, float]]:
    """Run the public CN/Newton stepper from a supplied phi(0) and control."""

    nx, ny = physical.nx_intervals, physical.ny_intervals
    if phi0.shape != (nx + 1, ny + 1) or control.shape != (FRAME_COUNT, nx + 1, ny + 1):
        raise ValueError("V3 solver inputs do not match the declared 128-node, 101-frame contract.")
    hx, hy = physical.lx / nx, physical.ly / ny
    laplacian = forward.laplacian_matrix_neumann(nx, ny, hx, hy)
    trapz = np.outer(forward.trapz_weights(nx + 1), forward.trapz_weights(ny + 1))
    weights_h = hx * hy * trapz
    phi = np.asarray(phi0, dtype=np.float64).copy()
    w = np.zeros_like(phi)
    mu = forward.initialize_mu(phi, w, physical.c1, physical.c2, physical.kappa, laplacian, nx, ny, physical.delta_separation)
    target_mass = float(np.sum(weights_h * phi))
    history = np.empty((FRAME_COUNT, nx + 1, ny + 1), dtype=np.float64)
    history[0] = phi
    maximum_residual = 0.0
    maximum_mass_error = 0.0
    for step in range(FRAME_COUNT - 1):
        w_new = forward.solve_w(w, physical.dt, physical.gamma, control[step], control[step + 1])
        phi_new, mu_new, residuals = forward.newton_raphson(
            phi,
            mu,
            w,
            w_new,
            physical.dt,
            physical.tau,
            physical.c1,
            physical.c2,
            physical.kappa,
            physical.delta_separation,
            laplacian,
            nx,
            ny,
            hx,
            hy,
            return_residual_history=True,
        )
        if not residuals or not np.isfinite(residuals[-1]) or residuals[-1] > 1.0e-5:
            raise RuntimeError(f"Newton solve failed at step {step + 1}: final residual={residuals[-1] if residuals else 'missing'}.")
        phi = _mass_correct(phi_new, target_mass, weights_h, physical.delta_separation)
        mu, w = mu_new, w_new
        history[step + 1] = phi
        maximum_residual = max(maximum_residual, float(residuals[-1]))
        maximum_mass_error = max(maximum_mass_error, abs(float(np.sum(weights_h * phi) - target_mass)))
    if not bool(np.isfinite(history).all()):
        raise RuntimeError("V3 solver generated non-finite phi values.")
    return history, {
        "max_newton_residual": maximum_residual,
        "max_abs_mass_error": maximum_mass_error,
        "phi_min": float(history.min()),
        "phi_max": float(history.max()),
    }


# Worker state is deliberately process-local.  Each process opens the same
# preallocated memmap but writes only its assigned trajectory index, so there
# is no queueing of 13-MiB trajectory arrays back to the parent process.
_PARALLEL_FORWARD: Any | None = None
_PARALLEL_PHYSICAL: PhysicalConfig | None = None
_PARALLEL_TARGET: np.memmap | None = None
_PARALLEL_TRAPZ: np.ndarray | None = None
_PARALLEL_INITIALS: dict[str, tuple[np.ndarray, dict[str, float]]] = {}


def _initialize_parallel_worker(
    solver_root_text: str,
    physical_values: dict[str, Any],
    target_path_text: str,
    trajectory_count: int,
) -> None:
    """Initialize an isolated solver instance and writable target view."""

    global _PARALLEL_FORWARD, _PARALLEL_PHYSICAL, _PARALLEL_TARGET, _PARALLEL_TRAPZ, _PARALLEL_INITIALS
    _PARALLEL_FORWARD = _load_forward_solver(Path(solver_root_text))
    _PARALLEL_PHYSICAL = PhysicalConfig(**physical_values)
    _PARALLEL_TARGET = np.lib.format.open_memmap(
        target_path_text,
        mode="r+",
        dtype=np.float32,
        shape=(trajectory_count, FRAME_COUNT, 2, NODES_PER_AXIS, NODES_PER_AXIS),
    )
    _PARALLEL_TRAPZ = np.outer(
        _PARALLEL_FORWARD.trapz_weights(NODES_PER_AXIS), _PARALLEL_FORWARD.trapz_weights(NODES_PER_AXIS)
    )
    _PARALLEL_INITIALS = {}


def _parallel_write_one(payload: tuple[int, TrajectorySpec]) -> tuple[dict[str, Any], float, float, int]:
    """Simulate and persist one non-overlapping trajectory slice in a worker."""

    index, spec = payload
    if _PARALLEL_FORWARD is None or _PARALLEL_PHYSICAL is None or _PARALLEL_TARGET is None or _PARALLEL_TRAPZ is None:
        raise RuntimeError("V3 parallel worker was used before initialization.")
    cached = _PARALLEL_INITIALS.get(spec.initial_id)
    if cached is None:
        cached = initial_field(spec, _PARALLEL_PHYSICAL, _PARALLEL_TRAPZ)
        _PARALLEL_INITIALS[spec.initial_id] = cached
    phi0, initial_meta = cached
    control, control_meta = control_history(spec, _PARALLEL_PHYSICAL, phi0.shape)
    phi, quality = simulate_trajectory(_PARALLEL_FORWARD, phi0, control, _PARALLEL_PHYSICAL)
    trajectory = np.stack((phi, control), axis=1).astype(np.float32)
    _PARALLEL_TARGET[index] = trajectory
    record: dict[str, Any] = {
        "split": spec.split,
        "output_index": index,
        "initial_id": spec.initial_id,
        "initial_family": spec.initial_family,
        "initial_realization": spec.initial_realization,
        "control_id": spec.control_id,
        "initial_seed": spec.initial_seed,
        "control_seed": spec.control_seed,
        "initial_sha256": _trajectory_digest(trajectory[0, 0]),
        "trajectory_sha256": _trajectory_digest(trajectory),
        **initial_meta,
        **control_meta,
        **quality,
    }
    if spec.split != "train":
        return record, 0.0, 0.0, 0
    control64 = control.astype(np.float64, copy=False)
    return record, float(control64.sum()), float(np.square(control64).sum()), int(control64.size)


def _write_split_parallel(
    *,
    split: str,
    specs: list[TrajectorySpec],
    output_root: Path,
    solver_root: Path,
    physical: PhysicalConfig,
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], float, float, int]:
    """Write one split with deterministic, non-overlapping multiprocessing."""

    path = output_root / f"vch_{split}_phi_u_float32.npy"
    target = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(specs), FRAME_COUNT, 2, NODES_PER_AXIS, NODES_PER_AXIS)
    )
    target.flush()
    del target
    records: list[dict[str, Any]] = []
    control_sum = control_square_sum = 0.0
    control_count = 0
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    try:
        with context.Pool(
            processes=workers,
            initializer=_initialize_parallel_worker,
            initargs=(str(solver_root), asdict(physical), str(path), len(specs)),
        ) as pool:
            for completed, (record, part_sum, part_square_sum, part_count) in enumerate(
                pool.imap_unordered(_parallel_write_one, enumerate(specs), chunksize=1), start=1
            ):
                records.append(record)
                control_sum += part_sum
                control_square_sum += part_square_sum
                control_count += part_count
                if completed % 4 == 0 or completed == len(specs):
                    elapsed = time.monotonic() - started
                    print(f"[{split}] {completed}/{len(specs)} trajectories | {elapsed / completed:.2f}s/trajectory", flush=True)
    finally:
        # After every worker closes its mapping, flush a fresh parent mapping
        # before computing a whole-file SHA-256 for the manifest.
        target = np.lib.format.open_memmap(path, mode="r+")
        target.flush()
        del target
    records.sort(key=lambda row: int(row["output_index"]))
    return (
        {"path": str(path), "sha256": sha256_file(path), "count": len(specs)},
        records,
        control_sum,
        control_square_sum,
        control_count,
    )


def _write_split(
    *,
    split: str,
    specs: list[TrajectorySpec],
    output_root: Path,
    forward: Any,
    physical: PhysicalConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], float, float, int]:
    path = output_root / f"vch_{split}_phi_u_float32.npy"
    target = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(specs), FRAME_COUNT, 2, NODES_PER_AXIS, NODES_PER_AXIS)
    )
    trapz = np.outer(forward.trapz_weights(NODES_PER_AXIS), forward.trapz_weights(NODES_PER_AXIS))
    records: list[dict[str, Any]] = []
    control_sum = control_square_sum = 0.0
    control_count = 0
    started = time.monotonic()
    try:
        cached_initials: dict[str, tuple[np.ndarray, dict[str, float]]] = {}
        for index, spec in enumerate(specs):
            if spec.initial_id not in cached_initials:
                cached_initials[spec.initial_id] = initial_field(spec, physical, trapz)
            phi0, initial_meta = cached_initials[spec.initial_id]
            control, control_meta = control_history(spec, physical, phi0.shape)
            phi, quality = simulate_trajectory(forward, phi0, control, physical)
            # V3 deliberately keeps float32 fields.  The extra disk space is
            # modest (~16 GiB for 1,200 phi/u trajectories) and avoids making
            # conservation checks a half-precision artefact.
            trajectory = np.stack((phi, control), axis=1).astype(np.float32)
            target[index] = trajectory
            records.append(
                {
                    "split": split,
                    "output_index": index,
                    "initial_id": spec.initial_id,
                    "initial_family": spec.initial_family,
                    "initial_realization": spec.initial_realization,
                    "control_id": spec.control_id,
                    "initial_seed": spec.initial_seed,
                    "control_seed": spec.control_seed,
                    "initial_sha256": _trajectory_digest(trajectory[0, 0]),
                    "trajectory_sha256": _trajectory_digest(trajectory),
                    **initial_meta,
                    **control_meta,
                    **quality,
                }
            )
            if split == "train":
                control64 = control.astype(np.float64, copy=False)
                control_sum += float(control64.sum())
                control_square_sum += float(np.square(control64).sum())
                control_count += int(control64.size)
            if (index + 1) % 4 == 0 or index + 1 == len(specs):
                elapsed = time.monotonic() - started
                print(f"[{split}] {index + 1}/{len(specs)} trajectories | {elapsed / (index + 1):.2f}s/trajectory", flush=True)
    finally:
        target.flush()
        del target
    return ({"path": str(path), "sha256": sha256_file(path), "count": len(specs)}, records, control_sum, control_square_sum, control_count)


def _assert_output_audit(records: list[dict[str, Any]], *, full: bool) -> None:
    trajectories = [str(row["trajectory_sha256"]) for row in records]
    if len(trajectories) != len(set(trajectories)):
        raise RuntimeError("V3 output contains an exact duplicate full trajectory.")
    if full:
        initial_owner: dict[str, str] = {}
        for row in records:
            owner = initial_owner.setdefault(str(row["initial_sha256"]), str(row["split"]))
            if owner != row["split"]:
                raise RuntimeError("V3 output leaked an initial field between splits.")
        if max(float(row["max_newton_residual"]) for row in records) > 1.0e-5:
            raise RuntimeError("V3 output did not meet the registered Newton residual threshold.")
        if max(float(row["max_abs_mass_error"]) for row in records) > 1.0e-8:
            raise RuntimeError("V3 output did not meet the registered mass-conservation threshold.")


def validate_full_preflight(solver_root: Path, pilot_manifest: Path | None) -> tuple[PhysicalConfig, dict[str, str]]:
    """Validate all immutable prerequisites before a full run writes any data."""

    if pilot_manifest is None or not pilot_manifest.is_file():
        raise RuntimeError("Full V3 generation requires --pilot-manifest from a completed V3.1 pilot.")
    physical = PhysicalConfig()
    source_identity = validate_pinned_solver(solver_root)
    pilot = json.loads(pilot_manifest.read_text(encoding="utf-8"))
    if pilot.get("status") != PILOT_STATUS or pilot.get("audit", {}).get("passed") is not True:
        raise RuntimeError("Full V3 generation is refused until a passing, registered V3.1 pilot manifest is supplied.")
    if pilot.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Full V3 generation requires a passing pilot from the same VCH-v3.1 protocol version.")
    if pilot.get("physical_config") != asdict(physical):
        raise RuntimeError("Full V3 generation requires a pilot with the same frozen physical configuration.")
    pilot_source = pilot.get("source_solver", {})
    if any(pilot_source.get(key) != value for key, value in source_identity.items() if key != "repository"):
        raise RuntimeError("Full V3 generation requires a pilot generated with the same verified solver source hashes.")
    return physical, source_identity


def validate_output_target(output_root: Path) -> int:
    """Refuse an existing target and require room for full float32 arrays."""

    if output_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing V3 output root: {output_root}")
    existing_parent = output_root.parent
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    if not existing_parent.exists():
        raise FileNotFoundError(f"No existing parent volume was found for output root: {output_root}")
    free_bytes = shutil.disk_usage(existing_parent).free
    if free_bytes < MINIMUM_OUTPUT_FREE_BYTES:
        required_gib = MINIMUM_OUTPUT_FREE_BYTES / 1024**3
        available_gib = free_bytes / 1024**3
        raise RuntimeError(
            f"V3 output requires at least {required_gib:.0f} GiB free near {output_root}; only {available_gib:.1f} GiB is available."
        )
    return free_bytes


def materialize(
    *, output_root: Path, solver_root: Path, mode: str, pilot_manifest: Path | None, workers: int
) -> dict[str, Any]:
    validate_output_target(output_root)
    if mode not in {"pilot", "full"}:
        raise ValueError("materialize mode must be pilot or full.")
    if not 1 <= workers <= 8:
        raise ValueError("--workers must be between 1 and 8; start a new server run with 4 workers.")
    if mode == "pilot" and workers != 1:
        raise ValueError("The mandatory 48-trajectory pilot must use --workers 1 for a directly comparable audit.")
    if mode == "full":
        physical, source_identity = validate_full_preflight(solver_root, pilot_manifest)
    else:
        physical = PhysicalConfig()
        source_identity = validate_pinned_solver(solver_root)
    forward = _load_forward_solver(solver_root)
    if int(round(physical.total_time / physical.dt)) + 1 != FRAME_COUNT:
        raise RuntimeError("V3 time grid no longer produces exactly 101 frames.")
    plan = pilot_plan() if mode == "pilot" else full_plan()
    output_root.mkdir(parents=True, exist_ok=False)
    all_records: list[dict[str, Any]] = []
    files: dict[str, Any] = {}
    control_sum = control_square_sum = 0.0
    control_count = 0
    try:
        for split, specs in plan.items():
            if workers == 1:
                file_info, records, part_sum, part_square_sum, part_count = _write_split(
                    split=split, specs=specs, output_root=output_root, forward=forward, physical=physical
                )
            else:
                file_info, records, part_sum, part_square_sum, part_count = _write_split_parallel(
                    split=split,
                    specs=specs,
                    output_root=output_root,
                    solver_root=solver_root,
                    physical=physical,
                    workers=workers,
                )
            files[split] = file_info
            all_records.extend(records)
            control_sum += part_sum
            control_square_sum += part_square_sum
            control_count += part_count
        _assert_output_audit(all_records, full=mode == "full")
        if mode == "full" and control_count <= 0:
            raise RuntimeError("V3 training-control moments were not collected.")
        control_mean = control_sum / control_count if control_count else None
        control_variance = max(control_square_sum / control_count - float(control_mean) ** 2, 0.0) if control_count else None
        metadata_path = output_root / "vch_v3_trajectory_metadata.json"
        atomic_json_write(metadata_path, {"records": all_records})
        manifest: dict[str, Any] = {
            "status": PILOT_STATUS if mode == "pilot" else V3_STATUS,
            "dataset": "VCH-v3.1 independently generated multi-initial-condition trajectories",
            "protocol_version": PROTOCOL_VERSION,
            "purpose": "Pilot numerical verification only" if mode == "pilot" else "Model development on train/valid only; primary test is frozen.",
            "source_solver": {**source_identity, "local_checkout": str(solver_root.resolve())},
            "generator_sha256": sha256_file(Path(__file__).resolve()),
            "physical_config": asdict(physical),
            "generation": {"trajectory_workers": workers, "worker_start_method": "spawn" if workers > 1 else None},
            "runtime_environment": runtime_environment(),
            "storage": {"dtype": "float32", "layout": "[trajectory,time,channel(phi,u),height,width]", "files": files},
            "split": {
                "unit": "complete initial field; all controls from an initial field remain in one split",
                "counts": {split: len(items) for split, items in plan.items()},
                "development_arrays": ["vch_train_phi_u_float32.npy", "vch_valid_phi_u_float32.npy"] if mode == "full" else [],
                "primary_test": "vch_test_phi_u_float32.npy is frozen and must never be uploaded for development" if mode == "full" else None,
            },
            "normalization": {
                "phi": "No learned normalization; retain physical scale.",
                "control": None
                if control_mean is None
                else {"u_mean_train_only": control_mean, "u_std_train_only": float(np.sqrt(control_variance)), "pixel_count": control_count},
            },
            "audit": {
                "passed": True,
                "trajectory_count": len(all_records),
                "unique_trajectory_count": len({row["trajectory_sha256"] for row in all_records}),
                "max_newton_residual": max(float(row["max_newton_residual"]) for row in all_records),
                "max_abs_mass_error": max(float(row["max_abs_mass_error"]) for row in all_records),
                "phi_min": min(float(row["phi_min"]) for row in all_records),
                "phi_max": max(float(row["phi_max"]) for row in all_records),
                "metadata_file": metadata_path.name,
            },
            "test_data_accessed_by_model": False,
            "test_data_opened_for_metrics": False,
        }
        atomic_json_write(output_root / "vch_v3_manifest.json", manifest)
        return manifest
    except Exception:
        # Preserve partial output for forensic debugging but never label it as a usable dataset.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an audited VCH-v3.1 multi-initial-condition dataset.")
    parser.add_argument("--solver-root", required=True, type=Path, help="Pinned public VCH solver checkout at commit 0f46a5f.")
    parser.add_argument("--output-root", required=True, type=Path, help="New non-existent output directory.")
    parser.add_argument("--mode", choices=("plan", "pilot", "preflight", "full"), default="plan")
    parser.add_argument("--pilot-manifest", type=Path, help="Required passing pilot manifest before --mode full.")
    parser.add_argument("--workers", type=int, default=1, help="Trajectory workers (1-8); server full generation begins at 4.")
    parser.add_argument("--execute", action="store_true", help="Required before any arrays are written.")
    args = parser.parse_args()
    plan = pilot_plan() if args.mode == "pilot" else full_plan()
    print(f"[PLAN] mode={args.mode} counts={{" + ", ".join(f"{key}:{len(value)}" for key, value in plan.items()) + "}")
    if args.mode == "preflight":
        free_bytes = validate_output_target(args.output_root.resolve())
        physical, source_identity = validate_full_preflight(args.solver_root.resolve(), args.pilot_manifest)
        print(
            "[PASS] V3.1 full preflight: pilot, physical configuration, and verified solver source agree "
            f"(commit={source_identity['pinned_commit']}, dt={physical.dt}, workers={args.workers}, free_gib={free_bytes / 1024**3:.1f})."
        )
        return
    if args.mode == "plan" or not args.execute:
        if args.mode == "full":
            print("[STOP] dry run only; full generation additionally requires --execute and a passing --pilot-manifest.")
        else:
            print("[STOP] dry run only; use --mode pilot --execute for the mandatory 48-trajectory numerical pilot.")
        return
    manifest = materialize(
        output_root=args.output_root.resolve(),
        solver_root=args.solver_root.resolve(),
        mode=args.mode,
        pilot_manifest=None if args.pilot_manifest is None else args.pilot_manifest.resolve(),
        workers=args.workers,
    )
    print(f"[DONE] status={manifest['status']} root={args.output_root.resolve()}")
    print(f"[DONE] audit={manifest['audit']}")


if __name__ == "__main__":
    main()
