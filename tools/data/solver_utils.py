"""Selected helpers used for reference-solver timing."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any
import numpy as np

SOLVER_COMMIT = "0f46a5f23412da1dc23208d20f2414f35656e587"

FORWARD_SOLVER_SHA256 = "a885d407b2438a5bf91c981f13b9383fcf5e55b20317358fc1de3e90bc0baf42"

SOLVER_CONFIG_SHA256 = "06e4a4a57ad01fd98d594a8ab5b9767b8bac987f312622eb4e3f71164cc05f39"

PINNED_SOURCE_ATTESTATION = "VCH_PINNED_SOURCE.json"

NODES_PER_AXIS = 128

DT = 0.01

TOTAL_TIME = 1.0

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
