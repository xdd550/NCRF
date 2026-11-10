from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from ..utils import read_json, sha256_file


def data_root(value=None):
    from ..config import project_root
    return Path(value or os.environ.get("NCRF_DATA_ROOT") or project_root() / "data").resolve()


def manifest(root, config):
    path = Path(root) / config["data"]["manifest"]
    if sha256_file(path) != config["data"]["frozen_manifest_sha256"]:
        raise RuntimeError("Not the frozen VCH-v3.1 manifest. Do not silently substitute a regenerated dataset.")
    value = read_json(path)
    control = value["normalization"]["control"]
    mean, std = float(control["u_mean_train_only"]), float(control["u_std_train_only"])
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid training-only control normalization.")
    return value, mean, std


def open_split(root, config, split, *, training=False):
    if split not in {"train", "valid", "test"} or (training and split not in {"train", "valid"}):
        raise ValueError("Training may read train/valid only.")
    info, mean, std = manifest(root, config)
    record = info["storage"]["files"][split]
    path = Path(root) / f"vch_{split}_phi_u_float32.npy"
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"Array SHA-256 mismatch: {split}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = (config["data"]["expected_counts"][split], 101, 2, 128, 128)
    if array.shape != shape or array.dtype != np.float32:
        raise ValueError(f"Incorrect {split} geometry/dtype: {array.shape}, {array.dtype}")
    return array, mean, std


def metadata(root, config, split):
    path = Path(root) / config["data"]["metadata"]
    if sha256_file(path) != config["data"]["frozen_metadata_sha256"]:
        raise RuntimeError("Frozen trajectory metadata SHA-256 mismatch.")
    value = read_json(path)
    records = value["records"]
    selected_records = [r for r in records if r["split"] == split]
    selected = {int(r["output_index"]):r for r in selected_records}
    count = config["data"]["expected_counts"][split]
    from collections import Counter
    groups = Counter(r["initial_id"] for r in selected_records)
    if len(selected_records) != count or set(selected) != set(range(count)) or len(groups) != count // 10 or set(groups.values()) != {10}:
        raise RuntimeError("Initial-field grouping or trajectory indexing differs from the frozen protocol.")
    owners = {}
    for record in records:
        owner = owners.setdefault(record["initial_id"],record["split"])
        if owner != record["split"]:
            raise RuntimeError("An initial field is shared between data splits.")
    for initial in groups:
        controls = [r["control_id"] for r in selected_records if r["initial_id"] == initial]
        if len(set(controls)) != 10:
            raise RuntimeError("Repeated control ID within an initial-field group.")
    return selected


def open_ood(path, expected_sha256):
    if not expected_sha256 or sha256_file(path) != expected_sha256:
        raise RuntimeError("An explicit matching OOD array SHA-256 is required.")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (180,101,2,128,128) or array.dtype != np.float32:
        raise ValueError("OOD array must follow [180,101,2,128,128] float32.")
    return array
