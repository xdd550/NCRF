from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refuse an empty result table.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_backend(*, strict_determinism=False):
    import torch
    if strict_determinism:
        # Set before the first cuBLAS operation in this process.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(strict_determinism)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def runtime_identity():
    import platform
    import torch
    return dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32)


def source_identity(project):
    project = Path(project)
    files = list((project / "src").rglob("*.py")) + list((project / "configs").rglob("*.json"))
    files += list((project / "third_party/fourier_neural_operator/neuralop").rglob("*.py"))
    return {p.relative_to(project).as_posix():sha256_file(p) for p in sorted(files)}


def reserve_directory(path):
    """Never merge a new evaluation into an existing output directory."""
    Path(path).mkdir(parents=True, exist_ok=False)
