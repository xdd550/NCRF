from pathlib import Path
from .utils import read_json

METHODS = ("ncrf", "plain_rescnn", "unet", "convlstm", "fno", "persistence", "no_memory", "generic_update", "single_scale")


def project_root():
    import os
    if os.environ.get("NCRF_PROJECT_ROOT"):
        return Path(os.environ["NCRF_PROJECT_ROOT"]).resolve()
    root = Path(__file__).resolve().parents[2]
    if not (root / "configs/common.json").is_file():
        raise RuntimeError("Use editable installation, or set NCRF_PROJECT_ROOT to the project folder.")
    return root


def load_config(method, project=None):
    root = Path(project) if project else project_root()
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    common = read_json(root / "configs/common.json")
    matches = [read_json(p) for p in (root / "configs").rglob("*.json") if p.name != "common.json"]
    method_config = next(c for c in matches if c.get("method") == method)
    return {**common, **method_config}
