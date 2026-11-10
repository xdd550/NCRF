"""Prepare the pinned NeuralOperator source for the NCRF FNO baseline."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

REPOSITORY = "https://github.com/neuraloperator/neuraloperator.git"
COMMIT = "00b7d86f8d74ff0af55da53eb585fe26df9c71f0"
FNO_SHA256_LF = "8b753f855ce6c08e16c949155b11a81e0ab6353cf4e54b07834edbf96db1f760"


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=REPOSITORY,
                        help="Upstream repository URL or local Git mirror of the pinned commit")
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent
    package = destination / "neuralop"
    if package.exists():
        raise SystemExit("neuralop already exists; move or remove it after checking its contents, then retry.")
    with tempfile.TemporaryDirectory(prefix="ncrf_fno_") as temp:
        checkout = Path(temp) / "upstream"
        run("git", "clone", "--no-checkout", "--", args.source, str(checkout))
        run("git", "-c", "core.autocrlf=false", "checkout", "--detach", COMMIT, cwd=checkout)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        if actual != COMMIT:
            raise RuntimeError("Unexpected NeuralOperator commit")
        source = checkout / "neuralop"
        content = (source / "models" / "fno.py").read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(content).hexdigest()
        if digest != FNO_SHA256_LF:
            raise RuntimeError("The pinned FNO source hash does not match the recorded experiment")
        staged = destination / ".neuralop-staging"
        if staged.exists():
            raise SystemExit("A staging directory already exists; inspect it before retrying.")
        shutil.copytree(source, staged, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        staged.rename(package)
    print(f"Prepared NeuralOperator {COMMIT} in {package}")


if __name__ == "__main__":
    main()
