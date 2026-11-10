from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import sys
from torch import nn


@lru_cache(maxsize=4)
def _locked_fno_class(source_directory: str) -> type[nn.Module]:
    root = Path(source_directory).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from neuralop.models import FNO  # type: ignore[import-not-found]
        import neuralop  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The locked NeuralOperator source needs tensorly, tensorly-torch and opt-einsum. "
            "Run scripts/development/bootstrap_v29_literature_sources.py without --skip-dependencies."
        ) from error
    module_path = Path(str(getattr(neuralop, "__file__", ""))).resolve()
    if root not in module_path.parents:
        raise RuntimeError("A different installed neuralop package shadowed the locked V29 FNO source.")
    return FNO

