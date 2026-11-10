from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from .ncrf import V31NCRF, core_config
from .ablations.single_scale import V31SingleScaleNCRF, single_scale_config
from .baselines.common import FairComparatorConfig
from .baselines.rescnn import PlainResCNNForecaster
from .baselines.unet import UNetForecaster
from .baselines.convlstm import ConvLSTMForecaster


class Persistence(nn.Module):
    def forward(self, context_phi, context_u, future_u):
        return context_phi[:, -1:].expand(-1, future_u.shape[1], -1, -1, -1).clone()


def build_model(method, *, image_size=128, project=None, checkpoint_steps=True):
    if method in {"ncrf", "no_memory", "generic_update"}:
        return V31NCRF(core_config(size=image_size), variant="full" if method == "ncrf" else method,
                       checkpoint_steps=checkpoint_steps)
    if method == "single_scale":
        return V31SingleScaleNCRF(single_scale_config(size=image_size), checkpoint_steps=checkpoint_steps)
    config = FairComparatorConfig(image_size=image_size)
    classes = {"plain_rescnn":PlainResCNNForecaster, "unet":UNetForecaster, "convlstm":ConvLSTMForecaster}
    if method in classes:
        return classes[method](config)
    if method == "persistence":
        return Persistence()
    if method == "fno":
        from .baselines.fno import ReflectiveFNOForecaster
        from ..config import project_root
        root = Path(project) if project else project_root()
        return ReflectiveFNOForecaster(config, source_directory=root / "third_party/fourier_neural_operator")
    raise ValueError(f"Unknown method: {method}")


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
