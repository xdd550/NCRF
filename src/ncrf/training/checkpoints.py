"""Tensor-only checkpoints, including exact RNG/optimizer state for resuming."""
from __future__ import annotations
from collections import OrderedDict
import os
from pathlib import Path
import random
import numpy as np
import torch


def model_state(model):
    # NeuralOperator attaches Python classes in informational _metadata.
    # Tensor names/values are unchanged; do not pickle executable classes.
    state = model.state_dict()
    unexpected = [key for key,value in state.items() if not torch.is_tensor(value) and key != "_metadata"]
    if unexpected:
        raise TypeError(f"Unexpected non-tensor model state: {unexpected}")
    return OrderedDict((key, value) for key, value in state.items() if torch.is_tensor(value))


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), *state[2:]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema") != "ncrf_clean_checkpoint_v1":
        raise ValueError("Use a checkpoint from this reorganized runner; archived checkpoints need an explicit audited migration.")
    return value


def load_for_inference(method, checkpoint, *, device="cpu"):
    from ..models.registry import build_model
    model = build_model(method).to(device).float()
    if method == "persistence":
        if checkpoint:
            raise ValueError("Persistence has no checkpoint.")
    else:
        if not checkpoint:
            raise ValueError("A trained checkpoint is required; random weights are not evaluation results.")
        value = load_checkpoint(checkpoint)
        if value["method"] != method or not value["complete"]:
            raise ValueError("Method mismatch or unfinished training checkpoint.")
        model.load_state_dict(value["model_state"], strict=True)
    return model.eval()
