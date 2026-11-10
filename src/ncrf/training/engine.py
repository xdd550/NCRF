"""Clean runner for the locked V31/V32 architecture, loss and window schedule.

No archive-result gates and no checkpoint selection using test data.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from ..config import load_config, project_root
from ..data.dataset import data_root, open_split, metadata
from ..data.sampling import _coverage_cycle_windows, _to_device
from ..data.windows import sample_window
from ..losses.long_horizon import VCHV15NeumannLongHorizonLoss
from ..models.registry import build_model
from ..utils import configure_backend, fingerprint, reserve_directory, runtime_identity, seed_all, source_identity, write_json, sha256_file
from .checkpoints import load_checkpoint, model_state, restore_rng, rng_state, save_checkpoint


def update(model, objective, optimizer, scaler, batch, *, amp=False, clip=1.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=batch[0].device.type, dtype=torch.float16, enabled=amp):
        prediction = model(*batch[:3])
        if not bool(torch.isfinite(prediction).all()):
            raise RuntimeError("Nonfinite prediction; stopped.")
        loss, parts = objective(prediction.float(), batch[3].float())
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Nonfinite loss; stopped.")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    connected = sum(p.numel() for p in model.parameters() if p.grad is not None)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=False)
    successful = bool(torch.isfinite(norm))
    if not scaler.is_enabled() and not successful:
        raise RuntimeError("Nonfinite FP32 gradient; stopped.")
    if successful:
        scaler.step(optimizer)
        scaler.update()
    else:
        scaler.update(new_scale=float(scaler.get_scale()) * scaler.get_backoff_factor())
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise RuntimeError("Nonfinite parameter; stopped.")
    row = dict(loss=float(loss.detach()), gradient_norm=float(norm) if successful else None,
               successful=successful, connected_parameters=connected, amp_scale=float(scaler.get_scale()))
    row.update({k:float(v.detach()) for k,v in parts.items()})
    row.update(getattr(model, "last_step_audit", {}))
    return row


def train(method, seed, root, output, *, resume=False, device="cuda", execute=False, strict_determinism=False):
    if not execute:
        raise ValueError("Training is opt-in: add --execute after reviewing the plan.")
    if method == "persistence":
        raise ValueError("Persistence is evaluated directly; it is not trained.")
    if device != "cuda" or not torch.cuda.is_available():
        raise ValueError("The formal training protocol requires CUDA. Use smoke for CPU checks.")
    config = load_config(method)
    if seed not in config["seeds"]:
        raise ValueError("Seed is outside the registered three-seed protocol.")
    configure_backend(strict_determinism=strict_determinism)
    seed_all(seed)
    root, output = data_root(root), Path(output).resolve()
    train_array, mean, std = open_split(root, config, "train", training=True)
    metadata(root,config,"train")
    # Validation is available as a separate command. It never selects the final checkpoint.
    identity = dict(config=config, seed=seed, source=source_identity(project_root()), runtime=runtime_identity(),
                    manifest_sha256=config["data"]["frozen_manifest_sha256"])
    signature = fingerprint(identity)
    if not resume:
        reserve_directory(output)
        write_json(output / "run_manifest.json", {**identity, "signature":signature})
    elif not output.is_dir():
        raise ValueError("Resume directory does not exist.")
    if (output / "final.pt").exists():
        raise FileExistsError("A completed run is immutable; choose a new output directory.")
    # An exclusive lock prevents concurrent writers. An interrupted process may
    # leave this file; inspect the process before manually removing that lock.
    with (output / "RUNNING.lock").open("x", encoding="utf-8") as handle:
        import os
        handle.write(str(os.getpid()))
    try:
        model = build_model(method).to(device)
        objective = VCHV15NeumannLongHorizonLoss(**config["loss"]).to(device)
        settings = config["training"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        amp = method != "fno"
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=settings["amp_initial_scale"])
        stage_index, done, stage_skips, consecutive = 0, 0, 0, 0
        saved = None
        if resume:
            saved = load_checkpoint(output / "latest.pt")
            if saved["signature"] != signature:
                raise RuntimeError("Source/config/data/runtime differs from checkpoint; refusing an inexact resume.")
            model.load_state_dict(saved["model_state"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state"])
            scaler.load_state_dict(saved["scaler_state"])
            stage_index, done = saved["stage_index"], saved["done"]
            stage_skips, consecutive = saved["stage_skips"], saved["consecutive_skips"]
            restore_rng(saved["rng_state"])
        state = saved
        stages = settings["stages"]
        for si in range(stage_index, len(stages)):
            stage = stages[si]
            resuming_stage = saved is not None and si == stage_index
            if not resuming_stage:
                done, stage_skips = 0, 0
                if stage["phase"] == "continuation" and stage["stage"] == 0:
                    seed_all(seed + settings["continuation_seed_offset"])
                    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
                    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=settings["amp_initial_scale"])
            # The offset resets stochastic training RNG only. The registered
            # stateless window schedules ALWAYS use the original experiment seed.
            sampling_seed = seed
            for attempt in range(done + 1, stage["updates"] + 1):
                indices, starts = _coverage_cycle_windows(seed=sampling_seed, stage_index=stage["stage"],
                    update=attempt, batch_size=settings["batch_size"], trajectory_count=len(train_array),
                    maximum_start=91-stage["horizon"])
                batch = _to_device(sample_window(train_array, indices, starts, stage["horizon"]), torch.device(device), mean, std)
                row = update(model, objective, optimizer, scaler, batch, amp=amp, clip=settings["gradient_clip"])
                stage_skips += int(not row["successful"])
                consecutive = 0 if row["successful"] else consecutive + 1
                row.update(stage_index=si, phase=stage["phase"], attempt=attempt, horizon=stage["horizon"],
                           trajectory_indices=indices.tolist(), starts=starts.tolist())
                with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                if consecutive >= settings["max_consecutive_skips"] or stage_skips > settings["max_stage_skip_fraction"] * stage["updates"]:
                    raise RuntimeError("Gradient-skip guard exceeded; inspect the log, do not silently change precision/budget.")
                if attempt % settings["checkpoint_interval"] == 0 or attempt == stage["updates"]:
                    state = dict(schema="ncrf_clean_checkpoint_v1", method=method, seed=seed,
                        signature=signature, config=config, complete=False, stage_index=si, done=attempt,
                        stage_skips=stage_skips, consecutive_skips=consecutive, model_state=model_state(model),
                        optimizer_state=optimizer.state_dict(), scaler_state=scaler.state_dict(), rng_state=rng_state())
                    save_checkpoint(output / "latest.pt", state)
                    print(f"{method} seed={seed} stage={si+1}/6 H{stage['horizon']} {attempt}/{stage['updates']} loss={row['loss']:.6g}", flush=True)
            saved = None
        state["complete"] = True
        save_checkpoint(output / "final.pt", state)
        write_json(output / "artifacts.json", dict(schema="ncrf_training_artifacts_v1",method=method,seed=seed,
            complete=True,run_signature=signature,selection="last_fixed_budget",automatic_cleanup=False,
            files={name:dict(path=name,sha256=sha256_file(output/name))
                   for name in ("final.pt","latest.pt","run_manifest.json","training.jsonl")}))
    finally:
        (output / "RUNNING.lock").unlink()
    return str(output / "final.pt")
