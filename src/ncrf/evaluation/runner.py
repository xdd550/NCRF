from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from ..config import load_config
from ..data.dataset import data_root, metadata, open_ood, open_split, manifest
from ..data.sampling import _to_device
from ..data.windows import sample_window
from ..training.checkpoints import load_checkpoint, load_for_inference
from ..utils import configure_backend, read_json, reserve_directory, runtime_identity, sha256_file, write_csv, write_json
from .metrics import field_metrics


def evaluation_data(method, root, split, *, authorize_test=False, ood_array=None, ood_sha256=None, ood_metadata=None):
    config, root = load_config(method), data_root(root)
    if split in {"test", "ood"} and not authorize_test:
        raise ValueError("Held-out data access requires --authorize-test.")
    if split != "ood":
        array, mean, std = open_split(root, config, split)
        groups = metadata(root, config, split)
        array_hash = sha256_file(root / f"vch_{split}_phi_u_float32.npy")
    else:
        if not ood_array or not ood_metadata:
            raise ValueError("OOD needs --ood-array, --ood-sha256 and --ood-metadata.")
        _, mean, std = manifest(root, config)
        array = open_ood(ood_array, ood_sha256)
        records = read_json(ood_metadata)["records"]
        groups = {int(r["output_index"]):r for r in records}
        from collections import Counter
        counts = Counter(r["initial_id"] for r in records)
        if len(records) != 180 or set(groups) != set(range(180)) or len(counts) != 18 or set(counts.values()) != {10}:
            raise ValueError("OOD metadata must identify 18 initial fields, 180 unique trajectories.")
        if any(len({r.get("control_id") for r in records if r["initial_id"] == initial}) != 10 for initial in counts):
            raise ValueError("OOD controls must be unique within each initial-field group.")
        array_hash = ood_sha256
    return array, mean, std, groups, array_hash


def evaluate(method, checkpoint, root, output, *, split="valid", horizons=(10,20,50,90),
             device="cuda", authorize_test=False, limit=None, save_predictions=True,
             scenario="main", ood_array=None, ood_sha256=None, ood_metadata=None):
    if split == "ood" and scenario == "main":
        raise ValueError("Provide an explicit OOD scenario, e.g. OOD-1.")
    if not horizons or len(set(horizons)) != len(horizons) or any(h not in (10,20,50,90) for h in horizons):
        raise ValueError("Use unique registered horizons: 10,20,50,90.")
    if limit is not None and (limit < 1 or limit > 180):
        raise ValueError("limit must be 1..180; limited evaluations are diagnostic only.")
    configure_backend()
    array, mean, std, groups, array_hash = evaluation_data(method, root, split, authorize_test=authorize_test,
        ood_array=ood_array, ood_sha256=ood_sha256, ood_metadata=ood_metadata)
    model = load_for_inference(method, checkpoint, device=device)
    seed = load_checkpoint(checkpoint)["seed"] if checkpoint else None
    output = Path(output)
    reserve_directory(output)
    identity = dict(schema="ncrf_evaluation_v1", method=method, seed=seed, split=split, scenario=scenario,
        array_sha256=array_hash, normalization=dict(mean=mean,std=std),
        checkpoint_sha256=sha256_file(checkpoint) if checkpoint else None,
        horizons=list(horizons), window_start=0, independent_rollouts=True, precision="fp32",
        trajectory_count=limit or len(array), diagnostic=limit is not None, runtime=runtime_identity(),
        save_predictions=save_predictions, checkpoint_path=str(Path(checkpoint).resolve()) if checkpoint else None,
        prediction_contents="prediction, truth, phase/control history, raw future control, IDs, window indices",
        complete=False)
    write_json(output / "evaluation_manifest.json", identity)
    rows, prediction_index = [], []
    with torch.inference_mode():
        for h in horizons:
            for index in range(limit or len(array)):
                batch = _to_device(sample_window(array, np.array([index]), np.array([0]), h), torch.device(device), mean, std)
                predicted = model(*batch[:3]).float().cpu().numpy()[0,:,0]
                truth = batch[3].cpu().numpy()[0,:,0]
                if not np.isfinite(predicted).all():
                    raise RuntimeError("Nonfinite prediction; no silent sample exclusion.")
                metrics = field_metrics(predicted, truth, batch[0].cpu().numpy()[0,-1,0])
                if any(v is not None and not np.isfinite(v) for v in metrics.values()):
                    raise RuntimeError("Nonfinite metric; inspect the sample instead of dropping it.")
                rows.append(dict(method=method, seed=seed, split=split, scenario=scenario, horizon=h,
                    trajectory_index=index, initial_id=groups[index]["initial_id"],
                    control_id=groups[index].get("control_id"), **metrics))
                if save_predictions:
                    directory = output / "predictions" / f"H{h}"
                    directory.mkdir(parents=True, exist_ok=True)
                    prediction_path = directory / f"trajectory_{index:03d}.npz"
                    np.savez_compressed(prediction_path, prediction=predicted, truth=truth,
                        last_context=batch[0].cpu().numpy()[0,-1,0],
                        history_phi=np.asarray(array[index,:10,0]), history_control=np.asarray(array[index,:10,1]),
                        # Raw (not normalized) prescribed control, same coordinate orientation.
                        control=np.asarray(array[index,10:10+h,1]), window_start=np.array(0),
                        observed_frame_indices=np.arange(10), predicted_frame_indices=np.arange(10,10+h),
                        method=np.array(method), seed=np.array(str(seed) if seed is not None else ""),
                        scenario=np.array(scenario), split=np.array(split), trajectory_index=np.array(index),
                        initial_id=np.array(str(groups[index]["initial_id"])),
                        control_id=np.array(str(groups[index].get("control_id", ""))),
                        checkpoint_sha256=np.array(identity["checkpoint_sha256"] or ""))
                    prediction_index.append(dict(method=method,seed=seed,scenario=scenario,split=split,horizon=h,
                        trajectory_index=index,initial_id=groups[index]["initial_id"],control_id=groups[index].get("control_id"),
                        path=prediction_path.relative_to(output).as_posix(),sha256=sha256_file(prediction_path)))
            # Keep completed horizons even if a later horizon is interrupted.
            write_csv(output / "per_trajectory.csv",rows)
            if prediction_index:
                write_csv(output / "prediction_index.csv",prediction_index)
            print(f"{method} {scenario}/{split} H{h}: {limit or len(array)} trajectories", flush=True)
    write_csv(output / "per_trajectory.csv", rows)
    identity["complete"] = True
    identity["table_sha256"] = sha256_file(output / "per_trajectory.csv")
    if prediction_index:
        identity["prediction_index_sha256"] = sha256_file(output / "prediction_index.csv")
        identity["saved_prediction_count"] = len(prediction_index)
    write_json(output / "evaluation_manifest.json", identity)
    return str(output / "per_trajectory.csv")
