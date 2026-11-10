"""Timing is separate from accuracy evaluation and from solver regeneration."""
from __future__ import annotations
from pathlib import Path
import sys
import time
import numpy as np
import torch
from ..config import project_root
from ..data.sampling import _to_device
from ..data.windows import sample_window
from ..training.checkpoints import load_for_inference
from ..utils import configure_backend, reserve_directory, runtime_identity, sha256_file, write_csv, write_json
from .runner import evaluation_data


def benchmark(method, checkpoint, root, output, *, device="cuda", split="valid", authorize_test=False,
              trajectories=10, warmup=10, solver=False):
    if not 2 <= trajectories <= 180 or warmup < 1:
        raise ValueError("Use 2..180 timing trajectories and at least one untimed warmup.")
    configure_backend()
    array, mean, std, groups, array_hash = evaluation_data(method, root, split, authorize_test=authorize_test)
    model = load_for_inference(method, checkpoint, device=device)
    indices = np.linspace(0,len(array)-1,trajectories,dtype=int)
    def prepare(index):
        return _to_device(sample_window(array,np.array([index]),np.array([0]),90),torch.device(device),mean,std)[:3]
    def synchronize():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(torch.device(device))
    if solver:
        sys.path.insert(0,str(project_root() / "tools/data"))
        from solver_utils import PhysicalConfig, _load_forward_solver, validate_pinned_solver
        from .solver_kernels import trapezoidal_weights, reconstruct_context_state, time_solver_rollout
        solver_root = project_root() / "third_party/vch_solver"
        solver_identity = validate_pinned_solver(solver_root)
        forward, physical = _load_forward_solver(solver_root), PhysicalConfig()
        laplacian, weights_h = trapezoidal_weights(forward,physical)
    output = Path(output)
    reserve_directory(output)
    identity = dict(schema="ncrf_runtime_v1", method=method, split=split, horizon=90,
        checkpoint_sha256=sha256_file(checkpoint) if checkpoint else None,array_sha256=array_hash,
        checkpoint_path=str(Path(checkpoint).resolve()) if checkpoint else None,
        normalization=dict(mean=mean,std=std),window_start=0,context_steps=10,
        trajectory_indices=indices.tolist(),warmup_excluded=warmup,precision="fp32",batch_size=1,
        scope="history encoding plus 90 autoregressive steps; excludes reading/hashing/data transfer/metrics",
        runtime=runtime_identity(),complete=False)
    if solver:
        identity.update(solver=solver_identity,solver_scope="90 CN/Newton steps frame9 to99; excludes context reconstruction")
    write_json(output / "runtime_manifest.json",identity)
    rows=[]
    with torch.inference_mode():
        inputs=prepare(indices[0])
        for _ in range(warmup):
            model(*inputs)
        synchronize()
        for index in indices:
            inputs=prepare(index)
            synchronize()
            started=time.perf_counter()
            result=model(*inputs)
            synchronize()
            seconds=time.perf_counter()-started
            if not bool(torch.isfinite(result).all()):
                raise RuntimeError("Nonfinite timed prediction.")
            row=dict(trajectory_index=int(index),model_seconds=seconds)
            if int(index) in groups:
                row.update(initial_id=groups[int(index)]["initial_id"],control_id=groups[int(index)].get("control_id"))
            if solver:
                state=reconstruct_context_state(forward,array[index],laplacian=laplacian,weights_h=weights_h,physical=physical)
                elapsed,residual,mass_error=time_solver_rollout(forward,state,laplacian=laplacian,weights_h=weights_h,physical=physical)
                row.update(solver_seconds=elapsed,solver_max_residual=residual,solver_max_mass_error=mass_error)
            rows.append(row)
            # Preserve completed trajectories during potentially long solver runs.
            write_csv(output / "per_trajectory_runtime.csv",rows)
            print(row,flush=True)
    values=np.asarray([r["model_seconds"] for r in rows])
    summary=dict(model_mean_seconds=float(values.mean()),model_trajectory_std_seconds=float(values.std(ddof=1)),
                 model_median_seconds=float(np.median(values)),count=len(rows),speedup_available=solver)
    if solver:
        summary["solver_mean_seconds"]=float(np.mean([r["solver_seconds"] for r in rows]))
        summary["speedup_ratio_of_unrounded_means"]=summary["solver_mean_seconds"]/summary["model_mean_seconds"]
    write_csv(output / "per_trajectory_runtime.csv",rows)
    if solver:
        write_json(output / "speedup_basis.json",dict(
            formula="mean(solver_seconds) / mean(model_seconds)",rounded_before_division=False,
            solver_mean_seconds=summary["solver_mean_seconds"],model_mean_seconds=summary["model_mean_seconds"],
            speedup=summary["speedup_ratio_of_unrounded_means"],paired_trajectory_indices=indices.tolist(),
            raw_table="per_trajectory_runtime.csv",raw_table_sha256=sha256_file(output/"per_trajectory_runtime.csv"),
            model_scope=identity["scope"],solver_scope=identity["solver_scope"]))
    write_json(output / "runtime_summary.json",summary)
    identity["complete"]=True
    identity["raw_table_sha256"]=sha256_file(output/"per_trajectory_runtime.csv")
    identity["summary_sha256"]=sha256_file(output/"runtime_summary.json")
    if solver:
        identity["speedup_basis_sha256"]=sha256_file(output/"speedup_basis.json")
    write_json(output / "runtime_manifest.json",identity)
    return str(output / "runtime_summary.json")
