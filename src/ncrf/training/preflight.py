"""Real training-array CUDA checks, never a substitute for a complete run."""
from __future__ import annotations
import gc
from pathlib import Path
import time
import numpy as np
import torch
from ..config import load_config
from ..data.dataset import data_root,open_split
from ..data.sampling import _to_device
from ..data.windows import sample_window
from ..models.registry import build_model,parameter_count
from ..losses.long_horizon import VCHV15NeumannLongHorizonLoss
from ..utils import configure_backend,reserve_directory,runtime_identity,seed_all,write_json
from .engine import update
from .checkpoints import model_state,save_checkpoint,load_checkpoint


def _check(method,horizon,array,mean,std,output):
    config=load_config(method)
    seed_all(config["seeds"][0])
    model=build_model(method).cuda()
    objective=VCHV15NeumannLongHorizonLoss(**config["loss"]).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    amp=method!="fno"
    scaler=torch.amp.GradScaler("cuda",enabled=amp,init_scale=1024)
    batch=_to_device(sample_window(array,np.array([0,140]),np.array([0,0]),horizon),torch.device("cuda"),mean,std)
    initial={k:v.detach().cpu().clone() for k,v in model_state(model).items()}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started=time.perf_counter()
    row=update(model,objective,optimizer,scaler,batch,amp=amp)
    torch.cuda.synchronize()
    row.update(method=method,horizon=horizon,batch_size=2,parameters=parameter_count(model),
        precision="amp_fp16" if amp else "fp32",seconds=time.perf_counter()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        changed_parameters=sum(not torch.equal(v.detach().cpu(),initial[k]) for k,v in model_state(model).items()))
    if not row["successful"] or not row["changed_parameters"]:
        raise RuntimeError("The short training check did not apply a finite optimizer update.")
    model.eval()
    with torch.inference_mode():
        predicted=model(*batch[:3]).detach().cpu()
    path=output/f"{method}_H{horizon}_diagnostic.pt"
    # Diagnostic weights cannot be mistaken for a completed formal run.
    save_checkpoint(path,dict(schema="ncrf_clean_checkpoint_v1",method=method,complete=False,
        diagnostic=True,model_state=model_state(model),optimizer_state=optimizer.state_dict(),scaler_state=scaler.state_dict()))
    saved=load_checkpoint(path)
    fresh=build_model(method).cuda().eval()
    fresh.load_state_dict(saved["model_state"],strict=True)
    with torch.inference_mode():
        restored=fresh(*batch[:3]).detach().cpu()
    if not torch.equal(predicted,restored):
        raise RuntimeError("Checkpoint round-trip changed the FP32 prediction.")
    row["checkpoint_prediction_max_abs_difference"]=0.
    row["status"]="passed"
    return row


def preflight(methods,root,output,*,horizons=(10,90)):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for actual training preflight; smoke is the CPU alternative.")
    configure_backend()
    output=Path(output)
    reserve_directory(output)
    config=load_config("ncrf")
    print("Verifying frozen TRAIN array; validation/test/OOD are not opened.",flush=True)
    array,mean,std=open_split(data_root(root),config,"train",training=True)
    result=dict(schema="ncrf_train_preflight_v1",diagnostic=True,runtime=runtime_identity(),
        manifest_sha256=config["data"]["frozen_manifest_sha256"],checks=[],
        scope="One real batch optimizer update per method/horizon, not a scientific result")
    for method in methods:
        if method=="persistence":
            continue
        for horizon in horizons:
            try:
                row=_check(method,horizon,array,mean,std,output)
            except torch.cuda.OutOfMemoryError:
                row=dict(method=method,horizon=horizon,status="insufficient_local_vram",
                         message="The registered batch/precision was NOT reduced; retry on the training server.")
            result["checks"].append(row)
            write_json(output/"preflight_report.json",result)
            print(row,flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    result["all_requested_checks_passed"]=all(r["status"]=="passed" for r in result["checks"])
    write_json(output/"preflight_report.json",result)
    return result
