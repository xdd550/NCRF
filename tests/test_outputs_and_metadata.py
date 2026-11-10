import csv
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from ncrf.config import project_root
from ncrf.data.dataset import metadata
from ncrf.utils import sha256_file,write_json


def make_metadata(tmp_path,records,count):
    path=tmp_path/"metadata.json"
    write_json(path,dict(records=records))
    config=dict(data=dict(metadata=path.name,frozen_metadata_sha256=sha256_file(path),expected_counts={"train":count}))
    return config


def test_metadata_rejects_duplicate_and_cross_split_initials(tmp_path):
    records=[dict(split="train",output_index=i,initial_id="a",control_id=i) for i in range(10)]
    config=make_metadata(tmp_path,records,10)
    assert len(metadata(tmp_path,config,"train"))==10
    config=make_metadata(tmp_path,records+[records[0]],10)
    with pytest.raises(RuntimeError):
        metadata(tmp_path,config,"train")
    config=make_metadata(tmp_path,records+[dict(split="valid",output_index=0,initial_id="a",control_id=0)],10)
    with pytest.raises(RuntimeError,match="shared"):
        metadata(tmp_path,config,"train")


def test_benchmark_writes_separate_complete_outputs(tmp_path,monkeypatch):
    import ncrf.evaluation.benchmark as module
    array=np.random.default_rng(1).normal(0,.1,(4,101,2,16,16)).astype(np.float32)
    monkeypatch.setattr(module,"evaluation_data",lambda *args,**kwargs:(array,0.,1.,{},"fixture"))
    output=tmp_path/"timing"
    module.benchmark("persistence",None,None,output,device="cpu",trajectories=2,warmup=1)
    report=json.loads((output/"runtime_summary.json").read_text())
    manifest=json.loads((output/"runtime_manifest.json").read_text())
    assert manifest["complete"] and manifest["trajectory_indices"]==[0,3]
    with (output/"per_trajectory_runtime.csv").open(newline="") as handle:
        rows=list(csv.DictReader(handle))
    times=np.array([float(r["model_seconds"]) for r in rows])
    assert report["model_mean_seconds"]==float(times.mean())
    assert report["model_trajectory_std_seconds"]==float(times.std(ddof=1))
    assert report["speedup_available"] is False
    assert manifest["raw_table_sha256"]==sha256_file(output/"per_trajectory_runtime.csv")


def test_paired_benchmark_retains_speedup_basis(tmp_path,monkeypatch):
    import ncrf.evaluation.benchmark as module
    import ncrf.evaluation.solver_kernels as kernels
    sys.path.insert(0,str(project_root()/"tools/data"))
    import solver_utils
    array=np.zeros((4,101,2,16,16),dtype=np.float32)
    array[3,0,0,0,0]=.3
    monkeypatch.setattr(module,"evaluation_data",lambda *args,**kwargs:(array,0.,1.,{},"fixture"))
    monkeypatch.setattr(solver_utils,"_load_forward_solver",lambda *args:object())
    monkeypatch.setattr(kernels,"trapezoidal_weights",lambda *args:(None,None))
    monkeypatch.setattr(kernels,"reconstruct_context_state",lambda forward,trajectory,**kwargs:float(trajectory[0,0,0,0]))
    monkeypatch.setattr(kernels,"time_solver_rollout",lambda forward,state,**kwargs:(2.+state,0.,0.))
    output=tmp_path/"paired"
    module.benchmark("persistence",None,None,output,device="cpu",trajectories=2,warmup=1,solver=True)
    basis=json.loads((output/"speedup_basis.json").read_text())
    summary=json.loads((output/"runtime_summary.json").read_text())
    assert summary["speedup_available"] is True
    assert basis["speedup"]==summary["solver_mean_seconds"]/summary["model_mean_seconds"]
    assert basis["rounded_before_division"] is False
    assert basis["raw_table_sha256"]==sha256_file(output/"per_trajectory_runtime.csv")


def test_prediction_saving_is_default_and_can_be_explicitly_disabled():
    from ncrf.cli import parser
    assert parser().parse_args(["evaluate","--output","unused"]).save_predictions is True
    assert parser().parse_args(["evaluate","--output","unused","--no-save-predictions"]).save_predictions is False


def test_solver_is_importable_and_pinned():
    path=project_root()/"tools/data"
    sys.path.insert(0,str(path))
    try:
        from solver_utils import validate_pinned_solver,_load_forward_solver,PhysicalConfig
        root=project_root()/"third_party/vch_solver"
        identity=validate_pinned_solver(root)
        module=_load_forward_solver(root)
        assert callable(module.solve_w) and callable(module.newton_raphson)
        assert identity["pinned_commit"].startswith("0f46a5f")
        from ncrf.evaluation.solver_kernels import trapezoidal_weights
        laplacian,weights=trapezoidal_weights(module,PhysicalConfig())
        assert laplacian.shape==(128*128,128*128) and weights.shape==(128,128)
        assert weights.sum()==pytest.approx(1.)
    finally:
        sys.path.remove(str(path))


def test_plot_writes_editable_svg_text_without_touching_arrays(tmp_path):
    from matplotlib import font_manager
    from ncrf.reporting.plots import plot_prediction
    try:
        font_manager.findfont("Times New Roman",fallback_to_default=False)
    except ValueError:
        pytest.skip("Install Times New Roman to check publication SVG fonts.")
    data=np.zeros((2,16,16),dtype=np.float32)
    path=tmp_path/"prediction.npz"
    np.savez(path,prediction=data,truth=data,control=data)
    digest=sha256_file(path)
    output=tmp_path/"plot.svg"
    plot_prediction(path,output)
    text=output.read_text(encoding="utf-8")
    assert "<text" in text and "Times New Roman" in text
    assert sha256_file(path)==digest
    with pytest.raises(FileExistsError):
        plot_prediction(path,output)
