"""Exercise the real training orchestrator, not only its one-step function.

Only the fixture geometry/budgets are reduced; GPU, optimizer, phase reset,
serialization, interrupted resume, evaluation and aggregation are real.
"""
import copy
import csv
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import ncrf.training.engine as engine
import ncrf.models.registry as registry
import ncrf.evaluation.runner as evaluation
from ncrf.config import load_config
from ncrf.reporting.summary import summarize
from ncrf.training.checkpoints import load_checkpoint
from ncrf.evaluation.metrics import field_metrics


@pytest.fixture
def reduced_workflow(monkeypatch):
    config=copy.deepcopy(load_config("ncrf"))
    config["image_size"]=16
    config["training"]["checkpoint_interval"]=1
    for stage in config["training"]["stages"]:
        stage.update(horizon=2,updates=2)
    rng=np.random.default_rng(17)
    array=rng.normal(0,.1,(4,101,2,16,16)).astype(np.float32)
    original=registry.build_model
    def factory(method,**kwargs):
        kwargs["image_size"]=16
        return original(method,**kwargs)
    monkeypatch.setattr(engine,"load_config",lambda _:copy.deepcopy(config))
    monkeypatch.setattr(engine,"open_split",lambda *args,**kwargs:(array,0.,1.))
    monkeypatch.setattr(engine,"metadata",lambda *args,**kwargs:{})
    monkeypatch.setattr(engine,"build_model",factory)
    monkeypatch.setattr(registry,"build_model",factory)
    return config,array


def assert_nested_equal(a,b):
    if torch.is_tensor(a):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:
            assert_nested_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            assert_nested_equal(x,y)
    else:
        assert a==b


@pytest.mark.skipif(not torch.cuda.is_available(),reason="Actual CUDA workflow check")
def test_six_stage_training_resume_and_evaluation(tmp_path,monkeypatch,reduced_workflow):
    _,array=reduced_workflow
    complete=tmp_path/"continuous"
    engine.train("ncrf",20261002,None,complete,execute=True,strict_determinism=True)
    expected=load_checkpoint(complete/"final.pt")
    assert expected["complete"] and expected["stage_index"]==5 and expected["done"]==2
    artifacts=json.loads((complete/"artifacts.json").read_text(encoding="utf-8"))
    assert set(artifacts["files"])=={"final.pt","latest.pt","run_manifest.json","training.jsonl"}
    assert artifacts["automatic_cleanup"] is False
    original_save=engine.save_checkpoint
    # Before/at the continuation reset, within continuation, and after its last update.
    for boundary in ((0,1),(3,2),(4,1),(5,2)):
        directory=tmp_path/f"resume_{boundary[0]}_{boundary[1]}"
        def interrupted_save(path,state):
            original_save(path,state)
            if Path(path).name=="latest.pt" and (state["stage_index"],state["done"])==boundary:
                raise KeyboardInterrupt("Injected immediately after a durable checkpoint")
        monkeypatch.setattr(engine,"save_checkpoint",interrupted_save)
        with pytest.raises(KeyboardInterrupt):
            engine.train("ncrf",20261002,None,directory,execute=True,strict_determinism=True)
        assert not (directory/"RUNNING.lock").exists()
        monkeypatch.setattr(engine,"save_checkpoint",original_save)
        engine.train("ncrf",20261002,None,directory,execute=True,resume=True,strict_determinism=True)
        resumed=load_checkpoint(directory/"final.pt")
        for key in ("model_state","optimizer_state","scaler_state","rng_state"):
            assert_nested_equal(expected[key],resumed[key])
    groups={i:dict(initial_id=f"initial_{i}",control_id=0) for i in range(4)}
    monkeypatch.setattr(evaluation,"evaluation_data",lambda *args,**kwargs:(array,0.,1.,groups,"fixture_hash"))
    output=tmp_path/"evaluation"
    evaluation.evaluate("ncrf",complete/"final.pt",None,output,device="cuda",horizons=(10,20),limit=2)
    assert (output/"prediction_index.csv").exists()
    with (output/"per_trajectory.csv").open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==4
    for row in rows:
        h,i=int(row["horizon"]),int(row["trajectory_index"])
        with np.load(output/f"predictions/H{h}/trajectory_{i:03d}.npz") as data:
            calculated=field_metrics(data["prediction"],data["truth"],data["last_context"])
            assert np.array_equal(data["history_phi"],array[i,:10,0])
            assert np.array_equal(data["history_control"],array[i,:10,1])
            assert np.array_equal(data["control"],array[i,10:10+h,1])
            assert np.array_equal(data["predicted_frame_indices"],np.arange(10,10+h))
            assert str(data["initial_id"])==f"initial_{i}"
        assert float(row["mae"])==calculated["mae"]
        assert float(row["zero_contour_assd"])==calculated["zero_contour_assd"]
    summarize([output],tmp_path/"summary",allow_diagnostic=True)
    with (tmp_path/"summary/across_seeds.csv").open(encoding="utf-8",newline="") as handle:
        summaries=list(csv.DictReader(handle))
    assert len(summaries)==2 and all(r["mae_seed_std"]=="" for r in summaries)
    for row in summaries:
        selected=[float(r["mae"]) for r in rows if r["horizon"]==row["horizon"]]
        assert float(row["mae_mean"])==float(np.mean(selected))
    with pytest.raises(ValueError):
        summarize([output],tmp_path/"wrong_formal_summary")


def test_summary_rejects_changed_grouping_and_identity(tmp_path):
    from ncrf.utils import write_csv,write_json,sha256_file
    directories=[]
    for seed in (20261002,20261003,20261004):
        path=tmp_path/str(seed);path.mkdir()
        rows=[dict(method="ncrf",seed=seed,scenario="main",split="test",horizon=90,
            trajectory_index=i,initial_id=f"initial_{i//10}",control_id=i%10,mae=float(seed-20261000)) for i in range(180)]
        write_csv(path/"per_trajectory.csv",rows)
        write_json(path/"evaluation_manifest.json",dict(complete=True,diagnostic=False,method="ncrf",seed=seed,
            scenario="main",split="test",horizons=[90],array_sha256="same",normalization={},trajectory_count=180,
            table_sha256=sha256_file(path/"per_trajectory.csv")))
        directories.append(path)
    summarize(directories,tmp_path/"summary")
    with (tmp_path/"summary/across_seeds.csv").open(encoding="utf-8",newline="") as handle:
        row=list(csv.DictReader(handle))[0]
    assert float(row["mae_mean"])==3 and float(row["mae_seed_std"])==1
    rows[0]["initial_id"]="wrong_group"
    write_csv(path/"per_trajectory.csv",rows)
    manifest=json.loads((path/"evaluation_manifest.json").read_text(encoding="utf-8"))
    manifest["table_sha256"]=sha256_file(path/"per_trajectory.csv")
    write_json(path/"evaluation_manifest.json",manifest)
    with pytest.raises(ValueError,match="grouping"):
        summarize(directories,tmp_path/"bad")
