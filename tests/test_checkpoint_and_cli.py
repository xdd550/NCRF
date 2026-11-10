import random
import numpy as np
import torch
import pytest
from ncrf.cli import main
from ncrf.models.registry import build_model
from ncrf.training.checkpoints import model_state,rng_state,restore_rng,save_checkpoint,load_checkpoint,load_for_inference
from ncrf.training.engine import update
from ncrf.losses.long_horizon import VCHV15NeumannLongHorizonLoss


def test_rng_round_trip():
    state=rng_state()
    first=(random.random(),np.random.rand(),torch.rand(2))
    restore_rng(state)
    second=(random.random(),np.random.rand(),torch.rand(2))
    assert first[:2]==second[:2] and torch.equal(first[2],second[2])


def test_safe_checkpoint_roundtrip_fno(tmp_path):
    model=build_model("fno",image_size=16)
    path=tmp_path/"weights.pt"
    state=model_state(model)
    save_checkpoint(path,dict(schema="ncrf_clean_checkpoint_v1",model_state=state))
    loaded=load_checkpoint(path)
    model.load_state_dict(loaded["model_state"],strict=True)
    assert all(torch.equal(v,loaded["model_state"][k]) for k,v in state.items())


def test_cpu_update_and_optimizer_resume(tmp_path):
    torch.manual_seed(7)
    model=build_model("ncrf",image_size=16,checkpoint_steps=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler("cpu",enabled=False)
    objective=VCHV15NeumannLongHorizonLoss()
    batch=(torch.randn(1,10,1,16,16)*.1,torch.randn(1,10,1,16,16),torch.randn(1,2,1,16,16),torch.randn(1,2,1,16,16)*.1)
    update(model,objective,optimizer,scaler,batch)
    path=tmp_path/"state.pt"
    save_checkpoint(path,dict(schema="ncrf_clean_checkpoint_v1",model_state=model_state(model),optimizer_state=optimizer.state_dict(),rng=rng_state()))
    expected=update(model,objective,optimizer,scaler,batch)
    expected_state=model_state(model)
    fresh=build_model("ncrf",image_size=16,checkpoint_steps=False)
    opt=torch.optim.AdamW(fresh.parameters(),lr=2e-4,weight_decay=.01)
    loaded=load_checkpoint(path)
    fresh.load_state_dict(loaded["model_state"]); opt.load_state_dict(loaded["optimizer_state"])
    restore_rng(loaded["rng"])
    actual=update(fresh,objective,opt,scaler,batch)
    assert actual["loss"]==expected["loss"]
    assert all(torch.equal(v,fresh.state_dict()[k]) for k,v in expected_state.items())


def test_list_and_plan_do_not_touch_data(capsys):
    main(["list"])
    assert "generic_update" in capsys.readouterr().out
    main(["plan","--method","fno"])
    assert '"updates": 700' in capsys.readouterr().out


def test_inference_rejects_missing_checkpoint():
    with pytest.raises(ValueError):
        load_for_inference("ncrf",None)
