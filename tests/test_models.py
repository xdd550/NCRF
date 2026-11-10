import pytest
import torch
from ncrf.cli import synthetic_smoke
from ncrf.config import METHODS,load_config
from ncrf.models.registry import build_model,parameter_count
from ncrf.models.operators import zero_flux_divergence


@pytest.mark.parametrize("method",METHODS)
def test_forward_backward_and_parameter_count(method):
    row=synthetic_smoke([method])[0]
    expected={"ncrf":186167,"plain_rescnn":210281,"unet":192331,"convlstm":169321,
              "fno":202105,"persistence":0,"no_memory":158325,"generic_update":186167,"single_scale":182646}
    assert row["parameters"]==expected[method]


def test_no_memory_and_single_scale_are_real_architectural_changes():
    b=build_model("no_memory",image_size=16)
    d=build_model("single_scale",image_size=16)
    assert b.relaxation_initializer is None and b.relaxation_encoder is None
    assert len(d.fine_blocks)==9
    assert d.coarse_input is None
    assert parameter_count(d)!=parameter_count(b)


def test_flux_divergence_telescopes_without_wraparound():
    q=torch.randn(2,1,16,16,requires_grad=True)
    m=torch.rand_like(q)
    divergence=zero_flux_divergence(q,m)
    assert torch.allclose(divergence.sum((-2,-1)),torch.zeros(2,1),atol=1e-5)
    divergence.square().mean().backward()
    assert torch.isfinite(q.grad).all()


def test_persistence_is_exact_last_observation():
    x=torch.rand(1,10,1,16,16)
    y=build_model("persistence")(x,torch.zeros_like(x),torch.zeros(1,3,1,16,16))
    assert torch.equal(y[:,0],x[:,-1]) and torch.equal(y[:,2],x[:,-1])


def test_training_budget_and_losses():
    c=load_config("ncrf")
    assert sum(s["updates"] for s in c["training"]["stages"])==2800
    assert sum(2*s["updates"]*s["horizon"] for s in c["training"]["stages"])==358000
    assert c["loss"]["interface_registration_max_weight"]==0
