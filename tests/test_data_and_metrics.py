import numpy as np
import pytest
from ncrf.data.windows import sample_window
from ncrf.data.sampling import _coverage_cycle_windows
from ncrf.evaluation.metrics import contour_metrics,field_metrics,neumann_spectral_length,relative_neumann_spectral_length_error


def test_causal_windows_keep_future_targets_separate():
    a=np.zeros((2,101,2,16,16),dtype=np.float32)
    a[:,:,0]=np.arange(101)[None,:,None,None]
    a[:,:,1]=-a[:,:,0]
    phi,u,future,target=sample_window(a,np.array([1]),np.array([0]),20)
    assert phi.shape==(1,10,1,16,16)
    assert phi[0,-1,0,0,0]==9 and future[0,0,0,0,0]==-10
    assert target[0,0,0,0,0]==10
    with pytest.raises(IndexError):
        sample_window(a,np.array([0]),np.array([2]),90)


def test_coverage_cycle_is_deterministic_and_without_replacement():
    parameters=dict(seed=20261002,stage_index=1,batch_size=2,trajectory_count=10,maximum_start=1)
    rows=[_coverage_cycle_windows(update=i,**parameters) for i in range(1,6)]
    assert len(set(np.concatenate([r[0] for r in rows]).tolist()))==10
    assert all(np.array_equal(x,y) for x,y in zip(rows[1],_coverage_cycle_windows(update=2,**parameters)))


def test_metrics_identical_fields_and_empty_contours():
    field=np.tile(np.linspace(-1,1,16),(16,1))
    metric=field_metrics(np.stack([field]*2),np.stack([field]*2),field)
    assert metric["mae"]==0 and metric["source_grid_mass_abs_drift_mean"]==0
    assert metric["zero_contour_assd"]==pytest.approx(0,abs=1e-14) and metric["zero_contour_hd95"]==pytest.approx(0,abs=1e-14)
    assert metric["terminal_band_dice_0.2"]==1
    assert contour_metrics(np.ones((16,16)),field)["zero_contour_hd95"]==pytest.approx(np.sqrt(2))
    assert contour_metrics(np.ones((16,16)),np.ones((16,16)))["zero_contour_assd"]==0


def test_zero_spectrum_rule_and_physical_frequency_units():
    zero=np.zeros((16,16))
    assert np.isinf(neumann_spectral_length(zero))
    assert relative_neumann_spectral_length_error(zero,zero)==0
    cosine=np.tile(np.cos(np.pi*np.linspace(0,1,16)),(16,1))
    assert neumann_spectral_length(cosine)==pytest.approx(2,abs=1e-12)
