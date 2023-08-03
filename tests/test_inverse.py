from test_emulator import default_emulator_init
from cardiomyocyte_emulator.inverse import match_Vm
import pytest
import numpy as np
import torch

@pytest.fixture
def reset_rand():
    np.random.seed(0)
    torch.manual_seed(0)

@pytest.fixture(scope="module")
def inverse_setup(default_emulator_init):
    np.random.seed(0)
    emulator = default_emulator_init
    t = np.linspace(-10, 500, num=1000)
    max_conds = np.concatenate([emulator.max_conds_center[np.newaxis], emulator.denormalize_max_conds_ranges_np(np.random.uniform(-0.1, 0.1, size=(4, emulator.nr_conds)))], axis=0)
    aps = emulator.forward_latent_np(t, max_conds)
    return emulator, t, aps, max_conds

def _inverse_error_asserts(emulator, aps, aps_emulated, max_conds, max_conds_est):
    aps_diff = aps - aps_emulated
    assert (aps_diff**2).mean() < 0.5 #MSE < 0.5 mV
    max_conds_err_normed = emulator.normalize_max_conds_ranges_np(max_conds) - emulator.normalize_max_conds_ranges_np(max_conds_est)
    assert (max_conds_err_normed**2).mean() < 0.1 #MSE < 10%

def test_match_Vm_simple(inverse_setup, reset_rand):
    emulator, t, aps, max_conds = inverse_setup
    max_conds_est, aps_emulated = match_Vm(emulator, t, aps, verbose=False, epochs=500, lr=2.5e-4)
    _inverse_error_asserts(emulator, aps, aps_emulated, max_conds, max_conds_est)

def test_match_Vm_simple_verbose(inverse_setup, reset_rand):
    emulator, t, aps, max_conds = inverse_setup
    max_conds_est, aps_emulated = match_Vm(emulator, t, aps, verbose=True, epochs=500, lr=2.5e-4)
    _inverse_error_asserts(emulator, aps, aps_emulated, max_conds, max_conds_est)

def test_match_Vm_single(inverse_setup, reset_rand):
    emulator, t, aps, max_conds = inverse_setup
    aps = aps[1]
    max_conds = max_conds[1]
    max_conds_est, aps_emulated = match_Vm(emulator, t, aps, verbose=False, epochs=500, lr=2.5e-4)
    _inverse_error_asserts(emulator, aps, aps_emulated, max_conds, max_conds_est)

def test_match_Vm_x0_prior(inverse_setup, reset_rand):
    emulator, t, aps, max_conds = inverse_setup
    max_conds_est, aps_emulated = match_Vm(emulator, t, aps, verbose=False, epochs=500, lr=2.5e-4, lambda_x0=10)
    _inverse_error_asserts(emulator, aps, aps_emulated, max_conds, max_conds_est)

def test_match_Vm_custom_x_init(inverse_setup, reset_rand):
    emulator, t, aps, max_conds = inverse_setup
    x_init_normed = np.tile([-0.1], (emulator.nr_conds))
    x_init = emulator.denormalize_max_conds_ranges_np(x_init_normed)
    max_conds_est, aps_emulated = match_Vm(emulator, t, aps[1], verbose=False, epochs=0, x_init=x_init)
    assert np.allclose(max_conds_est - x_init, 0., atol=1e-5)

def test_match_Vm_custom_x0(inverse_setup, reset_rand):
    emulator, t, aps, max_conds = inverse_setup
    x0_normed = np.tile([-0.1], (emulator.nr_conds))
    x0 = emulator.denormalize_max_conds_ranges_np(x0_normed)
    max_conds_est, aps_emulated = match_Vm(emulator, t, aps[1], verbose=False, epochs=100, lr=2.5e-4, x0=x0, lambda_x0=1e10, x_init=x0)
    assert np.allclose(max_conds_est - x0, 0., atol=1e-5)

def test_match_Vm_invalid_params(inverse_setup, reset_rand):
    emulator, t, aps, max_conds = inverse_setup
    with pytest.raises(Exception):
        max_conds_est, aps_emulated = match_Vm(emulator, t, aps, verbose=False, epochs=500, lr=2.5e-4, x_init="Invalid")
    
    with pytest.raises(Exception):
        max_conds_est, aps_emulated = match_Vm(emulator, t, aps, verbose=False, epochs=500, lr=2.5e-4, x0="Invalid")

def test_match_Vm_t_offset(inverse_setup, reset_rand):
    emulator, t, aps, max_conds = inverse_setup
    max_conds = emulator.max_conds_center
    ap = emulator.forward_latent_np(t + 0.25, max_conds)
    max_conds_est, aps_emulated, t_offset = match_Vm(emulator, t, ap, verbose=False, epochs=500, lr=1e-3, optimize_t_offset=True)
    assert np.isclose(t_offset, 0.25, atol=1e-2)
    _inverse_error_asserts(emulator, ap, aps_emulated[0], max_conds, max_conds_est)

    #Custom t-offset
    ap = emulator.forward_latent_np(t + 0.25, max_conds)
    max_conds_est, aps_emulated, t_offset = match_Vm(emulator, t, ap, verbose=False, epochs=500, lr=1e-3, optimize_t_offset=0.15)
    assert np.isclose(t_offset, 0.25, atol=1e-2)
    _inverse_error_asserts(emulator, ap, aps_emulated[0], max_conds, max_conds_est)