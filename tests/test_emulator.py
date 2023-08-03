import pytest
from cardiomyocyte_emulator import APEmulator, load_default_emulator_model
import torch
import numpy as np

available_devices = (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])

def test_import():
    pass

@pytest.fixture(scope="module", params=available_devices)
def default_emulator_init(request):
    return load_default_emulator_model(device=request.param)

def test_load_model(default_emulator_init):
    pass

def test_simple_forward_torch(default_emulator_init : APEmulator):
    emulator = default_emulator_init
    t = np.linspace(-5, 500, num=200)
    vm = emulator.forward_latent(torch.from_numpy(t).to(emulator.device, dtype=torch.float32), 
                                 torch.from_numpy(emulator.max_conds_extent / 2).to(emulator.device))
    assert vm.ndim in [1, 2]
    assert vm.shape[-1] == t.size
    assert vm.min() > -120
    assert vm.max() < 50
    assert issubclass(type(vm), (torch.Tensor,))
    assert vm.requires_grad == False

def test_simple_forward_np(default_emulator_init : APEmulator):
    emulator = default_emulator_init
    t = np.linspace(-5, 500, num=200)
    vm = emulator.forward_latent_np(t, emulator.max_conds_center)
    assert vm.ndim in [1, 2]
    assert vm.shape[-1] == t.size
    assert vm.min() > -120
    assert vm.max() < 50
    assert issubclass(type(vm), (np.ndarray,))

def test_inverse_gradient(default_emulator_init : APEmulator):
    emulator = default_emulator_init
    t = torch.from_numpy(np.linspace(-5, 500, num=200)).to(emulator.device, dtype=torch.float32)
    max_conds = emulator.max_conds_center_t
    t.requires_grad = True
    vm = emulator.forward_latent(t, max_conds)
    assert vm.requires_grad == True

    vm.sum().backward()
    assert t.grad is not None
    assert not np.allclose(t.grad.cpu().numpy(), 0.)
    assert max_conds.grad is None

    t.grad = None
    t.requires_grad = False
    max_conds.requires_grad = True
    vm = emulator.forward_latent(t, max_conds)
    vm.sum().backward()
    assert max_conds.grad is not None
    assert not np.allclose(max_conds.grad.cpu().numpy(), 0.)
    assert t.grad is None
