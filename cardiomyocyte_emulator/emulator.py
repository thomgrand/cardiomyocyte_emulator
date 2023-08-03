import torch
from torch.nn import Sequential, Linear, ELU, Tanh, Module
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np

class APEmulator(Module):
    nr_conds : int #No. max. conductances
    max_conds_names : Iterable[str]
    max_conds_ranges : Tuple[np.ndarray, np.ndarray]
    vm_ampl : float = 120.
    t_polynomial_degree : int = 8

    def __init__(self, device="cpu") -> None:
        super().__init__()
        self.device = device

        hyperparams = np.load(Path(__file__).parent / "emulator_hyperparams.npz", allow_pickle=True)
        max_conds_names = hyperparams["max_conds_names"]
        max_conds_ranges = hyperparams["max_conds_ranges"]

        self.max_conds_names = max_conds_names
        self.max_conds_ranges = max_conds_ranges
        self.max_conds_ranges_t = tuple(torch.from_numpy(r.astype(np.float32)).to(self.device)[None] for r in max_conds_ranges)
        self.max_conds_extent = self.max_conds_ranges[1] - self.max_conds_ranges[0]
        self.max_conds_extent_t = self.max_conds_ranges_t[1] - self.max_conds_ranges_t[0]
        self.nr_conds = len(max_conds_names)

        self.latent_encoder = Sequential(Linear(self.nr_conds, 256), ELU(),
                    Linear(256, 256), ELU(),
                    Linear(256, 256), ELU(),
                    Linear(256, 256), ELU(),
                    Linear(256, 256+3), ELU()
                    ).to(self.device)

        self.vm_emu = Sequential(Linear(256+self.t_polynomial_degree, 64), ELU(),
                                Linear(64, 64), ELU(),
                                Linear(64, 64), ELU(),
                                Linear(64, 64), Tanh(),
                                Linear(64, 1)).to(self.device)
        
    @property
    def max_conds_center(self):
        return 0.5 * (self.max_conds_ranges[0] + self.max_conds_ranges[1])

    @property
    def max_conds_center_t(self):
        return 0.5 * (self.max_conds_ranges_t[0] + self.max_conds_ranges_t[1])

    def normalized_time(self, x):
        return (x  + 10) / 1010 - 0.5
    
    def denormalized_time(self, x):
        return (x + 0.5) * 1010 - 10
    
    def depol_f(self, t, depol_ampl, depol_offset, depol_slope):   
        return torch.sigmoid(depol_slope**2 * (t - depol_offset)) * depol_ampl**2
    
    def normalize_max_conds_ranges(self, x : torch.Tensor) -> torch.Tensor:
        x_normed = (x - self.max_conds_ranges_t[0]) / self.max_conds_extent_t - 0.5
        return x_normed

    def normalize_max_conds_ranges_np(self, x : np.ndarray) -> np.ndarray:
        x_normed = (x - self.max_conds_ranges[0]) / self.max_conds_extent - 0.5
        return x_normed

    def denormalize_max_conds_ranges(self, x_normed : torch.Tensor) -> torch.Tensor:
        x = x_normed + 0.5
        x = x * self.max_conds_extent_t[0] + self.max_conds_ranges_t[0][0]
        return x

    def denormalize_max_conds_ranges_np(self, x_normed : np.ndarray) -> np.ndarray:
        x = x_normed + 0.5
        x = x * self.max_conds_extent + self.max_conds_ranges[0]
        return x

    def normalize_input_ranges(self, x : torch.Tensor, clamp=False) -> torch.Tensor:
        x_normed = self.normalize_max_conds_ranges(x[..., 1:], clamp)
        t = self.normalized_time(x)
        return torch.cat((t, x_normed), dim=-1)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        t = x[..., 0:1]
        t_normed = self.normalized_time(t)
        x = self.normalize_input_ranges(x)
        latent_code = self.latent_encoder(x[..., 1:])
        in2 = torch.cat((t_normed, latent_code), dim=-1)
        return self.vm_emu(in2)
    
    def forward_latent(self, t : torch.Tensor, max_conds : torch.Tensor) -> torch.Tensor:
        if max_conds.ndim == 1:
            max_conds = max_conds[None]

        assert t.ndim == 1
        assert max_conds.ndim == 2
        assert max_conds.shape[1] == len(self.max_conds_names), f"Expected {len(self.max_conds_names)} max_conds, but got {max_conds.shape[1]}"

        biomarkers_intermediate = self.latent_encoder(self.normalize_max_conds_ranges(max_conds))
        biomarkers_intermediate, depol_params = biomarkers_intermediate[..., :-3], biomarkers_intermediate[..., -3:]

        M, N = biomarkers_intermediate.shape
        T = t.numel()
        t_batched_ret = torch.tile(t, [M, 1])
        t_batched = t_batched_ret.T
        t_batched_norm = self.normalized_time(t_batched)
        t_batched_flat = t_batched_norm.reshape([-1, 1]) #[T] -> [M, T] -> [T, M] -> [TM, 1]
        x = torch.cat((*((t_batched_flat,) + tuple(t_batched_flat**i for i in range(2, self.t_polynomial_degree+1))),
                       torch.tile(biomarkers_intermediate, [T, 1, 1]).reshape([-1, N])), #[M, N] -> [T, M, N] -> [TM, N]
                       dim=-1)
        result = (self.vm_emu(x).reshape([T, M])  + self.depol_f(t_batched, 
                                                        depol_params[..., 0][None], 
                                                        depol_params[..., 1][None] + 1, 
                                                        depol_params[..., 2][None] + 3,
                                                    )).T
        
        #Magnify the network output
        result = result * self.vm_ampl

        return result

    
    def forward_latent_np(self, t : np.ndarray, max_conds : np.ndarray):
        with torch.no_grad():
            return self.forward_latent(torch.from_numpy(t.astype(np.float32)).to(self.device),
                                    torch.from_numpy(max_conds.astype(np.float32)).to(self.device)).detach().cpu().numpy()
    

def load_default_emulator_model(device="cpu"):
    state_path = Path(__file__).parent / "emulator_state.pt"
    assert state_path.exists(), f"Missing state file '{state_path:s}'. This should be provided with the repo"
    state_dict = torch.load(state_path, map_location=device)
    max_conds_names = state_dict["max_conds_names"]
    max_conds_ranges = state_dict["max_conds_ranges"]
    del state_dict["max_conds_names"]
    del state_dict["max_conds_ranges"]

    emulator = APEmulator(device=device)
    emulator.load_state_dict(state_dict)
    emulator.max_conds_names = max_conds_names
    emulator.max_conds_ranges = max_conds_ranges

    for l in emulator.latent_encoder + emulator.vm_emu:
        l.requires_grad_(False)
    
    return emulator
