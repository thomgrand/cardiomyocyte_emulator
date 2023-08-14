from .emulator import APEmulator
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import trange

def match_Vm(emulator : APEmulator, t : np.ndarray, traces : np.ndarray, 
                weights=None,
                x0=None, lambda_x0=10., x_init=None, 
                epochs=1000, lr=1e-4, batch_size=None, 
                optimize_t_offset=False,
                return_best=True,
                verbose=True) -> np.ndarray:

    if traces.ndim == 1:
        traces = traces[np.newaxis]

    if weights is None:
        weights = torch.ones(size=[t.size], device=emulator.device, dtype=torch.float32)
    
    if issubclass(type(weights), (np.ndarray,)):
        weights = torch.from_numpy(weights).to(emulator.device, dtype=torch.float32)

    if weights.ndim == 1:
        weights = torch.tile(weights, [traces.shape[0], 1])

    mean_fact = weights.sum()
    weights = weights / mean_fact #Mean normalization

    if x_init is None:
        x_init = torch.tile(emulator.max_conds_center_t, (traces.shape[0],1))
    elif issubclass(type(x_init), np.ndarray):
        x_init = torch.from_numpy(x_init.astype(np.float32)).to(emulator.device).clone()
    else:
        assert False, "Unknown format for x_init"

    if x0 is None:
        x0 = torch.tile(emulator.max_conds_center_t, (traces.shape[0],1))
    elif issubclass(type(x0), np.ndarray):
        x0 = torch.from_numpy(x0.astype(np.float32)).to(emulator.device).clone()
    else:
        assert False, "Unknown format for x0"

    if batch_size is None:
        batch_size = t.size

    param_vecs = x_init.detach().clone()

    #Prepare the data
    param_vecs.requires_grad = True
    traces_t = torch.from_numpy(traces.astype(np.float32)).to(emulator.device)
    t_t = torch.from_numpy(t.astype(np.float32)).to(emulator.device)
    data_time_loader = DataLoader(np.arange(t.size, dtype=np.int64), batch_size=min(batch_size, t.size), 
                                shuffle=batch_size != t.size)

    if optimize_t_offset:
        if issubclass(type(optimize_t_offset), float):
            t_init = float(np.squeeze(optimize_t_offset))
        else:
            t_init = 0.

        t_offset = torch.tensor([t_init], dtype=torch.float32, device=emulator.device, requires_grad=True)
        opt = Adam((param_vecs,t_offset), lr=lr)
    else:
        opt = Adam((param_vecs,), lr=lr)
        t_offset = torch.tensor([0.], dtype=torch.float32, device=emulator.device)

    best_loss = np.inf
    best_params = param_vecs.detach().clone()

    pbar = (trange(epochs) if verbose else range(epochs))
    loss_f = lambda x, y: 0.5 * torch.sum(weights * (x - y)**2)
    for epoch_i in pbar: 
        total_loss = 0
        for t_ind_batch in data_time_loader: 
            t_ind_batch = t_ind_batch.to(emulator.device)

            opt.zero_grad()
            t_batch = t_t[t_ind_batch]
            
            if optimize_t_offset:
                t_batch += t_offset

            pred = emulator.forward_latent(t_batch, param_vecs)

            loss = loss_f(traces_t[..., t_ind_batch], pred)

            if lambda_x0 > 0:
                loss += lambda_x0/2. * torch.mean((emulator.normalize_max_conds_ranges(param_vecs) - emulator.normalize_max_conds_ranges(x0))**2)

            loss.backward()
            opt.step()

            #Projection on the feasible space
            with torch.no_grad():
                param_vecs[:] = param_vecs.clamp(min=emulator.max_conds_ranges_t[0], max=emulator.max_conds_ranges_t[1])

            total_loss += loss.detach().cpu().numpy()

        if total_loss < best_loss:
            best_loss = total_loss
            best_params = param_vecs.detach().clone()

        if verbose and epoch_i % 50 == 0:
            pbar.set_postfix_str(f"Epoch: {epoch_i}, Loss: {total_loss}")

    if return_best:
        param_vecs = best_params

    with torch.no_grad():
        full_pred = emulator.forward_latent(t_t + t_offset, param_vecs)


    if optimize_t_offset:
        return param_vecs.detach().cpu().numpy(), full_pred.detach().cpu().numpy(), t_offset.detach().cpu().numpy()
    else:
        return param_vecs.detach().cpu().numpy(), full_pred.detach().cpu().numpy()
