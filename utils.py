# Author: Qiwei Wang
"""Shared training, plotting, and model utility functions."""

import torch
from torch.nn import functional as F
from typing import Iterable


class FreezeParameters:
    """Freeze Parameters component."""
    def __init__(self, modules: Iterable[torch.nn.Module]):
        self.modules = modules
        self.params = []
        self.requires_grad_states = []

    def __enter__(self):
        for module in self.modules:
            self.params.extend(list(module.parameters()))
        self.requires_grad_states = [p.requires_grad for p in self.params]
        for p in self.params:
            p.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p, state in zip(self.params, self.requires_grad_states):
            p.requires_grad = state


def imagine_ahead(
    prev_state,
    prev_belief,
    prev_map_embedding,
    prev_semantic_map,
    policy,
    transition_model,
    map_encoder,
    map_transition_model,
    planning_horizon=12,
):
    """Roll out latent imagined trajectories."""
    flatten = lambda x: x.view([-1] + list(x.size())[2:])
    prev_belief = flatten(prev_belief)
    prev_state = flatten(prev_state)
    prev_map_embedding = flatten(prev_map_embedding)

    has_full_map = (prev_semantic_map is not None)
    if has_full_map:
        prev_semantic_map = flatten(prev_semantic_map)
    T = planning_horizon
    N = prev_belief.size(0)
    beliefs = [torch.empty(0)] * T
    prior_states = [torch.empty(0)] * T
    prior_means = [torch.empty(0)] * T
    prior_std_devs = [torch.empty(0)] * T
    map_embeddings = [torch.empty(0)] * T
    actions = [torch.empty(0)] * T
    current_belief = prev_belief
    current_state = prev_state
    current_map_embedding = prev_map_embedding
    current_map = prev_semantic_map

    for t in range(planning_horizon):
        current_belief        = torch.nan_to_num(current_belief,        nan=0.0, posinf=50.0, neginf=-50.0)
        current_state         = torch.nan_to_num(current_state,         nan=0.0, posinf=20.0, neginf=-20.0)
        current_map_embedding = torch.nan_to_num(current_map_embedding, nan=0.0, posinf=50.0, neginf=-50.0)

        _action = policy.get_action(current_belief, current_state, current_map_embedding)
        actions[t] = _action
        output = transition_model(
            current_state,
            _action.unsqueeze(0),
            current_belief,
            observations=None,
            nonterminals=None,
            map_embeddings=current_map_embedding.unsqueeze(0),
        )

        next_belief = output[0][0]
        next_state = output[1][0]
        _prior_mean = output[2][0]
        _prior_std_dev = output[3][0]

        if has_full_map:
            next_map = map_transition_model(
                current_map, next_belief, next_state, _action, current_map_embedding
            )
            next_map_embedding = map_encoder(next_map)
        else:
            next_map = current_map
            next_map_embedding = current_map_embedding

        current_belief = next_belief
        current_state = next_state
        current_map = next_map
        current_map_embedding = next_map_embedding

        beliefs[t] = current_belief
        prior_states[t] = current_state
        prior_means[t] = _prior_mean
        prior_std_devs[t] = _prior_std_dev
        map_embeddings[t] = current_map_embedding

    return (
        torch.stack(beliefs),
        torch.stack(prior_states),
        torch.stack(prior_means),
        torch.stack(prior_std_devs),
        torch.stack(map_embeddings),
        torch.stack(actions),
    )


def lambda_return(imged_reward, value_pred, bootstrap, cont_pred=None,
                  discount=0.99, lambda_=0.95):
    """Compute lambda-return targets."""
    if cont_pred is None:
        
        cont = torch.ones_like(imged_reward)
    else:
        cont = cont_pred

    next_values = torch.cat([value_pred[1:], bootstrap[None]], 0)
    inputs = imged_reward + discount * cont * next_values * (1 - lambda_)
    last = bootstrap
    indices = reversed(range(len(inputs)))
    outputs = []
    for index in indices:
        last = inputs[index] + discount * lambda_ * cont[index] * last
        outputs.append(last)
    outputs = list(reversed(outputs))
    returns = torch.stack(outputs, 0)
    return returns


def trk_loss_multi_peak(pred_occ, gt_occ, n_peaks, patch_size=5):
    """Trk loss multi peak."""
    B, Ng, _ = pred_occ.shape
    device = pred_occ.device
    half = patch_size // 2

    with torch.no_grad():
        pooled = F.max_pool2d(
            gt_occ.unsqueeze(1), kernel_size=3, stride=1, padding=1
        ).squeeze(1)
        nms_mask = (gt_occ >= pooled - 1e-8).float()
        gt_masked = gt_occ * nms_mask
        flat = gt_masked.reshape(B, -1)
        _, topk_idx = flat.topk(n_peaks, dim=-1)      

        gt_ys_int = (topk_idx // Ng).clamp(half, Ng - 1 - half)
        gt_xs_int = (topk_idx %  Ng).clamp(half, Ng - 1 - half)
        gt_coords = torch.stack([gt_xs_int.float(), gt_ys_int.float()], dim=-1)

    offs = torch.arange(-half, half + 1, device=device, dtype=torch.long)
    patch_y = gt_ys_int.unsqueeze(-1).unsqueeze(-1) + offs.view(1, 1, -1, 1)
    patch_x = gt_xs_int.unsqueeze(-1).unsqueeze(-1) + offs.view(1, 1, 1, -1)
    patch_y = patch_y.expand(B, n_peaks, patch_size, patch_size)
    patch_x = patch_x.expand(B, n_peaks, patch_size, patch_size)
    b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand_as(patch_y)
    pred_patches = pred_occ[b_idx, patch_y, patch_x]
    flat_patches = pred_patches.reshape(B, n_peaks, -1)        
    weights = flat_patches.softmax(dim=-1)                     
    local_x = patch_x.float().reshape(B, n_peaks, -1)          
    local_y = patch_y.float().reshape(B, n_peaks, -1)
    pred_x = (weights * local_x).sum(dim=-1)                   
    pred_y = (weights * local_y).sum(dim=-1)
    pred_coords = torch.stack([pred_x, pred_y], dim=-1)        

    return F.smooth_l1_loss(pred_coords, gt_coords, reduction='none')


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Apply the symlog transform."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Apply the inverse symlog transform."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def compute_boundary_target(grid_size: int, margin: float = 2.0,
                             device=None, dtype=torch.float32) -> torch.Tensor:
    """Compute boundary target."""
    g = torch.arange(grid_size, dtype=dtype, device=device)
    gx = g.view(-1, 1).expand(grid_size, grid_size)
    gy = g.view(1, -1).expand(grid_size, grid_size)
    d = torch.minimum(
        torch.minimum(gx, grid_size - 1 - gx),
        torch.minimum(gy, grid_size - 1 - gy),
    )
    bdry = (1.0 - d / max(margin, 1e-6)).clamp(min=0.0, max=1.0)
    return bdry.detach()


def lineplot(x, y, name, path, xaxis='Episodes'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        plt.figure()
        plt.plot(x, y)
        plt.xlabel(xaxis)
        plt.ylabel(name)
        plt.savefig(path + '/' + name + '.png')
    except Exception as e:
        print(f"Plotting error: {e}")
    finally:
        plt.close()


def write_video(frames, title, path):
    pass


def numpy_to_torch(array):
    return torch.tensor(array)