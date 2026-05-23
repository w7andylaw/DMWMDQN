# Author: Qiwei Wang
"""Neural network modules for UAV navigation agents and world models."""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributions
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.nn import functional as F
import random






def bottle(f, x_tuple):
    """Apply a module across flattened leading dimensions."""
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    y_size = y.size()
    output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
    return output


def bottle_semantic(f, x_tuple, extra_tuple):
    """Bottle semantic."""
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    T, B = x_sizes[0][0], x_sizes[0][1]

    flat_x = [x.view(T * B, *x.shape[2:]) for x in x_tuple]
    flat_extra = [e.view(T * B, *e.shape[2:]) for e in extra_tuple]

    y = f(*flat_x, *flat_extra)

    if isinstance(y, tuple):
        return tuple(yi.view(T, B, *yi.shape[1:]) for yi in y)
    return y.view(T, B, *y.shape[1:])






class MapEncoder(nn.Module):
    """Map Encoder component."""

    def __init__(self, n_channels=6, grid_size=30, map_embedding_size=256):
        super().__init__()
        self.map_embedding_size = map_embedding_size
        self.conv1 = nn.Conv2d(n_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.fc = nn.Linear(128 * 4 * 4, map_embedding_size)
        self.component_modules = [self.conv1, self.conv2, self.conv3, self.fc]

    def forward(self, semantic_map):
        """Run the module forward pass."""
        h = F.elu(self.conv1(semantic_map))
        h = F.elu(self.conv2(h))
        h = F.elu(self.conv3(h))
        h = h.view(h.size(0), -1)
        m = self.fc(h)
        return m


class DifferentiableMapUpdater(nn.Module):
    """Differentiable Map Updater component."""

    def __init__(self, embedding_size=1024, n_channels=6, grid_size=30):
        super().__init__()
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.evidence_head = nn.Sequential(
            nn.Linear(embedding_size + 2, 512),   
            nn.ELU(),
            nn.Linear(512, n_channels * grid_size * grid_size),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(embedding_size + 2, 512),
            nn.ELU(),
            nn.Linear(512, n_channels * grid_size * grid_size),
            nn.Sigmoid()  
        )
        self.component_modules = [self.evidence_head, self.gate_head]

    def forward(self, prev_map, embedding, position):
        """Run the module forward pass."""
        B = embedding.size(0)
        cond = torch.cat([embedding, position], dim=1)
        raw_candidate = self.evidence_head(cond).view(
            B, self.n_channels, self.grid_size, self.grid_size
        )
        candidate = torch.cat([
            torch.tanh(raw_candidate[:, :3]),       
            torch.sigmoid(raw_candidate[:, 3:4]),   
            torch.tanh(raw_candidate[:, 4:]),       
        ], dim=1)
        gate = self.gate_head(cond).view(B, self.n_channels, self.grid_size, self.grid_size)
        new_map = (1 - gate) * prev_map + gate * candidate
        return new_map, gate


class MapTransitionModel(nn.Module):
    """Map Transition Model component."""

    def __init__(self, belief_size=200, state_size=30, action_size=8,
                 map_embedding_size=256, n_channels=6, grid_size=30):
        super().__init__()
        self.n_channels = n_channels
        self.grid_size = grid_size
        cond_size = belief_size + state_size + action_size + map_embedding_size
        self.cond_fc = nn.Sequential(
            nn.Linear(cond_size, 512),
            nn.ELU(),
        )
        self.film_gamma1 = nn.Linear(512, 64)
        self.film_beta1 = nn.Linear(512, 64)
        self.film_gamma2 = nn.Linear(512, 64)
        self.film_beta2 = nn.Linear(512, 64)
        self.conv1 = nn.Conv2d(n_channels, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(64, n_channels, 3, stride=1, padding=1)
        self.component_modules = [
            self.cond_fc, self.film_gamma1, self.film_beta1,
            self.film_gamma2, self.film_beta2,
            self.conv1, self.conv2, self.conv_out
        ]

    def forward(self, prev_map, belief, state, action, map_embedding):
        """Run the module forward pass."""
        B = prev_map.size(0)
        cond = self.cond_fc(torch.cat([belief, state, action, map_embedding], dim=1))
        h = self.conv1(prev_map)
        gamma1 = self.film_gamma1(cond).unsqueeze(-1).unsqueeze(-1)  
        beta1 = self.film_beta1(cond).unsqueeze(-1).unsqueeze(-1)
        h = F.elu(gamma1 * h + beta1)
        h = self.conv2(h)
        gamma2 = self.film_gamma2(cond).unsqueeze(-1).unsqueeze(-1)
        beta2 = self.film_beta2(cond).unsqueeze(-1).unsqueeze(-1)
        h = F.elu(gamma2 * h + beta2)
        raw = self.conv_out(h) + prev_map
        next_map = torch.cat([
            torch.tanh(raw[:, :3]),           
            torch.sigmoid(raw[:, 3:4]),       
            torch.tanh(raw[:, 4:]),           
        ], dim=1)

        return next_map


class ObstacleForecaster(nn.Module):
    """Obstacle Forecaster component."""

    def __init__(self, belief_size=200, state_size=30, map_embedding_size=256,
                 forecast_horizon=5, grid_size=30):
        super().__init__()
        self.K = forecast_horizon
        self.grid_size = grid_size
        input_size = belief_size + state_size + map_embedding_size
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ELU(),
            nn.Linear(1024, 128 * 4 * 4),
            nn.ELU(),
        )
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.conv_adapt = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        
        self.occ_head = nn.Conv2d(32, self.K, 3, stride=1, padding=1)

        
        self.flow_head = nn.Conv2d(32, self.K * 2, 3, stride=1, padding=1)

        self.component_modules = [
            self.fc, self.deconv1, self.deconv2, self.conv_adapt,
            self.occ_head, self.flow_head
        ]

    def forward(self, belief, state, map_embedding):
        """Run the module forward pass."""
        B = belief.size(0)

        h = self.fc(torch.cat([belief, state, map_embedding], dim=1))
        h = h.view(B, 128, 4, 4)
        h = F.elu(self.deconv1(h))    
        h = F.elu(self.deconv2(h))    

        
        h = F.interpolate(h, size=(self.grid_size, self.grid_size), mode='bilinear', align_corners=False)
        h = F.elu(self.conv_adapt(h))  

        
        occ = torch.sigmoid(self.occ_head(h))  

        
        flow_raw = torch.tanh(self.flow_head(h))  
        flow = flow_raw.view(B, self.K, 2, self.grid_size, self.grid_size)

        return occ, flow


class BoundaryForecaster(nn.Module):
    """Boundary Forecaster component."""

    def __init__(self, belief_size=200, state_size=30, map_embedding_size=256,
                 grid_size=30):
        super().__init__()
        self.grid_size = grid_size
        input_size = belief_size + state_size + map_embedding_size
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ELU(),
            nn.Linear(1024, 128 * 4 * 4),
            nn.ELU(),
        )
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)   
        self.conv_adapt = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bdry_head = nn.Conv2d(32, 1, 3, stride=1, padding=1)
        self.component_modules = [
            self.fc, self.deconv1, self.deconv2, self.conv_adapt, self.bdry_head,
        ]

    def forward(self, belief, state, map_embedding):
        """Run the module forward pass."""
        B = belief.size(0)
        h = self.fc(torch.cat([belief, state, map_embedding], dim=1))
        h = h.view(B, 128, 4, 4)
        h = F.elu(self.deconv1(h))                              
        h = F.elu(self.deconv2(h))                              
        h = F.interpolate(h, size=(self.grid_size, self.grid_size),
                          mode='bilinear', align_corners=False)
        h = F.elu(self.conv_adapt(h))                           
        bdry = torch.sigmoid(self.bdry_head(h)).squeeze(1)      
        return bdry


class ContinuationModel(nn.Module):
    """Continuation Model component."""
    def __init__(self, belief_size=200, state_size=30, map_embedding_size=256, hidden_size=200):
        super().__init__()
        self._map_emb_size = map_embedding_size
        self.fc1 = nn.Linear(belief_size + state_size + map_embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.component_modules = [self.fc1, self.fc2, self.fc3]

    def forward(self, belief, state, map_embedding=None):
        """Run the module forward pass."""
        if map_embedding is None:
            map_embedding = torch.zeros(belief.size(0), self._map_emb_size, device=belief.device)
        x = torch.cat([belief, state, map_embedding], dim=1)
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        return torch.sigmoid(self.fc3(h)).squeeze(dim=1)


class UAVHybridEncoder(nn.Module):
    """U A V Hybrid Encoder component."""

    def __init__(self, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc_vec = nn.Linear(2, 64)
        self.fc_out = nn.Linear(1024 + 1024 + 64, embedding_size)
        self.component_modules = [
            self.conv1, self.conv2, self.conv3, self.conv4,
            self.fc_vec, self.fc_out
        ]
    def forward_visual(self, img):
        hidden = self.act_fn(self.conv1(img))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        return hidden.view(hidden.size(0), 1024)
    def forward(self, image, target_image, vector, safety_mask=None):
        curr_emb = self.forward_visual(image)
        targ_emb = self.forward_visual(target_image)
        vec_emb = self.act_fn(self.fc_vec(vector))
        combined = torch.cat([curr_emb, targ_emb, vec_emb], dim=1)
        out = self.fc_out(combined)
        return out, curr_emb, targ_emb


class SemanticFeatureExtractor(nn.Module):
    """Semantic Feature Extractor component."""

    def __init__(self, semantic_size=512, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.semantic_size = semantic_size
        self.map_conv1 = nn.Conv2d(6, 32, 3, stride=2, padding=1)
        self.map_conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.map_conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.map_fc = nn.Linear(64 * 4 * 4, 512)
        self.conv1 = nn.Conv2d(3, 16, 4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 4, stride=2)
        self.fc_pos = nn.Linear(2, 64)
        self.fc_fuse = nn.Linear(512 + 512 + 64 + 512, semantic_size)
        self.component_modules = [
            self.map_conv1, self.map_conv2, self.map_conv3, self.map_fc,
            self.conv1, self.conv2, self.conv3, self.conv4,
            self.fc_pos, self.fc_fuse
        ]
    def forward_visual(self, img: torch.Tensor) -> torch.Tensor:
        h = self.act_fn(self.conv1(img))
        h = self.act_fn(self.conv2(h))
        h = self.act_fn(self.conv3(h))
        h = self.act_fn(self.conv4(h))
        return h.view(h.size(0), -1)
    def forward_map(self, semantic_map: torch.Tensor) -> torch.Tensor:
        h = self.act_fn(self.map_conv1(semantic_map))
        h = self.act_fn(self.map_conv2(h))
        h = self.act_fn(self.map_conv3(h))
        h = h.view(h.size(0), -1)
        return self.act_fn(self.map_fc(h))
    def forward(self, image: torch.Tensor, target_image: torch.Tensor,
                position: torch.Tensor, semantic_map: torch.Tensor) -> torch.Tensor:
        """Run the module forward pass."""
        curr_vis = self.forward_visual(image)
        targ_vis = self.forward_visual(target_image)
        pos_emb = self.act_fn(self.fc_pos(position))
        map_emb = self.forward_map(semantic_map)
        combined = torch.cat([curr_vis, targ_vis, pos_emb, map_emb], dim=1)
        return self.act_fn(self.fc_fuse(combined))


class TransitionModel(nn.Module):
    """Transition Model component."""
    def __init__(
            self,
            belief_size,
            state_size,
            action_size,
            hidden_size,
            embedding_size,
            semantic_size=512,          
            semantic_state_size=512,    
            map_embedding_size=256,
            activation_function='relu',
            min_std_dev=0.1,
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.map_embedding_size = map_embedding_size
        self.fc_embed_state_action = nn.Linear(
            state_size + action_size + map_embedding_size, belief_size
        )
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(
            belief_size + embedding_size + map_embedding_size, hidden_size
        )
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
        self.component_modules = [
            self.fc_embed_state_action,
            self.fc_embed_belief_prior,
            self.fc_state_prior,
            self.fc_embed_belief_posterior,
            self.fc_state_posterior,
        ]
    def forward(
            self,
            prev_state: torch.Tensor,
            actions: torch.Tensor,
            prev_belief: torch.Tensor,
            observations: Optional[torch.Tensor] = None,
            nonterminals: Optional[torch.Tensor] = None,
            map_embeddings: Optional[torch.Tensor] = None,
            map_embeddings_post: Optional[torch.Tensor] = None,
            deterministic: bool = False,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the module forward pass."""
        T = actions.size(0) + 1
        B = actions.size(1)
        beliefs = [torch.empty(0)] * T
        prior_states = [torch.empty(0)] * T
        prior_means = [torch.empty(0)] * T
        prior_std_devs = [torch.empty(0)] * T
        posterior_states = [torch.empty(0)] * T
        posterior_means = [torch.empty(0)] * T
        posterior_std_devs = [torch.empty(0)] * T
        beliefs[0] = prev_belief
        prior_states[0] = prev_state
        posterior_states[0] = prev_state

        for t in range(T - 1):
            _state = prior_states[t] if observations is None else posterior_states[t]
            _state = _state if nonterminals is None else _state * nonterminals[t]
            t_ = t
            if map_embeddings is not None:
                map_emb = map_embeddings[t_]
            else:
                map_emb = torch.zeros(B, self.map_embedding_size, device=_state.device)

            hidden = self.act_fn(
                self.fc_embed_state_action(torch.cat([_state, actions[t], map_emb], dim=1))
            )
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])

            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)

            prior_std_devs[t + 1] = (F.softplus(_prior_std_dev) + self.min_std_dev).clamp(max=5.0)
            
            if deterministic:
                prior_states[t + 1] = prior_means[t + 1]
            else:
                prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])

            
            if observations is not None:
                
                if map_embeddings_post is not None:
                    _map_emb_post = map_embeddings_post[t_]
                elif map_embeddings is not None:
                    _map_emb_post = map_embeddings[t_]  
                else:
                    _map_emb_post = torch.zeros(B, self.map_embedding_size, device=_state.device)
                hidden = self.act_fn(
                    self.fc_embed_belief_posterior(
                        torch.cat([beliefs[t + 1], observations[t_], _map_emb_post], dim=1)
                    )
                )
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                
                posterior_std_devs[t + 1] = (F.softplus(_posterior_std_dev) + self.min_std_dev).clamp(max=5.0)
                
                if deterministic:
                    posterior_states[t + 1] = posterior_means[t + 1]
                else:
                    posterior_states[t + 1] = (
                        posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
                    )
            else:
                posterior_states[t + 1] = prior_states[t + 1]
                posterior_means[t + 1] = prior_means[t + 1]
                posterior_std_devs[t + 1] = prior_std_devs[t + 1]

        return (
            torch.stack(beliefs[1:], dim=0),
            torch.stack(prior_states[1:], dim=0),
            torch.stack(prior_means[1:], dim=0),
            torch.stack(prior_std_devs[1:], dim=0),
            torch.stack(posterior_states[1:], dim=0),
            torch.stack(posterior_means[1:], dim=0),
            torch.stack(posterior_std_devs[1:], dim=0),
        )


class SymbolicObservationModel(nn.Module):
    def __init__(self, observation_size, belief_size, state_size, embedding_size,
                 activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, observation_size)
        self.component_modules = [self.fc1, self.fc2, self.fc3]
    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        return self.fc3(hidden)


class VisualObservationModel(nn.Module):
    """Visual Observation Model component."""

    def __init__(self, belief_size, state_size, embedding_size,
                 map_embedding_size=256, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        input_size = belief_size + state_size + map_embedding_size
        self.fc_in = nn.Linear(input_size, 128 * 8 * 8)
        self.d_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   
        self.d_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    
        self.d_conv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        self.component_modules = [self.fc_in, self.d_conv1, self.d_conv2, self.d_conv3]

    def forward(self, belief, state, map_embedding=None):
        """Run the module forward pass."""
        if map_embedding is not None:
            x = torch.cat([belief, state, map_embedding], dim=1)
        else:
            B = belief.size(0)
            pad_size = self.fc_in.in_features - belief.size(1) - state.size(1)
            x = torch.cat([belief, state, torch.zeros(B, pad_size, device=belief.device)], dim=1)

        hidden = self.act_fn(self.fc_in(x))
        hidden = hidden.view(-1, 128, 8, 8)
        hidden = self.act_fn(self.d_conv1(hidden))
        hidden = self.act_fn(self.d_conv2(hidden))
        observation = 0.5 * torch.tanh(self.d_conv3(hidden))
        return observation


def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size,
                     map_embedding_size=256, activation_function='relu'):
    if symbolic:
        return SymbolicObservationModel(observation_size, belief_size, state_size,
                                        embedding_size, activation_function)
    else:
        return VisualObservationModel(belief_size, state_size, embedding_size,
                                      map_embedding_size, activation_function)



class RewardModel(nn.Module):
    """Reward Model component."""
    def __init__(self, belief_size, state_size, hidden_size,
                 map_embedding_size=256, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        input_size = belief_size + state_size + map_embedding_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.component_modules = [self.fc1, self.fc2, self.fc3]
        self._map_emb_size = map_embedding_size

    def forward(self, belief, state, map_embedding=None):
        if map_embedding is not None:
            x = torch.cat([belief, state, map_embedding], dim=1)
        else:
            B = belief.size(0)
            x = torch.cat([belief, state, torch.zeros(B, self._map_emb_size, device=belief.device)], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        return self.fc3(hidden).squeeze(dim=1)



class QNetwork(nn.Module):
    """Q Network component."""

    def __init__(self, belief_size, state_size, hidden_size, action_size,
                 map_embedding_size=256, activation_function='elu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        input_size = belief_size + state_size + map_embedding_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.adv_head = nn.Linear(hidden_size, action_size)
        self.component_modules = [
            self.fc1, self.fc2, self.fc3, self.value_head, self.adv_head,
        ]
        self._action_size = action_size
        self._map_emb_size = map_embedding_size

    def forward(self, belief, state, map_embedding=None):
        if map_embedding is not None:
            x = torch.cat([belief, state, map_embedding], dim=1)
        else:
            B = belief.size(0)
            x = torch.cat(
                [belief, state,
                 torch.zeros(B, self._map_emb_size, device=belief.device)],
                dim=1,
            )
        h = self.act_fn(self.fc1(x))
        h = self.act_fn(self.fc2(h))
        h = self.act_fn(self.fc3(h))
        V = self.value_head(h)                              
        A = self.adv_head(h)                                
        Q = V + (A - A.mean(dim=-1, keepdim=True))          
        return Q


class QPolicy:
    """Q Policy component."""
    def __init__(self, q_net, action_size, default_epsilon=0.0):
        self.q_net = q_net
        self.action_size = action_size
        self.default_epsilon = default_epsilon

    def set_epsilon(self, eps):
        """Set epsilon."""
        self.default_epsilon = float(eps)

    def get_action(self, belief, state, map_embedding=None, det=False, epsilon=None):
        """Get action."""
        with torch.no_grad():
            q = self.q_net(belief, state, map_embedding)    
            B = q.size(0)
            greedy = q.argmax(dim=-1)
            eps = 0.0 if det else (epsilon if epsilon is not None else self.default_epsilon)
            if eps <= 0.0:
                idx = greedy
            else:
                rand = torch.randint(0, self.action_size, (B,), device=q.device)
                mask = torch.rand(B, device=q.device) < eps
                idx = torch.where(mask, rand, greedy)
            return F.one_hot(idx, num_classes=self.action_size).float()


def sync_target(target_net: nn.Module, online_net: nn.Module, tau: float = 1.0):
    """Sync target."""
    if tau >= 1.0:
        target_net.load_state_dict(online_net.state_dict())
    else:
        with torch.no_grad():
            for p_tgt, p in zip(target_net.parameters(), online_net.parameters()):
                p_tgt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


class SymbolicEncoder(nn.Module):
    def __init__(self, observation_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(observation_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)
        self.component_modules = [self.fc1, self.fc2, self.fc3]
    def forward(self, observation):
        hidden = self.act_fn(self.fc1(observation))
        hidden = self.act_fn(self.fc2(hidden))
        return self.fc3(hidden)


class VisualEncoder(nn.Module):

    def __init__(self, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
        self.component_modules = [self.conv1, self.conv2, self.conv3, self.conv4]
    def forward(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        return self.fc(hidden)


def Encoder(symbolic, observation_size, embedding_size, activation_function='relu'):
    if symbolic:
        return SymbolicEncoder(observation_size, embedding_size, activation_function)
    else:
        return VisualEncoder(embedding_size, activation_function)


def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True
        self.domain = torch.distributions.constraints.real
        self.codomain = torch.distributions.constraints.interval(-1.0, 1.0)

    @property
    def sign(self):
        return 1.0

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where((torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y)
        return atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (np.log(2) - x - F.softplus(-2.0 * x))


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        return torch.mean(self._dist.rsample(), 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()