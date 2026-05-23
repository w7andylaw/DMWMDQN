# Author: Qiwei Wang
"""Experience replay storage for sequence-based world-model training."""

import numpy as np
import torch

from env import postprocess_observation, preprocess_observation_


class ExperienceReplay:
    def __init__(self, size, symbolic_env, observation_size, action_size, bit_depth, device):
        self.device = device
        self.symbolic_env = symbolic_env
        self.size = size

        if isinstance(observation_size, dict):
            
            if 'semantic_map' in observation_size:
                sem_map_shape = observation_size['semantic_map'].shape
            else:
                
                sem_map_shape = (6, 30, 30)

            self.observations = {
                'image': np.empty((size, 3, 64, 64), dtype=np.uint8),
                'target': np.empty((size, 3, 64, 64), dtype=np.uint8),
                'position': np.empty((size, 2), dtype=np.float32),
                'semantic_map': np.empty((size, *sem_map_shape), dtype=np.float16),
            }

        elif symbolic_env:
            self.observations = np.empty((size, observation_size), dtype=np.float32)
        else:
            self.observations = np.empty((size, 3, 64, 64), dtype=np.uint8)

        self.actions = np.empty((size,), dtype=np.int64)
        self.action_size = action_size
        self.rewards = np.empty((size,), dtype=np.float32)
        self.nonterminals = np.empty((size, 1), dtype=np.float32)
        self.idx = 0
        self.full = False
        self.steps, self.episodes = 0, 0
        self.bit_depth = bit_depth

    def append(self, observation, action, reward, done):
        if isinstance(self.observations, dict):
            obs_np = {
                k: v.cpu().detach().numpy().squeeze(0) if torch.is_tensor(v) else v
                for k, v in observation.items()
            }

            self.observations['image'][self.idx] = postprocess_observation(obs_np['image'], self.bit_depth)
            self.observations['target'][self.idx] = postprocess_observation(obs_np['target'], self.bit_depth)
            self.observations['position'][self.idx] = obs_np['position']
            self.observations['semantic_map'][self.idx] = obs_np['semantic_map']
            

        elif self.symbolic_env:
            self.observations[self.idx] = observation.numpy()
        else:
            self.observations[self.idx] = postprocess_observation(
                observation.numpy(), self.bit_depth
            )

        
        a_np = action.numpy() if torch.is_tensor(action) else np.asarray(action)
        a_np = np.asarray(a_np).flatten()
        if a_np.size == 1:
            self.actions[self.idx] = int(a_np[0])
        else:
            self.actions[self.idx] = int(np.argmax(a_np))

        self.rewards[self.idx] = reward
        self.nonterminals[self.idx] = not done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

    def _sample_idx(self, L):
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.size if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.size
            valid_idx = not self.idx in idxs[1:]
        return idxs

    def _retrieve_batch(self, idxs, n, L):
        vec_idxs = idxs.transpose().reshape(-1)

        
        act_idx = torch.as_tensor(self.actions[vec_idxs], dtype=torch.int64)
        actions_onehot = torch.nn.functional.one_hot(
            act_idx, num_classes=self.action_size
        ).float().reshape(L, n, self.action_size)

        if isinstance(self.observations, dict):
            obs_batch = {}

            
            for key in ['image', 'target']:
                raw_data = self.observations[key][vec_idxs]
                tensor_data = torch.as_tensor(raw_data.astype(np.float32))
                preprocess_observation_(tensor_data, self.bit_depth)
                obs_batch[key] = tensor_data.reshape(L, n, *tensor_data.shape[1:])

            
            pos_data = self.observations['position'][vec_idxs]
            obs_batch['position'] = torch.as_tensor(pos_data).reshape(L, n, -1)

            

            
            sem_map_data = self.observations['semantic_map'][vec_idxs]
            sem_map_tensor = torch.as_tensor(sem_map_data.astype(np.float32))
            obs_batch['semantic_map'] = sem_map_tensor.reshape(L, n, *sem_map_tensor.shape[1:])

            return (
                obs_batch,
                actions_onehot,
                self.rewards[vec_idxs].reshape(L, n),
                self.nonterminals[vec_idxs].reshape(L, n, 1),
            )

        else:
            observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
            if not self.symbolic_env:
                preprocess_observation_(observations, self.bit_depth)
            return (
                observations.reshape(L, n, *observations.shape[1:]),
                actions_onehot,
                self.rewards[vec_idxs].reshape(L, n),
                self.nonterminals[vec_idxs].reshape(L, n, 1),
            )

    def sample(self, n, L):
        batch = self._retrieve_batch(np.asarray([self._sample_idx(L) for _ in range(n)]), n, L)
        processed_batch = []
        for item in batch:
            if isinstance(item, dict):
                processed_batch.append({k: v if torch.is_tensor(v) else torch.as_tensor(v)
                                        for k, v in item.items()})
            else:
                processed_batch.append(torch.as_tensor(item))

        return processed_batch