"""
memory.py — Experience Replay Buffer

修改:
  - semantic_map 使用 6 通道 (论文公式 9: Mexp, Mrel, Mtrv, Mocc, Mu, Mv)
  - [L2 修复] 修正值域注释: 各通道值域不同, 非统一 [-1, 1]
"""

import numpy as np
import torch

from env import postprocess_observation, preprocess_observation_


class ExperienceReplay:
    def __init__(self, size, symbolic_env, observation_size, action_size, bit_depth, device):
        self.device = device
        self.symbolic_env = symbolic_env
        self.size = size

        if isinstance(observation_size, dict):
            # 从 observation_size (gym.spaces.Dict) 中提取语义地图形状
            if 'semantic_map' in observation_size:
                sem_map_shape = observation_size['semantic_map'].shape
            else:
                # [论文公式 9] 6 通道: Mexp, Mrel, Mtrv, Mocc, Mu, Mv
                sem_map_shape = (6, 30, 30)

            self.observations = {
                'image': np.empty((size, 3, 64, 64), dtype=np.uint8),
                'target': np.empty((size, 3, 64, 64), dtype=np.uint8),
                'position': np.empty((size, 2), dtype=np.float32),
                'semantic_map': np.empty((size, *sem_map_shape), dtype=np.float16),
                # 安全掩码: 8 方向安全度 ∈ [0,1], 融合边界/Mtrv/Mocc 信息
                'safety_mask': np.empty((size, 8), dtype=np.float32),
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
            # 安全掩码
            if 'safety_mask' in obs_np:
                self.observations['safety_mask'][self.idx] = obs_np['safety_mask']
            else:
                self.observations['safety_mask'][self.idx] = np.ones(8, dtype=np.float32)

        elif self.symbolic_env:
            self.observations[self.idx] = observation.numpy()
        else:
            self.observations[self.idx] = postprocess_observation(
                observation.numpy(), self.bit_depth
            )

        # 离散动作: 存储为 int64 索引
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

        # 离散动作: int64 → one-hot float32
        act_idx = torch.as_tensor(self.actions[vec_idxs], dtype=torch.int64)
        actions_onehot = torch.nn.functional.one_hot(
            act_idx, num_classes=self.action_size
        ).float().reshape(L, n, self.action_size)

        if isinstance(self.observations, dict):
            obs_batch = {}

            # Image & Target: uint8 → float32 + preprocess
            for key in ['image', 'target']:
                raw_data = self.observations[key][vec_idxs]
                tensor_data = torch.as_tensor(raw_data.astype(np.float32))
                preprocess_observation_(tensor_data, self.bit_depth)
                obs_batch[key] = tensor_data.reshape(L, n, *tensor_data.shape[1:])

            # Position
            pos_data = self.observations['position'][vec_idxs]
            obs_batch['position'] = torch.as_tensor(pos_data).reshape(L, n, -1)

            # safety_mask: 8 维安全掩码
            sm_data = self.observations['safety_mask'][vec_idxs]
            obs_batch['safety_mask'] = torch.as_tensor(sm_data).reshape(L, n, -1)

            # semantic_map: 存储为 float16, 取出时转 float32
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

        # [OOM 修复] 不在此处将整个 batch 搬上 GPU.
        # 返回 CPU tensors, 由训练循环按需搬运, 避免峰值显存过大.
        processed_batch = []
        for item in batch:
            if isinstance(item, dict):
                # 观测字典: 保持在 CPU, 仅转为 torch tensor
                processed_batch.append({k: v if torch.is_tensor(v) else torch.as_tensor(v)
                                        for k, v in item.items()})
            else:
                processed_batch.append(torch.as_tensor(item))

        return processed_batch