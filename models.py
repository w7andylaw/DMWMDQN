"""
models.py — 完整复现论文所有模块。

修改清单 (对照论文公式):
──────────────────────────────────────────────────────────────
[1]  MapEncoder           (公式 12)  m_t = E_m(M_t)
[2]  DifferentiableMapUpdater (公式 11) 可微门控地图更新
[3]  MapTransitionModel   (公式 19, 23)  T_ω: 想象中地图转移 + 语义预测头 D_M
[4]  ObstacleForecaster   (公式 24-25)  D_o: 占用场 + 运动流预测
[5]  ContinuationModel    (公式 22, 38) D_c: episode 继续概率
[6]  TransitionModel      (公式 14-18)  RSSM — m_{t-1} 注入 GRU 输入
[7]  RewardModel          (公式 22)  D_r(h, z, m)
[8]  ValueModel           (公式 30)  V_ψ(h, z, m)
[9]  ActorModel           (公式 29)  π_η(a | h, z, m)
[10] ObservationModel     (公式 21)  p_θ(I | h, z, m)
──────────────────────────────────────────────────────────────
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributions
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.nn import functional as F
import random


# ============================================================================
#  Utility
# ============================================================================

def bottle(f, x_tuple):
    """Flatten (T, B, ...) → (T*B, ...), apply f, reshape back."""
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    y_size = y.size()
    output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
    return output


def bottle_semantic(f, x_tuple, extra_tuple):
    """
    与 bottle 类似, 但分别处理标准张量和额外张量 (如 semantic map).
    x_tuple: 标准输入 (T, B, ...)
    extra_tuple: 额外输入 (T, B, C, H, W) 等
    """
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    T, B = x_sizes[0][0], x_sizes[0][1]

    flat_x = [x.view(T * B, *x.shape[2:]) for x in x_tuple]
    flat_extra = [e.view(T * B, *e.shape[2:]) for e in extra_tuple]

    y = f(*flat_x, *flat_extra)

    if isinstance(y, tuple):
        return tuple(yi.view(T, B, *yi.shape[1:]) for yi in y)
    return y.view(T, B, *y.shape[1:])


# ============================================================================
#  [公式 12] MapEncoder: E_m(M_t) → m_t
# ============================================================================

class MapEncoder(nn.Module):
    """
    [论文公式 12] 语义地图编码器:
        m_t = E_m(M_t)

    将 6×Ng×Ng 结构化语义地图压缩为紧凑向量 m_t ∈ R^{map_embedding_size},
    供 RSSM 转移和 policy 使用.
    """

    def __init__(self, n_channels=6, grid_size=30, map_embedding_size=256):
        super().__init__()
        self.map_embedding_size = map_embedding_size

        # 6×30×30 → 32×15×15 → 64×8×8 → 128×4×4
        self.conv1 = nn.Conv2d(n_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        # 128 * 4 * 4 = 2048
        self.fc = nn.Linear(128 * 4 * 4, map_embedding_size)

        self.component_modules = [self.conv1, self.conv2, self.conv3, self.fc]

    def forward(self, semantic_map):
        """
        Args:
            semantic_map: (B, 6, Ng, Ng) float32, 值域 [-1, 1]
        Returns:
            m: (B, map_embedding_size)
        """
        h = F.elu(self.conv1(semantic_map))
        h = F.elu(self.conv2(h))
        h = F.elu(self.conv3(h))
        h = h.view(h.size(0), -1)
        m = self.fc(h)
        return m


# ============================================================================
#  [公式 11] DifferentiableMapUpdater: 可微门控地图更新
# ============================================================================

class DifferentiableMapUpdater(nn.Module):
    """
    [论文公式 11] 可微地图更新:
        M_t = (1 - G_t) ⊙ M_{t-1} + G_t ⊙ M̄_t

    其中:
        M̄_t: 由语义证据和位姿投影生成的候选地图
        G_t ∈ [0,1]^{6×Ng×Ng}: 学习的写入门控

    在训练期间, 从 embedding e_t (encoder 输出) 生成局部证据,
    然后通过位姿投影写入全局地图. 不同通道有不同的更新速率和置信度.
    """

    def __init__(self, embedding_size=1024, n_channels=6, grid_size=30):
        super().__init__()
        self.n_channels = n_channels
        self.grid_size = grid_size

        # 从 embedding 预测局部证据 (6 通道)
        # [公式 11 修复] 移除统一 Tanh — 不同通道值域不同, 在 forward 中按通道激活:
        #   CH0-2 (Mexp/Mrel/Mtrv)  ∈ [-1, 1]  → tanh
        #   CH3   (Mocc)            ∈ [ 0, 1]  → sigmoid
        #   CH4-5 (Mu/Mv)           ∈ [-1, 1]  → tanh
        # 统一 tanh 会把 Mocc 逼到 [-1,1], 与 GT 的 Gaussian 扩散 [0,1] 系统性失配,
        # 导致 L_upd 无法收敛到零, 并通过公式 (11) 的 gate 污染后续 M_t.
        self.evidence_head = nn.Sequential(
            nn.Linear(embedding_size + 2, 512),   # +2 for position
            nn.ELU(),
            nn.Linear(512, n_channels * grid_size * grid_size),
        )

        # 从 embedding 预测写入门 (6 通道)
        self.gate_head = nn.Sequential(
            nn.Linear(embedding_size + 2, 512),
            nn.ELU(),
            nn.Linear(512, n_channels * grid_size * grid_size),
            nn.Sigmoid()  # 输出 [0, 1] 范围
        )

        self.component_modules = [self.evidence_head, self.gate_head]

    def forward(self, prev_map, embedding, position):
        """
        Args:
            prev_map:  (B, 6, Ng, Ng) — 上一步地图 M_{t-1}
            embedding: (B, embedding_size) — encoder 输出 e_t
            position:  (B, 2) — 当前位置 p_t
        Returns:
            new_map: (B, 6, Ng, Ng) — 更新后的地图 M_t
            gate: (B, 6, Ng, Ng) — 写入门 G_t (用于可视化/分析)
        """
        B = embedding.size(0)

        # 拼接 embedding 和 position
        cond = torch.cat([embedding, position], dim=1)

        # 生成候选地图和门控
        # [公式 11 修复] 按通道激活: 与 MapTransitionModel (L244-248) 保持一致
        raw_candidate = self.evidence_head(cond).view(
            B, self.n_channels, self.grid_size, self.grid_size
        )
        candidate = torch.cat([
            torch.tanh(raw_candidate[:, :3]),       # CH0-2: Mexp, Mrel, Mtrv → [-1, 1]
            torch.sigmoid(raw_candidate[:, 3:4]),   # CH3:   Mocc            → [ 0, 1]
            torch.tanh(raw_candidate[:, 4:]),       # CH4-5: Mu, Mv          → [-1, 1]
        ], dim=1)
        gate = self.gate_head(cond).view(B, self.n_channels, self.grid_size, self.grid_size)

        # 公式 11: 门控更新
        new_map = (1 - gate) * prev_map + gate * candidate

        return new_map, gate


# ============================================================================
#  [公式 19, 23] MapTransitionModel: T_ω + 语义预测头 D_M
# ============================================================================

class MapTransitionModel(nn.Module):
    """
    [论文公式 19] 想象中的语义地图转移:
        M̃_{t+1} = T_ω(M̃_t, h̃_{t+1}, z̃_{t+1}, a_t)

    [论文公式 23] 语义预测头:
        M̂_{t+1} = D_M(h_t, z_t, a_t, m_t)

    实现方式: 将隐状态条件注入到一个轻量级 CNN, 在地图空间进行转移.
    """

    def __init__(self, belief_size=200, state_size=30, action_size=8,
                 map_embedding_size=256, n_channels=6, grid_size=30):
        super().__init__()
        self.n_channels = n_channels
        self.grid_size = grid_size

        # 条件向量 → 空间特征 (用于 FiLM 条件化)
        cond_size = belief_size + state_size + action_size + map_embedding_size
        self.cond_fc = nn.Sequential(
            nn.Linear(cond_size, 512),
            nn.ELU(),
        )
        # FiLM: scale & bias for each conv layer
        self.film_gamma1 = nn.Linear(512, 64)
        self.film_beta1 = nn.Linear(512, 64)
        self.film_gamma2 = nn.Linear(512, 64)
        self.film_beta2 = nn.Linear(512, 64)

        # 地图转移卷积 (保持空间大小)
        self.conv1 = nn.Conv2d(n_channels, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(64, n_channels, 3, stride=1, padding=1)

        self.component_modules = [
            self.cond_fc, self.film_gamma1, self.film_beta1,
            self.film_gamma2, self.film_beta2,
            self.conv1, self.conv2, self.conv_out
        ]

    def forward(self, prev_map, belief, state, action, map_embedding):
        """
        Args:
            prev_map:      (B, 6, Ng, Ng)
            belief:        (B, belief_size)
            state:         (B, state_size)
            action:        (B, action_size)
            map_embedding: (B, map_embedding_size) — m_t
        Returns:
            next_map: (B, 6, Ng, Ng) — 预测的下一步地图
        """
        B = prev_map.size(0)

        # 条件向量
        cond = self.cond_fc(torch.cat([belief, state, action, map_embedding], dim=1))

        # FiLM 条件化的 CNN
        h = self.conv1(prev_map)
        gamma1 = self.film_gamma1(cond).unsqueeze(-1).unsqueeze(-1)  # (B, 64, 1, 1)
        beta1 = self.film_beta1(cond).unsqueeze(-1).unsqueeze(-1)
        h = F.elu(gamma1 * h + beta1)

        h = self.conv2(h)
        gamma2 = self.film_gamma2(cond).unsqueeze(-1).unsqueeze(-1)
        beta2 = self.film_beta2(cond).unsqueeze(-1).unsqueeze(-1)
        h = F.elu(gamma2 * h + beta2)

        # 残差连接 + 逐通道激活 (非 inplace, 保持 autograd 安全)
        # [T-1 修复] Mocc (CH3) 是占用概率 ∈ [0,1], 用 sigmoid;
        # 其余通道 (Mexp, Mrel, Mtrv, Mu, Mv) ∈ [-1,1], 用 tanh.
        raw = self.conv_out(h) + prev_map
        next_map = torch.cat([
            torch.tanh(raw[:, :3]),           # CH0-2: Mexp, Mrel, Mtrv → [-1, 1]
            torch.sigmoid(raw[:, 3:4]),       # CH3:   Mocc → [0, 1]
            torch.tanh(raw[:, 4:]),           # CH4-5: Mu, Mv → [-1, 1]
        ], dim=1)

        return next_map


# ============================================================================
#  [公式 24-25] ObstacleForecaster: D_o
# ============================================================================

class ObstacleForecaster(nn.Module):
    """
    [论文公式 24] 动态障碍物预测:
        {Ô_{t+k}, Û_{t+k}}_{k=1}^{K} = D_o(h_t, z_t, m_t)

    从隐状态解码未来 K 步的:
        - 占用概率场 Ô ∈ [0,1]^{Ng×Ng}
        - 运动流场 Û ∈ R^{2×Ng×Ng}

    [论文公式 25] 轨迹提取 (后处理, 不在此模块中):
        Ξ̂ = Γ(Ô, Û)
    """

    def __init__(self, belief_size=200, state_size=30, map_embedding_size=256,
                 forecast_horizon=5, grid_size=30):
        super().__init__()
        self.K = forecast_horizon
        self.grid_size = grid_size

        input_size = belief_size + state_size + map_embedding_size

        # 将隐状态映射到空间特征
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ELU(),
            nn.Linear(1024, 128 * 4 * 4),
            nn.ELU(),
        )

        # 上采样到 Ng×Ng 并预测 K 步
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 4→8
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)   # 8→16
        # 自适应到 30×30
        self.conv_adapt = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        # 占用概率: K 个通道
        self.occ_head = nn.Conv2d(32, self.K, 3, stride=1, padding=1)

        # 运动流: K * 2 个通道 (u, v for each step)
        self.flow_head = nn.Conv2d(32, self.K * 2, 3, stride=1, padding=1)

        self.component_modules = [
            self.fc, self.deconv1, self.deconv2, self.conv_adapt,
            self.occ_head, self.flow_head
        ]

    def forward(self, belief, state, map_embedding):
        """
        Args:
            belief:        (B, belief_size)
            state:         (B, state_size)
            map_embedding: (B, map_embedding_size)
        Returns:
            occ_pred:  (B, K, Ng, Ng) — 未来 K 步占用概率
            flow_pred: (B, K, 2, Ng, Ng) — 未来 K 步运动流
        """
        B = belief.size(0)

        h = self.fc(torch.cat([belief, state, map_embedding], dim=1))
        h = h.view(B, 128, 4, 4)
        h = F.elu(self.deconv1(h))    # (B, 64, 8, 8)
        h = F.elu(self.deconv2(h))    # (B, 32, 16, 16)

        # 插值到 Ng×Ng
        h = F.interpolate(h, size=(self.grid_size, self.grid_size), mode='bilinear', align_corners=False)
        h = F.elu(self.conv_adapt(h))  # (B, 32, Ng, Ng)

        # 占用概率 [0, 1]
        occ = torch.sigmoid(self.occ_head(h))  # (B, K, Ng, Ng)

        # 运动流 [-1, 1]
        flow_raw = torch.tanh(self.flow_head(h))  # (B, K*2, Ng, Ng)
        flow = flow_raw.view(B, self.K, 2, self.grid_size, self.grid_size)

        return occ, flow


# ============================================================================
#  [工程化扩展 — 非论文项] BoundaryForecaster: D_b
# ----------------------------------------------------------------------------
#  动机:
#      论文 (26) 的 -λ_out·I[p_{t+1} ∉ A] 是反应式惩罚 —— UAV 真的飞出
#      边界后才扣分. 在 latent imagination 里, actor 没有任何梯度信号
#      能提前告知"朝角落/边缘方向前进有风险", 导致靠近边界的想象轨迹
#      只能依赖 collision/boundary 的稀疏终止 reward 学习.
#
#      本模块与 ObstacleForecaster (公式 24) 架构对称, 但有两处关键简化:
#        (a) 输出单通道静态图 B̂ ∈ [0,1]^{Ng×Ng} —— 边界不随时间变化,
#            无 K 步展开也无 flow.
#        (b) 在 risk 注入时对同一张 B̂ 沿想象轨迹索引 K 个位置:
#                χ_bdry_t = max_{1≤k≤K} B̂(ĩ_{t+k})
#            与 obstacle χ_t 完全同构.
#
#  该模块不是论文公式, 属于工程扩展 —— 等价于给 actor 一个可微、
#  前瞻性的"靠近边界风险"正则项.
# ============================================================================

class BoundaryForecaster(nn.Module):
    """
    [工程化扩展] 静态边界占用预测:
        B̂ = D_b(h_t, z_t, m_t),  B̂ ∈ [0, 1]^{Ng×Ng}

    与 ObstacleForecaster 的三处差异:
        * 输出单通道 (不是 K 通道) —— 边界是时不变的静态风险场.
        * 无 flow 头 —— 边界不流动.
        * GT 由环境几何确定 (距边界距离的软衰减), 调用方提供常量张量.

    风险注入 (由 main 训练循环负责):
        χ_bdry_t = max_{1≤k≤K} B̂(ĩ_{t+k})
        imged_reward -= λ_risk_bdry · χ_bdry
    """

    def __init__(self, belief_size=200, state_size=30, map_embedding_size=256,
                 grid_size=30):
        super().__init__()
        self.grid_size = grid_size

        input_size = belief_size + state_size + map_embedding_size

        # 隐状态 → 空间特征 (与 ObstacleForecaster 同一 trunk 拓扑)
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ELU(),
            nn.Linear(1024, 128 * 4 * 4),
            nn.ELU(),
        )
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 4→8
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)   # 8→16
        self.conv_adapt = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        # 单通道头: B̂ ∈ [0,1]^{Ng×Ng}
        self.bdry_head = nn.Conv2d(32, 1, 3, stride=1, padding=1)

        self.component_modules = [
            self.fc, self.deconv1, self.deconv2, self.conv_adapt, self.bdry_head,
        ]

    def forward(self, belief, state, map_embedding):
        """
        Args:
            belief:        (B, belief_size)
            state:         (B, state_size)
            map_embedding: (B, map_embedding_size)
        Returns:
            bdry_pred: (B, Ng, Ng) —— 静态边界风险图 ∈ [0, 1]
        """
        B = belief.size(0)
        h = self.fc(torch.cat([belief, state, map_embedding], dim=1))
        h = h.view(B, 128, 4, 4)
        h = F.elu(self.deconv1(h))                              # (B, 64, 8, 8)
        h = F.elu(self.deconv2(h))                              # (B, 32, 16, 16)
        h = F.interpolate(h, size=(self.grid_size, self.grid_size),
                          mode='bilinear', align_corners=False)
        h = F.elu(self.conv_adapt(h))                           # (B, 32, Ng, Ng)
        bdry = torch.sigmoid(self.bdry_head(h)).squeeze(1)      # (B, Ng, Ng)
        return bdry


# ============================================================================
#  [公式 22, 38] ContinuationModel: D_c
# ============================================================================

class ContinuationModel(nn.Module):
    """
    [论文公式 22] Episode 继续概率预测:
        ĉ_t = D_c(h_t, z_t, m_t)

    用于 λ-return 的 discount 加权 (公式 31-32).
    """

    def __init__(self, belief_size=200, state_size=30, map_embedding_size=256, hidden_size=200):
        super().__init__()
        self._map_emb_size = map_embedding_size
        self.fc1 = nn.Linear(belief_size + state_size + map_embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.component_modules = [self.fc1, self.fc2, self.fc3]

    def forward(self, belief, state, map_embedding=None):
        """
        Returns:
            cont_prob: (B,) — 继续概率 ∈ (0, 1)
        """
        if map_embedding is None:
            map_embedding = torch.zeros(belief.size(0), self._map_emb_size, device=belief.device)
        x = torch.cat([belief, state, map_embedding], dim=1)
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        return torch.sigmoid(self.fc3(h)).squeeze(dim=1)


# ============================================================================
#  UAVHybridEncoder (公式 10 的一部分: E_φ)
# ============================================================================

class UAVHybridEncoder(nn.Module):
    """
    [论文公式 10] 目标条件语义编码器 (扩展版):
        e_t = E_φ(I_t, I^g, p_t, σ_t)

    σ_t ∈ [0,1]^8 为局部安全上下文, 从语义地图的 M^occ/M^trv 通道派生,
    编码 8 个运动方向的即时安全性. 进入表征学习使世界模型能预测未来安全状态.
    """

    def __init__(self, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size

        # Siamese visual stream
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        # Position stream
        self.fc_vec = nn.Linear(2, 64)

        # [shield 改造] fc_safety 已移除 — 边界由 hard shield 物理保证, 不需软警告

        # Fusion: curr(1024) + targ(1024) + pos(64)
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
        # 注: safety_mask 参数保留是为了向后兼容旧 checkpoint 加载, 但已被忽略 (shield 改造).
        curr_emb = self.forward_visual(image)
        targ_emb = self.forward_visual(target_image)
        vec_emb = self.act_fn(self.fc_vec(vector))
        combined = torch.cat([curr_emb, targ_emb, vec_emb], dim=1)
        out = self.fc_out(combined)
        return out, curr_emb, targ_emb


# ============================================================================
#  SemanticFeatureExtractor (gθ)
# ============================================================================

class SemanticFeatureExtractor(nn.Module):
    """
    独立语义特征提取器 gθ, 接收完整观测 o_t = {I_t, I^g, p_t, M_t}.

    [论文 Section IV-B] gθ 独立于 encoder E_φ, 额外接收语义地图 M_t.
    输出供 semantic_rnn (TransitionModel 的 System 2) 使用.

    [修改] 输入 M_t 是 6 通道 (公式 9: Mexp, Mrel, Mtrv, Mocc, Mu, Mv).
    """

    def __init__(self, semantic_size=512, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.semantic_size = semantic_size

        # 语义地图流: 6×30×30 → 512
        self.map_conv1 = nn.Conv2d(6, 32, 3, stride=2, padding=1)
        self.map_conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.map_conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.map_fc = nn.Linear(64 * 4 * 4, 512)

        # 轻量级视觉流 (Siamese 共享权重)
        self.conv1 = nn.Conv2d(3, 16, 4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 4, stride=2)

        # 位置流
        self.fc_pos = nn.Linear(2, 64)

        # 融合: 512 + 512 + 64 + 512 = 1600 → semantic_size
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
        """
        Args:
            image:        (B, 3, 64, 64) — 当前观测 I_t
            target_image: (B, 3, 64, 64) — 目标图像 I^g
            position:     (B, 2) — 当前位置 p_t
            semantic_map: (B, 6, Ng, Ng) — 语义地图 M_t
        Returns:
            semantic_features: (B, semantic_size)
        """
        curr_vis = self.forward_visual(image)
        targ_vis = self.forward_visual(target_image)
        pos_emb = self.act_fn(self.fc_pos(position))
        map_emb = self.forward_map(semantic_map)
        combined = torch.cat([curr_vis, targ_vis, pos_emb, map_emb], dim=1)
        return self.act_fn(self.fc_fuse(combined))


# ============================================================================
#  [公式 14-20] TransitionModel (Semantic RSSM)
# ============================================================================

class TransitionModel(nn.Module):
    """
    Semantic RSSM (论文 Section IV-C).

    [公式 14] h_t = f_θ(h_{t-1}, z_{t-1}, a_{t-1}, m_{t-1})
    [公式 15] p_θ(z_t | h_t) — prior
    [公式 16] q_θ(z_t | h_t, e_t, m_t) — posterior

    [M1 修复] 移除论文未描述的 semantic_rnn / semantic_state (System 2).
    论文公式(13): s_t = {h_t, z_t, M_t, m_t}, 不含独立的 semantic_state.
    语义地图的影响通过 m_t (MapEncoder 输出) 注入 GRU 和 posterior.
    """

    def __init__(
            self,
            belief_size,
            state_size,
            action_size,
            hidden_size,
            embedding_size,
            semantic_size=512,          # 保留参数签名兼容性, 但不再使用
            semantic_state_size=512,    # 保留参数签名兼容性, 但不再使用
            map_embedding_size=256,
            activation_function='relu',
            min_std_dev=0.1,
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.map_embedding_size = map_embedding_size

        # --- RSSM ---
        # [公式 14] 输入: state + action + map_embedding (m_{t-1})
        self.fc_embed_state_action = nn.Linear(
            state_size + action_size + map_embedding_size, belief_size
        )
        self.rnn = nn.GRUCell(belief_size, belief_size)

        # [公式 15] Prior: p(z_t | h_t)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)

        # [公式 16] Posterior: q(z_t | h_t, e_t, m_t)
        self.fc_embed_belief_posterior = nn.Linear(
            belief_size + embedding_size + map_embedding_size, hidden_size
        )
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)

        # [M1 修复] 移除 System 2 (semantic_rnn, fc_embed_semantic)
        # 语义地图影响完全通过 m_t 进入 GRU 和 posterior

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
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            prev_state:     (B, state_size)
            actions:        (T, B, action_size)
            prev_belief:    (B, belief_size)
            observations:   (T, B, embedding_size) — encoder 输出 e_{t+1}
            nonterminals:   (T, B, 1)
            map_embeddings: (T, B, map_emb_size) — m_t, 用于 GRU 输入 (公式14)
            map_embeddings_post: (T, B, map_emb_size) — m_{t+1}, 用于 posterior (公式16)
                [P-1 修复] GRU 和 posterior 的地图嵌入时序不同:
                  公式14: h_{t+1} = f(h_t, z_t, a_t, m_t)     → 需要 m_t
                  公式16: q(z_{t+1}|h_{t+1}, e_{t+1}, m_{t+1}) → 需要 m_{t+1}
                若 map_embeddings_post 为 None, 退化为旧行为 (posterior 也用 m_t).
        Returns:
            beliefs, prior_states, prior_means, prior_std_devs,
            posterior_states, posterior_means, posterior_std_devs — 各 (T, B, *)
        """
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

            # [公式 14] h_{t+1} = f(h_t, z_t, a_t, m_t) — GRU 用 m_t
            t_ = t
            if map_embeddings is not None:
                map_emb = map_embeddings[t_]
            else:
                map_emb = torch.zeros(B, self.map_embedding_size, device=_state.device)

            hidden = self.act_fn(
                self.fc_embed_state_action(torch.cat([_state, actions[t], map_emb], dim=1))
            )
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])

            # [公式 15] Prior: p(z_{t+1} | h_{t+1})
            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            # [防线 3] std 上界 clamp — 防止 softplus 无上界导致想象中 state 采样爆炸
            # 下界 min_std_dev (通常 0.1) 已保证数值稳定, 上界 5.0 保留足够不确定性表达
            prior_std_devs[t + 1] = (F.softplus(_prior_std_dev) + self.min_std_dev).clamp(max=5.0)
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])

            # [公式 16] Posterior: q(z_{t+1} | h_{t+1}, e_{t+1}, m_{t+1})
            if observations is not None:
                # [P-1 修复] posterior 用 m_{t+1} (map_embeddings_post), 非 m_t
                if map_embeddings_post is not None:
                    _map_emb_post = map_embeddings_post[t_]
                elif map_embeddings is not None:
                    _map_emb_post = map_embeddings[t_]  # 回退: 用 m_t
                else:
                    _map_emb_post = torch.zeros(B, self.map_embedding_size, device=_state.device)
                hidden = self.act_fn(
                    self.fc_embed_belief_posterior(
                        torch.cat([beliefs[t + 1], observations[t_], _map_emb_post], dim=1)
                    )
                )
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                # [防线 3] posterior std 同样加上界, 对称处理
                posterior_std_devs[t + 1] = (F.softplus(_posterior_std_dev) + self.min_std_dev).clamp(max=5.0)
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


# ============================================================================
#  Observation Models (公式 21)
# ============================================================================

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
    """
    [论文公式 21] 观测解码器:
        Î_t ~ p_θ(I_t | h_t, z_t, m_t)

    [修改] 加入 map_embedding_size, 拼接 belief + state + map_embedding.
    """

    def __init__(self, belief_size, state_size, embedding_size,
                 map_embedding_size=256, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        input_size = belief_size + state_size + map_embedding_size

        self.fc_in = nn.Linear(input_size, 128 * 8 * 8)
        self.d_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 8→16
        self.d_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 16→32
        self.d_conv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)     # 32→64

        self.component_modules = [self.fc_in, self.d_conv1, self.d_conv2, self.d_conv3]

    def forward(self, belief, state, map_embedding=None):
        """
        Args:
            belief: (B, belief_size)
            state:  (B, state_size)
            map_embedding: (B, map_embedding_size) — 可选
        """
        if map_embedding is not None:
            x = torch.cat([belief, state, map_embedding], dim=1)
        else:
            # 兼容性: 无 map_embedding 时补零
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


# ============================================================================
#  [公式 22] RewardModel: D_r(h, z, m)
# ============================================================================

class RewardModel(nn.Module):
    """
    [论文公式 22] 奖励预测:
        r̂_t = D_r(h_t, z_t, m_t)

    [修改] 加入 map_embedding_size.
    """

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


# ============================================================================
#  [D3QN 替换] QNetwork: Q_θ(h, z, m, ·) — Dueling Double DQN
# ----------------------------------------------------------------------------
#  原 ValueModel (V_ψ, 公式 30) + ActorModel (π_η, 公式 29) 已被下方 Q 网络
#  取代. 论文公式 31-34 的 λ-return + actor/value 更新对应替换为:
#      y_t = r̂_t + γ · ĉ_t · Q_target(s_{t+1}, argmax_a Q_online(s_{t+1}, a))
#      L_Q = Huber(Q_online(s_t, a_t), y_t)
#  其余 WM 组件 (TransitionModel / ObservationModel / RewardModel /
#  ContinuationModel / MapEncoder / MapTransitionModel / ObstacleForecaster /
#  DifferentiableMapUpdater / Encoder) 与论文公式 9-25, 35-43 完全保留.
# ============================================================================

class QNetwork(nn.Module):
    """
    Dueling Q-Network: Q(h, z, m, a) for a ∈ {0, ..., action_size-1}.

    输入接口与原 ValueModel 完全一致:
        forward(belief, state, map_embedding=None) → (B, action_size)

    Dueling 头:
        shared MLP → [V(s) ∈ R, A(s, ·) ∈ R^|A|]
        Q(s, a) = V(s) + (A(s, a) - mean_a A(s, ·))
    """

    def __init__(self, belief_size, state_size, hidden_size, action_size,
                 map_embedding_size=256, activation_function='elu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        input_size = belief_size + state_size + map_embedding_size

        # 共享 trunk — 与原 ValueModel 一致的 3 层 FC
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        # Dueling heads
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

        V = self.value_head(h)                              # (B, 1)
        A = self.adv_head(h)                                # (B, |A|)
        Q = V + (A - A.mean(dim=-1, keepdim=True))          # (B, |A|)
        return Q


class QPolicy:
    """
    包装 QNetwork 为原 ActorModel 兼容接口.

    API 对齐:
        get_action(belief, state, map_embedding=None, det=False) → one-hot (B, |A|)

    这样 imagine_ahead 和 update_belief_and_act 中的
        action = planner.get_action(...)
    调用语义完全不变, 实现零侵入替换.

    参数:
        q_net:            QNetwork 实例
        action_size:      动作数 (= 8)
        default_epsilon:  非确定性 (det=False) 时的默认 ε.
                          在 imagine_ahead 中作为想象探索 ε (恒定, 如 0.3).
                          在 env 采集中通过 epsilon= 参数覆盖 (按 episode 衰减).
    """

    def __init__(self, q_net, action_size, default_epsilon=0.0):
        self.q_net = q_net
        self.action_size = action_size
        self.default_epsilon = default_epsilon

    def set_epsilon(self, eps):
        """外部可动态设置 ε (例如按 episode 衰减)."""
        self.default_epsilon = float(eps)

    def get_action(self, belief, state, map_embedding=None, det=False, epsilon=None):
        """
        返回 one-hot 动作 (B, action_size), 与原 ActorModel.get_action 接口一致.
        不回传梯度 — DQN 不需要对动作选择微分.

        Args:
            belief, state, map_embedding:  与原 ActorModel 相同
            det:       True → 纯 argmax (测试模式)
            epsilon:   覆盖 default_epsilon (可选)
        """
        with torch.no_grad():
            q = self.q_net(belief, state, map_embedding)    # (B, |A|)
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
    """
    同步 target 网络.
        tau = 1.0  — hard copy (标准 DQN)
        tau < 1.0  — soft Polyak update (θ_tgt ← τ·θ + (1-τ)·θ_tgt)
    """
    if tau >= 1.0:
        target_net.load_state_dict(online_net.state_dict())
    else:
        with torch.no_grad():
            for p_tgt, p in zip(target_net.parameters(), online_net.parameters()):
                p_tgt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


# ============================================================================
#  Legacy Encoders (non-UAV)
# ============================================================================

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


# ============================================================================
#  Legacy Utility Classes
# ============================================================================

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