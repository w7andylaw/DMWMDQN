import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F   # [L2 归一化需要]
import gymnasium as gym
from gymnasium import spaces
import mysql.connector
import pickle
import os
import time

# Constants
GYM_ENVS = ['Pendulum-v1', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2',
            'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2',
            'Walker2d-v2']
CONTROL_SUITE_ENVS = ['acrobot-swingup', 'cartpole-balance', 'cartpole-balance_sparse', 'cartpole-swingup',
                      'cartpole-swingup_sparse', 'ball_in_cup-catch', 'finger-spin', 'finger-turn_easy',
                      'finger-turn_hard', 'hopper-hop', 'hopper-stand', 'pendulum-swingup', 'quadruped-run',
                      'quadruped-walk', 'reacher-easy', 'reacher-hard', 'walker-run', 'walker-stand', 'walker-walk']
CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2, 'cheetah': 4, 'ball_in_cup': 6, 'walker': 2,
                                'humanoid': 2, 'fish': 2, 'acrobot': 4}


# ============================================================================
#  Unified Evaluation Protocol  (跨基线共用的测试任务列表)
#
#  所有基线 (WM / D3QN / SM-CERL / DreamerV3 / ...) 的评估循环都必须通过
#  get_eval_task_list() 获取测试任务, 严禁在各自的 main 文件里硬编码
#  `map_seed` 序列. 这样保证:
#    - 相同的 in-domain / OOD map_seed 集合
#    - 相同的 episode 数量
#    - 相同的顺序 (env 已全确定, 障碍物 / 起终点 / 图 全部可复现)
#
#  改评估协议时只改本函数, 所有基线自动对齐.
#
#  默认值:
#    in_domain_base = 42     → in-domain 种子 = [42, 43, ..., 42+N-1]
#    ood_base       = 9999   → OOD       种子 = [9999, 10000, ..., 9999+N-1]
#  两组不相交, 改动前请确保新值仍不相交 (否则 cache key train/ood
#  split tag 只够区分 split, 不够区分同 seed 不同 split 的场景下
#  的语义冲突 — 虽然代码不会出错, 但任务集会混淆).
# ============================================================================

def get_eval_task_list(n_episodes=10, in_domain_base=42, ood_base=9999):
    """
    返回按 split 分组的评估任务列表.

    参数:
        n_episodes:     每个 split 的 episode 数 (通常取 args.test_episodes).
        in_domain_base: in-domain 起始 map_seed.
        ood_base:       OOD 起始 map_seed.

    返回:
        dict[str, list[dict]] —
            {
              'in_domain': [{'map_seed': int, 'ood': False}, ...],  # 长度 n_episodes
              'ood':       [{'map_seed': int, 'ood': True},  ...],  # 长度 n_episodes
            }

    典型使用 (所有基线的评估循环都应是这个模板):
        from env import get_eval_task_list
        tasks = get_eval_task_list(n_episodes=args.test_episodes)
        for mode, mode_tasks in tasks.items():
            for task in mode_tasks:
                obs = env.reset(map_seed=task['map_seed'], ood=task['ood'])
                ...   # 跑 episode, 累计该 mode 的 SR / APLR / CR / MinClr
    """
    return {
        'in_domain': [{'map_seed': in_domain_base + k, 'ood': False}
                      for k in range(n_episodes)],
        'ood':       [{'map_seed': ood_base + k,       'ood': True}
                      for k in range(n_episodes)],
    }


# ============================================================================
#  Helper Functions
# ============================================================================

def preprocess_observation_(observation, bit_depth):
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)
    observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))


def postprocess_observation(observation, bit_depth):
    return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(
        np.uint8)


def _images_to_observation(images, bit_depth):
    if images is None: raise ValueError("Render returned None.")
    images = torch.tensor(cv2.resize(images, (64, 64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1),
                          dtype=torch.float32)
    preprocess_observation_(images, bit_depth)
    return images.unsqueeze(dim=0)


# ============================================================================
#  Siamese Network  (论文 Section III - SMM 语义认知子模块)
#
#  论文:
#    - 双分支 ResNet-18, 共享参数 φ, 每分支输出 128 维特征
#    - 特征距离  df = ||ft - fg||₂           (公式 4)
#    - 相似度    vs = max(0, 1 - df / df_max)
#    - 训练使用对比损失 Lc (公式 9), 需单独预训练
# ============================================================================

class SiameseNetwork(nn.Module):
    """论文中的 Siamese 网络: 双分支 ResNet-18, 共享参数, 128 维特征输出。"""

    def __init__(self, feature_dim=128):
        super(SiameseNetwork, self).__init__()
        try:
            import torchvision.models as models
            resnet = models.resnet18(weights=None)
        except ImportError:
            raise ImportError("需要安装 torchvision: pip install torchvision")
        # 移除最后的 FC 层, 保留到 AdaptiveAvgPool → 输出 512 维
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, feature_dim)  # 512 → 128 (论文: 128 outputs)

    def forward_one(self, x):
        """单分支: image (B,3,H,W) → 128 维特征 (已 L2 归一化)"""
        x = self.backbone(x)           # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)     # (B, 512)
        x = self.fc(x)                 # (B, 128)
        # [关键] L2 归一化, 必须和训练时 (train_siamese_v2.py normalize=True) 一致.
        # 否则推理端 df = ||f_t - f_g||_2 的分布和训练时不同, 相似度全错.
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2):
        """双分支 (共享参数)"""
        return self.forward_one(x1), self.forward_one(x2)


# ============================================================================
#  UAV Navigation Environment (Redesigned)
# ============================================================================

class DynamicObstacle:
    """
    动态障碍物。

    论文 Section III-A, 公式 (2)(3):
        状态 o^j_t = [q^j_t, v^j_t, ρ^j]
        运动: o^j_{t+1} ~ p(o^j_{t+1} | o^j_t, E)

    运动模式 (motion_type):
        'cv'       — 匀速直线 (constant-velocity), 碰壁后反射
        'random'   — 随机游走 (每步随机扰动速度方向)
        'road'     — 地图感知: 沿亮度较高区域 (道路/人行道) 行驶,
                     实现论文公式(3)的 p(o^j_{t+1} | o^j_t, E)
    """

    def __init__(self, q, v, rho, motion_type='cv', D=3000, cell_size=100.0,
                 satellite_map=None, rng=None):
        self.q = np.array(q, dtype=np.float32)
        self.v = np.array(v, dtype=np.float32)
        self.rho = float(rho)
        self.motion_type = motion_type
        self.D = D
        self.cell_size = cell_size
        self.satellite_map = satellite_map
        # 步计数器: 用于确定性运动模式的相位计算
        self.step_count = 0
        # [测试可复现性修复] Per-obstacle RNG.
        # - None (训练): 使用全局 np.random, 每步随机扰动 (仍随机).
        # - RandomState (测试): 同一 map_seed 下, 'random' 运动模式的每步
        #   方向漂移也完全可复现, 保证 test 指标曲线稳定.
        self.rng = rng

    def step(self, rng=None):
        """
        前进一步, 更新 q 和 v。

        [关键设计] 所有运动模式在 episode 内都是确定性的:
        给定初始状态 (q_0, v_0) 和地图 E, 轨迹完全可预测.
        这样世界模型才能从观测序列中学到障碍物动力学.
        随机性只存在于 episode 间的初始化 (_spawn_obstacles).
        """
        self.step_count += 1

        if self.motion_type == 'cv':
            # 匀速直线, 碰壁弹射 — 天然确定性
            self.q = self.q + self.v
            for i in range(2):
                if self.q[i] < self.rho:
                    self.q[i] = self.rho
                    self.v[i] = abs(self.v[i])
                elif self.q[i] > self.D - self.rho:
                    self.q[i] = self.D - self.rho
                    self.v[i] = -abs(self.v[i])

        elif self.motion_type == 'random':
            # [L1 修复] 论文公式(3): o^j_{t+1} ~ p(o^j_{t+1} | o^j_t, E)
            # 随机游走: 每步对速度方向施加随机扰动 ±30°
            # (旧版为确定性正弦, 与公式(3)的随机性描述不符)
            speed = np.linalg.norm(self.v)
            if speed < 1e-6:
                speed = self.cell_size * 0.5
            base_angle = np.arctan2(self.v[1], self.v[0])
            # 随机方向扰动: Uniform(-π/6, π/6)
            # [测试可复现性修复] 优先使用 self.rng, 否则退化到全局 np.random.
            _uniform = self.rng.uniform if self.rng is not None else np.random.uniform
            drift = _uniform(-np.pi / 6, np.pi / 6)
            new_angle = base_angle + drift
            self.v = np.array([np.cos(new_angle) * speed,
                               np.sin(new_angle) * speed], dtype=np.float32)
            self.q = self.q + self.v
            # 碰壁弹射 (与 cv 一致)
            for i in range(2):
                if self.q[i] < self.rho:
                    self.q[i] = self.rho
                    self.v[i] = abs(self.v[i])
                elif self.q[i] > self.D - self.rho:
                    self.q[i] = self.D - self.rho
                    self.v[i] = -abs(self.v[i])

        elif self.motion_type == 'road':
            # [论文公式(3)] 确定性地图感知运动:
            # 评估 3 个候选方向的地图亮度, 选最亮的 (无随机扰动).
            # 给定 (q_0, v_0, E), 轨迹确定.
            speed = np.linalg.norm(self.v)
            if speed < 1e-6:
                speed = self.cell_size * 0.5
            angle = np.arctan2(self.v[1], self.v[0])

            if self.satellite_map is not None:
                candidates = [angle, angle + np.pi / 4, angle - np.pi / 4]
                best_score = -1.0
                best_angle = angle

                map_h, map_w = self.satellite_map.shape[:2]
                for a in candidates:
                    nq = self.q + np.array([np.cos(a), np.sin(a)], dtype=np.float32) * speed
                    nq = np.clip(nq, self.rho, self.D - self.rho)
                    px = int(np.clip(nq[0], 0, map_h - 1))
                    py = int(np.clip(nq[1], 0, map_w - 1))
                    # 纯确定性: 只看亮度, 不加随机扰动
                    brightness = float(np.mean(self.satellite_map[px, py]))
                    if brightness > best_score:
                        best_score = brightness
                        best_angle = a

                self.v = np.array([np.cos(best_angle) * speed,
                                   np.sin(best_angle) * speed], dtype=np.float32)
                self.q = self.q + self.v
                self.q = np.clip(self.q, self.rho, self.D - self.rho)
            else:
                # 无地图: 退化为确定性缓慢偏转 (每 10 步右转 90°)
                if self.step_count % 10 == 0:
                    angle += np.pi / 2
                self.v = np.array([np.cos(angle) * speed,
                                   np.sin(angle) * speed], dtype=np.float32)
                self.q = self.q + self.v
                self.q = np.clip(self.q, self.rho, self.D - self.rho)

    @property
    def grid_pos(self):
        """返回网格坐标 (gx, gy)"""
        gx = int(np.clip(self.q[0] / self.cell_size, 0, int(self.D / self.cell_size) - 1))
        gy = int(np.clip(self.q[1] / self.cell_size, 0, int(self.D / self.cell_size) - 1))
        return gx, gy

    @property
    def grid_vel(self):
        """返回网格速度 (vx_grid, vy_grid) — 单位: 格/步"""
        return self.v / self.cell_size


class UAVNavigationEnv(gym.Env):
    """
    UAV视觉导航环境 —— 训练阶段随机地图 + 随机起终点 + 动态障碍物。

    严格参照论文修改的部分:
    ──────────────────────────────────────────────────────────────
    [1] 地图/网格 (Section III-A):
        D = 3km → map_size = 3000, Ng = 30, cell = 100px

    [2] 语义地图 Mt ∈ R^{6×Ng×Ng} (公式 8, 9):
        通道 0: Mexp — 探索置信度  (当前=1, 已访=0, 未探=-1)
        通道 1: Mrel — 目标相关性  (Siamese 相似度, 未探=-1)
        通道 2: Mtrv — 可通行度    (初始=1, 碰撞=-1)
        通道 3: Mocc — 动态占用概率 (障碍物预测, ∈[0,1])
        通道 4: Mu  — 障碍物水平运动流
        通道 5: Mv  — 障碍物垂直运动流

    [3] 奖励函数 (公式 26):
        rg·I[到达] − λ_step − λ_out·I[出界] − λ_col·I[碰撞]
        + λ_rel·Δs_t − λ_risk·χ_t

    [4] 到达判定 (论文: pt = pG):
        agent 所在网格 == 目标网格

    [5] 障碍物碰撞判定 (公式 4):
        ‖pt − q^j_t‖₂ ≤ ρ_u + ρ^j
    ──────────────────────────────────────────────────────────────
    """

    # 语义地图通道索引 (公式 9)
    CH_EXP = 0   # Mexp: 探索置信度
    CH_REL = 1   # Mrel: 目标相关性
    CH_TRV = 2   # Mtrv: 可通行度
    CH_OCC = 3   # Mocc: 动态占用概率
    CH_U   = 4   # Mu:   障碍物水平运动流
    CH_V   = 5   # Mv:   障碍物垂直运动流
    N_MAP_CHANNELS = 6

    def __init__(
        self,
        map_size=3000,             # [论文] D=3km → 3000 像素
        grid_size=30,              # [论文] Ng=30 → 30×30=900 格
        max_steps=500,             # [论文 V-A] Nm=500
        db_config=None,
        semantic_patch_size=30,
        pos_margin_cells=1,
        min_grid_distance=25,
        fixed_position_seed=12345,
        speed_scale=1.0,
        # --- [论文公式 26 扩展] 奖励参数 ---
        reward_reach=20.0,            # rg — 到达目标区域
        reward_boundary=-10.0,        # λ_out — 出界惩罚
        reward_step_penalty=-0.01,    # λ_step — 每步时间代价
        reward_collision=-10.0,       # λ_col — 碰撞惩罚
        reward_rel_scale=0.5,         # λ_rel — 检测置信度增益 (Δs_t)
        reward_risk_scale=0.01,       # λ_risk — 预测风险惩罚
        # --- 新增密集奖励项 (正则化, 解决稀疏奖励) ---
        reward_explore_scale=0.1,     # 新覆盖区域奖励
        reward_revisit_penalty=-0.05, # 重复访问惩罚
        reward_position_scale=0.05,   # 位置进展奖励 (靠近高 M^rel 区域)
        reach_radius=3,               # 到达判定半径 (切比雪夫距离, 格)
        # --- 可选 shaping ---
        similarity_reward_scale=0.0,
        # --- Siamese 相似度参数 ---
        siamese_model_path=None,
        # [匹配 train_siamese_v2.py normalize=True] L2 归一化后两个单位向量的
        # L2 距离上界是 2.0. 实际 val 集结果:
        #   df_pos = 0.073 ± 0.058
        #   df_neg = 1.366 ± 0.124
        #   sep    = +1.293  (AUC = 0.9994)
        # 中点取法 df_max = (df_pos + df_neg) / 2 ≈ 0.72
        # 这样正对 df < df_max 给出 vs ∈ [0.80, 1.00] 的平滑信号,
        # 负对 df > df_max 全部 clip 到 vs = 0.
        df_max=0.72,
        siamese_device='cuda',
        # --- [论文 Section III-A] 动态障碍物参数 ---
        # 碰撞判定距离 = obstacle_rho + uav_rho
        # 原始值 120+60=180px=1.8格, 碰撞直径3.6格 → 250步存活率≈0%
        # 调整后 40+20=60px=0.6格, 碰撞直径1.2格 → 250步存活率≈20%
        num_obstacles=3,
        obstacle_speed=20.0,       # 慢速更易预测和规避
        obstacle_rho=40.0,         # 障碍物安全半径
        uav_rho=20.0,              # UAV 安全半径
        obstacle_motion_types=('cv', 'random', 'road'),
        forecast_horizon=5,
    ):
        super(UAVNavigationEnv, self).__init__()

        # ---------- 地图与网格 ----------
        self.D = map_size
        self.Ng = grid_size
        self.max_steps = max_steps
        self.cell_size = self.D / self.Ng   # 3000/30 = 100
        self.semantic_patch_size = semantic_patch_size

        # ---------- 起终点生成参数 ----------
        self.pos_margin_cells = int(pos_margin_cells)
        self.min_grid_distance = int(min_grid_distance)
        self.fixed_position_seed = int(fixed_position_seed)
        self._fixed_agent_pos = None
        self._fixed_target_pos = None

        # ---------- 奖励参数 ----------
        self.speed_scale = speed_scale
        self.reward_reach = reward_reach
        self.reward_boundary = reward_boundary
        self.reward_step_penalty = reward_step_penalty
        self.reward_collision = float(reward_collision)
        self.reward_rel_scale = float(reward_rel_scale)
        self.reward_risk_scale = float(reward_risk_scale)
        self.similarity_reward_scale = float(similarity_reward_scale)
        # 新增密集奖励参数
        self.reward_explore_scale = float(reward_explore_scale)
        self.reward_revisit_penalty = float(reward_revisit_penalty)
        self.reward_position_scale = float(reward_position_scale)
        self.reach_radius = int(reach_radius)

        # ---------- [论文 Section III-A] 动态障碍物参数 ----------
        self.num_obstacles = int(num_obstacles)
        self.obstacle_speed = float(obstacle_speed)
        self.obstacle_rho = float(obstacle_rho)
        self.uav_rho = float(uav_rho)
        self.obstacle_motion_types = list(obstacle_motion_types)
        self.forecast_horizon = int(forecast_horizon)   # K
        self.obstacles = []                              # List[DynamicObstacle]

        # ---------- 动作空间 ----------
        self.action_space = spaces.Discrete(8)
        self._dir8 = np.array(
            [
                ( 0,  1),   # α1  N
                (-1,  1),   # α2  NW
                (-1,  0),   # α3  W
                (-1, -1),   # α4  SW
                ( 0, -1),   # α5  S
                ( 1, -1),   # α6  SE
                ( 1,  0),   # α7  E
                ( 1,  1),   # α8  NE
            ],
            dtype=np.float32,
        )
        # ---------- 观测空间 ----------
        # [论文公式 5] y_t = {I_t, I^g, p_t} + 语义地图 M_t
        # [M-3 修复] target_position 已移除 — 论文公式(5)不含目标坐标
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
            'target': spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
            'position': spaces.Box(low=0, high=self.D, shape=(2,), dtype=np.float32),
            'semantic_map': spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.N_MAP_CHANNELS, self.Ng, self.Ng),
                dtype=np.float32
            ),
            'safety_mask': spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
        })

        # ---------- Siamese 网络 (论文方法, 必须加载) ----------
        self.df_max = df_max
        self.siamese_device = siamese_device
        self._target_feature = None   # 缓存目标图像特征 (每 episode 算一次)

        if not siamese_model_path or not os.path.exists(siamese_model_path):
            raise FileNotFoundError(
                f"[UAVEnv] 必须提供已训练的 Siamese 模型, 路径无效: {siamese_model_path}. "
            )
        self.siamese_net = SiameseNetwork(feature_dim=128)
        state_dict = torch.load(siamese_model_path, map_location=siamese_device, weights_only=True)
        self.siamese_net.load_state_dict(state_dict)
        self.siamese_net.to(siamese_device)
        self.siamese_net.eval()
        print(f"[UAVEnv] Siamese 模型已加载: {siamese_model_path}")

        # ---------- 地图状态 ----------
        self.satellite_map = np.zeros((self.D, self.D, 3), dtype=np.uint8)
        self.map_ids = []
        self.original_map_size = None

        # ---------- 计数器 ----------
        self.episode_counter = 0
        self._is_test = False
        self._prev_image = None   # 用于观测延迟机制
        self._last_info = {}      # 用于评测读取 reach/collision

        # ---------- 数据库 ----------
        self.db_config = db_config
        self.db_conn = None

        if self.db_config:
            try:
                self.db_conn = mysql.connector.connect(**self.db_config)
                print("[UAVEnv] DB Connected.")
                cur = self.db_conn.cursor()
                cur.execute("SELECT id FROM image_maps2")
                all_ids = [x[0] for x in cur.fetchall()]
                cur.close()

                # [论文 V-B] 显式划分训练/测试地图集, 保证 OOD 评测无泄漏
                # 80% 训练, 20% held-out 测试
                split_rng = np.random.RandomState(42)  # 固定划分种子, 保证可复现
                split_rng.shuffle(all_ids)
                split_idx = int(len(all_ids) * 0.8)
                self.train_map_ids = all_ids[:split_idx]
                self.test_map_ids = all_ids[split_idx:]
                self.map_ids = self.train_map_ids  # 默认用训练集
                print(f"[UAVEnv] Maps: {len(self.train_map_ids)} train, "
                      f"{len(self.test_map_ids)} test (held-out).")
            except Exception as e:
                print(f"[UAVEnv] DB Error: {e}")
                self.train_map_ids = []
                self.test_map_ids = []

    # ================================================================
    #  Position Generation
    # ================================================================

    def _generate_random_positions(self, rng=None):
        """
        [改] 起终点生成: **独立随机采样**, 但要求切比雪夫网格距离 ≥ min_grid_distance.

        和论文对齐的点: 起终点是 "random", 不是固定距离.
        额外约束: 切比雪夫距离 ≥ 25 (默认), 保证两点大致近对角, 任务足够有挑战性.
        切比雪夫最大值受 Ng 和 margin 约束: 30-1-1-1 = 27. 所以实际分布 ∈ {25, 26, 27}.

        算法:
            1. 均匀采样起点网格 ∈ [margin, Ng-margin)
            2. 均匀采样终点网格, 若切比雪夫距离 < min 则重采
            3. 起/终点返回对应 cell 中心的连续坐标

        参数:
            rng: numpy RandomState (测试时可复现) / None (训练时完全随机)
        返回:
            agent_pos, target_pos: shape=(2,) float32, cell 中心对齐
        """
        _randint = rng.randint if rng is not None else np.random.randint

        lo = int(self.pos_margin_cells)
        hi = int(self.Ng - self.pos_margin_cells)  # exclusive
        D_min = int(self.min_grid_distance)

        usable = hi - lo
        max_possible_cheb = usable - 1  # 对角端到端的切比雪夫距离
        if D_min > max_possible_cheb:
            raise ValueError(
                f"min_grid_distance={D_min} 超出可行上限 {max_possible_cheb} "
                f"(Ng={self.Ng}, margin={lo}). 请调小 min_grid_distance."
            )

        # 每次同时重采起点和终点, 直到 Chebyshev ≥ D_min.
        # (若只固定起点, 中心起点无论如何重采终点都不可能成功)
        max_attempts = 5000
        for _ in range(max_attempts):
            gx_s = int(_randint(lo, hi))
            gy_s = int(_randint(lo, hi))
            gx_t = int(_randint(lo, hi))
            gy_t = int(_randint(lo, hi))
            cheb = max(abs(gx_t - gx_s), abs(gy_t - gy_s))
            if cheb >= D_min:
                agent_pos = np.array(
                    [(gx_s + 0.5) * self.cell_size, (gy_s + 0.5) * self.cell_size],
                    dtype=np.float32,
                )
                target_pos = np.array(
                    [(gx_t + 0.5) * self.cell_size, (gy_t + 0.5) * self.cell_size],
                    dtype=np.float32,
                )
                return agent_pos, target_pos

        # Fallback: 极少触发; 把 start 丢到角落, target 丢到对角
        print(f"[UAVEnv] WARNING: position sampling exhausted {max_attempts} attempts.")
        gx_s, gy_s = lo, lo
        gx_t, gy_t = hi - 1, hi - 1
        return (
            np.array([(gx_s + 0.5) * self.cell_size, (gy_s + 0.5) * self.cell_size], dtype=np.float32),
            np.array([(gx_t + 0.5) * self.cell_size, (gy_t + 0.5) * self.cell_size], dtype=np.float32),
        )

    # ================================================================
    #  Map Loading
    # ================================================================

    def _load_map_from_db(self, map_seed=None, ood=False):
        """
        从数据库加载地图。

        参数:
            map_seed:
                - None: 训练模式, 每次随机选择不同地图
                - int:  测试模式, 使用固定 seed 确保选择相同地图
            ood:
                - False: 从 train_map_ids (in-domain) 中选
                - True:  从 test_map_ids  (OOD/held-out) 中选
                此参数仅用于缓存 key 消歧; 实际的 map_ids 切换在
                reset() 里完成, 此处只读 self.map_ids.

        返回:
            缩放后的地图 (self.D × self.D × 3) uint8

        [缓存 key 修复]
        ──────────────────────────────────────────────────────────────
        旧实现: 缓存文件名只含 map_seed (test_map_seed_{seed}.npy),
                不含 split 信息. 由于 train_map_ids 和 test_map_ids
                是同一 all_ids 的 80/20 不相交划分, 给定 selected_idx
                在两个列表里指向不同的物理 map_id. 因此同一 map_seed
                在 in-domain 和 OOD 下会选到完全不同的两张图 — 但
                它们共用同一个缓存文件名, 第二次调用会错误命中第一
                次缓存的另一 split 的图.
        新实现: 把 split tag 写进文件名 (_train / _ood), 两个 split
                互不串扰. 旧缓存文件 (无 split 后缀) 会被忽略, 新缓
                存首次调用时自动生成, 对使用方透明.
        ──────────────────────────────────────────────────────────────
        """
        cache_dir = './map_cache'
        os.makedirs(cache_dir, exist_ok=True)

        # 缓存 key: 含 split 防止 in-domain / OOD 串台
        split_tag = 'ood' if ood else 'train'

        # ===== 测试模式: 优先从文件缓存加载 =====
        if map_seed is not None:
            cache_path = os.path.join(
                cache_dir, f'test_map_seed_{map_seed}_{split_tag}.npy'
            )
            if os.path.exists(cache_path):
                cached_data = np.load(cache_path, allow_pickle=True).item()
                self.original_map_size = cached_data['original_size']
                return cached_data['map'].astype(np.uint8)

        # ===== 检查数据库连接 =====
        if not self.db_conn:
            raise ConnectionError("Database is not connected! Check your DB config.")
        if not self.map_ids:
            raise ValueError("No map IDs found in the database table.")

        try:
            start_time = time.time()

            if not self.db_conn.is_connected():
                self.db_conn.reconnect()

            total_maps = len(self.map_ids)
            mode = "TEST" if map_seed is not None else "TRAIN"

            # ===== 根据 map_seed 选择地图 =====
            if map_seed is not None:
                rng = np.random.RandomState(map_seed)
                selected_idx = rng.randint(0, total_maps)
            else:
                selected_idx = np.random.randint(0, total_maps)

            selected_id = self.map_ids[selected_idx]
            print(f"[MapLoader] {mode} mode - Map ID: {selected_id} ({selected_idx}/{total_maps})")

            # ===== 从数据库加载地图 =====
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT image_data FROM image_maps2 WHERE id = %s", (int(selected_id),))
            row = cursor.fetchone()
            cursor.close()

            if row is None:
                raise ValueError(f"Map ID {selected_id} not found in database.")

            blob = row[0]

            # 解析地图数据
            try:
                original_map = pickle.loads(blob)
            except:
                original_map = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)

            if original_map is None:
                raise ValueError(f"Failed to decode map ID {selected_id}")

            # 确保是 RGB 格式
            if len(original_map.shape) == 2:
                original_map = cv2.cvtColor(original_map, cv2.COLOR_GRAY2RGB)
            elif original_map.shape[2] == 4:
                original_map = cv2.cvtColor(original_map, cv2.COLOR_BGRA2RGB)

            self.original_map_size = original_map.shape[:2]

            # ===== 缩放地图到导航尺寸 =====
            if original_map.shape[0] != self.D or original_map.shape[1] != self.D:
                scaled_map = cv2.resize(original_map, (self.D, self.D), interpolation=cv2.INTER_LINEAR)
            else:
                scaled_map = original_map

            elapsed = time.time() - start_time
            print(f"[MapLoader] Map loaded in {elapsed:.2f}s")

            # ===== 测试模式: 保存到文件缓存 (key 含 split_tag, 避免串台) =====
            if map_seed is not None:
                cache_path = os.path.join(
                    cache_dir, f'test_map_seed_{map_seed}_{split_tag}.npy'
                )
                np.save(cache_path, {'map': scaled_map, 'original_size': self.original_map_size})

            return scaled_map.astype(np.uint8)

        except Exception as e:
            raise RuntimeError(f"Critical failure loading map from DB: {e}")

    # ================================================================
    #  Semantic Features
    # ================================================================

    def _get_semantic_patch(self, pos):
        """提取以 pos 为中心的语义切片 (3, patch_size, patch_size)"""
        patch_size = self.semantic_patch_size
        half_patch = patch_size // 2

        padded_map = np.pad(
            self.satellite_map,
            ((half_patch, half_patch), (half_patch, half_patch), (0, 0)),
            mode='constant'
        )

        center_x = int(pos[0]) + half_patch
        center_y = int(pos[1]) + half_patch

        patch = padded_map[
            center_x - half_patch: center_x + half_patch,
            center_y - half_patch: center_y + half_patch
        ].copy()

        if patch.shape[:2] != (patch_size, patch_size):
            patch = cv2.resize(patch, (patch_size, patch_size))

        return patch.transpose(2, 0, 1)  # HWC -> CHW

    # [新增] 网格坐标工具
    def _pos_to_grid(self, pos):
        """连续坐标 → 网格坐标 (gx, gy)"""
        gx = int(np.clip(pos[0] / self.cell_size, 0, self.Ng - 1))
        gy = int(np.clip(pos[1] / self.cell_size, 0, self.Ng - 1))
        return gx, gy

    # 提取网格单元图像, 用于 Siamese 输入
    def _get_grid_cell_image(self, gx, gy):
        """
        提取网格 (gx, gy) 对应的卫星图像, resize 到 64×64。
        用于 Siamese 网络的输入 (论文: 观测图像 ot 和目标图像 og)。
        返回: (3, 64, 64) uint8, CHW 格式
        """
        x_start = int(gx * self.cell_size)
        y_start = int(gy * self.cell_size)
        x_end = min(int((gx + 1) * self.cell_size), self.D)
        y_end = min(int((gy + 1) * self.cell_size), self.D)
        cell_img = self.satellite_map[x_start:x_end, y_start:y_end]
        if cell_img.shape[0] == 0 or cell_img.shape[1] == 0:
            return np.zeros((3, 64, 64), dtype=np.uint8)
        cell_img = cv2.resize(cell_img, (64, 64), interpolation=cv2.INTER_LINEAR)
        return cell_img.transpose(2, 0, 1)  # HWC → CHW

    def _compute_similarity(self, pos):
        """
        [论文公式 4] 使用 Siamese 网络计算当前网格图像与目标网格图像的相似度.

        流程:
          1. 提取当前位置所在网格图像 ot (_get_grid_cell_image)
          2. Siamese 前向得到 128 维特征 ft
          3. 特征距离 df = ||ft - fg||₂   (fg 已在 reset 时缓存)
          4. 归一化 vs = max(0, 1 - df / df_max) ∈ [0, 1]
        """
        assert self._target_feature is not None, \
            "target feature not cached; 请确认 reset() 被正确调用"

        gx, gy = self._pos_to_grid(pos)
        cell_img = self._get_grid_cell_image(gx, gy)

        with torch.no_grad():
            img_tensor = torch.tensor(
                cell_img, dtype=torch.float32
            ).unsqueeze(0).to(self.siamese_device) / 255.0
            f_current = self.siamese_net.forward_one(img_tensor)
            df = torch.norm(f_current - self._target_feature, p=2, dim=1).item()

        vs = max(0.0, 1.0 - df / self.df_max)
        return float(vs)

    # ================================================================
    #  Dynamic Obstacles
    # ================================================================

    def _spawn_obstacles(self, rng=None):
        """
        [论文 Section III-A, 公式(2)] 在每个 episode 开始时生成障碍物。
        障碍物不能出现在 agent 或 target 附近 (>= margin)。

        参数:
            rng: numpy.random.RandomState 或 None.
                 - None  → 训练模式: 使用全局 np.random, 每个 episode
                           初始位置/速度/运动类型完全随机, 且 'random'
                           运动模式每步扰动也完全随机.
                 - 非 None → 测试模式: 同一 rng 状态 → 同一组障碍物
                           (初始位置/速度/运动类型 + 每步扰动序列均固定).

        [测试可复现性修复]
        ──────────────────────────────────────────────────────────────
        旧实现: 直接用全局 np.random.choice / uniform, 绕开 map_seed.
        结果: 同一 map_seed 重复测试时, 地图/起终点一致, 但障碍物初
              始状态和运动序列每次不同 → SR/CR 曲线抖动 + 不同
              checkpoint 并非在相同任务集上比较.
        新实现: 所有随机采样走 rng. 同时为每个障碍物从 rng 派生独
              立子 seed, 用于 DynamicObstacle 的逐步随机扰动, 保证
              整个 episode 的障碍物轨迹完全确定.
        ──────────────────────────────────────────────────────────────
        """
        self.obstacles = []
        margin = self.obstacle_rho + self.uav_rho + self.cell_size  # 安全生成距离

        # 随机源分发: rng 提供时走 rng, 否则退化到全局 np.random.
        _choice  = rng.choice  if rng is not None else np.random.choice
        _uniform = rng.uniform if rng is not None else np.random.uniform
        _randint = rng.randint if rng is not None else np.random.randint

        for _ in range(self.num_obstacles):
            # 随机运动类型
            mtype = _choice(self.obstacle_motion_types)
            # 随机速度方向
            angle = _uniform(0, 2 * np.pi)
            speed = _uniform(self.obstacle_speed * 0.5, self.obstacle_speed * 1.5)
            v = np.array([np.cos(angle) * speed, np.sin(angle) * speed], dtype=np.float32)

            # 随机位置: 避开 agent/target 周围
            q = None
            for _try in range(200):
                q_cand = _uniform(self.obstacle_rho, self.D - self.obstacle_rho, size=2).astype(np.float32)
                d_agent = np.linalg.norm(q_cand - self.agent_pos)
                d_target = np.linalg.norm(q_cand - self.target_pos)
                if d_agent > margin and d_target > margin:
                    q = q_cand
                    break
            if q is None:
                # Fallback: 200 次都没采到合法位置 (极小概率).
                # 直接用最后一次候选, 避免 q 未定义.
                q = q_cand

            # Per-obstacle RNG: 用于 'random' 运动模式的逐步方向漂移.
            # 即便 rng is None (训练), 也可选给一个 None → 障碍物内部
            # 自己退化到全局 np.random. 测试时从 rng 派生独立流, 保证
            # 不同障碍物之间的随机序列也互不干扰且可复现.
            if rng is not None:
                per_obs_rng = np.random.RandomState(int(_randint(0, 2**31 - 1)))
            else:
                per_obs_rng = None

            self.obstacles.append(
                DynamicObstacle(q, v, self.obstacle_rho, mtype, self.D, self.cell_size,
                                satellite_map=self.satellite_map if mtype == 'road' else None,
                                rng=per_obs_rng)
            )

    def _step_obstacles(self):
        """[论文公式(3)] 所有障碍物前进一步。"""
        for obs in self.obstacles:
            obs.step()

    def _check_collision(self, pos):
        """
        [论文公式(4)] 检查 UAV 是否与任意障碍物发生碰撞。
        ‖pt − q^j_t‖₂ ≤ ρ_u + ρ^j
        返回: (bool) 是否碰撞
        """
        threshold = self.uav_rho + self.obstacle_rho
        for obs in self.obstacles:
            if np.linalg.norm(pos - obs.q) <= threshold:
                return True
        return False

    def _build_obstacle_channels(self):
        """
        [论文公式(9), Section III-C] 计算语义地图中的动态通道:
            Mocc ∈ [0,1]: 当前占用概率 (高斯扩散软化)
            Mu, Mv ∈ [-1,1]: 归一化运动流

        使用高斯核将点障碍物软化为占用概率场, sigma = 1.5 格。
        """
        Mocc = np.zeros((self.Ng, self.Ng), dtype=np.float32)
        Mu   = np.zeros((self.Ng, self.Ng), dtype=np.float32)
        Mv   = np.zeros((self.Ng, self.Ng), dtype=np.float32)

        sigma = 1.5  # 高斯扩散标准差 (格单位)
        radius = int(np.ceil(3 * sigma))

        for obs in self.obstacles:
            gx, gy = obs.grid_pos
            vx_n, vy_n = obs.grid_vel  # 格/步

            # 高斯扩散到周围格子
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.Ng and 0 <= ny < self.Ng:
                        dist2 = dx ** 2 + dy ** 2
                        weight = np.exp(-dist2 / (2 * sigma ** 2))
                        if weight > Mocc[nx, ny]:
                            Mocc[nx, ny] = weight
                            # 运动流用最强权重那个障碍物的速度
                            Mu[nx, ny] = np.clip(vx_n / (self.obstacle_speed / self.cell_size + 1e-6), -1, 1)
                            Mv[nx, ny] = np.clip(vy_n / (self.obstacle_speed / self.cell_size + 1e-6), -1, 1)

        return Mocc, Mu, Mv

    def _compute_risk(self, pos, action_direction=None):
        """
        [R-2 修复] 论文公式(28): χ_t = max_{1≤k≤K} Ô_{t+k}(ĩ_{t+k})

        调用时 obstacles 已在 t+1. 先 check 后 step, 覆盖 t+1..t+K:
          k=0: check(pos, obs_{t+1}), step→t+2
          k=1: check(pos+dir, obs_{t+2}), step→t+3
          ...
          k=K-1: check(pos+dir*(K-1), obs_{t+K}), step→t+K+1(不用)
        """
        import copy
        obs_copies = [copy.deepcopy(o) for o in self.obstacles]
        threshold = self.uav_rho + self.obstacle_rho
        max_occ = 0.0
        for k in range(self.forecast_horizon):  # k=0..K-1, 映射公式 k=1..K
            if action_direction is not None:
                future_pos = pos + action_direction * self.cell_size * k
                future_pos = np.clip(future_pos, 0, self.D)
            else:
                future_pos = pos
            # 先 check (obstacle 和 UAV 同步在 t+1+k)
            for o in obs_copies:
                dist = np.linalg.norm(future_pos - o.q)
                occ = np.exp(-(dist ** 2) / (2 * (threshold) ** 2))
                if occ > max_occ:
                    max_occ = occ
            # 后 step (为下一轮准备)
            for o in obs_copies:
                o.step()
        return float(np.clip(max_occ, 0.0, 1.0))

    # ================================================================
    #  Semantic Map Update (6-channel, 论文公式 9)
    # ================================================================

    def _update_map(self, pos, is_current):
        """
        更新语义地图 6 通道 (论文公式 9):

        CH_EXP (0): Mexp — 探索置信度: 当前=1, 已访=0, 未探=-1
        CH_REL (1): Mrel — 已在 step() 中直接更新 (避免重复 Siamese 推理)
        CH_TRV (2): Mtrv — 可通行度: 初始=1; 碰撞格置 -1
        CH_OCC (3-5): 由 _refresh_dynamic_channels 统一刷新
        """
        gx = int(np.clip(pos[0] / self.cell_size, 0, self.Ng - 1))
        gy = int(np.clip(pos[1] / self.cell_size, 0, self.Ng - 1))

        # Mexp
        self.semantic_map[self.CH_EXP, gx, gy] = 1.0 if is_current else 0.0

        # Mrel: 不在此处更新 — step() 的 Δs_t 计算中已经调用 _compute_similarity
        # 并直接写入了 self.semantic_map[CH_REL, new_gx, new_gy]

    def _refresh_dynamic_channels(self):
        """每步刷新 Mocc / Mu / Mv 通道 (公式 9 后三通道)。"""
        Mocc, Mu, Mv = self._build_obstacle_channels()
        self.semantic_map[self.CH_OCC] = Mocc
        self.semantic_map[self.CH_U]   = Mu
        self.semantic_map[self.CH_V]   = Mv

    # ================================================================
    #  Observation
    # ================================================================

    def _get_current_view(self, pos):
        """提取以 pos 为中心的 64×64 视图 (3, 64, 64)"""
        x, y = int(pos[0]) + 32, int(pos[1]) + 32
        padded = np.pad(self.satellite_map, ((32, 32), (32, 32), (0, 0)), mode='constant')
        view = padded[x - 32:x + 32, y - 32:y + 32]
        if view.shape != (64, 64, 3):
            view = cv2.resize(view, (64, 64))
        return view.transpose(2, 0, 1)

    def _apply_obs_noise(self, image):
        """
        [论文 Section V-A] 观测损坏:
            - 高斯图像噪声 (σ=5, uint8 级)
            - 5% 概率感知 dropout (整张图置零)
            - 10% 概率观测延迟 (返回上一步的缓存图像)
        仅训练模式生效, 测试模式返回原图.
        """
        if self._is_test:
            self._prev_image = image.copy()
            return image

        # 观测延迟: 10% 概率返回上一步的图像
        if hasattr(self, '_prev_image') and self._prev_image is not None:
            if np.random.random() < 0.10:
                stale = self._prev_image.copy()
                self._prev_image = image.copy()
                return stale
        self._prev_image = image.copy()

        # 感知 dropout: 5% 概率
        if np.random.random() < 0.05:
            return np.zeros_like(image)

        # 高斯噪声: σ=5
        noise = np.random.normal(0, 5, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    def _compute_safety_mask(self, pos):
        """
        计算 8 方向安全掩码 σ_t ∈ [0,1]^8.
        σ_t[i] = 1 表示方向 i 安全, 0 表示危险(障碍物/出界/不可通行).

        融合三种信息:
          - 边界: 下一步是否出界
          - M^trv: 目标格是否曾碰撞 (CH_TRV == -1)
          - M^occ: 目标格的动态占用概率
        """
        mask = np.ones(8, dtype=np.float32)
        for i, d in enumerate(self._dir8):
            next_pos = pos + d * self.cell_size
            # 出界检查
            if not (0 <= next_pos[0] <= self.D and 0 <= next_pos[1] <= self.D):
                mask[i] = 0.0
                continue
            ngx, ngy = self._pos_to_grid(next_pos)
            ngx = int(np.clip(ngx, 0, self.Ng - 1))
            ngy = int(np.clip(ngy, 0, self.Ng - 1))
            # 不可通行 (之前碰撞过)
            if self.semantic_map[self.CH_TRV, ngx, ngy] < 0:
                mask[i] = 0.0
                continue
            # 动态占用: 1 - occ (高占用 → 低安全)
            occ = float(self.semantic_map[self.CH_OCC, ngx, ngy])
            mask[i] = max(0.0, 1.0 - occ)
        return mask

    def _get_obs(self):
        img = self._get_current_view(self.agent_pos)
        tgt = self._get_current_view(self.target_pos)
        return {
            'image': self._apply_obs_noise(img),
            'target': tgt,
            'position': self.agent_pos.copy(),
            'semantic_map': self.semantic_map.copy(),
            'safety_mask': self._compute_safety_mask(self.agent_pos),
        }

    # ================================================================
    #  Core: reset & step
    # ================================================================

    def reset(self, seed=None, options=None, map_seed=None, ood=False):
        """
        重置环境。

        参数:
            map_seed:
                - None  → 训练模式: 训练地图集随机选 + 每 episode 随机起终点 + 观测噪声
                - int   → 测试模式: 固定地图 + 固定起终点 + 无噪声
            ood:
                - False → 使用训练地图集 (in-domain)
                - True  → 使用 held-out 测试地图集 (out-of-domain)
        """
        super().reset(seed=seed)
        self.episode_counter += 1
        self._is_test = (map_seed is not None)

        # [论文 V-B] 切换地图集: 训练用 train_map_ids, OOD 评测用 test_map_ids
        if ood and hasattr(self, 'test_map_ids') and self.test_map_ids:
            self.map_ids = self.test_map_ids
        elif hasattr(self, 'train_map_ids') and self.train_map_ids:
            self.map_ids = self.train_map_ids

        # 加载地图 (ood 标志同时用于 map_ids 切换与缓存 key 消歧)
        self.satellite_map = self._load_map_from_db(map_seed=map_seed, ood=ood)

        # --- 起终点生成 ---
        # [测试可复现性修复] 测试模式下维护一个与 map_seed 绑定的 RNG,
        # 同时用于起终点采样和障碍物生成, 保证完全可复现.
        if self._is_test:
            # [修复] 测试模式: 每个 episode 用 map_seed 派生独立的 RNG
            # 不同 map_seed → 不同起终点 + 不同障碍物, 保证测试覆盖多种配置
            # 相同 map_seed → 完全一致的任务 (map + start/goal + obstacles)
            test_rng = np.random.RandomState(map_seed if map_seed is not None
                                             else self.fixed_position_seed)
            self.agent_pos, self.target_pos = self._generate_random_positions(rng=test_rng)
            # 复用同一 RNG 的后续状态派生障碍物采样流. 这样:
            #  - 不同起终点 → 不同 pos_rng 消耗量 → 不同 obs 子流 (但仍确定)
            # 直接用 test_rng 继续消耗即可, 无需再拆一个 seed.
            self._obs_rng = test_rng
        else:
            # [论文 V-A] 训练模式: 每 episode 随机采样新的 start-goal pair
            self.agent_pos, self.target_pos = self._generate_random_positions(rng=None)
            self._obs_rng = None

        self.init_dist = float(np.linalg.norm(self.agent_pos - self.target_pos))
        self.steps = 0

        # 缓存目标图像 Siamese 特征
        tgx, tgy = self._pos_to_grid(self.target_pos)
        target_img = self._get_grid_cell_image(tgx, tgy)
        with torch.no_grad():
            target_tensor = torch.tensor(
                target_img, dtype=torch.float32
            ).unsqueeze(0).to(self.siamese_device) / 255.0
            self._target_feature = self.siamese_net.forward_one(target_tensor)

        # [论文公式 9] 初始化 6 通道语义地图
        self.semantic_map = np.zeros((self.N_MAP_CHANNELS, self.Ng, self.Ng), dtype=np.float32)
        self.semantic_map[self.CH_EXP] = -1.0
        self.semantic_map[self.CH_REL] = -1.0
        self.semantic_map[self.CH_TRV] = 1.0

        # [论文 Section III-A] 生成动态障碍物
        # 测试模式传入 self._obs_rng → 可复现; 训练模式传 None → 全随机.
        self._spawn_obstacles(rng=self._obs_rng)
        self._refresh_dynamic_channels()
        self._update_map(self.agent_pos, True)

        return self._get_obs(), {}

    def step(self, action):
        """
        执行一步。

        奖励设计 (论文公式 26 扩展, 增加密集正则项):
            rt = rg·I[到达区域]
               − λ_col·I[碰撞] − λ_out·I[出界]
               − λ_step                           # 每步时间代价
               + λ_explore · new_covered           # 新覆盖区域奖励
               + λ_rel · max(0, Δs_t)             # 检测置信度增益 (仅正向)
               − λ_revisit · I[重复访问]            # 重复访问惩罚
               + λ_pos · position_progress         # 位置进展 (M^rel 方向梯度)
               − λ_risk · χ_t                     # 预测风险惩罚
        """
        self.steps += 1

        # ===== 1. 解析动作 =====
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        action = np.asarray(action).flatten()
        if action.size == 1:
            a_idx = int(action[0])
        else:
            a_idx = int(np.argmax(action))
        a_idx = int(np.clip(a_idx, 0, 7))
        direction = self._dir8[a_idx]
        new_pos = self.agent_pos + direction * self.cell_size

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # ===== 2. 先让障碍物移动 (公式 3) =====
        self._step_obstacles()

        # ===== 3. 每步时间代价 (所有情况都扣) =====
        reward += self.reward_step_penalty   # -0.01

        # ===== 4. 边界检查 =====
        if not (0 <= new_pos[0] <= self.D and 0 <= new_pos[1] <= self.D):
            reward += self.reward_boundary   # -10
            terminated = True
            new_pos = np.clip(new_pos, 0, self.D)
            info['boundary'] = True
            # [CR 语义统一] 出界也算"碰撞"(不安全终止).
            # 所有下游评估器都读 _last_info['collision'] 来累计 CR —
            # 在唯一真源 (env.step) 处把出界并进去, 就避免每个 main 里
            # 都得记得去 or 一下 info['boundary']. 'boundary' 键保留,
            # 如果后面要做"障碍物碰撞率 vs 出界率"的细分统计, 仍可区分:
            #   obs_coll_only = info['collision'] and not info.get('boundary')
            info['collision'] = True
        else:
            new_gx, new_gy = self._pos_to_grid(new_pos)
            target_gx, target_gy = self._pos_to_grid(self.target_pos)

            # ===== 5. 到达检查 (区域判定, 切比雪夫距离 ≤ reach_radius) =====
            cheb_to_target = max(abs(new_gx - target_gx), abs(new_gy - target_gy))
            if cheb_to_target <= self.reach_radius:
                reward += self.reward_reach   # +20
                terminated = True
                info['reach'] = True
            else:
                # ===== 6. 碰撞检查 (论文公式 4) =====
                collision = self._check_collision(new_pos)
                if collision:
                    reward += self.reward_collision   # -10
                    self.semantic_map[self.CH_TRV, new_gx, new_gy] = -1.0
                    terminated = True
                    info['collision'] = True

            # ===== 7. 新覆盖区域奖励 + 重复访问惩罚 =====
            is_first_visit = bool(self.semantic_map[self.CH_EXP, new_gx, new_gy] == -1.0)
            if is_first_visit:
                # 新探索: 正奖励
                reward += self.reward_explore_scale   # +0.1
            else:
                # 重复访问: 负奖励
                reward += self.reward_revisit_penalty  # -0.05

            # ===== 8. Mrel 更新 + 检测置信度增益 (公式 27 改进) =====
            old_gx, old_gy = self._pos_to_grid(self.agent_pos)
            mrel_old = self.semantic_map[self.CH_REL, old_gx, old_gy]

            vs_new = self._compute_similarity(new_pos)
            self.semantic_map[self.CH_REL, new_gx, new_gy] = vs_new

            if self.reward_rel_scale > 0.0:
                # 仅奖励正向增益, 不惩罚下降 (避免探索新区域时的负 Δs_t)
                delta_s = float(vs_new) - float(mrel_old)
                reward += self.reward_rel_scale * max(0.0, delta_s)   # +0.5 * max(0, Δs)

            # ===== 9. 位置进展奖励 (朝 M^rel 高值区域移动) =====
            if self.reward_position_scale > 0.0:
                # 计算 8 邻域 M^rel 的加权方向梯度
                # 正向: 当前移动方向上 M^rel 增加 → 正奖励
                rel_new = float(self.semantic_map[self.CH_REL, new_gx, new_gy])
                rel_old = float(self.semantic_map[self.CH_REL, old_gx, old_gy])
                # 仅在两个格子都被探索过时 (非 -1) 才计算
                if rel_new >= 0 and rel_old >= 0:
                    pos_progress = rel_new - rel_old
                    reward += self.reward_position_scale * max(0.0, pos_progress)

            # ===== 10. 预测风险惩罚 χ_t (公式 28) =====
            if self.reward_risk_scale > 0.0:
                chi = self._compute_risk(new_pos, action_direction=direction)
                reward -= self.reward_risk_scale * chi
                info['risk'] = chi

            # 可选 shaping: 首访新格相似度
            if self.similarity_reward_scale > 0.0:
                if is_first_visit:
                    reward += self.similarity_reward_scale * vs_new

        # ===== 11. 更新语义地图 =====
        self._update_map(self.agent_pos, False)   # 旧位置 → 已探索
        self.agent_pos = new_pos.astype(np.float32)
        self._update_map(self.agent_pos, True)    # 新位置 → 当前
        self._refresh_dynamic_channels()

        # ===== 12. 截断检查 =====
        if self.steps >= self.max_steps:
            truncated = True

        self._last_info = info
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if self.db_conn:
            self.db_conn.close()


# ============================================================================
#  Wrappers (保持与 main.py / main_ddpg.py 完全兼容)
# ============================================================================

class UAVEnvWrapper:
    """包装 UAVNavigationEnv, 输出 torch Tensor + bit-depth 预处理 + action_repeat。"""

    def __init__(self, env, bit_depth, action_repeat=1):
        self._env = env
        self.bit_depth = bit_depth
        self.action_repeat = action_repeat
        self.observation_size = dict(env.observation_space)
        if hasattr(env.action_space, 'n'):
            self.action_size = int(env.action_space.n)
        else:
            self.action_size = int(env.action_space.shape[0])
        self.reward_reach = getattr(env, 'reward_reach', 30.0)

    def reset(self, map_seed=None, ood=False):
        return self._process_obs(self._env.reset(map_seed=map_seed, ood=ood)[0])

    def step(self, action):
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy().flatten()
        total_reward = 0.0
        done = False
        for _ in range(self.action_repeat):
            obs, r, term, trunc, _ = self._env.step(action)
            total_reward += r
            done = term or trunc
            if done:
                break
        return self._process_obs(obs), total_reward, done

    def _process_obs(self, obs):
        processed = {}
        for k in ['image', 'target']:
            t = torch.tensor(obs[k], dtype=torch.float32)
            preprocess_observation_(t, self.bit_depth)
            processed[k] = t.unsqueeze(0)
        _D = self._env.D if hasattr(self._env, 'D') else 3000.0
        processed['position'] = torch.tensor(obs['position'] / _D, dtype=torch.float32).unsqueeze(0)
        processed['semantic_map'] = torch.tensor(obs['semantic_map'], dtype=torch.float32).unsqueeze(0)
        # 安全掩码: 8 维, 值域 [0,1], 无需额外归一化
        processed['safety_mask'] = torch.tensor(obs['safety_mask'], dtype=torch.float32).unsqueeze(0)
        return processed

    def sample_random_action(self):
        # [论文] Discrete(8): 返回长度为 1 的 int64 tensor
        a = self._env.action_space.sample()
        return torch.tensor([int(a)], dtype=torch.int64)

    def close(self):
        self._env.close()


# ============================================================================
#  Other Env Wrappers (ControlSuite / Gym — unchanged)
# ============================================================================

class ControlSuiteEnv:
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
        from dm_control import suite
        from dm_control.suite.wrappers import pixels
        domain, task = env.split('-')
        self._env = suite.load(domain, task, task_kwargs={'random': seed})
        if not symbolic: self._env = pixels.Wrapper(self._env)
        self.max_episode_length, self.action_repeat, self.bit_depth = max_episode_length, action_repeat, bit_depth
        self.symbolic = symbolic

    def reset(self):
        self.t = 0
        state = self._env.reset()
        if self.symbolic: return torch.tensor(
            np.concatenate([np.asarray([o]) if isinstance(o, float) else o for o in state.observation.values()],
                           axis=0), dtype=torch.float32).unsqueeze(0)
        return _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)

    def step(self, action):
        reward = 0
        for _ in range(self.action_repeat):
            state = self._env.step(action.detach().numpy())
            reward += state.reward
            self.t += 1
            if state.last() or self.t == self.max_episode_length: break
        if self.symbolic:
            obs = torch.tensor(
                np.concatenate([np.asarray([o]) if isinstance(o, float) else o for o in state.observation.values()],
                               axis=0), dtype=torch.float32).unsqueeze(0)
        else:
            obs = _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)
        return obs, reward, (state.last() or self.t == self.max_episode_length)

    def render(self):
        pass

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return sum([(1 if len(o.shape) == 0 else o.shape[0]) for o in
                    self._env.observation_spec().values()]) if self.symbolic else (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_spec().shape[0]

    def sample_random_action(self):
        spec = self._env.action_spec()
        return torch.from_numpy(np.random.uniform(spec.minimum, spec.maximum, spec.shape))


class GymEnv:
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
        self.symbolic, self.max_episode_length, self.action_repeat, self.bit_depth = symbolic, max_episode_length, action_repeat, bit_depth
        try:
            import gymnasium; self._env = gymnasium.make(env, render_mode='rgb_array'); self.is_gym = True
        except:
            import gym; self._env = gym.make(env); self._env.seed(seed); self.is_gym = False
        self.seed = seed

    def reset(self):
        self.t = 0
        state = self._env.reset(seed=self.seed)[0] if self.is_gym else self._env.reset()
        if self.symbolic: return torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return _images_to_observation(self._env.render() if self.is_gym else self._env.render(mode='rgb_array'),
                                      self.bit_depth)

    def step(self, action):
        reward = 0
        for _ in range(self.action_repeat):
            if self.is_gym:
                state, r, term, trunc, _ = self._env.step(action.detach().numpy()); done = term or trunc
            else:
                state, r, done, _ = self._env.step(action.detach().numpy())
            reward += r;
            self.t += 1
            if done or self.t == self.max_episode_length: break
        if self.symbolic: return torch.tensor(state, dtype=torch.float32).unsqueeze(0), reward, done
        return _images_to_observation(self._env.render() if self.is_gym else self._env.render(mode='rgb_array'),
                                      self.bit_depth), reward, done

    def render(self):
        pass

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return self._env.observation_space.shape[0] if self.symbolic else (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    def sample_random_action(self):
        return torch.from_numpy(self._env.action_space.sample())


# ============================================================================
#  Factory Function
# ============================================================================

def Env(env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
    if env == 'UAV-v0':
        db_cfg = {
            'user': 'root', 'password': 'Wqw030221',
            'host': 'localhost', 'database': 'senmap',
            'raise_on_warnings': True
        }
        return UAVEnvWrapper(
            UAVNavigationEnv(max_steps=max_episode_length, db_config=db_cfg,
                             siamese_model_path='siamese_model.pth'),
            bit_depth,
            action_repeat=action_repeat,
        )
    elif env in GYM_ENVS:
        return GymEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)
    elif env in CONTROL_SUITE_ENVS:
        return ControlSuiteEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)


# ============================================================================
#  EnvBatcher (unchanged)
# ============================================================================

class EnvBatcher:
    def __init__(self, env_class, env_args, env_kwargs, n):
        self.n, self.envs, self.dones = n, [env_class(*env_args, **env_kwargs) for _ in range(n)], [True] * n

    def reset(self):
        obs = [e.reset() for e in self.envs];
        self.dones = [False] * self.n
        if isinstance(obs[0], dict): return {k: torch.cat([o[k] for o in obs], 0) for k in obs[0]}
        return torch.cat(obs)

    def step(self, actions):
        obs, rs, ds = zip(*[e.step(a) for e, a in zip(self.envs, actions)])
        self.dones = [d or pd for d, pd in zip(ds, self.dones)]
        done_mask = torch.tensor(self.dones)
        if isinstance(obs[0], dict):
            final_obs = {k: torch.cat([o[k] for o in obs], 0) for k in obs[0]}
            # [修复] 对 dict 观测也需置零已结束环境的数据, 防止残留污染
            if done_mask.any():
                for k in final_obs:
                    final_obs[k][done_mask] = 0
        else:
            final_obs = torch.cat(obs)
            final_obs[done_mask] = 0
        return final_obs, torch.tensor(rs, dtype=torch.float32), torch.tensor(ds, dtype=torch.uint8)

    def close(self):
        [e.close() for e in self.envs]