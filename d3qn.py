"""
d3qn.py — Dueling Double DQN Baseline for UAV Navigation
=================================================================
对齐 main.py 的世界模型代码, 提供 D3QN 作为第一个 baseline.

与 WM 完全一致的部分:
  1. 环境 (UAVEnvWrapper, UAV-v0)
  2. 字典观测接口 (image / target / position / semantic_map / safety_mask)
  3. 训练-评测-采样循环结构 (seed → train → eval → collect → ckpt)
  4. metrics 字典键名, lineplot/TensorBoard 写入格式
  5. 评测指标口径: SR / CR / APLR / MinClr / Reward (in-domain + OOD)
  6. 论文 Table I 风格最终评测 (每集详表 + 平均行)

D3QN 专属:
  - 损失指标用 q_loss 代替 WM 的 9 项损失; 不生成世界模型预测指标
    (Occ-IoU / ADE / FDE 在最终 Table I 中打印为 "-", 标注 N/A for D3QN).
  - 一步 TD 更新; 经验回放按单步 (s, a, r, s', done).

Usage:
    python d3qn.py --id d3qn_run1 --episodes 1000
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from env import CONTROL_SUITE_ENVS, GYM_ENVS, Env, EnvBatcher, postprocess_observation, preprocess_observation_, get_eval_task_list
from utils import lineplot


# ============================================================================
#  Args
# ============================================================================
parser = argparse.ArgumentParser(description='D3QN Baseline for UAV Navigation')
parser.add_argument('--id', type=str, default='d3qn', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S')
parser.add_argument('--disable-cuda', action='store_true')
parser.add_argument('--env', type=str, default='UAV-v0',
                    choices=GYM_ENVS + CONTROL_SUITE_ENVS + ['UAV-v0'])
parser.add_argument('--symbolic-env', action='store_true')
parser.add_argument('--max-episode-length', type=int, default=100)
parser.add_argument('--experience-size', type=int, default=1000000,
                    help='Replay buffer size')
parser.add_argument('--action-repeat', type=int, default=1)
parser.add_argument('--bit-depth', type=int, default=5)
parser.add_argument('--episodes', type=int, default=1000, metavar='E')
parser.add_argument('--seed-episodes', type=int, default=5)
parser.add_argument('--collect-interval', type=int, default=100,
                    help='Gradient steps per episode')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--hidden-size', type=int, default=256)
parser.add_argument('--learning-rate', type=float, default=1e-4)
parser.add_argument('--adam-epsilon', type=float, default=1e-7)
parser.add_argument('--grad-clip-norm', type=float, default=10.0)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--target-update', type=int, default=1000,
                    help='Target network update interval (gradient steps)')
parser.add_argument('--eps-start', type=float, default=1.0)
parser.add_argument('--eps-end', type=float, default=0.05)
parser.add_argument('--eps-decay-episodes', type=int, default=300,
                    help='Linear decay episodes (ε_start → ε_end)')
parser.add_argument('--test', action='store_true')
parser.add_argument('--test-interval', type=int, default=25)
parser.add_argument('--test-episodes', type=int, default=10)
parser.add_argument('--checkpoint-interval', type=int, default=25)
parser.add_argument('--models', type=str, default='')
parser.add_argument('--render', action='store_true')
args = parser.parse_args()

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))


# ============================================================================
#  Setup
# ============================================================================
results_dir = os.path.join('results', '{}_{}'.format(args.env, args.id))
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available() and not args.disable_cuda:
    print("Using CUDA")
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(args.seed)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    print("Using CPU")
    args.device = torch.device('cpu')

# 指标 dict — 和 WM 一致的键结构 (仅 WM 特有损失改为 q_loss)
metrics = {
    'steps': [], 'episodes': [], 'train_rewards': [],
    'test_episodes': [], 'test_rewards': [], 'test_avg_rewards': [],
    'q_loss': [],
}

writer = SummaryWriter(results_dir + "/{}_{}_log".format(args.env, args.id))
print("Writer is ready.")

# Env
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length,
          args.action_repeat, args.bit_depth)
if env is None:
    raise ValueError(f"Environment '{args.env}' not found.")
print("Environment is loaded.")


# ============================================================================
#  Replay Buffer (按单步转移; dict 观测紧凑存储)
# ============================================================================
class D3QNReplayBuffer:
    """
    单步 (s, a, r, s', done) 回放池.
    dict 观测: image/target 存 uint8, semantic_map 存 float16, 其余 float32.
    这样 200k 条数据下内存可控.
    """
    def __init__(self, size, obs_space, action_size, bit_depth, device):
        self.size = size
        self.device = device
        self.bit_depth = bit_depth
        self.action_size = action_size

        sem_shape = obs_space['semantic_map'].shape   # (6, 30, 30)
        pos_shape = obs_space['position'].shape       # (2,)
        sm_shape = obs_space['safety_mask'].shape     # (8,)

        self.obs = {
            'image':        np.empty((size, 3, 64, 64), dtype=np.uint8),
            'target':       np.empty((size, 3, 64, 64), dtype=np.uint8),
            'position':     np.empty((size, *pos_shape), dtype=np.float32),
            'semantic_map': np.empty((size, *sem_shape), dtype=np.float16),
            'safety_mask':  np.empty((size, *sm_shape), dtype=np.float32),
        }
        # 下一观测 — 同样结构
        self.next_obs = {
            'image':        np.empty((size, 3, 64, 64), dtype=np.uint8),
            'target':       np.empty((size, 3, 64, 64), dtype=np.uint8),
            'position':     np.empty((size, *pos_shape), dtype=np.float32),
            'semantic_map': np.empty((size, *sem_shape), dtype=np.float16),
            'safety_mask':  np.empty((size, *sm_shape), dtype=np.float32),
        }
        self.actions = np.empty((size,), dtype=np.int64)
        self.rewards = np.empty((size,), dtype=np.float32)
        self.dones   = np.empty((size,), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.steps = 0
        self.episodes = 0

    def _obs_to_np(self, obs):
        """torch dict → numpy dict (单样本)."""
        out = {}
        for k, v in obs.items():
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy().squeeze(0)
            out[k] = v
        return out

    def append(self, obs, action, reward, next_obs, done):
        o  = self._obs_to_np(obs)
        no = self._obs_to_np(next_obs)

        # image/target: preprocess [-0.5,0.5] → uint8
        self.obs['image'][self.idx]      = postprocess_observation(o['image'],  self.bit_depth)
        self.obs['target'][self.idx]     = postprocess_observation(o['target'], self.bit_depth)
        self.obs['position'][self.idx]   = o['position']
        self.obs['semantic_map'][self.idx] = o['semantic_map']
        self.obs['safety_mask'][self.idx]  = o['safety_mask']

        self.next_obs['image'][self.idx]      = postprocess_observation(no['image'],  self.bit_depth)
        self.next_obs['target'][self.idx]     = postprocess_observation(no['target'], self.bit_depth)
        self.next_obs['position'][self.idx]   = no['position']
        self.next_obs['semantic_map'][self.idx] = no['semantic_map']
        self.next_obs['safety_mask'][self.idx]  = no['safety_mask']

        # action: int index
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        a = np.asarray(action).flatten()
        self.actions[self.idx] = int(a[0]) if a.size == 1 else int(np.argmax(a))

        self.rewards[self.idx] = float(reward)
        self.dones[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps += 1
        if done:
            self.episodes += 1

    def __len__(self):
        return self.size if self.full else self.idx

    def _fetch_dict(self, d, idxs):
        """取出 idxs 对应的一批观测, 转 torch 并预处理 image/target."""
        batch = {}
        # image / target: uint8 → float32 + preprocess_observation_
        for key in ['image', 'target']:
            raw = d[key][idxs].astype(np.float32)
            t = torch.as_tensor(raw)
            preprocess_observation_(t, self.bit_depth)
            batch[key] = t
        batch['position']     = torch.as_tensor(d['position'][idxs])
        batch['semantic_map'] = torch.as_tensor(d['semantic_map'][idxs].astype(np.float32))
        batch['safety_mask']  = torch.as_tensor(d['safety_mask'][idxs])
        return batch

    def sample(self, batch_size):
        n = len(self)
        idxs = np.random.randint(0, n, size=batch_size)
        s_batch  = self._fetch_dict(self.obs,      idxs)
        ns_batch = self._fetch_dict(self.next_obs, idxs)
        a = torch.as_tensor(self.actions[idxs], dtype=torch.int64)
        r = torch.as_tensor(self.rewards[idxs], dtype=torch.float32)
        d = torch.as_tensor(self.dones[idxs],   dtype=torch.float32)
        return s_batch, a, r, ns_batch, d


# ============================================================================
#  Dueling Q-Network (处理 dict 观测)
# ============================================================================
class DuelingQNet(nn.Module):
    """
    输入 dict 观测 → Q(s, a) ∈ R^|A|.
    融合:
        - image (3,64,64)        → CNN → 512
        - target (3,64,64)       → CNN → 512
        - semantic_map (6,30,30) → CNN → 512
        - [position(2) ‖ safety_mask(8)]  → MLP → 64
    Dueling 头: Q = V(s) + (A(s,·) − mean_a A(s,·))
    """
    def __init__(self, action_size, hidden=256):
        super().__init__()
        # 图像编码器 (64x64 → 2x2)
        def make_img_cnn():
            return nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2),  nn.ReLU(),   # 31
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),   # 14
                nn.Conv2d(64, 128, 4, stride=2),nn.ReLU(),   # 6
                nn.Conv2d(128, 128, 4, stride=2), nn.ReLU(), # 2
                nn.Flatten(),
                nn.Linear(128 * 2 * 2, 256), nn.ReLU(),
            )
        self.img_cnn = make_img_cnn()
        self.tgt_cnn = make_img_cnn()

        # 语义地图 CNN (6,30,30)
        self.map_cnn = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=2, padding=1),  nn.ReLU(),   # 15
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 8
            nn.Conv2d(64, 128, 3, stride=2, padding=1),nn.ReLU(),   # 4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
        )

        # 向量特征 (position + safety_mask)
        self.vec_mlp = nn.Sequential(
            nn.Linear(2 + 8, 64), nn.ReLU(),
            nn.Linear(64, 64),    nn.ReLU(),
        )

        feat_dim = 256 + 256 + 256 + 64  # = 832
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )

        # Dueling heads
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, obs):
        img_f = self.img_cnn(obs['image'])
        tgt_f = self.tgt_cnn(obs['target'])
        map_f = self.map_cnn(obs['semantic_map'])

        pos = obs['position']
        sm  = obs['safety_mask']
        # squeeze 可能的多余维度 (B, 1, 2) → (B, 2) etc.
        if pos.dim() == 3: pos = pos.squeeze(1)
        if sm.dim() == 3:  sm  = sm.squeeze(1)
        vec = torch.cat([pos, sm], dim=-1)
        vec_f = self.vec_mlp(vec)

        h = torch.cat([img_f, tgt_f, map_f, vec_f], dim=-1)
        h = self.fusion(h)

        V = self.value_head(h)                          # (B, 1)
        A = self.adv_head(h)                            # (B, |A|)
        Q = V + (A - A.mean(dim=-1, keepdim=True))
        return Q


# ============================================================================
#  D3QN Agent
# ============================================================================
class D3QNAgent:
    def __init__(self, action_size, device, hidden=256, lr=1e-4,
                 gamma=0.99, target_update=1000, grad_clip_norm=10.0,
                 adam_epsilon=1e-7):
        self.device = device
        self.action_size = action_size
        self.gamma = gamma
        self.target_update = target_update
        self.grad_clip_norm = grad_clip_norm

        self.q_net = DuelingQNet(action_size, hidden).to(device)
        self.target_q_net = DuelingQNet(action_size, hidden).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_q_net.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.q_net.parameters(),
                                    lr=lr, eps=adam_epsilon)
        self.update_count = 0

    def _obs_to_device(self, obs):
        return {k: v.to(self.device, non_blocking=True) for k, v in obs.items()}

    @torch.no_grad()
    def select_action(self, obs, epsilon=0.0):
        """ε-greedy 选择动作. obs: 单样本 dict (已在 CPU, batch=1)."""
        if np.random.rand() < epsilon:
            return torch.tensor([np.random.randint(self.action_size)],
                                dtype=torch.int64)
        obs_d = self._obs_to_device(obs)
        q = self.q_net(obs_d)
        a = q.argmax(dim=-1).detach().cpu()   # (1,)
        return a.to(torch.int64)

    def update(self, batch):
        s, a, r, s_next, d = batch
        s      = self._obs_to_device(s)
        s_next = self._obs_to_device(s_next)
        a = a.to(self.device)
        r = r.to(self.device)
        d = d.to(self.device)

        # Online Q(s, a)
        q_sa = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)  # (B,)

        # Double DQN: online 选 argmax, target 估 value
        with torch.no_grad():
            next_a = self.q_net(s_next).argmax(dim=-1, keepdim=True)   # (B,1)
            next_q = self.target_q_net(s_next).gather(1, next_a).squeeze(1)
            target = r + self.gamma * next_q * (1.0 - d)

        loss = F.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())


# ============================================================================
#  Replay buffer 初始化 + seed episodes
# ============================================================================
D = D3QNReplayBuffer(args.experience_size, env.observation_size,
                     env.action_size, args.bit_depth, args.device)

if not args.test:
    for s in range(1, args.seed_episodes + 1):
        observation, done, t = env.reset(), False, 0
        while not done:
            action = env.sample_random_action()
            next_observation, reward, done = env.step(action)
            D.append(observation, action, reward, next_observation, done)
            observation = next_observation
            t += 1
        metrics['steps'].append(t * args.action_repeat +
                                (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
        metrics['episodes'].append(s)
print("Experience replay buffer is ready. (size={})".format(len(D)))


# ============================================================================
#  Agent
# ============================================================================
agent = D3QNAgent(
    action_size=env.action_size,
    device=args.device,
    hidden=args.hidden_size,
    lr=args.learning_rate,
    gamma=args.gamma,
    target_update=args.target_update,
    grad_clip_norm=args.grad_clip_norm,
    adam_epsilon=args.adam_epsilon,
)

# 加载预训练 (可选)
if args.models != '' and os.path.exists(args.models):
    print("Loading pre-trained D3QN weights")
    ckpt = torch.load(args.models, map_location=args.device, weights_only=True)
    agent.q_net.load_state_dict(ckpt['q_net'])
    agent.target_q_net.load_state_dict(ckpt['target_q_net'])
    agent.optimizer.load_state_dict(ckpt['optimizer'])

print("D3QN agent is ready.")


def current_epsilon(ep):
    """线性衰减 ε_start → ε_end, 之后保持 ε_end."""
    if args.eps_decay_episodes <= 0:
        return args.eps_end
    frac = min(1.0, max(0.0, (ep - 1) / args.eps_decay_episodes))
    return args.eps_start + frac * (args.eps_end - args.eps_start)


# ============================================================================
#  Episode runner (测试 / 收集 都复用)
# ============================================================================
def run_episode(env, agent, epsilon, train=False):
    """
    运行一个 episode, 返回:
        total_reward, steps, reached, collided, path_length, min_clearance, start_pos
    若 train=True, 同时写入 replay buffer.
    """
    observation = env.reset() if not train else env.reset()  # train 用默认 reset
    done = False
    total_reward = 0.0
    path_length = 0.0
    min_clearance = float('inf')
    reached, collided = False, False
    t = 0

    _inner = env._env if hasattr(env, '_env') else env
    start_pos = _inner.agent_pos.copy() if hasattr(_inner, 'agent_pos') else None

    while not done:
        action = agent.select_action(observation, epsilon=epsilon)
        next_observation, reward, done = env.step(action)

        r_val = reward.item() if torch.is_tensor(reward) else float(reward)
        total_reward += r_val
        path_length += 1.0

        # 最小间距 (与 main.py 一致的口径)
        if hasattr(_inner, 'obstacles') and hasattr(_inner, 'agent_pos'):
            for obs_j in _inner.obstacles:
                d = float(np.linalg.norm(_inner.agent_pos - obs_j.q))
                if d < min_clearance:
                    min_clearance = d

        # 到达/碰撞判定 (优先 info dict, 回退 reward 阈值)
        _info = getattr(_inner, '_last_info', {}) or {}
        if not _info:
            _rt = getattr(_inner, 'reward_reach', 100.0)
            _rc = getattr(_inner, 'reward_collision', -10.0)
            if r_val >= _rt * 0.9:
                _info = {'reach': True}
            if r_val <= _rc * 0.9:
                _info['collision'] = True
        if _info.get('reach', False):
            reached = True
        if _info.get('collision', False):
            collided = True

        if train:
            D.append(observation, action, reward, next_observation, done)

        observation = next_observation
        t += 1
        if args.render and train:
            env.render()

    return {
        'total_reward': total_reward, 'steps': t,
        'reached': reached, 'collided': collided,
        'path_length': path_length, 'min_clearance': min_clearance,
        'start_pos': start_pos,
    }


def run_test_episode(env, agent, map_seed, ood):
    """评测单集 (ε=0)."""
    observation = env.reset(map_seed=map_seed, ood=ood)
    done = False
    ep_reward = 0.0
    path_length = 0.0
    min_clearance = float('inf')
    reached, collided = False, False

    _inner = env._env if hasattr(env, '_env') else env
    start_pos = _inner.agent_pos.copy() if hasattr(_inner, 'agent_pos') else None

    while not done:
        action = agent.select_action(observation, epsilon=0.0)
        next_observation, reward, done = env.step(action)

        r_val = reward.item() if torch.is_tensor(reward) else float(reward)
        ep_reward += r_val
        path_length += 1.0

        if hasattr(_inner, 'obstacles') and hasattr(_inner, 'agent_pos'):
            for obs_j in _inner.obstacles:
                d = float(np.linalg.norm(_inner.agent_pos - obs_j.q))
                if d < min_clearance:
                    min_clearance = d

        _info = getattr(_inner, '_last_info', {}) or {}
        if not _info:
            _rt = getattr(_inner, 'reward_reach', 100.0)
            _rc = getattr(_inner, 'reward_collision', -10.0)
            if r_val >= _rt * 0.9:
                _info = {'reach': True}
            if r_val <= _rc * 0.9:
                _info['collision'] = True
        if _info.get('reach', False):
            reached = True
        if _info.get('collision', False):
            collided = True

        observation = next_observation

    # 单集 APLR: 到达时为 路径步数 / 最短切比雪夫距离, 未到达 NaN
    ep_aplr = float('nan')
    if reached and start_pos is not None and hasattr(_inner, 'target_pos'):
        _tgt = _inner.target_pos
        _shortest = max(abs(start_pos[0] - _tgt[0]),
                        abs(start_pos[1] - _tgt[1])) / _inner.cell_size
        if _shortest > 0:
            ep_aplr = path_length / _shortest

    return {
        'SR':     100.0 if reached else 0.0,
        'CR':     100.0 if collided else 0.0,
        'APLR':   ep_aplr,
        'MinClr': (min_clearance if min_clearance < float('inf') else 0.0),
        'Reward': ep_reward,
    }


# ============================================================================
#  Training Loop
# ============================================================================
start_episode = metrics['episodes'][-1] + 1 if metrics['episodes'] else 1
for episode in tqdm(range(start_episode, args.episodes + 1),
                    total=args.episodes, initial=start_episode - 1):

    print(f"Training loop EP:{episode}")

    # ── 1. 梯度更新 (collect_interval 步) ───────────────────────────────────
    losses = []
    if len(D) >= args.batch_size:
        agent.q_net.train()
        for s in tqdm(range(args.collect_interval), leave=False):
            batch = D.sample(args.batch_size)
            loss = agent.update(batch)
            losses.append(loss)

    mean_q_loss = float(np.mean(losses)) if losses else 0.0
    metrics['q_loss'].append(mean_q_loss)

    lineplot(metrics['episodes'][-len(metrics['q_loss']):],
             metrics['q_loss'], 'Q Loss', results_dir)
    writer.add_scalar('Loss/q', mean_q_loss, episode)

    # ── 2. 评测 (每 test_interval 集) ───────────────────────────────────────
    if episode % args.test_interval == 0:
        agent.q_net.eval()

        eval_results = {}
        # [测试协议统一] 种子列表从 env.get_eval_task_list 获取,
        # 与 WM 主 main 及其他基线保持严格一致 (同 map + 同起终点 +
        # 同障碍物). 修改协议只改 env.get_eval_task_list 一处.
        for eval_mode, mode_tasks in get_eval_task_list(n_episodes=args.test_episodes).items():
            ep_records = []
            for task in mode_tasks:
                rec = run_test_episode(env, agent,
                                       map_seed=task['map_seed'], ood=task['ood'])
                ep_records.append(rec)

            sr     = float(np.mean([r['SR']     for r in ep_records]))
            cr     = float(np.mean([r['CR']     for r in ep_records]))
            aplrs  = [r['APLR']  for r in ep_records if not np.isnan(r['APLR'])]
            aplr   = float(np.mean(aplrs)) if aplrs else 0.0
            mclrs  = [r['MinClr'] for r in ep_records if r['MinClr'] > 0]
            minclr = float(np.mean(mclrs)) if mclrs else 0.0
            avg_rw = float(np.mean([r['Reward'] for r in ep_records]))

            eval_results[eval_mode] = {
                'SR': sr, 'CR': cr, 'APLR': aplr,
                'MinClr': minclr, 'Reward': avg_rw,
            }

            steps_now = metrics['steps'][-1] if metrics['steps'] else 0
            writer.add_scalar(f'Eval/{eval_mode}_SR',     sr,     steps_now)
            writer.add_scalar(f'Eval/{eval_mode}_CR',     cr,     steps_now)
            writer.add_scalar(f'Eval/{eval_mode}_APLR',   aplr,   steps_now)
            writer.add_scalar(f'Eval/{eval_mode}_MinClr', minclr, steps_now)
            writer.add_scalar(f'Eval/{eval_mode}_Reward', avg_rw, steps_now)

        _avg_test_reward = float(np.mean(
            [eval_results[m]['Reward'] for m in eval_results]))
        writer.add_scalar('Eval/Average_Test_Rewards', _avg_test_reward,
                          metrics['steps'][-1] if metrics['steps'] else 0)

        for mode, res in eval_results.items():
            print(f"  [{mode}] SR={res['SR']:.1f}% CR={res['CR']:.1f}% "
                  f"APLR={res['APLR']:.2f} MinClr={res['MinClr']:.0f} "
                  f"Reward={res['Reward']:.1f}")
        print(f"  [Average Test Rewards] {_avg_test_reward:.1f}")

        # 存入 metrics
        metrics['test_episodes'].append(episode)
        for mode in ['in_domain', 'ood']:
            for key in ['SR', 'CR', 'APLR', 'MinClr', 'Reward']:
                mk = f'test_{mode}_{key}'
                if mk not in metrics:
                    metrics[mk] = []
                metrics[mk].append(eval_results[mode][key])

        metrics['test_rewards'].append(eval_results['in_domain']['SR'])
        metrics['test_avg_rewards'].append(eval_results['in_domain']['Reward'])

        lineplot(metrics['test_episodes'],
                 metrics.get('test_in_domain_SR', []), 'SR_InDomain', results_dir)
        lineplot(metrics['test_episodes'],
                 metrics.get('test_ood_SR', []),       'SR_OOD',       results_dir)
        lineplot(metrics['test_episodes'],
                 metrics.get('test_in_domain_CR', []), 'CR_InDomain', results_dir)
        lineplot(metrics['test_episodes'],
                 metrics.get('test_ood_CR', []),       'CR_OOD',       results_dir)
        lineplot(metrics['test_episodes'],
                 metrics.get('test_in_domain_Reward', []),
                 'Reward_InDomain', results_dir)
        lineplot(metrics['test_episodes'],
                 metrics.get('test_ood_Reward', []),
                 'Reward_OOD', results_dir)
        # [补图] APLR / MinClr 在 in-domain + OOD 两域都出图,
        # 与 main_wm_d3qn.py / main.py 保持完全一致的图文件清单.
        lineplot(metrics['test_episodes'],
                 metrics.get('test_in_domain_APLR', []),
                 'APLR_InDomain', results_dir)
        lineplot(metrics['test_episodes'],
                 metrics.get('test_ood_APLR', []),
                 'APLR_OOD', results_dir)
        lineplot(metrics['test_episodes'],
                 metrics.get('test_in_domain_MinClr', []),
                 'MinClr_InDomain', results_dir)
        lineplot(metrics['test_episodes'],
                 metrics.get('test_ood_MinClr', []),
                 'MinClr_OOD', results_dir)
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        agent.q_net.train()

    # ── 3. 采集新一集 (ε-greedy) ───────────────────────────────────────────
    eps = current_epsilon(episode)
    with torch.no_grad():
        agent.q_net.eval()
        ep_stats = run_episode(env, agent, epsilon=eps, train=True)
        agent.q_net.train()

    total_reward = ep_stats['total_reward']
    metrics['steps'].append(ep_stats['steps'] + metrics['steps'][-1])
    metrics['episodes'].append(episode)
    metrics['train_rewards'].append(total_reward)

    writer.add_scalar('Train/episode_reward', total_reward, episode)
    writer.add_scalar('Train/epsilon', eps, episode)
    writer.add_scalar('Train/episode_length', ep_stats['steps'], episode)

    lineplot(metrics['episodes'][-len(metrics['train_rewards']):],
             metrics['train_rewards'], 'Train Rewards', results_dir)

    print(f"  ep={episode} reward={total_reward:.1f} steps={ep_stats['steps']} "
          f"reached={ep_stats['reached']} collided={ep_stats['collided']} "
          f"ε={eps:.3f} q_loss={mean_q_loss:.4f}")

    # ── 4. Checkpoint ─────────────────────────────────────────────────────
    if episode % args.checkpoint_interval == 0:
        torch.save({
            'q_net':        agent.q_net.state_dict(),
            'target_q_net': agent.target_q_net.state_dict(),
            'optimizer':    agent.optimizer.state_dict(),
            'episode':      episode,
            'update_count': agent.update_count,
        }, os.path.join(results_dir, f'models_{episode}.pth'))


# ============================================================================
#  POST-TRAINING FINAL EVALUATION  (论文 Table I 风格)
# ============================================================================
print("\n" + "=" * 92)
print(" " * 26 + "POST-TRAINING FINAL EVALUATION  (D3QN)")
print("=" * 92)

agent.q_net.eval()

# WM 专属预测指标: 对 D3QN 无定义 → 置 NaN, 打印为 "-"
final_occ_iou = float('nan')
final_ade     = float('nan')
final_fde     = float('nan')

_final_eval_results = {}
# [测试协议统一] 同上.
for _eval_mode, _mode_tasks in get_eval_task_list(n_episodes=args.test_episodes).items():
    _per_ep_records = []
    for _task in _mode_tasks:
        rec = run_test_episode(env, agent,
                               map_seed=_task['map_seed'], ood=_task['ood'])
        _per_ep_records.append(rec)
    _final_eval_results[_eval_mode] = _per_ep_records


# ── 打印 Table I ─────────────────────────────────────────────────────────────
_col = "{:>4} {:>8} {:>8} {:>8} {:>10} {:>10} {:>8} {:>8} {:>10}"

def _fmt(v, spec):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "   -   "
    return spec.format(v)

print("\n" + "-" * 92)
print(" In-Domain test episodes (SR-in / APLR / CR-in / MinClr)")
print("-" * 92)
print(_col.format("Ep", "SR(%)", "APLR", "CR(%)", "MinClr",
                  "Occ-IoU", "ADE", "FDE", "Reward"))
for _i, _rec in enumerate(_final_eval_results['in_domain'], 1):
    print(_col.format(
        _i,
        _fmt(_rec['SR'],     "{:8.1f}"),
        _fmt(_rec['APLR'],   "{:8.3f}"),
        _fmt(_rec['CR'],     "{:8.1f}"),
        _fmt(_rec['MinClr'], "{:10.1f}"),
        _fmt(final_occ_iou,  "{:10.3f}"),
        _fmt(final_ade,      "{:8.2f}"),
        _fmt(final_fde,      "{:8.2f}"),
        _fmt(_rec['Reward'], "{:10.1f}"),
    ))

print("-" * 92)
print(" OOD test episodes (SR-OOD / APLR / CR-OOD / MinClr)")
print("-" * 92)
print(_col.format("Ep", "SR(%)", "APLR", "CR(%)", "MinClr",
                  "Occ-IoU", "ADE", "FDE", "Reward"))
for _i, _rec in enumerate(_final_eval_results['ood'], 1):
    print(_col.format(
        _i,
        _fmt(_rec['SR'],     "{:8.1f}"),
        _fmt(_rec['APLR'],   "{:8.3f}"),
        _fmt(_rec['CR'],     "{:8.1f}"),
        _fmt(_rec['MinClr'], "{:10.1f}"),
        _fmt(final_occ_iou,  "{:10.3f}"),
        _fmt(final_ade,      "{:8.2f}"),
        _fmt(final_fde,      "{:8.2f}"),
        _fmt(_rec['Reward'], "{:10.1f}"),
    ))


# ── 汇总平均行 ──────────────────────────────────────────────────────────────
def _mean_ignore_nan(vals):
    vs = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vs)) if vs else 0.0

_id_recs = _final_eval_results['in_domain']
_od_recs = _final_eval_results['ood']

avg_SR      = _mean_ignore_nan([r['SR']     for r in _id_recs])
avg_APLR    = _mean_ignore_nan([r['APLR']   for r in _id_recs])
avg_CR      = _mean_ignore_nan([r['CR']     for r in _id_recs])
avg_MinClr  = _mean_ignore_nan([r['MinClr'] for r in _id_recs])
avg_SR_OOD  = _mean_ignore_nan([r['SR']     for r in _od_recs])
avg_CR_OOD  = _mean_ignore_nan([r['CR']     for r in _od_recs])
avg_Rew_ID  = _mean_ignore_nan([r['Reward'] for r in _id_recs])
avg_Rew_OD  = _mean_ignore_nan([r['Reward'] for r in _od_recs])
avg_Rew_all = _mean_ignore_nan([r['Reward'] for r in _id_recs + _od_recs])

print("\n" + "=" * 92)
print(" Paper Table I — D3QN Baseline — Averaged over all test episodes")
print("=" * 92)
print("  SR (%)↑          = {:>7.2f}".format(avg_SR))
print("  APLR↓            = {:>7.3f}".format(avg_APLR))
print("  CR (%)↓          = {:>7.2f}".format(avg_CR))
print("  MinClr↑          = {:>7.2f}".format(avg_MinClr))
print("  Occ-IoU↑         =    -     (N/A for D3QN)")
print("  ADE↓             =    -     (N/A for D3QN)")
print("  FDE↓             =    -     (N/A for D3QN)")
print("  SR-OOD (%)↑      = {:>7.2f}".format(avg_SR_OOD))
print("  CR-OOD (%)↓      = {:>7.2f}".format(avg_CR_OOD))
print("  --------------------------------")
print("  Avg Test Reward (In-Domain)  = {:>7.2f}".format(avg_Rew_ID))
print("  Avg Test Reward (OOD)        = {:>7.2f}".format(avg_Rew_OD))
print("  Average Test Rewards (all)   = {:>7.2f}".format(avg_Rew_all))
print("=" * 92)


# ── TensorBoard 写入 ────────────────────────────────────────────────────────
_final_step = metrics['steps'][-1] if metrics['steps'] else args.episodes

for _mode_name in ['in_domain', 'ood']:
    for _i, _rec in enumerate(_final_eval_results[_mode_name], 1):
        for _k, _v in _rec.items():
            if _v is None or (isinstance(_v, float) and np.isnan(_v)):
                continue
            writer.add_scalar(f'FinalTest/{_mode_name}_ep{_i}_{_k}',
                              float(_v), _final_step)

writer.add_scalar('FinalTest/Average_SR',       avg_SR,      _final_step)
writer.add_scalar('FinalTest/Average_APLR',     avg_APLR,    _final_step)
writer.add_scalar('FinalTest/Average_CR',       avg_CR,      _final_step)
writer.add_scalar('FinalTest/Average_MinClr',   avg_MinClr,  _final_step)
writer.add_scalar('FinalTest/Average_SR_OOD',   avg_SR_OOD,  _final_step)
writer.add_scalar('FinalTest/Average_CR_OOD',   avg_CR_OOD,  _final_step)
writer.add_scalar('FinalTest/Average_Test_Rewards',          avg_Rew_all, _final_step)
writer.add_scalar('FinalTest/Average_Test_Rewards_InDomain', avg_Rew_ID,  _final_step)
writer.add_scalar('FinalTest/Average_Test_Rewards_OOD',      avg_Rew_OD,  _final_step)

_table_text = (
    "| Method | SR (%)↑ | APLR↓ | CR (%)↓ | MinClr↑ | Occ-IoU↑ | ADE↓ | FDE↓ | SR-OOD (%)↑ | CR-OOD (%)↓ |\n"
    "|--------|---------|-------|---------|---------|----------|------|------|-------------|-------------|\n"
    f"| D3QN   | {avg_SR:.2f}   | {avg_APLR:.3f} | {avg_CR:.2f}   "
    f"| {avg_MinClr:.2f}   |    -     |  -   |  -   "
    f"| {avg_SR_OOD:.2f}       | {avg_CR_OOD:.2f}       |"
)
try:
    writer.add_text('FinalTest/PaperTableI_D3QN', _table_text, _final_step)
except Exception as _e:
    print(f"  [warn] writer.add_text failed: {_e}")

# 持久化
metrics['final_eval'] = {
    'in_domain':              _final_eval_results['in_domain'],
    'ood':                    _final_eval_results['ood'],
    'avg_SR':                 avg_SR,
    'avg_APLR':               avg_APLR,
    'avg_CR':                 avg_CR,
    'avg_MinClr':             avg_MinClr,
    'occ_iou':                None,   # N/A
    'ade':                    None,   # N/A
    'fde':                    None,   # N/A
    'avg_SR_OOD':             avg_SR_OOD,
    'avg_CR_OOD':             avg_CR_OOD,
    'avg_test_rewards':       avg_Rew_all,
    'avg_test_rewards_id':    avg_Rew_ID,
    'avg_test_rewards_ood':   avg_Rew_OD,
}
torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

writer.flush()
writer.close()
env.close()
print("\nDone.")