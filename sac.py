# Author: Qiwei Wang
"""Discrete Soft Actor-Critic baseline for UAV navigation."""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
from env import (
    CONTROL_SUITE_ENVS,
    GYM_ENVS,
    Env,
    postprocess_observation,
    preprocess_observation_,
    get_eval_task_list,
)
from utils import lineplot


parser = argparse.ArgumentParser(description='Discrete SAC Baseline for UAV Navigation')
parser.add_argument('--id', type=str, default='sac', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S')
parser.add_argument('--disable-cuda', action='store_true')
parser.add_argument('--env', type=str, default='UAV-v0',
                    choices=GYM_ENVS + CONTROL_SUITE_ENVS + ['UAV-v0'])
parser.add_argument('--symbolic-env', action='store_true')
parser.add_argument('--max-episode-length', type=int, default=1000)
parser.add_argument('--experience-size', type=int, default=500000,
                    help='Replay buffer size')
parser.add_argument('--action-repeat', type=int, default=1)
parser.add_argument('--bit-depth', type=int, default=5)
parser.add_argument('--episodes', type=int, default=1000, metavar='E')
parser.add_argument('--seed-episodes', type=int, default=5)
parser.add_argument('--collect-interval', type=int, default=100,
                    help='Gradient steps per episode')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--hidden-size', type=int, default=256)
parser.add_argument('--actor-lr', type=float, default=1e-5)
parser.add_argument('--critic-lr', type=float, default=2e-5)
parser.add_argument('--alpha-lr', type=float, default=1e-5)
parser.add_argument('--adam-epsilon', type=float, default=1e-7)
parser.add_argument('--grad-clip-norm', type=float, default=10.0)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.005,
                    help='Polyak soft-update coefficient for target critics')
parser.add_argument('--init-alpha', type=float, default=0.01,
                    help='Initial entropy temperature')
parser.add_argument('--target-entropy', type=float, default=None,
                    help='Target entropy. Default: 0.30 * log(action_size).')
parser.add_argument('--stochastic-eval', action='store_true',
                    help='Sample from policy during evaluation instead of argmax.')
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

metrics = {
    'steps': [], 'episodes': [], 'train_rewards': [],
    'test_episodes': [], 'test_rewards': [], 'test_avg_rewards': [],
    'q_loss': [], 'critic_1_loss': [], 'critic_2_loss': [],
    'actor_loss': [], 'alpha_loss': [], 'alpha': [], 'entropy': [],
    'train_success': [], 'train_collision': [],
    'train_sr': [], 'train_cr': [],
}
TRAIN_METRIC_WINDOW = 100

writer = SummaryWriter(results_dir + "/{}_{}_log".format(args.env, args.id))
print("Writer is ready.")

env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length,
          args.action_repeat, args.bit_depth)
if env is None:
    raise ValueError(f"Environment '{args.env}' not found.")
if args.env != 'UAV-v0':
    raise ValueError("sac.py implements discrete SAC for UAV-v0's 8-action space.")
print("Environment is loaded.")





class SACReplayBuffer:
    """Replay buffer for single-step SAC transitions."""

    def __init__(self, size, obs_space, action_size, bit_depth, device):
        self.size = size
        self.device = device
        self.bit_depth = bit_depth
        self.action_size = action_size

        sem_shape = obs_space['semantic_map'].shape
        pos_shape = obs_space['position'].shape
        self.obs = {
            'image': np.empty((size, 3, 64, 64), dtype=np.uint8),
            'target': np.empty((size, 3, 64, 64), dtype=np.uint8),
            'position': np.empty((size, *pos_shape), dtype=np.float32),
            'semantic_map': np.empty((size, *sem_shape), dtype=np.float16),
        }
        self.next_obs = {
            'image': np.empty((size, 3, 64, 64), dtype=np.uint8),
            'target': np.empty((size, 3, 64, 64), dtype=np.uint8),
            'position': np.empty((size, *pos_shape), dtype=np.float32),
            'semantic_map': np.empty((size, *sem_shape), dtype=np.float16),
        }
        self.actions = np.empty((size,), dtype=np.int64)
        self.rewards = np.empty((size,), dtype=np.float32)
        self.dones = np.empty((size,), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.steps = 0
        self.episodes = 0

    def _obs_to_np(self, obs):
        out = {}
        for k, v in obs.items():
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy().squeeze(0)
            out[k] = v
        return out

    def append(self, obs, action, reward, next_obs, done):
        o = self._obs_to_np(obs)
        no = self._obs_to_np(next_obs)

        self.obs['image'][self.idx] = postprocess_observation(
            o['image'], self.bit_depth)
        self.obs['target'][self.idx] = postprocess_observation(
            o['target'], self.bit_depth)
        self.obs['position'][self.idx] = o['position']
        self.obs['semantic_map'][self.idx] = o['semantic_map']

        self.next_obs['image'][self.idx] = postprocess_observation(
            no['image'], self.bit_depth)
        self.next_obs['target'][self.idx] = postprocess_observation(
            no['target'], self.bit_depth)
        self.next_obs['position'][self.idx] = no['position']
        self.next_obs['semantic_map'][self.idx] = no['semantic_map']

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
        batch = {}
        for key in ['image', 'target']:
            raw = d[key][idxs].astype(np.float32)
            t = torch.as_tensor(raw)
            preprocess_observation_(t, self.bit_depth)
            batch[key] = t
        batch['position'] = torch.as_tensor(d['position'][idxs])
        batch['semantic_map'] = torch.as_tensor(
            d['semantic_map'][idxs].astype(np.float32))
        return batch

    def sample(self, batch_size):
        n = len(self)
        idxs = np.random.randint(0, n, size=batch_size)
        s_batch = self._fetch_dict(self.obs, idxs)
        ns_batch = self._fetch_dict(self.next_obs, idxs)
        a = torch.as_tensor(self.actions[idxs], dtype=torch.int64)
        r = torch.as_tensor(self.rewards[idxs], dtype=torch.float32)
        d = torch.as_tensor(self.dones[idxs], dtype=torch.float32)
        return s_batch, a, r, ns_batch, d





class DictObsEncoder(nn.Module):
    """Encoder for UAV dictionary observations."""

    def __init__(self, hidden=256):
        super().__init__()

        def make_img_cnn():
            return nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
                nn.Conv2d(128, 128, 4, stride=2), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128 * 2 * 2, 256), nn.ReLU(),
            )

        self.img_cnn = make_img_cnn()
        self.tgt_cnn = make_img_cnn()
        self.map_cnn = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
        )
        self.vec_mlp = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256 + 256 + 64, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

    def forward(self, obs):
        img_f = self.img_cnn(obs['image'])
        tgt_f = self.tgt_cnn(obs['target'])
        map_f = self.map_cnn(obs['semantic_map'])

        pos = obs['position']
        if pos.dim() == 3:
            pos = pos.squeeze(1)
        x = pos[:, 0:1]
        y = pos[:, 1:2]
        dist_x = torch.minimum(x, 1.0 - x)
        dist_y = torch.minimum(y, 1.0 - y)
        danger_x = (1.0 - 2.0 * dist_x).pow(2)
        danger_y = (1.0 - 2.0 * dist_y).pow(2)
        pos_feat = torch.cat([pos, dist_x, dist_y, danger_x, danger_y], dim=-1)
        vec_f = self.vec_mlp(pos_feat)

        return self.fusion(torch.cat([img_f, tgt_f, map_f, vec_f], dim=-1))


class PolicyNet(nn.Module):
    """Categorical policy network for discrete actions."""

    def __init__(self, action_size, hidden=256):
        super().__init__()
        self.encoder = DictObsEncoder(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, obs):
        return F.softmax(self.head(self.encoder(obs)), dim=-1)


class QValueNet(nn.Module):
    """Critic network that predicts Q-values for all actions."""

    def __init__(self, action_size, hidden=256):
        super().__init__()
        self.encoder = DictObsEncoder(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, obs):
        return self.head(self.encoder(obs))


class SACAgent:
    def __init__(self, action_size, device, hidden=256,
                 actor_lr=1e-5, critic_lr=2e-5, alpha_lr=1e-5,
                 init_alpha=0.01, target_entropy=None, tau=0.005,
                 gamma=0.99, grad_clip_norm=10.0, adam_epsilon=1e-7):
        self.device = device
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.grad_clip_norm = grad_clip_norm
        self.target_entropy = (
            0.30 * np.log(action_size) if target_entropy is None
            else float(target_entropy)
        )

        self.actor = PolicyNet(action_size, hidden).to(device)
        self.critic_1 = QValueNet(action_size, hidden).to(device)
        self.critic_2 = QValueNet(action_size, hidden).to(device)
        self.target_critic_1 = QValueNet(action_size, hidden).to(device)
        self.target_critic_2 = QValueNet(action_size, hidden).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        for p in self.target_critic_1.parameters():
            p.requires_grad = False
        for p in self.target_critic_2.parameters():
            p.requires_grad = False

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_lr, eps=adam_epsilon)
        self.critic_1_optimizer = optim.Adam(
            self.critic_1.parameters(), lr=critic_lr, eps=adam_epsilon)
        self.critic_2_optimizer = optim.Adam(
            self.critic_2.parameters(), lr=critic_lr, eps=adam_epsilon)

        init_alpha = max(float(init_alpha), 1e-8)
        self.log_alpha = torch.tensor(
            np.log(init_alpha), dtype=torch.float32, device=device,
            requires_grad=True)
        self.log_alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=alpha_lr, eps=adam_epsilon)
        self.update_count = 0

    def _obs_to_device(self, obs):
        return {k: v.to(self.device, non_blocking=True) for k, v in obs.items()}

    def train(self):
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()

    def eval(self):
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()

    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        obs_d = self._obs_to_device(obs)
        probs = self.actor(obs_d)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = torch.distributions.Categorical(probs).sample()
        return action.detach().cpu().to(torch.int64)

    def calc_target(self, rewards, next_obs, dones):
        next_probs = self.actor(next_obs)
        next_log_probs = torch.log(next_probs + 1e-8)
        next_entropy = -torch.sum(
            next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_obs)
        q2_value = self.target_critic_2(next_obs)
        min_qvalue = torch.sum(
            next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * next_entropy
        return rewards + self.gamma * next_value * (1.0 - dones)

    def soft_update(self, net, target_net):
        with torch.no_grad():
            for param_target, param in zip(target_net.parameters(),
                                           net.parameters()):
                param_target.data.mul_(1.0 - self.tau)
                param_target.data.add_(param.data, alpha=self.tau)

    def update(self, batch):
        s, a, r, s_next, d = batch
        s = self._obs_to_device(s)
        s_next = self._obs_to_device(s_next)
        a = a.to(self.device).view(-1, 1)
        r = r.to(self.device).view(-1, 1)
        d = d.to(self.device).view(-1, 1)

        with torch.no_grad():
            td_target = self.calc_target(r, s_next, d)

        critic_1_q = self.critic_1(s).gather(1, a)
        critic_2_q = self.critic_2(s).gather(1, a)
        critic_1_loss = F.mse_loss(critic_1_q, td_target)
        critic_2_loss = F.mse_loss(critic_2_q, td_target)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_1.parameters(),
                                 self.grad_clip_norm)
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_2.parameters(),
                                 self.grad_clip_norm)
        self.critic_2_optimizer.step()

        probs = self.actor(s)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        with torch.no_grad():
            q1_value = self.critic_1(s)
            q2_value = self.critic_2(s)
        min_qvalue = torch.sum(
            probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()

        alpha_loss = torch.mean(
            (entropy.detach() - self.target_entropy) * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        self.update_count += 1

        q_loss = 0.5 * (critic_1_loss.item() + critic_2_loss.item())
        return {
            'q_loss': float(q_loss),
            'critic_1_loss': float(critic_1_loss.item()),
            'critic_2_loss': float(critic_2_loss.item()),
            'actor_loss': float(actor_loss.item()),
            'alpha_loss': float(alpha_loss.item()),
            'alpha': float(self.log_alpha.exp().detach().cpu().item()),
            'entropy': float(entropy.mean().detach().cpu().item()),
        }





D = SACReplayBuffer(args.experience_size, env.observation_size,
                    env.action_size, args.bit_depth, args.device)

if not args.test:
    for s in range(1, args.seed_episodes + 1):
        observation, done, t = env.reset(), False, 0
        seed_reached = False
        seed_collided = False
        seed_inner = env._env if hasattr(env, '_env') else env
        while not done:
            action = env.sample_random_action()
            next_observation, reward, done = env.step(action)
            D.append(observation, action, reward, next_observation, done)
            observation = next_observation
            t += 1
            info = getattr(seed_inner, '_last_info', {}) or {}
            seed_reached = seed_reached or info.get('reach', False)
            seed_collided = seed_collided or info.get('collision', False)
        prev_steps = metrics['steps'][-1] if metrics['steps'] else 0
        metrics['steps'].append(t * args.action_repeat + prev_steps)
        metrics['episodes'].append(s)
        metrics['train_success'].append(1 if seed_reached else 0)
        metrics['train_collision'].append(1 if seed_collided else 0)
print("Experience replay buffer is ready. (size={})".format(len(D)))





agent = SACAgent(
    action_size=env.action_size,
    device=args.device,
    hidden=args.hidden_size,
    actor_lr=args.actor_lr,
    critic_lr=args.critic_lr,
    alpha_lr=args.alpha_lr,
    init_alpha=args.init_alpha,
    target_entropy=args.target_entropy,
    tau=args.tau,
    gamma=args.gamma,
    grad_clip_norm=args.grad_clip_norm,
    adam_epsilon=args.adam_epsilon,
)

if args.models != '' and os.path.exists(args.models):
    print("Loading pre-trained SAC weights")
    ckpt = torch.load(args.models, map_location=args.device, weights_only=True)
    agent.actor.load_state_dict(ckpt['actor'])
    agent.critic_1.load_state_dict(ckpt['critic_1'])
    agent.critic_2.load_state_dict(ckpt['critic_2'])
    agent.target_critic_1.load_state_dict(ckpt['target_critic_1'])
    agent.target_critic_2.load_state_dict(ckpt['target_critic_2'])
    agent.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
    agent.critic_1_optimizer.load_state_dict(ckpt['critic_1_optimizer'])
    agent.critic_2_optimizer.load_state_dict(ckpt['critic_2_optimizer'])
    agent.log_alpha.data.copy_(ckpt['log_alpha'].to(args.device))
    agent.log_alpha_optimizer.load_state_dict(ckpt['log_alpha_optimizer'])
    agent.update_count = int(ckpt.get('update_count', 0))

print("SAC agent is ready.")





def run_episode(env, agent, train=False, deterministic=False):
    observation = env.reset()
    done = False
    total_reward = 0.0
    path_length = 0.0
    min_clearance = float('inf')
    reached, collided = False, False
    t = 0

    inner = env._env if hasattr(env, '_env') else env
    start_pos = inner.agent_pos.copy() if hasattr(inner, 'agent_pos') else None

    while not done:
        action = agent.select_action(observation, deterministic=deterministic)
        next_observation, reward, done = env.step(action)

        r_val = reward.item() if torch.is_tensor(reward) else float(reward)
        total_reward += r_val
        path_length += 1.0

        if hasattr(inner, 'obstacles') and hasattr(inner, 'agent_pos'):
            for obs_j in inner.obstacles:
                d = float(np.linalg.norm(inner.agent_pos - obs_j.q))
                if d < min_clearance:
                    min_clearance = d

        info = getattr(inner, '_last_info', {}) or {}
        reached = reached or info.get('reach', False)
        collided = collided or info.get('collision', False)

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
    observation = env.reset(map_seed=map_seed, ood=ood)
    done = False
    ep_reward = 0.0
    path_length = 0.0
    min_clearance = float('inf')
    reached, collided = False, False

    inner = env._env if hasattr(env, '_env') else env
    start_pos = inner.agent_pos.copy() if hasattr(inner, 'agent_pos') else None

    while not done:
        action = agent.select_action(
            observation, deterministic=not args.stochastic_eval)
        next_observation, reward, done = env.step(action)

        r_val = reward.item() if torch.is_tensor(reward) else float(reward)
        ep_reward += r_val
        path_length += 1.0

        if hasattr(inner, 'obstacles') and hasattr(inner, 'agent_pos'):
            for obs_j in inner.obstacles:
                d = float(np.linalg.norm(inner.agent_pos - obs_j.q))
                if d < min_clearance:
                    min_clearance = d

        info = getattr(inner, '_last_info', {}) or {}
        reached = reached or info.get('reach', False)
        collided = collided or info.get('collision', False)

        observation = next_observation

    ep_aplr = float('nan')
    if reached and start_pos is not None and hasattr(inner, 'target_pos'):
        target = inner.target_pos
        shortest = max(abs(start_pos[0] - target[0]),
                       abs(start_pos[1] - target[1])) / inner.cell_size
        if shortest > 0:
            ep_aplr = path_length / shortest

    return {
        'SR': 100.0 if reached else 0.0,
        'CR': 100.0 if collided else 0.0,
        'APLR': ep_aplr,
        'MinClr': (min_clearance if min_clearance < float('inf') else 0.0),
        'Reward': ep_reward,
    }


def mean_ignore_nan(vals):
    vs = [v for v in vals if v is not None and not (
        isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vs)) if vs else float('nan')





start_episode = metrics['episodes'][-1] + 1 if metrics['episodes'] else 1
for episode in tqdm(range(start_episode, args.episodes + 1),
                    total=args.episodes, initial=start_episode - 1):

    print(f"Training loop EP:{episode}")

    update_logs = []
    if len(D) >= args.batch_size:
        agent.train()
        for _ in tqdm(range(args.collect_interval), leave=False):
            batch = D.sample(args.batch_size)
            update_logs.append(agent.update(batch))

    for key in ['q_loss', 'critic_1_loss', 'critic_2_loss',
                'actor_loss', 'alpha_loss', 'alpha', 'entropy']:
        value = float(np.mean([x[key] for x in update_logs])) if update_logs else 0.0
        metrics[key].append(value)
        writer.add_scalar(f'Loss/{key}', value, episode)

    loss_x = list(range(start_episode, episode + 1))
    lineplot(loss_x, metrics['q_loss'], 'Q Loss', results_dir)
    lineplot(loss_x, metrics['actor_loss'], 'Actor Loss', results_dir)
    lineplot(loss_x, metrics['alpha'], 'Alpha', results_dir)
    lineplot(loss_x, metrics['entropy'], 'Entropy', results_dir)

    if episode % args.test_interval == 0:
        agent.eval()
        eval_results = {}
        for eval_mode, mode_tasks in get_eval_task_list(
                n_episodes=args.test_episodes).items():
            ep_records = []
            for task in mode_tasks:
                rec = run_test_episode(env, agent,
                                       map_seed=task['map_seed'],
                                       ood=task['ood'])
                ep_records.append(rec)

            sr = float(np.mean([r['SR'] for r in ep_records]))
            cr = float(np.mean([r['CR'] for r in ep_records]))
            aplr = mean_ignore_nan([r['APLR'] for r in ep_records])
            mclrs = [r['MinClr'] for r in ep_records if r['MinClr'] > 0]
            minclr = float(np.mean(mclrs)) if mclrs else 0.0
            avg_rw = float(np.mean([r['Reward'] for r in ep_records]))

            eval_results[eval_mode] = {
                'SR': sr, 'CR': cr, 'APLR': aplr,
                'MinClr': minclr, 'Reward': avg_rw,
            }

            steps_now = metrics['steps'][-1] if metrics['steps'] else 0
            writer.add_scalar(f'Eval/{eval_mode}_SR', sr, steps_now)
            writer.add_scalar(f'Eval/{eval_mode}_CR', cr, steps_now)
            writer.add_scalar(f'Eval/{eval_mode}_APLR', aplr, steps_now)
            writer.add_scalar(f'Eval/{eval_mode}_MinClr', minclr, steps_now)
            writer.add_scalar(f'Eval/{eval_mode}_Reward', avg_rw, steps_now)

        avg_test_reward = float(np.mean(
            [eval_results[m]['Reward'] for m in eval_results]))
        writer.add_scalar('Eval/Average_Test_Rewards', avg_test_reward,
                          metrics['steps'][-1] if metrics['steps'] else 0)

        for mode, res in eval_results.items():
            print(f"  [{mode}] SR={res['SR']:.1f}% CR={res['CR']:.1f}% "
                  f"APLR={res['APLR']:.2f} MinClr={res['MinClr']:.0f} "
                  f"Reward={res['Reward']:.1f}")
        print(f"  [Average Test Rewards] {avg_test_reward:.1f}")

        metrics['test_episodes'].append(episode)
        for mode in ['in_domain', 'ood']:
            for key in ['SR', 'CR', 'APLR', 'MinClr', 'Reward']:
                mk = f'test_{mode}_{key}'
                metrics.setdefault(mk, [])
                metrics[mk].append(eval_results[mode][key])

        metrics['test_rewards'].append(eval_results['in_domain']['SR'])
        metrics['test_avg_rewards'].append(eval_results['in_domain']['Reward'])

        lineplot(metrics['test_episodes'],
                 metrics.get('test_in_domain_SR', []), 'SR_InDomain', results_dir)
        lineplot(metrics['test_episodes'],
                 metrics.get('test_ood_SR', []), 'SR_OOD', results_dir)
        lineplot(metrics['test_episodes'],
                 metrics.get('test_in_domain_CR', []), 'CR_InDomain', results_dir)
        lineplot(metrics['test_episodes'],
                 metrics.get('test_ood_CR', []), 'CR_OOD', results_dir)
        lineplot(metrics['test_episodes'],
                 metrics.get('test_in_domain_Reward', []),
                 'Reward_InDomain', results_dir)
        lineplot(metrics['test_episodes'],
                 metrics.get('test_ood_Reward', []), 'Reward_OOD', results_dir)
        lineplot(metrics['test_episodes'],
                 metrics.get('test_in_domain_APLR', []),
                 'APLR_InDomain', results_dir)
        lineplot(metrics['test_episodes'],
                 metrics.get('test_ood_APLR', []), 'APLR_OOD', results_dir)
        lineplot(metrics['test_episodes'],
                 metrics.get('test_in_domain_MinClr', []),
                 'MinClr_InDomain', results_dir)
        lineplot(metrics['test_episodes'],
                 metrics.get('test_ood_MinClr', []), 'MinClr_OOD', results_dir)
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

    with torch.no_grad():
        agent.eval()
        ep_stats = run_episode(env, agent, train=True, deterministic=False)
        agent.train()

    total_reward = ep_stats['total_reward']
    prev_steps = metrics['steps'][-1] if metrics['steps'] else 0
    metrics['steps'].append(ep_stats['steps'] + prev_steps)
    metrics['episodes'].append(episode)
    metrics['train_rewards'].append(total_reward)

    metrics['train_success'].append(1 if ep_stats['reached'] else 0)
    metrics['train_collision'].append(1 if ep_stats['collided'] else 0)

    win = min(TRAIN_METRIC_WINDOW, len(metrics['train_success']))
    train_sr = float(np.mean(metrics['train_success'][-win:])) * 100.0
    train_cr = float(np.mean(metrics['train_collision'][-win:])) * 100.0
    metrics['train_sr'].append(train_sr)
    metrics['train_cr'].append(train_cr)

    writer.add_scalar('Train/episode_reward', total_reward, episode)
    writer.add_scalar('Train/episode_length', ep_stats['steps'], episode)
    writer.add_scalar('Train/Success', metrics['train_success'][-1], episode)
    writer.add_scalar('Train/Collision', metrics['train_collision'][-1], episode)
    writer.add_scalar('Train/SR', train_sr, episode)
    writer.add_scalar('Train/CR', train_cr, episode)

    lineplot(metrics['episodes'][-len(metrics['train_rewards']):],
             metrics['train_rewards'], 'Train Rewards', results_dir)
    sr_x = metrics['episodes'][-len(metrics['train_sr']):]
    lineplot(sr_x, metrics['train_sr'], 'Train SR', results_dir)
    lineplot(sr_x, metrics['train_cr'], 'Train CR', results_dir)

    print(f"  ep={episode} reward={total_reward:.1f} steps={ep_stats['steps']} "
          f"reached={ep_stats['reached']} collided={ep_stats['collided']} "
          f"SR={train_sr:.1f}% CR={train_cr:.1f}% "
          f"q_loss={metrics['q_loss'][-1]:.4f} "
          f"actor_loss={metrics['actor_loss'][-1]:.4f} "
          f"alpha={metrics['alpha'][-1]:.4f} "
          f"entropy={metrics['entropy'][-1]:.3f}")

    if episode % args.checkpoint_interval == 0:
        torch.save({
            'actor': agent.actor.state_dict(),
            'critic_1': agent.critic_1.state_dict(),
            'critic_2': agent.critic_2.state_dict(),
            'target_critic_1': agent.target_critic_1.state_dict(),
            'target_critic_2': agent.target_critic_2.state_dict(),
            'actor_optimizer': agent.actor_optimizer.state_dict(),
            'critic_1_optimizer': agent.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': agent.critic_2_optimizer.state_dict(),
            'log_alpha': agent.log_alpha.detach().cpu(),
            'log_alpha_optimizer': agent.log_alpha_optimizer.state_dict(),
            'episode': episode,
            'update_count': agent.update_count,
        }, os.path.join(results_dir, f'models_{episode}.pth'))





print("\n" + "=" * 92)
print(" " * 26 + "POST-TRAINING FINAL EVALUATION  (SAC)")
print("=" * 92)

agent.eval()
final_occ_iou = float('nan')
final_ade = float('nan')
final_fde = float('nan')

final_eval_results = {}
for eval_mode, mode_tasks in get_eval_task_list(
        n_episodes=args.test_episodes).items():
    per_ep_records = []
    for task in mode_tasks:
        rec = run_test_episode(env, agent,
                               map_seed=task['map_seed'], ood=task['ood'])
        per_ep_records.append(rec)
    final_eval_results[eval_mode] = per_ep_records

col = "{:>4} {:>8} {:>8} {:>8} {:>10} {:>10} {:>8} {:>8} {:>10}"


def fmt(v, spec):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "   -   "
    return spec.format(v)


print("\n" + "-" * 92)
print(" In-Domain test episodes (SR / APLR / CR / MinClr)")
print("-" * 92)
print(col.format("Ep", "SR(%)", "APLR", "CR(%)", "MinClr",
                 "Occ-IoU", "ADE", "FDE", "Reward"))
for i, rec in enumerate(final_eval_results['in_domain'], 1):
    print(col.format(
        i,
        fmt(rec['SR'], "{:8.1f}"),
        fmt(rec['APLR'], "{:8.3f}"),
        fmt(rec['CR'], "{:8.1f}"),
        fmt(rec['MinClr'], "{:10.1f}"),
        fmt(final_occ_iou, "{:10.3f}"),
        fmt(final_ade, "{:8.2f}"),
        fmt(final_fde, "{:8.2f}"),
        fmt(rec['Reward'], "{:10.1f}"),
    ))

print("-" * 92)
print(" OOD test episodes (SR-OOD / APLR / CR-OOD / MinClr)")
print("-" * 92)
print(col.format("Ep", "SR(%)", "APLR", "CR(%)", "MinClr",
                 "Occ-IoU", "ADE", "FDE", "Reward"))
for i, rec in enumerate(final_eval_results['ood'], 1):
    print(col.format(
        i,
        fmt(rec['SR'], "{:8.1f}"),
        fmt(rec['APLR'], "{:8.3f}"),
        fmt(rec['CR'], "{:8.1f}"),
        fmt(rec['MinClr'], "{:10.1f}"),
        fmt(final_occ_iou, "{:10.3f}"),
        fmt(final_ade, "{:8.2f}"),
        fmt(final_fde, "{:8.2f}"),
        fmt(rec['Reward'], "{:10.1f}"),
    ))

id_recs = final_eval_results['in_domain']
od_recs = final_eval_results['ood']

avg_SR_indomain = mean_ignore_nan([r['SR'] for r in id_recs])
avg_CR_indomain = mean_ignore_nan([r['CR'] for r in id_recs])
avg_SR = metrics['train_sr'][-1] if metrics['train_sr'] else float('nan')
avg_CR = metrics['train_cr'][-1] if metrics['train_cr'] else float('nan')

avg_APLR = mean_ignore_nan([r['APLR'] for r in id_recs])
avg_MinClr = mean_ignore_nan([r['MinClr'] for r in id_recs])
avg_SR_OOD = mean_ignore_nan([r['SR'] for r in od_recs])
avg_CR_OOD = mean_ignore_nan([r['CR'] for r in od_recs])
avg_Rew_ID = mean_ignore_nan([r['Reward'] for r in id_recs])
avg_Rew_OD = mean_ignore_nan([r['Reward'] for r in od_recs])
avg_Rew_all = mean_ignore_nan([r['Reward'] for r in id_recs + od_recs])

print("\n" + "=" * 92)
print(" Paper Table I - SAC Baseline - Averaged over all test episodes")
print("=" * 92)
print("  SR (%)            = {:>7.2f}    [train, sliding-{} ep]".format(
    avg_SR, TRAIN_METRIC_WINDOW))
print("  APLR              = {:>7.3f}".format(avg_APLR))
print("  CR (%)            = {:>7.2f}    [train, sliding-{} ep]".format(
    avg_CR, TRAIN_METRIC_WINDOW))
print("  MinClr            = {:>7.2f}".format(avg_MinClr))
print("  Occ-IoU           =    -     (N/A for SAC)")
print("  ADE               =    -     (N/A for SAC)")
print("  FDE               =    -     (N/A for SAC)")
print("  SR-OOD (%)        = {:>7.2f}".format(avg_SR_OOD))
print("  CR-OOD (%)        = {:>7.2f}".format(avg_CR_OOD))
print("  --------------------------------")
print("  [aux] In-Domain Eval SR  = {:>7.2f}".format(avg_SR_indomain))
print("  [aux] In-Domain Eval CR  = {:>7.2f}".format(avg_CR_indomain))
print("  Avg Test Reward (In-Domain)  = {:>7.2f}".format(avg_Rew_ID))
print("  Avg Test Reward (OOD)        = {:>7.2f}".format(avg_Rew_OD))
print("  Average Test Rewards (all)   = {:>7.2f}".format(avg_Rew_all))
print("=" * 92)

final_step = metrics['steps'][-1] if metrics['steps'] else args.episodes

for mode_name in ['in_domain', 'ood']:
    for i, rec in enumerate(final_eval_results[mode_name], 1):
        for k, v in rec.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            writer.add_scalar(f'FinalTest/{mode_name}_ep{i}_{k}',
                              float(v), final_step)

writer.add_scalar('FinalTest/Average_SR', avg_SR, final_step)
writer.add_scalar('FinalTest/Average_CR', avg_CR, final_step)
writer.add_scalar('FinalTest/InDomain_SR_aux', avg_SR_indomain, final_step)
writer.add_scalar('FinalTest/InDomain_CR_aux', avg_CR_indomain, final_step)
writer.add_scalar('FinalTest/Average_APLR', avg_APLR, final_step)
writer.add_scalar('FinalTest/Average_MinClr', avg_MinClr, final_step)
writer.add_scalar('FinalTest/Average_SR_OOD', avg_SR_OOD, final_step)
writer.add_scalar('FinalTest/Average_CR_OOD', avg_CR_OOD, final_step)
writer.add_scalar('FinalTest/Average_Test_Rewards', avg_Rew_all, final_step)
writer.add_scalar('FinalTest/Average_Test_Rewards_InDomain',
                  avg_Rew_ID, final_step)
writer.add_scalar('FinalTest/Average_Test_Rewards_OOD',
                  avg_Rew_OD, final_step)

table_text = (
    "| Method | SR (%) | APLR | CR (%) | MinClr | Occ-IoU | ADE | FDE | "
    "SR-OOD (%) | CR-OOD (%) |\n"
    "|--------|--------|------|--------|--------|---------|-----|-----|"
    "------------|------------|\n"
    f"| SAC | {avg_SR:.2f} | {avg_APLR:.3f} | {avg_CR:.2f} | "
    f"{avg_MinClr:.2f} | - | - | - | {avg_SR_OOD:.2f} | "
    f"{avg_CR_OOD:.2f} |"
)
try:
    writer.add_text('FinalTest/PaperTableI_SAC', table_text, final_step)
except Exception as exc:
    print(f"  [warn] writer.add_text failed: {exc}")

metrics['final_eval'] = {
    'in_domain': final_eval_results['in_domain'],
    'ood': final_eval_results['ood'],
    'avg_SR': avg_SR,
    'avg_CR': avg_CR,
    'avg_SR_indomain': avg_SR_indomain,
    'avg_CR_indomain': avg_CR_indomain,
    'train_metric_window': TRAIN_METRIC_WINDOW,
    'avg_APLR': avg_APLR,
    'avg_MinClr': avg_MinClr,
    'occ_iou': None,
    'ade': None,
    'fde': None,
    'avg_SR_OOD': avg_SR_OOD,
    'avg_CR_OOD': avg_CR_OOD,
    'avg_test_rewards': avg_Rew_all,
    'avg_test_rewards_id': avg_Rew_ID,
    'avg_test_rewards_ood': avg_Rew_OD,
}
torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

writer.flush()
writer.close()
env.close()
print("\nDone.")