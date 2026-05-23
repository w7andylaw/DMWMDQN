# Author: Qiwei Wang
"""Dueling Double DQN baseline for UAV navigation."""

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


parser = argparse.ArgumentParser(description='D3QN Baseline for UAV Navigation')
parser.add_argument('--id', type=str, default='d3qn', help='Experiment ID')
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
parser.add_argument('--learning-rate', type=float, default=2e-5)
parser.add_argument('--adam-epsilon', type=float, default=1e-7)
parser.add_argument('--grad-clip-norm', type=float, default=10.0)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--target-update', type=int, default=1000,
                    help='Target network update interval (gradient steps)')
parser.add_argument('--epsilon', type=float, default=0.01,
                    help='Fixed epsilon for ε-greedy exploration during training.')
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
    'q_loss': [],
    'train_success': [],
    'train_collision': [],
    'train_sr': [],
    'train_cr': [],
}

TRAIN_METRIC_WINDOW = 100

writer = SummaryWriter(results_dir + "/{}_{}_log".format(args.env, args.id))
print("Writer is ready.")


env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length,
          args.action_repeat, args.bit_depth)
if env is None:
    raise ValueError(f"Environment '{args.env}' not found.")
print("Environment is loaded.")



class D3QNReplayBuffer:
    """Replay buffer for single-step D3QN transitions."""
    def __init__(self, size, obs_space, action_size, bit_depth, device):
        self.size = size
        self.device = device
        self.bit_depth = bit_depth
        self.action_size = action_size
        sem_shape = obs_space['semantic_map'].shape   
        pos_shape = obs_space['position'].shape
        self.obs = {
            'image':        np.empty((size, 3, 64, 64), dtype=np.uint8),
            'target':       np.empty((size, 3, 64, 64), dtype=np.uint8),
            'position':     np.empty((size, *pos_shape), dtype=np.float32),
            'semantic_map': np.empty((size, *sem_shape), dtype=np.float16),
        }
        self.next_obs = {
            'image':        np.empty((size, 3, 64, 64), dtype=np.uint8),
            'target':       np.empty((size, 3, 64, 64), dtype=np.uint8),
            'position':     np.empty((size, *pos_shape), dtype=np.float32),
            'semantic_map': np.empty((size, *sem_shape), dtype=np.float16),
        }
        self.actions = np.empty((size,), dtype=np.int64)
        self.rewards = np.empty((size,), dtype=np.float32)
        self.dones   = np.empty((size,), dtype=np.float32)
        self.idx = 0
        self.full = False
        self.steps = 0
        self.episodes = 0

    def _obs_to_np(self, obs):
        """Convert one observation dictionary to NumPy arrays."""
        out = {}
        for k, v in obs.items():
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy().squeeze(0)
            out[k] = v
        return out

    def append(self, obs, action, reward, next_obs, done):
        o  = self._obs_to_np(obs)
        no = self._obs_to_np(next_obs)
        self.obs['image'][self.idx]      = postprocess_observation(o['image'],  self.bit_depth)
        self.obs['target'][self.idx]     = postprocess_observation(o['target'], self.bit_depth)
        self.obs['position'][self.idx]   = o['position']
        self.obs['semantic_map'][self.idx] = o['semantic_map']
        self.next_obs['image'][self.idx]      = postprocess_observation(no['image'],  self.bit_depth)
        self.next_obs['target'][self.idx]     = postprocess_observation(no['target'], self.bit_depth)
        self.next_obs['position'][self.idx]   = no['position']
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
        """Fetch and preprocess a batch of dictionary observations."""
        batch = {}
        
        for key in ['image', 'target']:
            raw = d[key][idxs].astype(np.float32)
            t = torch.as_tensor(raw)
            preprocess_observation_(t, self.bit_depth)
            batch[key] = t
        batch['position']     = torch.as_tensor(d['position'][idxs])
        batch['semantic_map'] = torch.as_tensor(d['semantic_map'][idxs].astype(np.float32))
        
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


class DuelingQNet(nn.Module):
    """Dueling Q-network for dictionary observations."""
    def __init__(self, action_size, hidden=256):
        super().__init__()
        
        def make_img_cnn():
            return nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2),  nn.ReLU(),   
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),   
                nn.Conv2d(64, 128, 4, stride=2),nn.ReLU(),   
                nn.Conv2d(128, 128, 4, stride=2), nn.ReLU(), 
                nn.Flatten(),
                nn.Linear(128 * 2 * 2, 256), nn.ReLU(),
            )
        self.img_cnn = make_img_cnn()
        self.tgt_cnn = make_img_cnn()
        self.map_cnn = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=2, padding=1),  nn.ReLU(),   
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   
            nn.Conv2d(64, 128, 3, stride=2, padding=1),nn.ReLU(),   
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
        )
        self.vec_mlp = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        feat_dim = 256 + 256 + 256 + 64  
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
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
        if pos.dim() == 3: pos = pos.squeeze(1)
        x = pos[:, 0:1]
        y = pos[:, 1:2]
        dist_x = torch.minimum(x, 1.0 - x)
        dist_y = torch.minimum(y, 1.0 - y)
        danger_x = (1.0 - 2.0 * dist_x).pow(2)
        danger_y = (1.0 - 2.0 * dist_y).pow(2)
        pos_feat = torch.cat([pos, dist_x, dist_y, danger_x, danger_y], dim=-1)
        vec_f = self.vec_mlp(pos_feat)
        h = torch.cat([img_f, tgt_f, map_f, vec_f], dim=-1)
        h = self.fusion(h)
        V = self.value_head(h)                          
        A = self.adv_head(h)                            
        Q = V + (A - A.mean(dim=-1, keepdim=True))
        return Q

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
        """Select an action from the current policy."""
        if np.random.rand() < epsilon:
            return torch.tensor([np.random.randint(self.action_size)],
                                dtype=torch.int64)
        obs_d = self._obs_to_device(obs)
        q = self.q_net(obs_d)
        a = q.argmax(dim=-1).detach().cpu()   
        return a.to(torch.int64)

    def update(self, batch):
        s, a, r, s_next, d = batch
        s      = self._obs_to_device(s)
        s_next = self._obs_to_device(s_next)
        a = a.to(self.device)
        r = r.to(self.device)
        d = d.to(self.device)
        q_sa = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)  

        with torch.no_grad():
            next_a = self.q_net(s_next).argmax(dim=-1, keepdim=True)   
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

D = D3QNReplayBuffer(args.experience_size, env.observation_size,
                     env.action_size, args.bit_depth, args.device)

if not args.test:
    for s in range(1, args.seed_episodes + 1):
        observation, done, t = env.reset(), False, 0
        _seed_reached = False
        _seed_collided = False
        _seed_inner = env._env if hasattr(env, '_env') else env
        while not done:
            action = env.sample_random_action()
            next_observation, reward, done = env.step(action)
            D.append(observation, action, reward, next_observation, done)
            observation = next_observation
            t += 1
            _info = getattr(_seed_inner, '_last_info', {}) or {}
            if _info.get('reach', False):
                _seed_reached = True
            if _info.get('collision', False):
                _seed_collided = True
        metrics['steps'].append(t * args.action_repeat +
                                (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
        metrics['episodes'].append(s)
        metrics['train_success'].append(1 if _seed_reached else 0)
        metrics['train_collision'].append(1 if _seed_collided else 0)
print("Experience replay buffer is ready. (size={})".format(len(D)))

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

if args.models != '' and os.path.exists(args.models):
    print("Loading pre-trained D3QN weights")
    ckpt = torch.load(args.models, map_location=args.device, weights_only=True)
    agent.q_net.load_state_dict(ckpt['q_net'])
    agent.target_q_net.load_state_dict(ckpt['target_q_net'])
    agent.optimizer.load_state_dict(ckpt['optimizer'])

print("D3QN agent is ready.")

def current_epsilon(ep):
    """Return the fixed exploration rate."""
    return float(args.epsilon)

def run_episode(env, agent, epsilon, train=False):
    """Run one training or collection episode."""
    observation = env.reset() if not train else env.reset()  
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
        
        if hasattr(_inner, 'obstacles') and hasattr(_inner, 'agent_pos'):
            for obs_j in _inner.obstacles:
                d = float(np.linalg.norm(_inner.agent_pos - obs_j.q))
                if d < min_clearance:
                    min_clearance = d

        
        _info = getattr(_inner, '_last_info', {}) or {}
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
    """Run one deterministic evaluation episode."""
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
        if _info.get('reach', False):
            reached = True
        if _info.get('collision', False):
            collided = True

        observation = next_observation

    
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

start_episode = metrics['episodes'][-1] + 1 if metrics['episodes'] else 1
for episode in tqdm(range(start_episode, args.episodes + 1),
                    total=args.episodes, initial=start_episode - 1):

    print(f"Training loop EP:{episode}")
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

    
    if episode % args.test_interval == 0:
        agent.q_net.eval()
        eval_results = {}
        
        for eval_mode, mode_tasks in get_eval_task_list(n_episodes=args.test_episodes).items():
            ep_records = []
            for task in mode_tasks:
                rec = run_test_episode(env, agent,
                                       map_seed=task['map_seed'], ood=task['ood'])
                ep_records.append(rec)

            sr     = float(np.mean([r['SR']     for r in ep_records]))
            cr     = float(np.mean([r['CR']     for r in ep_records]))
            aplrs  = [r['APLR']  for r in ep_records if not np.isnan(r['APLR'])]
            
            
            aplr   = float(np.mean(aplrs)) if aplrs else float('nan')
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

    
    eps = current_epsilon(episode)
    with torch.no_grad():
        agent.q_net.eval()
        ep_stats = run_episode(env, agent, epsilon=eps, train=True)
        agent.q_net.train()

    total_reward = ep_stats['total_reward']
    metrics['steps'].append(ep_stats['steps'] + metrics['steps'][-1])
    metrics['episodes'].append(episode)
    metrics['train_rewards'].append(total_reward)

    
    
    metrics['train_success'].append(1 if ep_stats['reached'] else 0)
    metrics['train_collision'].append(1 if ep_stats['collided'] else 0)

    
    
    _win = min(TRAIN_METRIC_WINDOW, len(metrics['train_success']))
    _train_sr = float(np.mean(metrics['train_success'][-_win:])) * 100.0
    _train_cr = float(np.mean(metrics['train_collision'][-_win:])) * 100.0
    metrics['train_sr'].append(_train_sr)
    metrics['train_cr'].append(_train_cr)

    writer.add_scalar('Train/episode_reward', total_reward, episode)
    writer.add_scalar('Train/epsilon', eps, episode)
    writer.add_scalar('Train/episode_length', ep_stats['steps'], episode)
    
    writer.add_scalar('Train/Success',   metrics['train_success'][-1],   episode)
    writer.add_scalar('Train/Collision', metrics['train_collision'][-1], episode)
    writer.add_scalar('Train/SR', _train_sr, episode)
    writer.add_scalar('Train/CR', _train_cr, episode)

    lineplot(metrics['episodes'][-len(metrics['train_rewards']):],
             metrics['train_rewards'], 'Train Rewards', results_dir)
    
    _sr_x = metrics['episodes'][-len(metrics['train_sr']):]
    lineplot(_sr_x, metrics['train_sr'], 'Train SR', results_dir)
    lineplot(_sr_x, metrics['train_cr'], 'Train CR', results_dir)

    print(f"  ep={episode} reward={total_reward:.1f} steps={ep_stats['steps']} "
          f"reached={ep_stats['reached']} collided={ep_stats['collided']} "
          f"SR={_train_sr:.1f}% CR={_train_cr:.1f}% "
          f"ε={eps:.3f} q_loss={mean_q_loss:.4f}")

    
    if episode % args.checkpoint_interval == 0:
        torch.save({
            'q_net':        agent.q_net.state_dict(),
            'target_q_net': agent.target_q_net.state_dict(),
            'optimizer':    agent.optimizer.state_dict(),
            'episode':      episode,
            'update_count': agent.update_count,
        }, os.path.join(results_dir, f'models_{episode}.pth'))

print("\n" + "=" * 92)
print(" " * 26 + "POST-TRAINING FINAL EVALUATION  (D3QN)")
print("=" * 92)

agent.q_net.eval()


final_occ_iou = float('nan')
final_ade     = float('nan')
final_fde     = float('nan')

_final_eval_results = {}

for _eval_mode, _mode_tasks in get_eval_task_list(n_episodes=args.test_episodes).items():
    _per_ep_records = []
    for _task in _mode_tasks:
        rec = run_test_episode(env, agent,
                               map_seed=_task['map_seed'], ood=_task['ood'])
        _per_ep_records.append(rec)
    _final_eval_results[_eval_mode] = _per_ep_records



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



def _mean_ignore_nan(vals):
    vs = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    
    
    
    
    return float(np.mean(vs)) if vs else float('nan')

_id_recs = _final_eval_results['in_domain']
_od_recs = _final_eval_results['ood']




avg_SR_indomain = _mean_ignore_nan([r['SR'] for r in _id_recs])
avg_CR_indomain = _mean_ignore_nan([r['CR'] for r in _id_recs])
avg_SR = metrics['train_sr'][-1] if metrics['train_sr'] else float('nan')
avg_CR = metrics['train_cr'][-1] if metrics['train_cr'] else float('nan')

avg_APLR    = _mean_ignore_nan([r['APLR']   for r in _id_recs])
avg_MinClr  = _mean_ignore_nan([r['MinClr'] for r in _id_recs])
avg_SR_OOD  = _mean_ignore_nan([r['SR']     for r in _od_recs])
avg_CR_OOD  = _mean_ignore_nan([r['CR']     for r in _od_recs])
avg_Rew_ID  = _mean_ignore_nan([r['Reward'] for r in _id_recs])
avg_Rew_OD  = _mean_ignore_nan([r['Reward'] for r in _od_recs])
avg_Rew_all = _mean_ignore_nan([r['Reward'] for r in _id_recs + _od_recs])

print("\n" + "=" * 92)
print(" Paper Table I — D3QN Baseline — Averaged over all test episodes")
print("=" * 92)
print("  SR (%)↑          = {:>7.2f}    [train, sliding-{} ep]".format(avg_SR, TRAIN_METRIC_WINDOW))
print("  APLR↓            = {:>7.3f}".format(avg_APLR))
print("  CR (%)↓          = {:>7.2f}    [train, sliding-{} ep]".format(avg_CR, TRAIN_METRIC_WINDOW))
print("  MinClr↑          = {:>7.2f}".format(avg_MinClr))
print("  Occ-IoU↑         =    -     (N/A for D3QN)")
print("  ADE↓             =    -     (N/A for D3QN)")
print("  FDE↓             =    -     (N/A for D3QN)")
print("  SR-OOD (%)↑      = {:>7.2f}".format(avg_SR_OOD))
print("  CR-OOD (%)↓      = {:>7.2f}".format(avg_CR_OOD))
print("  --------------------------------")
print("  [aux] In-Domain Eval SR  = {:>7.2f}".format(avg_SR_indomain))
print("  [aux] In-Domain Eval CR  = {:>7.2f}".format(avg_CR_indomain))
print("  Avg Test Reward (In-Domain)  = {:>7.2f}".format(avg_Rew_ID))
print("  Avg Test Reward (OOD)        = {:>7.2f}".format(avg_Rew_OD))
print("  Average Test Rewards (all)   = {:>7.2f}".format(avg_Rew_all))
print("=" * 92)



_final_step = metrics['steps'][-1] if metrics['steps'] else args.episodes

for _mode_name in ['in_domain', 'ood']:
    for _i, _rec in enumerate(_final_eval_results[_mode_name], 1):
        for _k, _v in _rec.items():
            if _v is None or (isinstance(_v, float) and np.isnan(_v)):
                continue
            writer.add_scalar(f'FinalTest/{_mode_name}_ep{_i}_{_k}',
                              float(_v), _final_step)



writer.add_scalar('FinalTest/Average_SR',       avg_SR,      _final_step)   
writer.add_scalar('FinalTest/Average_CR',       avg_CR,      _final_step)   
writer.add_scalar('FinalTest/InDomain_SR_aux',  avg_SR_indomain, _final_step)  
writer.add_scalar('FinalTest/InDomain_CR_aux',  avg_CR_indomain, _final_step)
writer.add_scalar('FinalTest/Average_APLR',     avg_APLR,    _final_step)
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




metrics['final_eval'] = {
    'in_domain':              _final_eval_results['in_domain'],
    'ood':                    _final_eval_results['ood'],
    'avg_SR':                 avg_SR,             
    'avg_CR':                 avg_CR,             
    'avg_SR_indomain':        avg_SR_indomain,    
    'avg_CR_indomain':        avg_CR_indomain,    
    'train_metric_window':    TRAIN_METRIC_WINDOW,
    'avg_APLR':               avg_APLR,
    'avg_MinClr':             avg_MinClr,
    'occ_iou':                None,   
    'ade':                    None,   
    'fde':                    None,   
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