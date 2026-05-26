# Author: Qiwei Wang
"""World-model D3QN training entry point for UAV navigation."""

import argparse
import os
import gymnasium as gym
import numpy as np
import torch
try:
    from torch.amp import autocast as _autocast_new, GradScaler
    def autocast(enabled=True):
        return _autocast_new('cuda', enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from env import CONTROL_SUITE_ENVS, GYM_ENVS, Env, EnvBatcher, get_eval_task_list
from memory import ExperienceReplay
from models import (
    Encoder, ObservationModel, RewardModel, TransitionModel,
    bottle, UAVHybridEncoder, SemanticFeatureExtractor,
    MapEncoder, MapTransitionModel, ObstacleForecaster, ContinuationModel,
    DifferentiableMapUpdater,
    QNetwork, QPolicy, sync_target,
)
from utils import (
    FreezeParameters, imagine_ahead, lambda_return, lineplot, write_video,
    trk_loss_multi_peak,
    symlog, symexp,
)


def safe_clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    for p in parameters:
        torch.nan_to_num_(p.grad, nan=0.0, posinf=1e4, neginf=-1e4)
    return nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)


parser = argparse.ArgumentParser(description='Semantic World Model for UAV Navigation')
parser.add_argument('--id', type=str, default='wm + senmap', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument(
    '--env', type=str, default='UAV-v0',
    choices=GYM_ENVS + CONTROL_SUITE_ENVS + ['UAV-v0'],
    help='Environment',
)
parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
parser.add_argument('--max-episode-length', type=int, default=500, metavar='T')
parser.add_argument('--experience-size', type=int, default=1000000, metavar='D',
                    help='Experience replay size')
parser.add_argument('--cnn-activation-function', type=str, default='relu', choices=dir(F))
parser.add_argument('--dense-activation-function', type=str, default='elu', choices=dir(F))
parser.add_argument('--embedding-size', type=int, default=1024, metavar='E')
parser.add_argument('--hidden-size', type=int, default=200, metavar='H')
parser.add_argument('--belief-size', type=int, default=200, metavar='H')
parser.add_argument('--state-size', type=int, default=30, metavar='Z')
parser.add_argument('--semantic-size', type=int, default=512, metavar='S', help='Semantic feature size (gθ output)')
parser.add_argument('--map-embedding-size', type=int, default=256, metavar='M', help='Map embedding size (E_m output)')
parser.add_argument('--action-repeat', type=int, default=1, metavar='R')
parser.add_argument('--episodes', type=int, default=1000, metavar='E')
parser.add_argument('--seed-episodes', type=int, default=5, metavar='S')
parser.add_argument('--wm-pretrain-episodes', type=int, default=5, metavar='WP',
                    help='Number of warm-up episodes that train only the world model with random actions; Q/planner is not updated.')
parser.add_argument('--skip-eval-during-wm-pretrain', action='store_true', default=True,
                    help='Skip policy evaluation during WM-only pretraining because the planner is intentionally untrained.')
parser.add_argument('--eval-during-wm-pretrain', dest='skip_eval_during_wm_pretrain',
                    action='store_false', help='Run evaluation even during WM-only pretraining.')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C')
parser.add_argument('--batch-size', type=int, default=16, metavar='B',
                    help='Batch size per gradient step')
parser.add_argument('--chunk-size', type=int, default=50, metavar='L',
                    help='Chunk length')
parser.add_argument('--grad-accumulate', type=int, default=1, metavar='GA',
                    help='Gradient accumulation steps (effective batch = batch-size * grad-accumulate)')
parser.add_argument('--worldmodel-LogProbLoss', action='store_true')
parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D')
parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β')
parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R')
parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg')
parser.add_argument('--bit-depth', type=int, default=5, metavar='B')
parser.add_argument('--model-learning-rate', type=float, default=5e-4, metavar='α')
parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS')
parser.add_argument('--adam-epsilon', type=float, default=1e-7, metavar='ε')
parser.add_argument('--grad-clip-norm', type=float, default=10.0, metavar='C')
parser.add_argument('--planning-horizon', type=int, default=3, metavar='H',
                    help='Planning horizon for imagination rollout length.')
parser.add_argument('--n-steps', type=int, default=1, metavar='N',
                    help='N-step TD horizon for Q/planner update. '
                         'This is separated from --planning-horizon: '
                         '--planning-horizon controls imagined rollout length, '
                         '--n-steps controls how many imagined rewards are used in each TD target.')
parser.add_argument('--discount', type=float, default=0.99, metavar='H')
parser.add_argument('--disclam', type=float, default=0.95, metavar='H')
parser.add_argument('--test', action='store_true', help='Test only')
parser.add_argument('--test-interval', type=int, default=25, metavar='I')
parser.add_argument('--test-episodes', type=int, default=10, metavar='E')
parser.add_argument('--checkpoint-interval', type=int, default=25, metavar='I')
parser.add_argument('--checkpoint-experience', action='store_true')
parser.add_argument('--models', type=str, default='', metavar='M')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER')
parser.add_argument('--render', action='store_true')
parser.add_argument('--semantic-loss-scale', type=float, default=10.0)
parser.add_argument('--contrastive-margin', type=float, default=10.0)
parser.add_argument('--similarity-dist-thresh', type=float, default=300.0)
parser.add_argument('--dyn-scale', type=float, default=1.0, help='Weight for dynamics loss (公式39 L_dyn)')
parser.add_argument('--rep-scale', type=float, default=0.2, help='Weight for representation loss (公式39 L_rep)')
parser.add_argument('--lambda-img', type=float, default=0.5, help='Weight for image loss (公式36)')
parser.add_argument('--lambda-r', type=float, default=15.0, help='Weight for reward loss (公式37)')
parser.add_argument('--lambda-c', type=float, default=10.0, help='Weight for continuation loss (公式38)')
parser.add_argument('--lambda-map', type=float, default=4.0, help='Weight for map prediction loss (公式40)')
parser.add_argument('--lambda-occ', type=float, default=1.5, help='Weight for occupancy loss (公式41)')
parser.add_argument('--lambda-flow', type=float, default=1.0, help='Weight for motion flow loss (公式42)')
parser.add_argument('--lambda-trk', type=float, default=2.0, help='Weight for trajectory loss (公式43)')
parser.add_argument('--lambda-upd', type=float, default=4.0, help='[C-2] Weight for map updater loss (公式11, 独立于 L_map)')
parser.add_argument('--lambda-risk-imag', type=float, default=0.01,
                    help='[C-1] Extra risk penalty in imagination. Default 0 avoids double-counting because env reward already includes risk.')
parser.add_argument('--reward-symlog', dest='reward_symlog',
                    action='store_true', default=True,
                    help='[DreamerV3] symlog-transform reward targets for training, '
                         'symexp reward_model output in imagination. 默认开启.')
parser.add_argument('--no-reward-symlog', dest='reward_symlog',
                    action='store_false',
                    help='关闭 symlog, reward_model 在原始尺度训练 (论文原始设置).')
parser.add_argument('--forecast-horizon', type=int, default=3, help='Obstacle forecasting K steps')
parser.add_argument('--trk-patch-size', type=int, default=5,
                    help='[公式 43] 软质心 patch 大小, 奇数. 建议 3~7.')
parser.add_argument('--encode-batch', type=int, default=100, metavar='EB',
                    help='Mini-batch size for encoder forward')
parser.add_argument('--eval-steps', type=int, default=500, metavar='ES')
parser.add_argument('--q-learning-rate', type=float, default=5e-5, metavar='αQ',
                    help='Q-network 学习率')
parser.add_argument('--q-target-update', type=int, default=100,
                    help='Target Q 网络硬同步周期 (单位: 梯度步)')
parser.add_argument('--q-target-tau', type=float, default=0.005,
                    help='1.0 → hard copy; <1.0 → Polyak soft update')
parser.add_argument('--q-epsilon-imag', type=float, default=0.20,
                    help='想象 rollout 中的 ε')
parser.add_argument('--q-epsilon-env', type=float, default=0.05,
                    help='Fixed epsilon for QPolicy during real-environment collection after WM pretraining.')
args = parser.parse_args()
args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)
args.n_steps = max(1, min(int(args.n_steps), int(args.planning_horizon)))
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
    print(f"  CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}, PyTorch: {torch.__version__}")
    _cc = torch.cuda.get_device_capability(0)
    print(f"  Compute capability: {_cc[0]}.{_cc[1]}")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
else:
    print("Using CPU")
    args.device = torch.device('cpu')

metrics = {
    'steps': [], 'episodes': [], 'train_rewards': [],
    'test_episodes': [], 'test_rewards': [], 'test_avg_rewards': [],
    'observation_loss': [], 'reward_loss': [], 'kl_loss': [],
    'continuation_loss': [], 'map_loss': [], 'occ_loss': [], 'flow_loss': [],
    'q_loss': [],
    'train_success': [],
    'train_collision': [],
    'train_sr': [],
    'train_cr': [],
}




TRAIN_METRIC_WINDOW = 100
summary_name = results_dir + "/{}_{}_log"
writer = SummaryWriter(summary_name.format(args.env, args.id))
print("Writer is ready.")


env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)
if env is None:
    raise ValueError(f"Environment '{args.env}' not found.")
print("Environment is loaded.")



_inner = getattr(env, '_env', None)
if _inner is None:
    _inner = getattr(env, 'env', env)
_env_nobs = getattr(_inner, 'num_obstacles', None)
args.num_obstacles = _env_nobs if isinstance(_env_nobs, int) else 0


if args.experience_replay != '' and os.path.exists(args.experience_replay):
    D = torch.load(args.experience_replay, weights_only=False)  
    metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
elif not args.test:
    D = ExperienceReplay(
        args.experience_size, args.symbolic_env, env.observation_size,
        env.action_size, args.bit_depth, args.device
    )
    for s in range(1, args.seed_episodes + 1):
        observation, done, t = env.reset(), False, 0
        
        _seed_reached = False
        _seed_collided = False
        _seed_inner = env._env if hasattr(env, '_env') else env
        while not done:
            action = env.sample_random_action()
            next_observation, reward, done = env.step(action)
            D.append(observation, action, reward, done)
            observation = next_observation
            t += 1
            _info = getattr(_seed_inner, '_last_info', {}) or {}
            if _info.get('reach', False):
                _seed_reached = True
            if _info.get('collision', False):
                _seed_collided = True
        metrics['steps'].append(t * args.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
        metrics['episodes'].append(s)
        
        metrics['train_success'].append(1 if _seed_reached else 0)
        metrics['train_collision'].append(1 if _seed_collided else 0)
print("Experience replay buffer is ready.")



transition_model = TransitionModel(
    belief_size=args.belief_size,
    state_size=args.state_size,
    action_size=env.action_size,
    hidden_size=args.hidden_size,
    embedding_size=args.embedding_size,
    semantic_size=args.semantic_size,
    semantic_state_size=args.semantic_size,
    map_embedding_size=args.map_embedding_size,
    activation_function=args.dense_activation_function,
).to(device=args.device)


observation_model = ObservationModel(
    args.symbolic_env, env.observation_size,
    args.belief_size, args.state_size, args.embedding_size,
    map_embedding_size=args.map_embedding_size,
    activation_function=args.cnn_activation_function,
).to(device=args.device)


reward_model = RewardModel(
    args.belief_size, args.state_size, args.hidden_size,
    map_embedding_size=args.map_embedding_size,
    activation_function=args.dense_activation_function,
).to(device=args.device)


continuation_model = ContinuationModel(
    args.belief_size, args.state_size, args.map_embedding_size, args.hidden_size,
).to(device=args.device)


map_encoder = MapEncoder(
    n_channels=6, grid_size=30, map_embedding_size=args.map_embedding_size,
).to(device=args.device)


map_transition_model = MapTransitionModel(
    belief_size=args.belief_size, state_size=args.state_size,
    action_size=env.action_size, map_embedding_size=args.map_embedding_size,
    n_channels=6, grid_size=30,
).to(device=args.device)







obstacle_forecaster = ObstacleForecaster(
    belief_size=args.belief_size, state_size=args.state_size,
    map_embedding_size=args.map_embedding_size,
    forecast_horizon=args.forecast_horizon, grid_size=30,
).to(device=args.device)


map_updater = DifferentiableMapUpdater(
    embedding_size=args.embedding_size, n_channels=6, grid_size=30,
).to(device=args.device)


if args.env == 'UAV-v0':
    print("Using UAVHybridEncoder.")
    encoder = UAVHybridEncoder(args.embedding_size, args.cnn_activation_function).to(device=args.device)
    semantic_extractor = None
else:
    encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size,
                      args.cnn_activation_function).to(device=args.device)
    semantic_extractor = None









q_net = QNetwork(
    belief_size=args.belief_size,
    state_size=args.state_size,
    hidden_size=args.hidden_size,
    action_size=env.action_size,
    map_embedding_size=args.map_embedding_size,
    activation_function=args.dense_activation_function,
).to(device=args.device)

target_q_net = QNetwork(
    belief_size=args.belief_size,
    state_size=args.state_size,
    hidden_size=args.hidden_size,
    action_size=env.action_size,
    map_embedding_size=args.map_embedding_size,
    activation_function=args.dense_activation_function,
).to(device=args.device)
target_q_net.load_state_dict(q_net.state_dict())
for _p in target_q_net.parameters():
    _p.requires_grad = False


q_policy = QPolicy(q_net, env.action_size,
                   default_epsilon=args.q_epsilon_imag)



world_model_modules = [
    transition_model, observation_model, reward_model, continuation_model,
    encoder, map_encoder, map_transition_model, map_updater, obstacle_forecaster,
]


param_list = []
for m in world_model_modules:
    param_list.extend(list(m.parameters()))



model_optimizer = optim.Adam(
    param_list,
    lr=0 if args.learning_rate_schedule != 0 else args.model_learning_rate,
    eps=args.adam_epsilon,
)

q_optimizer = optim.Adam(
    q_net.parameters(),
    lr=0 if args.learning_rate_schedule != 0 else args.q_learning_rate,
    eps=args.adam_epsilon,
)

q_update_count = 0


if args.models != '' and os.path.exists(args.models):
    print("loading pre-trained models")
    model_dicts = torch.load(args.models, weights_only=True)
    transition_model.load_state_dict(model_dicts['transition_model'])
    observation_model.load_state_dict(model_dicts['observation_model'])
    reward_model.load_state_dict(model_dicts['reward_model'])
    encoder.load_state_dict(model_dicts['encoder'])
    
    if 'q_net' in model_dicts:
        q_net.load_state_dict(model_dicts['q_net'])
    if 'target_q_net' in model_dicts:
        target_q_net.load_state_dict(model_dicts['target_q_net'])
    else:
        target_q_net.load_state_dict(q_net.state_dict())
    
    
    if 'model_optimizer' in model_dicts:
        try:
            model_optimizer.load_state_dict(model_dicts['model_optimizer'])
        except ValueError as _e:
            print(f"[WARN] Skip loading model_optimizer due to architecture change: {_e}")
    if 'q_optimizer' in model_dicts:
        q_optimizer.load_state_dict(model_dicts['q_optimizer'])
    if 'q_update_count' in model_dicts:
        q_update_count = int(model_dicts['q_update_count'])
    if 'continuation_model' in model_dicts:
        continuation_model.load_state_dict(model_dicts['continuation_model'])
    if 'map_encoder' in model_dicts:
        map_encoder.load_state_dict(model_dicts['map_encoder'])
    if 'map_transition_model' in model_dicts:
        map_transition_model.load_state_dict(model_dicts['map_transition_model'])
    if 'obstacle_forecaster' in model_dicts:
        obstacle_forecaster.load_state_dict(model_dicts['obstacle_forecaster'])
    if 'map_updater' in model_dicts:
        map_updater.load_state_dict(model_dicts['map_updater'])
    if semantic_extractor is not None and 'semantic_extractor' in model_dicts:
        semantic_extractor.load_state_dict(model_dicts['semantic_extractor'])


planner = q_policy
global_prior = Normal(
    torch.zeros(args.batch_size, args.state_size, device=args.device),
    torch.ones(args.batch_size, args.state_size, device=args.device),
)



use_amp = False
print("AMP: OFF (forced FP32 for stability)")
model_scaler = GradScaler(enabled=False)
q_scaler = GradScaler(enabled=False)
print("Semantic World Model + D3QN is ready.")



def get_all_model_modules():
    """Get all model modules."""
    modules = []
    for m in world_model_modules:
        if hasattr(m, 'component_modules'):
            modules.extend(m.component_modules)
        else:
            modules.append(m)
    return modules






def update_belief_and_act(
    args, env, planner, transition_model, encoder,
    belief, posterior_state, action, observation,
    map_encoder, current_map_embedding=None,
    map_updater=None, prev_map=None,
    explore=False
):
    """Update belief and act."""
    prev_map_embedding = current_map_embedding

    if isinstance(observation, dict):
        obs_img = observation['image'].to(args.device)
        obs_tgt_img = observation['target'].to(args.device)
        pos = observation['position'].to(args.device)
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        obs_vec = pos
        full_output = encoder(obs_img, obs_tgt_img, obs_vec)
        embed = full_output[0] if isinstance(full_output, tuple) else full_output

        
        if map_encoder is not None and 'semantic_map' in observation:
            obs_sem_map = observation['semantic_map'].to(args.device)
            if obs_sem_map.dim() == 3:
                obs_sem_map = obs_sem_map.unsqueeze(0)
            new_map_embedding = map_encoder(obs_sem_map)
            current_map = obs_sem_map.detach()
        else:
            new_map_embedding = prev_map_embedding  
            current_map = prev_map
    else:
        obs_input = observation.to(args.device)
        full_output = encoder(obs_input)
        embed = full_output[0] if isinstance(full_output, tuple) else full_output
        new_map_embedding = prev_map_embedding
        current_map = prev_map
    if prev_map_embedding is None:
        prev_map_embedding = torch.zeros(1, args.map_embedding_size, device=args.device)
    if new_map_embedding is None:
        new_map_embedding = torch.zeros(1, args.map_embedding_size, device=args.device)
    belief, _, _, _, posterior_state, _, _ = transition_model(
        posterior_state,
        action.unsqueeze(dim=0),
        belief,
        embed.unsqueeze(dim=0),
        map_embeddings=prev_map_embedding.unsqueeze(dim=0),       
        map_embeddings_post=new_map_embedding.unsqueeze(dim=0),   
        deterministic=(not explore),                             
    )
    belief = belief.squeeze(dim=0)
    posterior_state = posterior_state.squeeze(dim=0)
    action = planner.get_action(
        belief, posterior_state, new_map_embedding,
        det=not explore,
        epsilon=(getattr(args, 'action_noise', getattr(args, 'q_epsilon_env', 0.0)) if explore else 0.0),
    )
    next_observation, reward, done = env.step(
        action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu()
    )
    return belief, posterior_state, new_map_embedding, current_map, action, next_observation, reward, done

for episode in tqdm(
    range(metrics['episodes'][-1] + 1, args.episodes + 1),
    total=args.episodes, initial=metrics['episodes'][-1] + 1
):
    model_modules = get_all_model_modules()
    wm_pretraining = (episode <= args.wm_pretrain_episodes)
    phase_name = 'WM_PRETRAIN_RANDOM' if wm_pretraining else 'WM_D3QN_JOINT'
    print(f"Training loop EP:{episode} [{phase_name}]")
    writer.add_scalar('Train/wm_pretraining', 1.0 if wm_pretraining else 0.0, episode)
    losses = []
    for s in tqdm(range(args.collect_interval)):
        observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size)

        _dev = args.device
        actions = actions.to(_dev)
        rewards = rewards.to(_dev)
        nonterminals = nonterminals.to(_dev)

        if isinstance(observations, dict):
            obs_img = observations['image']              
            obs_tgt_img = observations['target']         
            obs_pos = observations['position']           
            obs_vec = obs_pos
            obs_sem_map = observations['semantic_map']
            T, B = obs_img.shape[:2]
            TB = T * B
            ENCODE_BATCH = min(TB, args.encode_batch)
            flat_img_cpu = obs_img.reshape(TB, *obs_img.shape[2:])
            flat_tgt_cpu = obs_tgt_img.reshape(TB, *obs_tgt_img.shape[2:])
            flat_vec_cpu = obs_vec.reshape(TB, -1)
            flat_sem_map_cpu = obs_sem_map.reshape(TB, *obs_sem_map.shape[2:])

            
            embed_chunks = []
            for i in range(0, TB, ENCODE_BATCH):
                j = min(i + ENCODE_BATCH, TB)
                _img = flat_img_cpu[i:j].contiguous().float().to(_dev)
                _tgt = flat_tgt_cpu[i:j].contiguous().float().to(_dev)
                _vec = flat_vec_cpu[i:j].contiguous().float().to(_dev)
                with autocast(enabled=use_amp):
                    out = encoder(_img, _tgt, _vec)
                if isinstance(out, tuple):
                    embed_chunks.append(out[0])
                else:
                    embed_chunks.append(out)
                del _img, _tgt, _vec, out
            torch.cuda.empty_cache()
            flat_embed = torch.cat(embed_chunks, dim=0)
            del embed_chunks
            full_embed = flat_embed.float().view(T, B, -1)
            embed = full_embed[1:]
            sem_feat_for_tm = None
            map_emb_chunks = []
            for i in range(0, TB, ENCODE_BATCH):
                j = min(i + ENCODE_BATCH, TB)
                _sem = flat_sem_map_cpu[i:j].contiguous().float().to(_dev)
                with autocast(enabled=use_amp):
                    map_emb_chunks.append(map_encoder(_sem))
                del _sem
            torch.cuda.empty_cache()
            flat_map_emb = torch.cat(map_emb_chunks, dim=0).float()  
            del map_emb_chunks
            full_map_emb = flat_map_emb.view(T, B, -1)
            del flat_map_emb
            map_emb_for_tm = full_map_emb[:-1]
            map_emb_for_loss = full_map_emb[1:]
            updater_embed = full_embed[1:]  
            source_maps_cpu = obs_sem_map[:-1]   
            updater_pos_cpu = obs_pos[1:]
            target_obs_cpu = obs_img[1:]       
            target_maps_cpu = obs_sem_map[1:]
            del flat_img_cpu, flat_tgt_cpu, obs_img, obs_tgt_img
            del flat_vec_cpu, flat_sem_map_cpu
        else:
            obs_tensor = observations.to(_dev)
            embed = bottle(encoder, (obs_tensor[1:],))
            target_obs_cpu = obs_tensor[1:]  
            loss_curr_feat = None
            sem_feat_for_tm = None
            map_emb_for_tm = None
            map_emb_for_loss = None
            target_maps_cpu = None
            source_maps_cpu = None
            updater_embed = None
            updater_pos_cpu = None
            full_map_emb = None

        init_belief = torch.zeros(args.batch_size, args.belief_size, device=args.device)
        init_state = torch.zeros(args.batch_size, args.state_size, device=args.device)
        

        
        (
            beliefs, prior_states, prior_means, prior_std_devs,
            posterior_states, posterior_means, posterior_std_devs,
        ) = transition_model(
            init_state, actions[:-1], init_belief, embed, nonterminals[:-1],
            map_embeddings=map_emb_for_tm,
            map_embeddings_post=map_emb_for_loss,
        )

        
        if torch.isnan(beliefs).any() or torch.isnan(posterior_states).any():
            print(f"  [WARN] NaN detected in TransitionModel output at step {s}, skipping this batch.")
            model_optimizer.zero_grad()
            losses.append([0.0] * 9)
            torch.cuda.empty_cache()
            continue

        T_loss, B_loss = beliefs.shape[:2]
        flat_beliefs = beliefs.view(T_loss * B_loss, -1)
        flat_post_states = posterior_states.view(T_loss * B_loss, -1)

        if map_emb_for_loss is not None:
            flat_map_emb_loss = map_emb_for_loss.view(T_loss * B_loss, -1)
        elif map_emb_for_tm is not None:
            
            flat_map_emb_loss = map_emb_for_tm.view(T_loss * B_loss, -1)
        else:
            flat_map_emb_loss = None

        if flat_map_emb_loss is not None:
            obs_loss_accum = torch.tensor(0.0, device=_dev)
            obs_count = 0
            OBS_CHUNK = min(T_loss * B_loss, args.encode_batch)
            _obs_var_scalar = torch.tensor(1.0, device=_dev)   
            for oi in range(0, T_loss * B_loss, OBS_CHUNK):
                oj = min(oi + OBS_CHUNK, T_loss * B_loss)
                with autocast(enabled=use_amp):
                    _obs_mean = observation_model(
                        flat_beliefs[oi:oj], flat_post_states[oi:oj], flat_map_emb_loss[oi:oj]
                    )
                _target = target_obs_cpu.reshape(T_loss * B_loss, *target_obs_cpu.shape[2:])[oi:oj]
                _target = _target.contiguous().float().to(_dev)
                _obs_mean_f = _obs_mean.float()
                
                obs_loss_accum = obs_loss_accum + F.gaussian_nll_loss(
                    _obs_mean_f, _target, _obs_var_scalar.expand_as(_obs_mean_f),
                    reduction='sum', full=True,
                )
                obs_count += _obs_mean_f.numel()
                del _obs_mean, _obs_mean_f, _target
            observation_loss = obs_loss_accum / max(obs_count, 1)
            del obs_loss_accum, _obs_var_scalar
        else:
            with autocast(enabled=use_amp):
                obs_mean = bottle(observation_model, (beliefs, posterior_states))
            if isinstance(obs_mean, tuple):
                obs_mean = obs_mean[0]
            _target_gpu = target_obs_cpu.to(_dev) if not target_obs_cpu.is_cuda else target_obs_cpu
            _obs_mean_f = obs_mean.float()
            _obs_var_full = torch.ones_like(_obs_mean_f)
            observation_loss = F.gaussian_nll_loss(
                _obs_mean_f, _target_gpu, _obs_var_full, reduction='mean', full=True,
            )
            del obs_mean, _target_gpu, _obs_mean_f, _obs_var_full
        torch.cuda.empty_cache()

        with autocast(enabled=use_amp):
            if flat_map_emb_loss is not None:
                reward_pred = reward_model(flat_beliefs, flat_post_states, flat_map_emb_loss)
                reward_pred = reward_pred.view(T_loss, B_loss)
            else:
                _flat_b = beliefs.view(T_loss * B_loss, -1)
                _flat_s = posterior_states.view(T_loss * B_loss, -1)
                reward_pred = reward_model(_flat_b, _flat_s).view(T_loss, B_loss)
        _reward_target = rewards[:-1]
        if args.reward_symlog:
            _reward_target = symlog(_reward_target)
        reward_loss = F.mse_loss(reward_pred.float(), _reward_target, reduction='mean')
        del _reward_target

        with autocast(enabled=use_amp):
            if flat_map_emb_loss is not None:
                cont_pred = continuation_model(flat_beliefs, flat_post_states, flat_map_emb_loss)
                cont_pred = cont_pred.view(T_loss, B_loss)
            else:
                _flat_b = beliefs.view(T_loss * B_loss, -1)
                _flat_s = posterior_states.view(T_loss * B_loss, -1)
                cont_pred = continuation_model(_flat_b, _flat_s).view(T_loss, B_loss)
        cont_target = nonterminals[:-1].squeeze(-1)
        cont_pred_safe = cont_pred.float().clamp(1e-6, 1.0 - 1e-6)
        continuation_loss = F.binary_cross_entropy(cont_pred_safe, cont_target, reduction='mean')

        dist_post = Normal(posterior_means, posterior_std_devs)
        dist_prior = Normal(prior_means, prior_std_devs)
        dist_post_detached = Normal(posterior_means.detach(), posterior_std_devs.detach())
        kl_loss_dyn = kl_divergence(dist_post_detached, dist_prior).sum(dim=2).mean(dim=(0, 1))
        dist_prior_detached = Normal(prior_means.detach(), prior_std_devs.detach())
        kl_loss_rep = kl_divergence(dist_post, dist_prior_detached).sum(dim=2).mean(dim=(0, 1))
        kl_loss = args.dyn_scale * kl_loss_dyn + args.rep_scale * kl_loss_rep

        
        if target_maps_cpu is not None:
            T_u, B_u = source_maps_cpu.shape[:2]
            N_u = T_u * B_u
            _flat_src_cpu = source_maps_cpu.reshape(N_u, *source_maps_cpu.shape[2:])
            _flat_emb = updater_embed.reshape(N_u, -1)  
            _flat_pos_cpu = updater_pos_cpu.reshape(N_u, -1)
            _flat_tgt_cpu = target_maps_cpu.reshape(N_u, *target_maps_cpu.shape[2:])
            _upd_loss = torch.tensor(0.0, device=_dev)
            _upd_count = 0
            for ci in range(0, N_u, args.encode_batch):
                cj = min(ci + args.encode_batch, N_u)
                _src = _flat_src_cpu[ci:cj].float().to(_dev)
                _pos = _flat_pos_cpu[ci:cj].float().to(_dev)
                _tgt = _flat_tgt_cpu[ci:cj].float().to(_dev)
                with autocast(enabled=use_amp):
                    _pred, _ = map_updater(_src, _flat_emb[ci:cj], _pos)
                _upd_loss = _upd_loss + F.l1_loss(_pred.float(), _tgt, reduction='sum')
                _upd_count += _tgt.numel()
                del _pred, _src, _pos, _tgt
            map_updater_loss = _upd_loss / max(_upd_count, 1)
            del _flat_src_cpu, _flat_emb, _flat_pos_cpu, _flat_tgt_cpu, _upd_loss
            del updater_embed
        else:
            map_updater_loss = torch.tensor(0.0, device=_dev)
            if updater_embed is not None:
                del updater_embed

        
        
        
        
        
        
        
        if target_maps_cpu is not None and map_emb_for_tm is not None and full_map_emb is not None:
            
            src_maps_cpu = obs_sem_map[1:-1]   
            tgt_maps_cpu = obs_sem_map[2:]     
            T_m, B_m = src_maps_cpu.shape[:2]
            if T_m > 0:
                N_m = T_m * B_m
                _fp_cpu = src_maps_cpu.reshape(N_m, *src_maps_cpu.shape[2:])
                _fb = beliefs[:T_m].reshape(N_m, -1)               
                _fs = posterior_states[:T_m].reshape(N_m, -1)       
                _fa = actions[1:1+T_m].reshape(N_m, -1)            
                _fm = full_map_emb[1:1+T_m].reshape(N_m, -1)       
                _ft_cpu = tgt_maps_cpu.reshape(N_m, *tgt_maps_cpu.shape[2:])

                _map_loss = torch.tensor(0.0, device=_dev)
                _map_count = 0
                for ci in range(0, N_m, args.encode_batch):
                    cj = min(ci + args.encode_batch, N_m)
                    _fp_g = _fp_cpu[ci:cj].float().to(_dev)
                    _ft_g = _ft_cpu[ci:cj].float().to(_dev)
                    with autocast(enabled=use_amp):
                        _pred = map_transition_model(_fp_g, _fb[ci:cj], _fs[ci:cj], _fa[ci:cj], _fm[ci:cj])
                    _map_loss = _map_loss + F.l1_loss(_pred.float(), _ft_g, reduction='sum')
                    _map_count += _ft_g.numel()
                    del _pred, _fp_g, _ft_g
                map_loss = _map_loss / max(_map_count, 1)
                del _fp_cpu, _fb, _fs, _fa, _fm, _ft_cpu, _map_loss
            else:
                map_loss = torch.tensor(0.0, device=_dev)
        else:
            map_loss = torch.tensor(0.0, device=_dev)
        torch.cuda.empty_cache()

        
        if target_maps_cpu is not None and flat_map_emb_loss is not None:
            K = args.forecast_horizon

            
            
            
            
            nt_aligned = nonterminals[:-1].squeeze(-1)  
            forecast_mask_list = []
            for k in range(1, K + 1):
                mask_k = torch.ones(T_loss, B_loss, device=_dev)
                for j in range(1, k + 1):  
                    shifted_nt = torch.zeros_like(nt_aligned)
                    if j < T_loss:
                        shifted_nt[:-j] = nt_aligned[j:]  
                    mask_k = mask_k * shifted_nt
                forecast_mask_list.append(mask_k)
            
            forecast_mask = torch.stack(forecast_mask_list, dim=2).reshape(T_loss * B_loss, K)
            del forecast_mask_list

            gt_occ_list = []
            gt_flow_list = []
            for k in range(1, K + 1):
                shifted = torch.roll(target_maps_cpu, shifts=-k, dims=0)
                if k < T_loss:
                    shifted[-k:] = target_maps_cpu[-1:]
                else:
                    shifted = target_maps_cpu[-1:].expand_as(target_maps_cpu)
                gt_occ_list.append(shifted[:, :, 3, :, :])
                gt_flow_list.append(torch.stack([shifted[:, :, 4, :, :], shifted[:, :, 5, :, :]], dim=2))

            gt_occ_k = torch.stack(gt_occ_list, dim=2).reshape(T_loss * B_loss, K, 30, 30)
            gt_flow_k = torch.stack(gt_flow_list, dim=2).reshape(T_loss * B_loss, K, 2, 30, 30)
            del gt_occ_list, gt_flow_list

            
            
            _occ_loss = torch.tensor(0.0, device=_dev)
            _flow_loss = torch.tensor(0.0, device=_dev)
            _trk_loss = torch.tensor(0.0, device=_dev)
            _occ_count = torch.tensor(0.0, device=_dev)
            _flow_count = torch.tensor(0.0, device=_dev)
            _trk_count = torch.tensor(0.0, device=_dev)

            
            

            N_of = T_loss * B_loss
            for ci in range(0, N_of, args.encode_batch):
                cj = min(ci + args.encode_batch, N_of)
                with autocast(enabled=use_amp):
                    _occ_p, _flow_p = obstacle_forecaster(
                        flat_beliefs[ci:cj], flat_post_states[ci:cj], flat_map_emb_loss[ci:cj]
                    )
                _gt_occ = gt_occ_k[ci:cj].float().to(_dev)
                _gt_flow = gt_flow_k[ci:cj].float().to(_dev)
                _mask = forecast_mask[ci:cj].float().to(_dev)          
                _mask_occ = _mask.unsqueeze(-1).unsqueeze(-1)
                _occ_p_safe = _occ_p.float().clamp(1e-6, 1.0 - 1e-6)
                _gt_occ_safe = _gt_occ.clamp(0, 1)
                _valid_occ = _mask_occ.expand_as(_gt_occ_safe)
                _pos_occ = ((_gt_occ_safe > 0.05).float() * _valid_occ).sum()
                _valid_count_occ = _valid_occ.sum().clamp(min=1.0)
                _neg_occ = (_valid_count_occ - _pos_occ).clamp(min=1.0)
                _pos_weight_occ = (_neg_occ / _pos_occ.clamp(min=1.0)).clamp(min=1.0, max=20.0)
                _occ_weight = torch.where(
                    _gt_occ_safe > 0.05,
                    _pos_weight_occ,
                    torch.ones_like(_gt_occ_safe),
                )
                _occ_elem = F.binary_cross_entropy(
                    _occ_p_safe, _gt_occ_safe, reduction='none'
                )
                _occ_loss = _occ_loss + (_occ_elem * _occ_weight * _valid_occ).sum()
                _occ_count = _occ_count + (_occ_weight * _valid_occ).sum()
                _flow_elem = F.l1_loss(_flow_p.float(), _gt_flow, reduction='none')
                _flow_loss = _flow_loss + (_flow_elem * _mask_occ.unsqueeze(2)).sum()
                _flow_count = _flow_count + _mask_occ.unsqueeze(2).expand_as(_flow_elem).sum()

                for k_idx in range(K):
                    _pred_occ_k = _occ_p[:, k_idx].float()         
                    _gt_occ_k_s = _gt_occ[:, k_idx]                
                    _trk_elem = trk_loss_multi_peak(
                        _pred_occ_k, _gt_occ_k_s,
                        n_peaks=args.num_obstacles,
                        patch_size=args.trk_patch_size,
                    )  
                    _mk = _mask[:, k_idx].view(-1, 1, 1)            
                    _trk_loss = _trk_loss + (_trk_elem * _mk).sum()
                    _trk_count = _trk_count + _mk.expand_as(_trk_elem).sum()

                del _occ_p, _flow_p, _gt_occ, _gt_flow, _mask, _occ_elem, _flow_elem
                del _gt_occ_safe, _valid_occ, _pos_occ, _valid_count_occ
                del _neg_occ, _pos_weight_occ, _occ_weight
            occ_loss = _occ_loss / _occ_count.clamp(min=1)
            flow_loss = _flow_loss / _flow_count.clamp(min=1)
            trk_loss = _trk_loss / _trk_count.clamp(min=1)
            del gt_occ_k, gt_flow_k, _occ_loss, _flow_loss, _trk_loss
            del _occ_count, _flow_count, _trk_count
        else:
            occ_loss = torch.tensor(0.0, device=_dev)
            flow_loss = torch.tensor(0.0, device=_dev)
            trk_loss = torch.tensor(0.0, device=_dev)
        torch.cuda.empty_cache()

        
        if 'forecast_mask' in dir() and forecast_mask is not None:
            del forecast_mask

        model_loss = (
            args.lambda_img * observation_loss       
            + args.lambda_r * reward_loss            
            + args.lambda_c * continuation_loss      
            + kl_loss                                
            + args.lambda_map * map_loss             
            + args.lambda_upd * map_updater_loss     
            + args.lambda_occ * occ_loss             
            + args.lambda_flow * flow_loss           
            + args.lambda_trk * trk_loss             
        )

        model_loss = model_loss / args.grad_accumulate
        if args.learning_rate_schedule != 0:
            for group in model_optimizer.param_groups:
                group['lr'] = min(
                    group['lr'] + args.model_learning_rate / args.learning_rate_schedule,
                    args.model_learning_rate
                )

        if s % args.grad_accumulate == 0:
            model_optimizer.zero_grad()
        model_scaler.scale(model_loss).backward()

        
        if (s + 1) % args.grad_accumulate == 0 or (s + 1) == args.collect_interval:
            model_scaler.unscale_(model_optimizer)
            safe_clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
            model_scaler.step(model_optimizer)
            model_scaler.update()

        
        _loss_items = [
            observation_loss.item(), reward_loss.item(), kl_loss.item(),
            continuation_loss.item(), map_loss.item(), map_updater_loss.item(),
            occ_loss.item(), flow_loss.item(), trk_loss.item(),
        ]
        del model_loss, observation_loss, reward_loss, continuation_loss
        del kl_loss, map_loss, map_updater_loss, occ_loss, flow_loss, trk_loss
        try:
            del reward_pred, cont_pred
        except NameError:
            pass
        torch.cuda.empty_cache()

        
        if wm_pretraining:
            _loss_items.append(0.0)  
            losses.append(_loss_items)
            torch.cuda.empty_cache()
            continue

        
        with torch.no_grad():
            start_states = posterior_states.detach()
            start_beliefs = beliefs.detach()
            if map_emb_for_loss is not None:
                start_map_emb = map_emb_for_loss.detach()
            elif map_emb_for_tm is not None:
                start_map_emb = map_emb_for_tm.detach()
            else:
                start_map_emb = torch.zeros(
                    *start_beliefs.shape[:2], args.map_embedding_size, device=args.device
                )

            
            if isinstance(observations, dict) and obs_sem_map is not None:
                start_maps = obs_sem_map[1:].to(_dev).detach()
            else:
                start_maps = None

        
        del beliefs, prior_states, prior_means, prior_std_devs
        del posterior_states, posterior_means, posterior_std_devs
        del embed, sem_feat_for_tm, map_emb_for_tm, map_emb_for_loss
        if full_map_emb is not None:
            del full_map_emb
        torch.cuda.empty_cache()

        
        with torch.no_grad():
            with FreezeParameters(model_modules):
                imagination_traj = imagine_ahead(
                    start_states, start_beliefs, start_map_emb,
                    start_maps,
                    q_policy,  
                    transition_model, map_encoder, map_transition_model,
                    args.planning_horizon,
                )

            (imged_beliefs, imged_prior_states, _imged_pm, _imged_ps,
             imged_map_emb, imged_actions_onehot) = imagination_traj

            H, N = imged_beliefs.shape[:2]
            flat_ib = imged_beliefs.reshape(H * N, -1)
            flat_ips = imged_prior_states.reshape(H * N, -1)
            flat_ime = imged_map_emb.reshape(H * N, -1)

            with FreezeParameters(model_modules):
                _raw_r_pred = reward_model(flat_ib, flat_ips, flat_ime).view(H, N)
                imged_reward = symexp(_raw_r_pred) if args.reward_symlog else _raw_r_pred
                del _raw_r_pred
                imged_cont = continuation_model(flat_ib, flat_ips, flat_ime).view(H, N)

                _use_obs_risk  = args.lambda_risk_imag > 0
                if _use_obs_risk:
                    K_f = args.forecast_horizon
                    Ng  = 30
                    _dir8_t = torch.tensor([
                        [ 0,  1], [-1,  1], [-1,  0], [-1, -1],
                        [ 0, -1], [ 1, -1], [ 1,  0], [ 1,  1],
                    ], dtype=torch.float32, device=_dev)
                    action_dirs = imged_actions_onehot.float() @ _dir8_t
                    cum_disp = torch.cumsum(action_dirs, dim=0)
                    start_grid = (obs_pos[1:].reshape(-1, 2).to(_dev).float()
                                  * float(Ng))
                    t_idx = torch.arange(H, device=_dev).view(H, 1)
                    k_idx = torch.arange(K_f, device=_dev).view(1, K_f)
                    cum_i = (t_idx + k_idx + 1).clamp(max=H - 1)
                    disp_hk = cum_disp[cum_i]                        
                    pos_hk  = start_grid.view(1, 1, N, 2) + disp_hk  
                    pos_hk  = pos_hk.clamp(0, Ng - 1).round().long()
                    ix_first  = pos_hk[..., 0]                       
                    ix_second = pos_hk[..., 1]                       
                    h_ar = torch.arange(H,   device=_dev).view(H, 1, 1).expand(H, K_f, N)
                    n_ar = torch.arange(N,   device=_dev).view(1, 1, N).expand(H, K_f, N)

                
                if _use_obs_risk:
                    occ_pred, _ = obstacle_forecaster(flat_ib, flat_ips, flat_ime)
                    occ_pred = occ_pred.view(H, N, K_f, Ng, Ng)
                    occ_perm = occ_pred.permute(0, 2, 1, 3, 4)       
                    k_ar = torch.arange(K_f, device=_dev).view(1, K_f, 1).expand(H, K_f, N)
                    occ_at_pos = occ_perm[h_ar, k_ar, n_ar, ix_first, ix_second]  
                    chi = occ_at_pos.max(dim=1).values               
                    imged_reward = imged_reward - args.lambda_risk_imag * chi
                    del occ_pred, occ_perm, occ_at_pos, chi, k_ar

                
                if _use_obs_risk:
                    del action_dirs, cum_disp, disp_hk, pos_hk
                    del ix_first, ix_second, h_ar, n_ar
            
            
            start_b0 = start_beliefs.reshape(-1, start_beliefs.shape[-1]).unsqueeze(0)
            start_s0 = start_states.reshape(-1, start_states.shape[-1]).unsqueeze(0)
            start_m0 = start_map_emb.reshape(-1, start_map_emb.shape[-1]).unsqueeze(0)

            cur_b_seq = torch.cat([start_b0, imged_beliefs[:-1]], dim=0)       
            cur_s_seq = torch.cat([start_s0, imged_prior_states[:-1]], dim=0)  
            cur_m_seq = torch.cat([start_m0, imged_map_emb[:-1]], dim=0)       

            next_b = imged_beliefs.reshape(-1, imged_beliefs.shape[-1])
            next_s = imged_prior_states.reshape(-1, imged_prior_states.shape[-1])
            next_m = imged_map_emb.reshape(-1, imged_map_emb.shape[-1])

            q_next_online = q_net(next_b, next_s, next_m)              
            a_star = q_next_online.argmax(dim=-1, keepdim=True)        
            q_next_target = target_q_net(next_b, next_s, next_m).gather(1, a_star).squeeze(1)
            q_next_target = q_next_target.view(H, N)

            n_td = max(1, min(int(args.n_steps), H))
            td_targets = []
            for td_i in range(H):
                ret = torch.zeros_like(imged_reward[td_i])
                discount_prod = torch.ones_like(imged_reward[td_i])
                max_j = min(n_td, H - td_i)

                for j in range(max_j):
                    idx = td_i + j
                    ret = ret + discount_prod * imged_reward[idx]
                    discount_prod = discount_prod * args.discount * imged_cont[idx]

                bootstrap_idx = td_i + max_j - 1
                ret = ret + discount_prod * q_next_target[bootstrap_idx]
                td_targets.append(ret)

            td_target = torch.stack(td_targets, dim=0)
            td_target = td_target.clamp(-100.0, 100.0)

        cur_b = cur_b_seq.reshape(-1, cur_b_seq.shape[-1])
        cur_s = cur_s_seq.reshape(-1, cur_s_seq.shape[-1])
        cur_m = cur_m_seq.reshape(-1, cur_m_seq.shape[-1])
        cur_a_idx = imged_actions_onehot.argmax(dim=-1).reshape(-1)   

        with autocast(enabled=use_amp):
            q_all = q_net(cur_b, cur_s, cur_m)                         
            q_sa = q_all.gather(1, cur_a_idx.unsqueeze(1)).squeeze(1)
            q_sa = q_sa.view(H, N)
            q_loss = F.smooth_l1_loss(q_sa, td_target, reduction='mean')

        _q_loss_val = q_loss.item()
        q_optimizer.zero_grad()
        q_scaler.scale(q_loss).backward()
        q_scaler.unscale_(q_optimizer)
        safe_clip_grad_norm_(q_net.parameters(), args.grad_clip_norm, norm_type=2)
        q_scaler.step(q_optimizer)
        q_scaler.update()
        q_update_count += 1
        if args.q_target_tau >= 1.0:
            if q_update_count % args.q_target_update == 0:
                sync_target(target_q_net, q_net, tau=1.0)
        else:
            sync_target(target_q_net, q_net, tau=args.q_target_tau)

        _loss_items.append(_q_loss_val)
        losses.append(_loss_items)  

        
        del imagination_traj, imged_beliefs, imged_prior_states
        del imged_map_emb, imged_actions_onehot, imged_reward, imged_cont
        del q_next_online, a_star, q_next_target, td_target, q_all, q_sa, q_loss
        del cur_b_seq, cur_s_seq, cur_m_seq, start_b0, start_s0, start_m0
        if start_maps is not None:
            del start_maps
        torch.cuda.empty_cache()

    losses = tuple(zip(*losses))
    metrics['observation_loss'].append(np.mean(losses[0]))
    metrics['reward_loss'].append(np.mean(losses[1]))
    metrics['kl_loss'].append(np.mean(losses[2]))
    metrics['continuation_loss'].append(np.mean(losses[3]))
    metrics['map_loss'].append(np.mean(losses[4]))
    metrics['map_updater_loss'] = metrics.get('map_updater_loss', [])
    metrics['map_updater_loss'].append(np.mean(losses[5]))
    metrics['occ_loss'].append(np.mean(losses[6]))
    metrics['flow_loss'].append(np.mean(losses[7]))
    metrics['trk_loss'] = metrics.get('trk_loss', [])
    metrics['trk_loss'].append(np.mean(losses[8]))
    metrics['q_loss'].append(np.mean(losses[9]))
    lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'Observation Loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'Reward Loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['kl_loss']):], metrics['kl_loss'], 'KL Loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['continuation_loss']):], metrics['continuation_loss'], 'Continuation Loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['map_loss']):], metrics['map_loss'], 'Map Loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['map_updater_loss']):], metrics['map_updater_loss'], 'MapUpdater Loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['occ_loss']):], metrics['occ_loss'], 'Occ Loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['q_loss']):], metrics['q_loss'], 'Q Loss', results_dir)
    writer.add_scalar('Loss/observation', metrics['observation_loss'][-1], episode)
    writer.add_scalar('Loss/reward', metrics['reward_loss'][-1], episode)
    writer.add_scalar('Loss/kl', metrics['kl_loss'][-1], episode)
    writer.add_scalar('Loss/continuation', metrics['continuation_loss'][-1], episode)
    writer.add_scalar('Loss/map', metrics['map_loss'][-1], episode)
    writer.add_scalar('Loss/map_updater', metrics['map_updater_loss'][-1], episode)
    writer.add_scalar('Loss/occ', metrics['occ_loss'][-1], episode)
    writer.add_scalar('Loss/flow', metrics['flow_loss'][-1], episode)
    writer.add_scalar('Loss/trk', metrics['trk_loss'][-1], episode)
    writer.add_scalar('Loss/q', metrics['q_loss'][-1], episode)

    if episode % args.test_interval == 0 and not (wm_pretraining and args.skip_eval_during_wm_pretrain):
        for m in world_model_modules:
            m.eval()
        q_net.eval()
        eval_results = {}
        eval_tasks_dict = get_eval_task_list(n_episodes=args.test_episodes)
        for eval_mode, mode_tasks in eval_tasks_dict.items():
            ep_successes = 0
            ep_collisions = 0
            ep_total = 0
            ep_path_ratios = []
            ep_min_clearances = []
            ep_rewards = []

            with torch.no_grad():
                for task in mode_tasks:
                    observation = env.reset(map_seed=task['map_seed'], ood=task['ood'])
                    belief = torch.zeros(1, args.belief_size, device=args.device)
                    posterior_state = torch.zeros(1, args.state_size, device=args.device)
                    map_emb = torch.zeros(1, args.map_embedding_size, device=args.device)
                    prev_map = None  
                    action = torch.zeros(1, env.action_size, device=args.device)
                    done = False
                    ep_total += 1
                    path_length = 0.0
                    min_clearance = float('inf')
                    ep_reward = 0.0
                    reached = False
                    collided = False
                    _inner_env = env._env if hasattr(env, '_env') else env
                    start_pos = _inner_env.agent_pos.copy() if hasattr(_inner_env, 'agent_pos') else None

                    while not done:
                        (belief, posterior_state, map_emb, prev_map,
                         action, next_observation, reward, done) = update_belief_and_act(
                            args, env, planner, transition_model, encoder,
                            belief, posterior_state, action, observation,
                            map_encoder=map_encoder,
                            current_map_embedding=map_emb,
                            map_updater=map_updater, prev_map=prev_map,
                            explore=False,
                        )
                        r_val = reward.item() if torch.is_tensor(reward) else reward
                        ep_reward += r_val
                        path_length += 1.0

                        if hasattr(_inner_env, 'obstacles') and hasattr(_inner_env, 'agent_pos'):
                            for obs_j in _inner_env.obstacles:
                                d = float(np.linalg.norm(_inner_env.agent_pos - obs_j.q))
                                if d < min_clearance:
                                    min_clearance = d
                        _info = getattr(_inner_env, '_last_info', {}) or {}
                        if _info.get('reach', False):
                            reached = True
                        if _info.get('collision', False):
                            collided = True
                        observation = next_observation

                    if reached:
                        ep_successes += 1
                    if collided:
                        ep_collisions += 1
                    if reached and start_pos is not None and hasattr(_inner_env, 'target_pos'):
                        _target = _inner_env.target_pos
                        shortest = max(abs(start_pos[0] - _target[0]),
                                       abs(start_pos[1] - _target[1])) / _inner_env.cell_size
                        if shortest > 0:
                            ep_path_ratios.append(path_length / shortest)
                    if min_clearance < float('inf'):
                        ep_min_clearances.append(min_clearance)
                    ep_rewards.append(ep_reward)
            sr = ep_successes / max(ep_total, 1) * 100
            cr = ep_collisions / max(ep_total, 1) * 100
            aplr = float(np.mean(ep_path_ratios)) if ep_path_ratios else float('nan')
            min_clr = float(np.mean(ep_min_clearances)) if ep_min_clearances else 0.0
            avg_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0

            eval_results[eval_mode] = {
                'SR': sr, 'CR': cr, 'APLR': aplr, 'MinClr': min_clr, 'Reward': avg_reward
            }

            current_training_steps = metrics['steps'][-1] if len(metrics['steps']) > 0 else 0
            writer.add_scalar(f'Eval/{eval_mode}_SR', sr, current_training_steps)
            writer.add_scalar(f'Eval/{eval_mode}_CR', cr, current_training_steps)
            writer.add_scalar(f'Eval/{eval_mode}_APLR', aplr, current_training_steps)
            writer.add_scalar(f'Eval/{eval_mode}_MinClr', min_clr, current_training_steps)
            writer.add_scalar(f'Eval/{eval_mode}_Reward', avg_reward, current_training_steps)

        _avg_test_reward = float(np.mean([eval_results[m]['Reward'] for m in eval_results]))
        writer.add_scalar('Eval/Average_Test_Rewards', _avg_test_reward,
                          metrics['steps'][-1] if len(metrics['steps']) > 0 else 0)

        
        for mode, res in eval_results.items():
            print(f"  [{mode}] SR={res['SR']:.1f}% CR={res['CR']:.1f}% "
                  f"APLR={res['APLR']:.2f} MinClr={res['MinClr']:.0f} Reward={res['Reward']:.1f}")
        print(f"  [Average Test Rewards] {_avg_test_reward:.1f}")

        metrics['test_episodes'].append(episode)
        for mode in ['in_domain', 'ood']:
            for key in ['SR', 'CR', 'APLR', 'MinClr', 'Reward']:
                mkey = f'test_{mode}_{key}'
                if mkey not in metrics:
                    metrics[mkey] = []
                metrics[mkey].append(eval_results[mode][key])

        
        metrics['test_rewards'].append(eval_results['in_domain']['SR'])
        metrics['test_avg_rewards'].append(eval_results['in_domain']['Reward'])

        lineplot(metrics['test_episodes'], metrics.get('test_in_domain_SR', []), 'SR_InDomain', results_dir)
        lineplot(metrics['test_episodes'], metrics.get('test_ood_SR', []), 'SR_OOD', results_dir)
        lineplot(metrics['test_episodes'], metrics.get('test_in_domain_CR', []), 'CR_InDomain', results_dir)
        lineplot(metrics['test_episodes'], metrics.get('test_ood_CR', []), 'CR_OOD', results_dir)
        lineplot(metrics['test_episodes'], metrics.get('test_in_domain_Reward', []), 'Reward_InDomain', results_dir)
        lineplot(metrics['test_episodes'], metrics.get('test_ood_Reward', []),       'Reward_OOD',       results_dir)
        lineplot(metrics['test_episodes'], metrics.get('test_in_domain_APLR', []),   'APLR_InDomain',    results_dir)
        lineplot(metrics['test_episodes'], metrics.get('test_ood_APLR', []),         'APLR_OOD',         results_dir)
        lineplot(metrics['test_episodes'], metrics.get('test_in_domain_MinClr', []), 'MinClr_InDomain',  results_dir)
        lineplot(metrics['test_episodes'], metrics.get('test_ood_MinClr', []),       'MinClr_OOD',       results_dir)
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        if episode % args.checkpoint_interval == 0:
            with torch.no_grad():
                occ_ious, ades, fdes = [], [], []
                for _ in range(min(5, max(1, D.episodes))):
                    try:
                        _obs, _act, _rew, _nt = D.sample(min(4, args.batch_size), min(20, args.chunk_size))
                    except Exception:
                        break
                    if not isinstance(_obs, dict) or 'semantic_map' not in _obs:
                        break

                    _dev_e = args.device
                    _sem = _obs['semantic_map'].float().to(_dev_e)
                    _T, _B = _sem.shape[:2]
                    K = args.forecast_horizon
                    _img = _obs['image'].float().to(_dev_e)
                    _tgt = _obs['target'].float().to(_dev_e)
                    _pos = _obs['position'].float().to(_dev_e)
                    _TB = _T * _B
                    _flat_img = _img.view(_TB, *_img.shape[2:])
                    _flat_tgt = _tgt.view(_TB, *_tgt.shape[2:])
                    _flat_pos = _pos.view(_TB, -1)
                    _enc_out = encoder(_flat_img, _flat_tgt, _flat_pos)
                    _flat_emb = _enc_out[0] if isinstance(_enc_out, tuple) else _enc_out
                    _full_emb = _flat_emb.view(_T, _B, -1)
                    _embed = _full_emb[1:]
                    _flat_sem = _sem.view(_TB, *_sem.shape[2:])
                    _flat_me = map_encoder(_flat_sem).view(_T, _B, -1)
                    _me_for_tm = _flat_me[:-1]   
                    _me_for_post = _flat_me[1:]   
                    _me_for_eval = _flat_me[1:]
                    _init_b = torch.zeros(_B, args.belief_size, device=_dev_e)
                    _init_s = torch.zeros(_B, args.state_size, device=_dev_e)
                    _tm_out = transition_model(
                        _init_s, _act[:-1].to(_dev_e), _init_b, _embed,
                        _nt[:-1].to(_dev_e),
                        map_embeddings=_me_for_tm,
                        map_embeddings_post=_me_for_post,
                    )
                    _beliefs = _tm_out[0]           
                    _post_states = _tm_out[4]       
                    _T_loss = _beliefs.shape[0]
                    grid_y, grid_x = torch.meshgrid(
                        torch.arange(30, device=_dev_e, dtype=torch.float32),
                        torch.arange(30, device=_dev_e, dtype=torch.float32), indexing='ij')
                    _eval_coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

                    
                    for ti in range(0, max(1, _T_loss - K)):
                        _hh = _beliefs[ti]           
                        _zz = _post_states[ti]       
                        _me = _me_for_eval[ti]       
                        _occ_p, _ = obstacle_forecaster(_hh, _zz, _me)

                        for k in range(K):
                            fut_t = ti + k + 1
                            if fut_t >= _T_loss:
                                break
                            gt_occ = _sem[fut_t + 1, :, 3, :, :].float()
                            pred_occ = _occ_p[:, k]

                            gt_bin = (gt_occ > 0.3).float()
                            pred_bin = (pred_occ > 0.3).float()
                            intersection = (gt_bin * pred_bin).sum(dim=(1, 2))
                            union = ((gt_bin + pred_bin) > 0).float().sum(dim=(1, 2))
                            iou = (intersection / union.clamp(min=1)).mean().item()
                            occ_ious.append(iou)
                            gt_w = gt_occ.reshape(_B, -1).softmax(dim=-1)
                            pred_w = pred_occ.reshape(_B, -1).softmax(dim=-1)
                            gt_c = torch.matmul(gt_w, _eval_coords)
                            pred_c = torch.matmul(pred_w, _eval_coords)
                            disp = torch.norm(gt_c - pred_c, dim=-1).mean().item()
                            ades.append(disp)
                            if k == K - 1:
                                fdes.append(disp)

                    del _img, _tgt, _pos, _sem, _flat_img, _flat_tgt, _flat_pos
                    del _flat_emb, _full_emb, _embed, _flat_sem, _flat_me
                    del _beliefs, _post_states, _tm_out
                    torch.cuda.empty_cache()

                avg_occ_iou = float(np.mean(occ_ious)) if occ_ious else 0.0
                avg_ade = float(np.mean(ades)) if ades else 0.0
                avg_fde = float(np.mean(fdes)) if fdes else 0.0

                print(f"  [Forecast] Occ-IoU={avg_occ_iou:.3f} ADE={avg_ade:.2f} FDE={avg_fde:.2f}")
                writer.add_scalar('Eval/Occ_IoU', avg_occ_iou, episode)
                writer.add_scalar('Eval/ADE', avg_ade, episode)
                writer.add_scalar('Eval/FDE', avg_fde, episode)
                metrics.setdefault('forecast_eval_episodes', []).append(episode)
                metrics.setdefault('forecast_Occ_IoU', []).append(avg_occ_iou)
                metrics.setdefault('forecast_ADE', []).append(avg_ade)
                metrics.setdefault('forecast_FDE', []).append(avg_fde)
                lineplot(metrics['forecast_eval_episodes'], metrics['forecast_Occ_IoU'],
                         'Forecast_Occ_IoU', results_dir)
                lineplot(metrics['forecast_eval_episodes'], metrics['forecast_ADE'],
                         'Forecast_ADE', results_dir)
                lineplot(metrics['forecast_eval_episodes'], metrics['forecast_FDE'],
                         'Forecast_FDE', results_dir)
                torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        for m in world_model_modules:
            m.train()
        
        q_net.train()

    _current_eps = 1.0 if wm_pretraining else float(args.q_epsilon_env)
    args.action_noise = _current_eps
    writer.add_scalar('Train/epsilon', _current_eps, episode)

    with torch.no_grad():
        observation, total_reward = env.reset(), 0
        belief = torch.zeros(1, args.belief_size, device=args.device)
        posterior_state = torch.zeros(1, args.state_size, device=args.device)
        
        map_emb = torch.zeros(1, args.map_embedding_size, device=args.device)
        prev_map = None  
        action = torch.zeros(1, env.action_size, device=args.device)

        _train_reached = False
        _train_collided = False
        _train_inner = env._env if hasattr(env, '_env') else env

        pbar = tqdm(range(args.max_episode_length // args.action_repeat))
        for t in pbar:
            if wm_pretraining:
                action = env.sample_random_action()
                next_observation, reward, done = env.step(action)
            else:
                (belief, posterior_state, map_emb, prev_map,
                 action, next_observation, reward, done) = update_belief_and_act(
                    args, env, planner, transition_model, encoder,
                    belief, posterior_state, action, observation,
                    map_encoder=map_encoder,
                    current_map_embedding=map_emb,
                    map_updater=map_updater, prev_map=prev_map,
                    explore=True,
                )
            D.append(observation, action.cpu() if torch.is_tensor(action) else action, reward, done)
            total_reward += reward
            observation = next_observation

            _info = getattr(_train_inner, '_last_info', {}) or {}
            if _info.get('reach', False):
                _train_reached = True
            if _info.get('collision', False):
                _train_collided = True

            if args.render:
                env.render()
            if done:
                pbar.close()
                break

        metrics['steps'].append(t + metrics['steps'][-1])
        metrics['episodes'].append(episode)
        metrics['train_rewards'].append(total_reward)
        metrics['train_success'].append(1 if _train_reached else 0)
        metrics['train_collision'].append(1 if _train_collided else 0)

        _win = min(TRAIN_METRIC_WINDOW, len(metrics['train_success']))
        _train_sr = float(np.mean(metrics['train_success'][-_win:])) * 100.0
        _train_cr = float(np.mean(metrics['train_collision'][-_win:])) * 100.0
        metrics['train_sr'].append(_train_sr)
        metrics['train_cr'].append(_train_cr)

        writer.add_scalar('Train/Success', metrics['train_success'][-1], episode)
        writer.add_scalar('Train/Collision', metrics['train_collision'][-1], episode)
        writer.add_scalar('Train/SR',  _train_sr, episode)
        writer.add_scalar('Train/CR',  _train_cr, episode)

        lineplot(
            metrics['episodes'][-len(metrics['train_rewards']):],
            metrics['train_rewards'], 'Train Rewards', results_dir,
        )
        
        
        _sr_x = metrics['episodes'][-len(metrics['train_sr']):]
        lineplot(_sr_x, metrics['train_sr'], 'Train SR', results_dir)
        lineplot(_sr_x, metrics['train_cr'], 'Train CR', results_dir)

    if episode % args.checkpoint_interval == 0:
        save_dict = {
            'transition_model': transition_model.state_dict(),
            'observation_model': observation_model.state_dict(),
            'reward_model': reward_model.state_dict(),
            'continuation_model': continuation_model.state_dict(),
            'encoder': encoder.state_dict(),
            
            'q_net': q_net.state_dict(),
            'target_q_net': target_q_net.state_dict(),
            'q_update_count': q_update_count,
            'map_encoder': map_encoder.state_dict(),
            'map_transition_model': map_transition_model.state_dict(),
            'obstacle_forecaster': obstacle_forecaster.state_dict(),
            'map_updater': map_updater.state_dict(),
            'model_optimizer': model_optimizer.state_dict(),
            'q_optimizer': q_optimizer.state_dict(),
        }
        if semantic_extractor is not None:
            save_dict['semantic_extractor'] = semantic_extractor.state_dict()
        torch.save(save_dict, os.path.join(results_dir, 'models_%d.pth' % episode))
        if args.checkpoint_experience:
            torch.save(D, os.path.join(results_dir, 'experience.pth'))

print("\n" + "=" * 92)
print(" " * 26 + "POST-TRAINING FINAL EVALUATION")
print("=" * 92)
for _m in world_model_modules:
    _m.eval()
q_net.eval()
_final_occ_ious, _final_ades, _final_fdes = [], [], []
with torch.no_grad():
    for _ in range(min(10, max(1, D.episodes))):
        try:
            _obs_f, _act_f, _rew_f, _nt_f = D.sample(
                min(4, args.batch_size), min(20, args.chunk_size)
            )
        except Exception:
            break
        if not isinstance(_obs_f, dict) or 'semantic_map' not in _obs_f:
            break

        _dev_f = args.device
        _sem_f = _obs_f['semantic_map'].float().to(_dev_f)
        _T_f, _B_f = _sem_f.shape[:2]
        _K_f = args.forecast_horizon
        _TB_f = _T_f * _B_f

        _img_f = _obs_f['image'].float().to(_dev_f)
        _tgt_f = _obs_f['target'].float().to(_dev_f)
        _pos_f = _obs_f['position'].float().to(_dev_f)
        _enc_f = encoder(
            _img_f.view(_TB_f, *_img_f.shape[2:]),
            _tgt_f.view(_TB_f, *_tgt_f.shape[2:]),
            _pos_f.view(_TB_f, -1),
        )
        _flat_emb_f = _enc_f[0] if isinstance(_enc_f, tuple) else _enc_f
        _embed_f = _flat_emb_f.view(_T_f, _B_f, -1)[1:]

        _flat_me_f = map_encoder(_sem_f.view(_TB_f, *_sem_f.shape[2:])).view(_T_f, _B_f, -1)
        _me_tm_f, _me_post_f = _flat_me_f[:-1], _flat_me_f[1:]

        _init_b_f = torch.zeros(_B_f, args.belief_size, device=_dev_f)
        _init_s_f = torch.zeros(_B_f, args.state_size, device=_dev_f)
        _tm_f = transition_model(
            _init_s_f, _act_f[:-1].to(_dev_f), _init_b_f, _embed_f,
            _nt_f[:-1].to(_dev_f),
            map_embeddings=_me_tm_f, map_embeddings_post=_me_post_f,
        )
        _beliefs_f, _poststates_f = _tm_f[0], _tm_f[4]
        _T_loss_f = _beliefs_f.shape[0]

        _gy, _gx = torch.meshgrid(
            torch.arange(30, device=_dev_f, dtype=torch.float32),
            torch.arange(30, device=_dev_f, dtype=torch.float32), indexing='ij')
        _coords_f = torch.stack([_gx.reshape(-1), _gy.reshape(-1)], dim=-1)

        for ti in range(0, max(1, _T_loss_f - _K_f)):
            _occ_p, _ = obstacle_forecaster(
                _beliefs_f[ti], _poststates_f[ti], _me_post_f[ti]
            )
            for k in range(_K_f):
                fut = ti + k + 1
                if fut >= _T_loss_f:
                    break
                gt_occ = _sem_f[fut + 1, :, 3, :, :].float()
                pred_occ = _occ_p[:, k]

                gt_bin = (gt_occ > 0.3).float()
                pr_bin = (pred_occ > 0.3).float()
                inter = (gt_bin * pr_bin).sum(dim=(1, 2))
                union = ((gt_bin + pr_bin) > 0).float().sum(dim=(1, 2))
                _final_occ_ious.append((inter / union.clamp(min=1)).mean().item())

                gt_w = gt_occ.reshape(_B_f, -1).softmax(dim=-1)
                pr_w = pred_occ.reshape(_B_f, -1).softmax(dim=-1)
                disp = torch.norm(
                    torch.matmul(gt_w, _coords_f) - torch.matmul(pr_w, _coords_f),
                    dim=-1
                ).mean().item()
                _final_ades.append(disp)
                if k == _K_f - 1:
                    _final_fdes.append(disp)

        del _img_f, _tgt_f, _pos_f, _sem_f, _flat_emb_f, _embed_f
        del _flat_me_f, _beliefs_f, _poststates_f, _tm_f
        torch.cuda.empty_cache()

final_occ_iou = float(np.mean(_final_occ_ious)) if _final_occ_ious else 0.0
final_ade = float(np.mean(_final_ades)) if _final_ades else 0.0
final_fde = float(np.mean(_final_fdes)) if _final_fdes else 0.0


_final_eval_results = {}   


_eval_tasks_dict = get_eval_task_list(n_episodes=args.test_episodes)
for _eval_mode, _mode_tasks in _eval_tasks_dict.items():
    _per_ep_records = []
    with torch.no_grad():
        for _task in _mode_tasks:
            observation = env.reset(map_seed=_task['map_seed'], ood=_task['ood'])
            belief = torch.zeros(1, args.belief_size, device=args.device)
            posterior_state = torch.zeros(1, args.state_size, device=args.device)
            map_emb = torch.zeros(1, args.map_embedding_size, device=args.device)
            prev_map = None
            action = torch.zeros(1, env.action_size, device=args.device)
            done = False

            path_length = 0.0
            min_clearance = float('inf')
            ep_reward = 0.0
            reached, collided = False, False

            _inner_env = env._env if hasattr(env, '_env') else env
            start_pos = (_inner_env.agent_pos.copy()
                         if hasattr(_inner_env, 'agent_pos') else None)

            while not done:
                (belief, posterior_state, map_emb, prev_map,
                 action, next_observation, reward, done) = update_belief_and_act(
                    args, env, planner, transition_model, encoder,
                    belief, posterior_state, action, observation,
                    map_encoder=map_encoder,
                    current_map_embedding=map_emb,
                    map_updater=map_updater, prev_map=prev_map,
                    explore=False,
                )
                r_val = reward.item() if torch.is_tensor(reward) else float(reward)
                ep_reward += r_val
                path_length += 1.0

                if hasattr(_inner_env, 'obstacles') and hasattr(_inner_env, 'agent_pos'):
                    for _obs_j in _inner_env.obstacles:
                        _d = float(np.linalg.norm(_inner_env.agent_pos - _obs_j.q))
                        if _d < min_clearance:
                            min_clearance = _d

                
                _info = getattr(_inner_env, '_last_info', {}) or {}
                if _info.get('reach', False):
                    reached = True
                if _info.get('collision', False):
                    collided = True
                observation = next_observation

            
            ep_aplr = float('nan')
            if reached and start_pos is not None and hasattr(_inner_env, 'target_pos'):
                _tgt = _inner_env.target_pos
                _shortest = max(abs(start_pos[0] - _tgt[0]),
                                abs(start_pos[1] - _tgt[1])) / _inner_env.cell_size
                if _shortest > 0:
                    ep_aplr = path_length / _shortest

            _per_ep_records.append({
                'SR':     100.0 if reached else 0.0,
                'CR':     100.0 if collided else 0.0,
                'APLR':   ep_aplr,
                'MinClr': (min_clearance if min_clearance < float('inf') else 0.0),
                'Reward': ep_reward,
            })
    _final_eval_results[_eval_mode] = _per_ep_records


_col = "{:>4} {:>8} {:>8} {:>8} {:>10} {:>10} {:>8} {:>8} {:>10}"

def _fmt(v, spec):
    """Fmt."""
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
print(" Paper Table I — Averaged over all test episodes")
print("=" * 92)
print("  SR (%)↑          = {:>7.2f}    [train, sliding-{} ep]".format(avg_SR, TRAIN_METRIC_WINDOW))
print("  APLR↓            = {:>7.3f}".format(avg_APLR))
print("  CR (%)↓          = {:>7.2f}    [train, sliding-{} ep]".format(avg_CR, TRAIN_METRIC_WINDOW))
print("  MinClr↑          = {:>7.2f}".format(avg_MinClr))
print("  Occ-IoU↑         = {:>7.3f}".format(final_occ_iou))
print("  ADE↓             = {:>7.2f}".format(final_ade))
print("  FDE↓             = {:>7.2f}".format(final_fde))
print("  SR-OOD (%)↑      = {:>7.2f}".format(avg_SR_OOD))
print("  CR-OOD (%)↓      = {:>7.2f}".format(avg_CR_OOD))
print("  --------------------------------")
print("  [aux] In-Domain Eval SR  = {:>7.2f}".format(avg_SR_indomain))
print("  [aux] In-Domain Eval CR  = {:>7.2f}".format(avg_CR_indomain))
print("  Avg Test Reward (In-Domain)  = {:>7.2f}".format(avg_Rew_ID))
print("  Avg Test Reward (OOD)        = {:>7.2f}".format(avg_Rew_OD))
print("  Average Test Rewards (all)   = {:>7.2f}".format(avg_Rew_all))
print("=" * 92)

_final_step = metrics['steps'][-1] if len(metrics['steps']) > 0 else args.episodes

for _mode_name in ['in_domain', 'ood']:
    for _i, _rec in enumerate(_final_eval_results[_mode_name], 1):
        for _k, _v in _rec.items():
            if _v is None or (isinstance(_v, float) and np.isnan(_v)):
                continue
            writer.add_scalar(f'FinalTest/{_mode_name}_ep{_i}_{_k}', float(_v), _final_step)

writer.add_scalar('FinalTest/Average_SR',       avg_SR,      _final_step)   
writer.add_scalar('FinalTest/Average_CR',       avg_CR,      _final_step)   
writer.add_scalar('FinalTest/InDomain_SR_aux',  avg_SR_indomain, _final_step)  
writer.add_scalar('FinalTest/InDomain_CR_aux',  avg_CR_indomain, _final_step)
writer.add_scalar('FinalTest/Average_APLR',     avg_APLR,    _final_step)
writer.add_scalar('FinalTest/Average_MinClr',   avg_MinClr,  _final_step)
writer.add_scalar('FinalTest/Average_Occ_IoU',  final_occ_iou, _final_step)
writer.add_scalar('FinalTest/Average_ADE',      final_ade,   _final_step)
writer.add_scalar('FinalTest/Average_FDE',      final_fde,   _final_step)
writer.add_scalar('FinalTest/Average_SR_OOD',   avg_SR_OOD,  _final_step)
writer.add_scalar('FinalTest/Average_CR_OOD',   avg_CR_OOD,  _final_step)
writer.add_scalar('FinalTest/Average_Test_Rewards',          avg_Rew_all, _final_step)
writer.add_scalar('FinalTest/Average_Test_Rewards_InDomain', avg_Rew_ID,  _final_step)
writer.add_scalar('FinalTest/Average_Test_Rewards_OOD',      avg_Rew_OD,  _final_step)

_table_text = (
    "| SR (%)↑ | APLR↓ | CR (%)↓ | MinClr↑ | Occ-IoU↑ | ADE↓ | FDE↓ | SR-OOD (%)↑ | CR-OOD (%)↓ |\n"
    "|---------|-------|---------|---------|----------|------|------|-------------|-------------|\n"
    f"| {avg_SR:.2f}   | {avg_APLR:.3f} | {avg_CR:.2f}   "
    f"| {avg_MinClr:.2f}   | {final_occ_iou:.3f}    | {final_ade:.2f} | {final_fde:.2f} "
    f"| {avg_SR_OOD:.2f}       | {avg_CR_OOD:.2f}       |"
)
try:
    writer.add_text('FinalTest/PaperTableI', _table_text, _final_step)
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
    'occ_iou':                final_occ_iou,
    'ade':                    final_ade,
    'fde':                    final_fde,
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
