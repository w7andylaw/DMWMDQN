# Author: Qiwei Wang
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F   
import gymnasium as gym
from gymnasium import spaces
import mysql.connector
import pickle
import os
import time


GYM_ENVS = ['Pendulum-v1', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2',
            'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2',
            'Walker2d-v2']
CONTROL_SUITE_ENVS = ['acrobot-swingup', 'cartpole-balance', 'cartpole-balance_sparse', 'cartpole-swingup',
                      'cartpole-swingup_sparse', 'ball_in_cup-catch', 'finger-spin', 'finger-turn_easy',
                      'finger-turn_hard', 'hopper-hop', 'hopper-stand', 'pendulum-swingup', 'quadruped-run',
                      'quadruped-walk', 'reacher-easy', 'reacher-hard', 'walker-run', 'walker-stand', 'walker-walk']
CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2, 'cheetah': 4, 'ball_in_cup': 6, 'walker': 2,
                                'humanoid': 2, 'fish': 2, 'acrobot': 4}






















def get_eval_task_list(n_episodes=10, in_domain_base=42, ood_base=9999):
    """Return deterministic in-domain and OOD evaluation tasks."""
    return {
        'in_domain': [{'map_seed': in_domain_base + k, 'ood': False}
                      for k in range(n_episodes)],
        'ood':       [{'map_seed': ood_base + k,       'ood': True}
                      for k in range(n_episodes)],
    }






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












class SiameseNetwork(nn.Module):
    """Siamese visual feature network."""

    def __init__(self, feature_dim=128):
        super(SiameseNetwork, self).__init__()
        try:
            import torchvision.models as models
            resnet = models.resnet18(weights=None)
        except ImportError:
            raise ImportError("需要安装 torchvision: pip install torchvision")
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, feature_dim)  

    def forward_one(self, x):
        """Encode one image branch."""
        x = self.backbone(x)           
        x = x.view(x.size(0), -1)     
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2):
        """Run the module forward pass."""
        return self.forward_one(x1), self.forward_one(x2)






class DynamicObstacle:
    """Dynamic obstacle state and motion model."""

    def __init__(self, q, v, rho, motion_type='cv', D=3000, cell_size=100.0,
                 satellite_map=None, rng=None):
        self.q = np.array(q, dtype=np.float32)
        self.v = np.array(v, dtype=np.float32)
        self.rho = float(rho)
        self.motion_type = motion_type
        self.D = D
        self.cell_size = cell_size
        self.satellite_map = satellite_map
        self.step_count = 0
        self.rng = rng

    def step(self, rng=None):
        """Advance the environment by one action."""
        self.step_count += 1

        if self.motion_type == 'cv':
            self.q = self.q + self.v
            for i in range(2):
                if self.q[i] < self.rho:
                    self.q[i] = self.rho
                    self.v[i] = abs(self.v[i])
                elif self.q[i] > self.D - self.rho:
                    self.q[i] = self.D - self.rho
                    self.v[i] = -abs(self.v[i])

        elif self.motion_type == 'random':
            
            
            
            speed = np.linalg.norm(self.v)
            if speed < 1e-6:
                speed = self.cell_size * 0.5
            base_angle = np.arctan2(self.v[1], self.v[0])
            _uniform = self.rng.uniform if self.rng is not None else np.random.uniform
            drift = _uniform(-np.pi / 6, np.pi / 6)
            new_angle = base_angle + drift
            self.v = np.array([np.cos(new_angle) * speed,
                               np.sin(new_angle) * speed], dtype=np.float32)
            self.q = self.q + self.v
            
            for i in range(2):
                if self.q[i] < self.rho:
                    self.q[i] = self.rho
                    self.v[i] = abs(self.v[i])
                elif self.q[i] > self.D - self.rho:
                    self.q[i] = self.D - self.rho
                    self.v[i] = -abs(self.v[i])

        elif self.motion_type == 'road':
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
                    
                    brightness = float(np.mean(self.satellite_map[px, py]))
                    if brightness > best_score:
                        best_score = brightness
                        best_angle = a

                self.v = np.array([np.cos(best_angle) * speed,
                                   np.sin(best_angle) * speed], dtype=np.float32)
                self.q = self.q + self.v
                self.q = np.clip(self.q, self.rho, self.D - self.rho)
            else:
                
                if self.step_count % 10 == 0:
                    angle += np.pi / 2
                self.v = np.array([np.cos(angle) * speed,
                                   np.sin(angle) * speed], dtype=np.float32)
                self.q = self.q + self.v
                self.q = np.clip(self.q, self.rho, self.D - self.rho)

    @property
    def grid_pos(self):
        """Grid pos."""
        gx = int(np.clip(self.q[0] / self.cell_size, 0, int(self.D / self.cell_size) - 1))
        gy = int(np.clip(self.q[1] / self.cell_size, 0, int(self.D / self.cell_size) - 1))
        return gx, gy

    @property
    def grid_vel(self):
        """Grid vel."""
        return self.v / self.cell_size


class UAVNavigationEnv(gym.Env):
    """Gymnasium environment for UAV visual navigation."""
    CH_EXP = 0   
    CH_REL = 1   
    CH_TRV = 2   
    CH_OCC = 3   
    CH_U   = 4   
    CH_V   = 5   
    N_MAP_CHANNELS = 6

    def __init__(
        self,
        map_size=3000,             
        grid_size=30,              
        max_steps=500,             
        db_config=None,
        semantic_patch_size=30,
        pos_margin_cells=1,
        min_grid_distance=25,
        fixed_position_seed=12345,
        speed_scale=1.0,
        reward_reach=30.0,            
        reward_blocked=-0.2,
        reward_step_penalty=-0.01,    
        reward_collision=-10.0,       
        reward_rel_scale=1.0,         
        reward_risk_scale=0.03,
        reward_explore_scale=0.02,     
        reward_revisit_penalty=-0.02, 
        reward_position_scale=0.1,   
        reach_radius=5,
        similarity_reward_scale=0.0,
        siamese_model_path=None,
        df_max=0.72,
        siamese_device='cuda',
        num_obstacles=5,
        obstacle_speed=20.0,       
        obstacle_rho=40.0,         
        uav_rho=20.0,              
        obstacle_motion_types=('cv', 'random', 'road'),
        forecast_horizon=5,
    ):
        super(UAVNavigationEnv, self).__init__()
        self.D = map_size
        self.Ng = grid_size
        self.max_steps = max_steps
        self.cell_size = self.D / self.Ng   
        self.semantic_patch_size = semantic_patch_size
        self.pos_margin_cells = int(pos_margin_cells)
        self.min_grid_distance = int(min_grid_distance)
        self.fixed_position_seed = int(fixed_position_seed)
        self._fixed_agent_pos = None
        self._fixed_target_pos = None
        self.speed_scale = speed_scale
        self.reward_reach = reward_reach
        self.reward_blocked = float(reward_blocked)
        self.reward_step_penalty = reward_step_penalty
        self.reward_collision = float(reward_collision)
        self.reward_rel_scale = float(reward_rel_scale)
        self.reward_risk_scale = float(reward_risk_scale)
        self.similarity_reward_scale = float(similarity_reward_scale)
        self.reward_explore_scale = float(reward_explore_scale)
        self.reward_revisit_penalty = float(reward_revisit_penalty)
        self.reward_position_scale = float(reward_position_scale)
        self.reach_radius = int(reach_radius)
        self.num_obstacles = int(num_obstacles)
        self.obstacle_speed = float(obstacle_speed)
        self.obstacle_rho = float(obstacle_rho)
        self.uav_rho = float(uav_rho)
        self.obstacle_motion_types = list(obstacle_motion_types)
        self.forecast_horizon = int(forecast_horizon)   
        self.obstacles = []
        self.action_space = spaces.Discrete(8)
        self._dir8 = np.array(
            [
                ( 0,  1),   
                (-1,  1),   
                (-1,  0),   
                (-1, -1),   
                ( 0, -1),   
                ( 1, -1),   
                ( 1,  0),   
                ( 1,  1),   
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
            'target': spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
            'position': spaces.Box(low=0, high=self.D, shape=(2,), dtype=np.float32),
            'semantic_map': spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.N_MAP_CHANNELS, self.Ng, self.Ng),
                dtype=np.float32
            ),
        })
        self.df_max = df_max
        self.siamese_device = siamese_device
        self._target_feature = None
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
        self.satellite_map = np.zeros((self.D, self.D, 3), dtype=np.uint8)
        self.map_ids = []
        self.original_map_size = None
        self.episode_counter = 0
        self._is_test = False
        self._prev_image = None   
        self._last_info = {}
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

                
                
                split_rng = np.random.RandomState(42)  
                split_rng.shuffle(all_ids)
                split_idx = int(len(all_ids) * 0.8)
                self.train_map_ids = all_ids[:split_idx]
                self.test_map_ids = all_ids[split_idx:]
                self.map_ids = self.train_map_ids  
                print(f"[UAVEnv] Maps: {len(self.train_map_ids)} train, "
                      f"{len(self.test_map_ids)} test (held-out).")
            except Exception as e:
                print(f"[UAVEnv] DB Error: {e}")
                self.train_map_ids = []
                self.test_map_ids = []

    
    
    

    def _generate_random_positions(self, rng=None):
        """Generate random positions."""
        _randint = rng.randint if rng is not None else np.random.randint
        lo = int(self.pos_margin_cells)
        hi = int(self.Ng - self.pos_margin_cells)  
        D_min = int(self.min_grid_distance)
        usable = hi - lo
        max_possible_cheb = usable - 1  
        if D_min > max_possible_cheb:
            raise ValueError(
                f"min_grid_distance={D_min} 超出可行上限 {max_possible_cheb} "
                f"(Ng={self.Ng}, margin={lo}). 请调小 min_grid_distance."
            )
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
        print(f"WARNING: Position sampling exhausted {max_attempts} attempts.")
        gx_s, gy_s = lo, lo
        gx_t, gy_t = hi - 1, hi - 1
        return (
            np.array([(gx_s + 0.5) * self.cell_size, (gy_s + 0.5) * self.cell_size], dtype=np.float32),
            np.array([(gx_t + 0.5) * self.cell_size, (gy_t + 0.5) * self.cell_size], dtype=np.float32),
        )

    
    
    

    def _load_map_from_db(self, map_seed=None, ood=False):
        """Load map from db."""
        cache_dir = './map_cache'
        os.makedirs(cache_dir, exist_ok=True)
        split_tag = 'ood' if ood else 'train'

        
        if map_seed is not None:
            cache_path = os.path.join(
                cache_dir, f'test_map_seed_{map_seed}_{split_tag}.npy'
            )
            if os.path.exists(cache_path):
                cached_data = np.load(cache_path, allow_pickle=True).item()
                self.original_map_size = cached_data['original_size']
                return cached_data['map'].astype(np.uint8)

        
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

            
            if map_seed is not None:
                rng = np.random.RandomState(map_seed)
                selected_idx = rng.randint(0, total_maps)
            else:
                selected_idx = np.random.randint(0, total_maps)

            selected_id = self.map_ids[selected_idx]
            print(f"[MapLoader] {mode} mode - Map ID: {selected_id} ({selected_idx}/{total_maps})")

            
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT image_data FROM image_maps2 WHERE id = %s", (int(selected_id),))
            row = cursor.fetchone()
            cursor.close()

            if row is None:
                raise ValueError(f"Map ID {selected_id} not found in database.")

            blob = row[0]

            
            try:
                original_map = pickle.loads(blob)
            except:
                original_map = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)

            if original_map is None:
                raise ValueError(f"Failed to decode map ID {selected_id}")

            
            if len(original_map.shape) == 2:
                original_map = cv2.cvtColor(original_map, cv2.COLOR_GRAY2RGB)
            elif original_map.shape[2] == 4:
                original_map = cv2.cvtColor(original_map, cv2.COLOR_BGRA2RGB)

            self.original_map_size = original_map.shape[:2]

            
            if original_map.shape[0] != self.D or original_map.shape[1] != self.D:
                scaled_map = cv2.resize(original_map, (self.D, self.D), interpolation=cv2.INTER_LINEAR)
            else:
                scaled_map = original_map

            elapsed = time.time() - start_time
            print(f"[MapLoader] Map loaded in {elapsed:.2f}s")

            
            if map_seed is not None:
                cache_path = os.path.join(
                    cache_dir, f'test_map_seed_{map_seed}_{split_tag}.npy'
                )
                np.save(cache_path, {'map': scaled_map, 'original_size': self.original_map_size})

            return scaled_map.astype(np.uint8)

        except Exception as e:
            raise RuntimeError(f"Critical failure loading map from DB: {e}")

    
    
    

    def _get_semantic_patch(self, pos):
        """Get semantic patch."""
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

        return patch.transpose(2, 0, 1)  

    
    def _pos_to_grid(self, pos):
        """Pos to grid."""
        gx = int(np.clip(pos[0] / self.cell_size, 0, self.Ng - 1))
        gy = int(np.clip(pos[1] / self.cell_size, 0, self.Ng - 1))
        return gx, gy

    
    def _get_grid_cell_image(self, gx, gy):
        """Get grid cell image."""
        x_start = int(gx * self.cell_size)
        y_start = int(gy * self.cell_size)
        x_end = min(int((gx + 1) * self.cell_size), self.D)
        y_end = min(int((gy + 1) * self.cell_size), self.D)
        cell_img = self.satellite_map[x_start:x_end, y_start:y_end]
        if cell_img.shape[0] == 0 or cell_img.shape[1] == 0:
            return np.zeros((3, 64, 64), dtype=np.uint8)
        cell_img = cv2.resize(cell_img, (64, 64), interpolation=cv2.INTER_LINEAR)
        return cell_img.transpose(2, 0, 1)  

    def _compute_similarity(self, pos):
        """Compute similarity."""
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

    
    
    

    def _spawn_obstacles(self, rng=None):
        """Spawn obstacles."""
        self.obstacles = []
        margin = self.obstacle_rho + self.uav_rho + self.cell_size  

        
        _choice  = rng.choice  if rng is not None else np.random.choice
        _uniform = rng.uniform if rng is not None else np.random.uniform
        _randint = rng.randint if rng is not None else np.random.randint

        for _ in range(self.num_obstacles):
            mtype = _choice(self.obstacle_motion_types)
            angle = _uniform(0, 2 * np.pi)
            speed = _uniform(self.obstacle_speed * 0.5, self.obstacle_speed * 1.5)
            v = np.array([np.cos(angle) * speed, np.sin(angle) * speed], dtype=np.float32)

            
            q = None
            for _try in range(200):
                q_cand = _uniform(self.obstacle_rho, self.D - self.obstacle_rho, size=2).astype(np.float32)
                d_agent = np.linalg.norm(q_cand - self.agent_pos)
                d_target = np.linalg.norm(q_cand - self.target_pos)
                if d_agent > margin and d_target > margin:
                    q = q_cand
                    break
            if q is None:
                q = q_cand
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
        """Step obstacles."""
        for obs in self.obstacles:
            obs.step()

    def _check_collision(self, pos):
        """Check collision."""
        threshold = self.uav_rho + self.obstacle_rho
        for obs in self.obstacles:
            if np.linalg.norm(pos - obs.q) <= threshold:
                return True
        return False

    def _build_obstacle_channels(self):
        """Build obstacle channels."""
        Mocc = np.zeros((self.Ng, self.Ng), dtype=np.float32)
        Mu   = np.zeros((self.Ng, self.Ng), dtype=np.float32)
        Mv   = np.zeros((self.Ng, self.Ng), dtype=np.float32)
        sigma = 1.5  
        radius = int(np.ceil(3 * sigma))

        for obs in self.obstacles:
            gx, gy = obs.grid_pos
            vx_n, vy_n = obs.grid_vel
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.Ng and 0 <= ny < self.Ng:
                        dist2 = dx ** 2 + dy ** 2
                        weight = np.exp(-dist2 / (2 * sigma ** 2))
                        if weight > Mocc[nx, ny]:
                            Mocc[nx, ny] = weight
                            Mu[nx, ny] = np.clip(vx_n / (self.obstacle_speed / self.cell_size + 1e-6), -1, 1)
                            Mv[nx, ny] = np.clip(vy_n / (self.obstacle_speed / self.cell_size + 1e-6), -1, 1)
        return Mocc, Mu, Mv

    def _compute_risk(self, pos, action_direction=None):
        """Compute risk."""
        import copy
        obs_copies = [copy.deepcopy(o) for o in self.obstacles]
        threshold = self.uav_rho + self.obstacle_rho
        max_occ = 0.0
        for k in range(self.forecast_horizon):  
            if action_direction is not None:
                future_pos = pos + action_direction * self.cell_size * k
                future_pos = np.clip(future_pos, 0, self.D)
            else:
                future_pos = pos
            
            for o in obs_copies:
                dist = np.linalg.norm(future_pos - o.q)
                occ = np.exp(-(dist ** 2) / (2 * (threshold) ** 2))
                if occ > max_occ:
                    max_occ = occ
            
            for o in obs_copies:
                o.step()
        return float(np.clip(max_occ, 0.0, 1.0))

    
    
    

    def _update_map(self, pos, is_current):
        """Update map."""
        gx = int(np.clip(pos[0] / self.cell_size, 0, self.Ng - 1))
        gy = int(np.clip(pos[1] / self.cell_size, 0, self.Ng - 1))
        self.semantic_map[self.CH_EXP, gx, gy] = 1.0 if is_current else 0.0

        
        

    def _refresh_dynamic_channels(self):
        """Refresh dynamic channels."""
        Mocc, Mu, Mv = self._build_obstacle_channels()
        self.semantic_map[self.CH_OCC] = Mocc
        self.semantic_map[self.CH_U]   = Mu
        self.semantic_map[self.CH_V]   = Mv

    
    
    

    def _get_current_view(self, pos):
        """Get current view."""
        x, y = int(pos[0]) + 32, int(pos[1]) + 32
        padded = np.pad(self.satellite_map, ((32, 32), (32, 32), (0, 0)), mode='constant')
        view = padded[x - 32:x + 32, y - 32:y + 32]
        if view.shape != (64, 64, 3):
            view = cv2.resize(view, (64, 64))
        return view.transpose(2, 0, 1)

    def _apply_obs_noise(self, image):
        """Apply obs noise."""
        if self._is_test:
            self._prev_image = image.copy()
            return image

        if hasattr(self, '_prev_image') and self._prev_image is not None:
            if np.random.random() < 0.10:
                stale = self._prev_image.copy()
                self._prev_image = image.copy()
                return stale
        self._prev_image = image.copy()

        if np.random.random() < 0.05:
            return np.zeros_like(image)

        noise = np.random.normal(0, 5, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    
    
    

    def _get_obs(self):
        img = self._get_current_view(self.agent_pos)
        tgt = self._get_current_view(self.target_pos)
        return {
            'image': self._apply_obs_noise(img),
            'target': tgt,
            'position': self.agent_pos.copy(),
            'semantic_map': self.semantic_map.copy(),
        }

    
    
    

    def reset(self, seed=None, options=None, map_seed=None, ood=False):
        """Reset environment state."""
        super().reset(seed=seed)
        self.episode_counter += 1
        self._is_test = (map_seed is not None)

        
        if ood and hasattr(self, 'test_map_ids') and self.test_map_ids:
            self.map_ids = self.test_map_ids
        elif hasattr(self, 'train_map_ids') and self.train_map_ids:
            self.map_ids = self.train_map_ids
        self.satellite_map = self._load_map_from_db(map_seed=map_seed, ood=ood)
        if self._is_test:
            test_rng = np.random.RandomState(map_seed if map_seed is not None
                                             else self.fixed_position_seed)
            self.agent_pos, self.target_pos = self._generate_random_positions(rng=test_rng)
            self._obs_rng = test_rng
        else:
            self.agent_pos, self.target_pos = self._generate_random_positions(rng=None)
            self._obs_rng = None
        self.init_dist = float(np.linalg.norm(self.agent_pos - self.target_pos))
        self.steps = 0

        
        tgx, tgy = self._pos_to_grid(self.target_pos)
        target_img = self._get_grid_cell_image(tgx, tgy)
        with torch.no_grad():
            target_tensor = torch.tensor(
                target_img, dtype=torch.float32
            ).unsqueeze(0).to(self.siamese_device) / 255.0
            self._target_feature = self.siamese_net.forward_one(target_tensor)

        
        self.semantic_map = np.zeros((self.N_MAP_CHANNELS, self.Ng, self.Ng), dtype=np.float32)
        self.semantic_map[self.CH_EXP] = -1.0
        self.semantic_map[self.CH_REL] = -1.0
        self.semantic_map[self.CH_TRV] = 1.0
        self._spawn_obstacles(rng=self._obs_rng)
        self._refresh_dynamic_channels()
        self._update_map(self.agent_pos, True)
        return self._get_obs(), {}

    def step(self, action):
        """Advance the environment by one action."""
        self.steps += 1
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
        self._step_obstacles()
        reward += self.reward_step_penalty
        proposed_out_of_bound = not (0 <= new_pos[0] <= self.D and
                                      0 <= new_pos[1] <= self.D)
        if proposed_out_of_bound:
            new_pos = self.agent_pos.copy()
            reward += self.reward_blocked
            info['blocked'] = True

        new_gx, new_gy = self._pos_to_grid(new_pos)
        target_gx, target_gy = self._pos_to_grid(self.target_pos)
        cheb_to_target = max(abs(new_gx - target_gx), abs(new_gy - target_gy))
        if cheb_to_target <= self.reach_radius:
            reward += self.reward_reach   
            terminated = True
            info['reach'] = True
        else:
            collision = self._check_collision(new_pos)
            if collision:
                reward += self.reward_collision   
                self.semantic_map[self.CH_TRV, new_gx, new_gy] = -1.0
                terminated = True
                info['collision'] = True

        if not terminated:
            
            is_first_visit = bool(self.semantic_map[self.CH_EXP, new_gx, new_gy] == -1.0)
            if is_first_visit:
                reward += self.reward_explore_scale   
            else:
                reward += self.reward_revisit_penalty
            old_gx, old_gy = self._pos_to_grid(self.agent_pos)
            mrel_old = self.semantic_map[self.CH_REL, old_gx, old_gy]
            vs_new = self._compute_similarity(new_pos)
            self.semantic_map[self.CH_REL, new_gx, new_gy] = vs_new
            if self.reward_rel_scale > 0.0:
                delta_s = float(vs_new) - float(mrel_old)
                reward += self.reward_rel_scale * max(0.0, delta_s)
            if self.reward_position_scale > 0.0:
                rel_new = float(self.semantic_map[self.CH_REL, new_gx, new_gy])
                rel_old = float(self.semantic_map[self.CH_REL, old_gx, old_gy])
                if rel_new >= 0 and rel_old >= 0:
                    pos_progress = rel_new - rel_old
                    reward += self.reward_position_scale * max(0.0, pos_progress)
            if self.reward_risk_scale > 0.0:
                chi = self._compute_risk(new_pos, action_direction=direction)
                reward -= self.reward_risk_scale * chi
                info['risk'] = chi
            if self.similarity_reward_scale > 0.0:
                if is_first_visit:
                    reward += self.similarity_reward_scale * vs_new

        self._update_map(self.agent_pos, False)   
        self.agent_pos = new_pos.astype(np.float32)
        self._update_map(self.agent_pos, True)    
        self._refresh_dynamic_channels()

        if self.steps >= self.max_steps:
            truncated = True
        self._last_info = info
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if self.db_conn:
            self.db_conn.close()






class UAVEnvWrapper:
    """Torch-friendly wrapper for the UAV environment."""

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
        
        return processed

    def sample_random_action(self):
        
        a = self._env.action_space.sample()
        return torch.tensor([int(a)], dtype=torch.int64)

    def close(self):
        self._env.close()






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
            
            if done_mask.any():
                for k in final_obs:
                    final_obs[k][done_mask] = 0
        else:
            final_obs = torch.cat(obs)
            final_obs[done_mask] = 0
        return final_obs, torch.tensor(rs, dtype=torch.float32), torch.tensor(ds, dtype=torch.uint8)

    def close(self):
        [e.close() for e in self.envs]