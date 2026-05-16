"""
utils.py — 修改对照:

[1] imagine_ahead:
    - [严格论文时序] 先 RSSM transition (公式 17-18)，再 MapTransitionModel (公式 19)，最后 MapEncoder (公式 20)
    - 在想象中维护完整 6×30×30 语义地图
    - 删除 _imagine_map_embedding 残差近似

[2] lambda_return:
    - 加入 continuation predictions ĉ (公式 31-32)

[3] FreezeParameters: 不变

[4] trk_loss_multi_peak (公式 25, 43):
    - [新增] 多障碍物 tracking 损失
    - GT peak (NMS + top-K, 无梯度) 作 anchor
    - Pred 在每个 anchor 周围 patch 内做 softmax 软质心 (可微)
    - smooth-L1 逐 anchor 计算
    - 替代原 main.py 里的单质心实现 (多障碍时坍缩为几何平均)
"""

import torch
from torch.nn import functional as F
from typing import Iterable


class FreezeParameters:
    """Context manager to locally freeze gradients for a list of modules."""
    def __init__(self, modules: Iterable[torch.nn.Module]):
        self.modules = modules
        self.params = []
        self.requires_grad_states = []

    def __enter__(self):
        for module in self.modules:
            self.params.extend(list(module.parameters()))
        self.requires_grad_states = [p.requires_grad for p in self.params]
        for p in self.params:
            p.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p, state in zip(self.params, self.requires_grad_states):
            p.requires_grad = state


def imagine_ahead(
    prev_state,
    prev_belief,
    prev_map_embedding,
    prev_semantic_map,
    policy,
    transition_model,
    map_encoder,
    map_transition_model,
    planning_horizon=12,
):
    """
    [论文公式 17-20] 想象阶段: 在隐空间中推演未来轨迹.

    [方案A] 完整地图传播 — 严格实现公式 (19)-(20).
    [M1 修复] 移除 prev_semantic_state — 论文公式(13)不含独立 semantic_state.

    Args:
        prev_state:          (1, B*T_chunk, state_size)
        prev_belief:         (1, B*T_chunk, belief_size)
        prev_map_embedding:  (1, B*T_chunk, map_embedding_size)
        prev_semantic_map:   (1, B*T_chunk, 6, Ng, Ng) or None
        policy:              ActorModel
        transition_model:    TransitionModel
        map_encoder:         MapEncoder
        map_transition_model: MapTransitionModel
        planning_horizon:    int

    Returns:
        beliefs, prior_states, prior_means, prior_std_devs,
        map_embeddings, actions — 各 (H, N, *)
    """
    flatten = lambda x: x.view([-1] + list(x.size())[2:])
    prev_belief = flatten(prev_belief)
    prev_state = flatten(prev_state)
    prev_map_embedding = flatten(prev_map_embedding)

    has_full_map = (prev_semantic_map is not None)
    if has_full_map:
        prev_semantic_map = flatten(prev_semantic_map)

    T = planning_horizon
    N = prev_belief.size(0)

    beliefs = [torch.empty(0)] * T
    prior_states = [torch.empty(0)] * T
    prior_means = [torch.empty(0)] * T
    prior_std_devs = [torch.empty(0)] * T
    map_embeddings = [torch.empty(0)] * T
    actions = [torch.empty(0)] * T

    current_belief = prev_belief
    current_state = prev_state
    current_map_embedding = prev_map_embedding
    current_map = prev_semantic_map

    for t in range(planning_horizon):
        # ============================================================
        # [防线 4] sanitize 想象循环的状态输入 — 切断 NaN 污染链路
        # ------------------------------------------------------------
        # 上游任一环节产生 NaN (map_transition / map_encoder / 前一步
        # GRU), 会通过 current_* 传到 actor (导致 logits NaN, Categorical
        # 构造失败) 和下一步 transition_model (产生更多 NaN).
        # 每步入口统一清理, 确保下游模块永远拿到 finite 输入.
        # ============================================================
        current_belief        = torch.nan_to_num(current_belief,        nan=0.0, posinf=50.0, neginf=-50.0)
        current_state         = torch.nan_to_num(current_state,         nan=0.0, posinf=20.0, neginf=-20.0)
        current_map_embedding = torch.nan_to_num(current_map_embedding, nan=0.0, posinf=50.0, neginf=-50.0)

        # [公式 29] 动作选择：a_t ~ π/Q(h_t, z_t, m_t)
        _action = policy.get_action(current_belief, current_state, current_map_embedding)
        actions[t] = _action

        # [严格论文时序修复]
        # 公式 (17)-(18): 先用当前地图嵌入 m_t 进入 RSSM，得到 h_{t+1}, z_{t+1}
        #   h_{t+1} = f(h_t, z_t, a_t, m_t)
        #   z_{t+1} ~ p(z_{t+1} | h_{t+1})
        output = transition_model(
            current_state,
            _action.unsqueeze(0),
            current_belief,
            observations=None,
            nonterminals=None,
            map_embeddings=current_map_embedding.unsqueeze(0),
        )

        next_belief = output[0][0]
        next_state = output[1][0]
        _prior_mean = output[2][0]
        _prior_std_dev = output[3][0]

        # 公式 (19)-(20): 再用 h_{t+1}, z_{t+1} 更新语义地图，并编码 m_{t+1}
        #   M_{t+1} = T_omega(M_t, h_{t+1}, z_{t+1}, a_t)
        #   m_{t+1} = E_m(M_{t+1})
        if has_full_map:
            next_map = map_transition_model(
                current_map, next_belief, next_state, _action, current_map_embedding
            )
            next_map_embedding = map_encoder(next_map)
        else:
            next_map = current_map
            next_map_embedding = current_map_embedding

        current_belief = next_belief
        current_state = next_state
        current_map = next_map
        current_map_embedding = next_map_embedding

        beliefs[t] = current_belief
        prior_states[t] = current_state
        prior_means[t] = _prior_mean
        prior_std_devs[t] = _prior_std_dev
        map_embeddings[t] = current_map_embedding

    return (
        torch.stack(beliefs),
        torch.stack(prior_states),
        torch.stack(prior_means),
        torch.stack(prior_std_devs),
        torch.stack(map_embeddings),
        torch.stack(actions),
    )


def lambda_return(imged_reward, value_pred, bootstrap, cont_pred=None,
                  discount=0.99, lambda_=0.95):
    """
    [论文公式 31-32] λ-return 计算.

    G^(n)_τ = Σ_{i=0}^{n-1} γ^i (Π_{j=0}^{i-1} ĉ_{τ+j}) r̂_{τ+i}
              + γ^n (Π_{j=0}^{n-1} ĉ_{τ+j}) V_ψ(s̃_{τ+n})

    V^λ(s̃_τ) = (1-λ) Σ_{n=1}^{H-1} λ^{n-1} G^(n)_τ + λ^{H-1} G^(H)_τ

    简化实现: 标准的 TD(λ) 递推形式, 加入 continuation ĉ.

    Args:
        imged_reward: (H, N) — 想象中的奖励
        value_pred:   (H, N) — 想象中的价值预测
        bootstrap:    (N,)   — 最后一步的 value 估计
        cont_pred:    (H, N) — 继续概率 ĉ (可选, None 时全部设为 1)
        discount:     float  — γ
        lambda_:      float  — λ
    Returns:
        returns: (H, N)
    """
    if cont_pred is None:
        # 无 continuation 预测时, 假设全部继续 (退化为原始 lambda_return)
        cont = torch.ones_like(imged_reward)
    else:
        cont = cont_pred

    # 标准 TD(λ) 递推
    next_values = torch.cat([value_pred[1:], bootstrap[None]], 0)
    # 每步的 one-step target: r + γ * ĉ * V(s')
    # λ-return 混合: inputs = r + γ * ĉ * (1-λ) * V(s')
    inputs = imged_reward + discount * cont * next_values * (1 - lambda_)

    last = bootstrap
    indices = reversed(range(len(inputs)))
    outputs = []
    for index in indices:
        last = inputs[index] + discount * lambda_ * cont[index] * last
        outputs.append(last)
    outputs = list(reversed(outputs))
    returns = torch.stack(outputs, 0)
    return returns


def trk_loss_multi_peak(pred_occ, gt_occ, n_peaks, patch_size=5):
    """
    [论文公式 25, 43] 多障碍物轨迹一致性损失.

    Γ 的合理近似:
        (a) 从 GT 占用图中用 NMS + top-K 提取 K 个峰值中心 (无梯度).
        (b) 在 pred 占用图上, 以每个 GT 峰值为 anchor, 取 patch_size×patch_size
            窗口, 做 softmax 加权软质心 (可微).
        (c) smooth-L1 逐 anchor 计算 pred_soft_centroid ↔ gt_peak_integer_coord.

    说明:
        - 原实现 `_pred_w = _pred_occ.softmax(-1); _pred_c = _pred_w @ _coords`
          对多障碍物而言是**整张图**的软质心, 会坍缩到 K 个障碍物的几何平均.
          在 K=3 场景下 gt_c ≈ geometric mean(3 obs), 失去 per-track 语义,
          loss 无法指导 pred 产生分离的峰值, 也不可直接作为 ADE/FDE 指标.
        - 新实现按 GT peak 空间分治, pred 只需在每个 anchor 局部对齐, 既可微
          又天然避开 Hungarian 匹配的不可微难题.
        - GT 侧 argmax 不可微是 OK 的 — GT 本来就不需要梯度.

    Args:
        pred_occ:   (B, Ng, Ng) 预测占用概率 ∈ [0, 1], 带梯度.
        gt_occ:     (B, Ng, Ng) GT 占用 ∈ [0, 1], detached.
        n_peaks:    int, 障碍物数量 K (= env.num_obstacles).
        patch_size: int, 局部软质心窗口大小 (奇数, 默认 5).
    Returns:
        loss_elem: (B, n_peaks, 2) — 未 reduce 的 smooth-L1. 调用处乘上
                   forecast mask 后 sum / count 得标量 L_trk.
    """
    B, Ng, _ = pred_occ.shape
    device = pred_occ.device
    half = patch_size // 2

    # --- 1. GT peak extraction: NMS (3x3 max-pool) + top-K ------------------
    with torch.no_grad():
        # NMS: 标记局部最大像元
        pooled = F.max_pool2d(
            gt_occ.unsqueeze(1), kernel_size=3, stride=1, padding=1
        ).squeeze(1)
        nms_mask = (gt_occ >= pooled - 1e-8).float()
        gt_masked = gt_occ * nms_mask
        flat = gt_masked.reshape(B, -1)
        _, topk_idx = flat.topk(n_peaks, dim=-1)      # (B, K)
        # 展开为 (row, col) = (y, x). clamp 到 patch 合法中心范围.
        gt_ys_int = (topk_idx // Ng).clamp(half, Ng - 1 - half)
        gt_xs_int = (topk_idx %  Ng).clamp(half, Ng - 1 - half)
        # (B, K, 2) = (x, y) — 与 main.py 里 _coords 的 (grid_x, grid_y) 约定一致
        gt_coords = torch.stack([gt_xs_int.float(), gt_ys_int.float()], dim=-1)

    # --- 2. 在 pred 上每个 GT anchor 附近提取 patch ---------------------------
    offs = torch.arange(-half, half + 1, device=device, dtype=torch.long)
    # patch_y[b,k,i,j] = gt_ys_int[b,k] + offs[i]
    # patch_x[b,k,i,j] = gt_xs_int[b,k] + offs[j]
    patch_y = gt_ys_int.unsqueeze(-1).unsqueeze(-1) + offs.view(1, 1, -1, 1)
    patch_x = gt_xs_int.unsqueeze(-1).unsqueeze(-1) + offs.view(1, 1, 1, -1)
    patch_y = patch_y.expand(B, n_peaks, patch_size, patch_size)
    patch_x = patch_x.expand(B, n_peaks, patch_size, patch_size)

    # advanced indexing: pred_occ[b, patch_y, patch_x]
    b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand_as(patch_y)
    pred_patches = pred_occ[b_idx, patch_y, patch_x]   # (B, K, ps, ps), 梯度畅通

    # --- 3. 软质心 (每 patch 内独立归一化) ----------------------------------
    flat_patches = pred_patches.reshape(B, n_peaks, -1)        # (B, K, ps²)
    weights = flat_patches.softmax(dim=-1)                     # (B, K, ps²)
    local_x = patch_x.float().reshape(B, n_peaks, -1)          # (B, K, ps²)
    local_y = patch_y.float().reshape(B, n_peaks, -1)
    pred_x = (weights * local_x).sum(dim=-1)                   # (B, K)
    pred_y = (weights * local_y).sum(dim=-1)
    pred_coords = torch.stack([pred_x, pred_y], dim=-1)        # (B, K, 2)

    # --- 4. smooth-L1 (不 reduce, 留给调用方乘 mask) -------------------------
    return F.smooth_l1_loss(pred_coords, gt_coords, reduction='none')


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    [DreamerV3 标准] Symmetric log transform:
        symlog(x) = sign(x) · log(1 + |x|)

    用于 reward head 训练目标的尺度压缩. 解决的问题:
        环境奖励跨 ~3 个数量级 (step penalty -0.01, goal +20, collision -10).
        在原始尺度下, reward_model 的 MSE 损失被大奖励主导, -0.01 的梯度
        占比 ~(0.01)² / (10)² ≈ 10⁻⁶, 网络对它完全不敏感 —— reward_model
        预测倾向于稳定在 0 附近, 丢失了"每步都有小惩罚"这一关键信号.

    symlog 后:
        -0.01 → -0.00995       (几乎不压缩)
         0.00 →  0
        10.00 → +2.40
        20.00 → +3.04
        -10.0 → -2.40

    大奖励被压到 ~O(1) 量级, 小奖励保持 ~O(0.01), 梯度比变成 10⁻⁴, 更平衡.

    数值稳定性: 用 log1p 避免 log(1+x) 在小 x 时精度损失.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    [DreamerV3 标准] symlog 的逆变换:
        symexp(y) = sign(y) · (exp(|y|) - 1)

    使用时机: reward_model 的裸输出 tilde_r_t ∈ symlog 空间, 在 imagination
    中消费前先 symexp 回原始尺度, 以便:
        * 风险项 λ_risk · χ 保持原来的权重语义
        * TD target γ·ĉ·Q 的折扣尺度不变
        * Q 网络仍然输出"累积真实奖励"而不需学习 symlog 逆变换

    数值稳定性: 用 expm1 避免 exp(x)-1 在小 x 时精度损失.
    """
    return torch.sign(x) * torch.expm1(torch.abs(x))


def compute_boundary_target(grid_size: int, margin: float = 2.0,
                             device=None, dtype=torch.float32) -> torch.Tensor:
    """
    [工程化扩展] 预计算静态边界风险 GT:

        d(gx, gy)    = min(gx, gy, Ng-1-gx, Ng-1-gy)        # 到最近边的格距
        B_gt(gx, gy) = max(0, 1 - d / margin)

    物理直觉:
        * 贴边 (d=0)       → B=1.0  —— 最高风险
        * 向内 1 格 (d=1)  → B=0.5  —— margin=2 时
        * 内部 (d ≥ margin)→ B=0.0  —— 安全

    该张量是**时不变常量**, 每训练步复用, 不参与梯度流 (detach 语义).

    Args:
        grid_size: Ng, 地图网格边长 (e.g. 30).
        margin:    风险衰减宽度 (格数), 2.0 通常合理.
        device:    目标设备.
        dtype:     目标精度.
    Returns:
        B_gt: (Ng, Ng) float tensor ∈ [0, 1], 不需梯度.
    """
    g = torch.arange(grid_size, dtype=dtype, device=device)
    gx = g.view(-1, 1).expand(grid_size, grid_size)
    gy = g.view(1, -1).expand(grid_size, grid_size)
    d = torch.minimum(
        torch.minimum(gx, grid_size - 1 - gx),
        torch.minimum(gy, grid_size - 1 - gy),
    )
    bdry = (1.0 - d / max(margin, 1e-6)).clamp(min=0.0, max=1.0)
    return bdry.detach()


def lineplot(x, y, name, path, xaxis='Episodes'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        plt.figure()
        plt.plot(x, y)
        plt.xlabel(xaxis)
        plt.ylabel(name)
        plt.savefig(path + '/' + name + '.png')
    except Exception as e:
        print(f"Plotting error: {e}")
    finally:
        plt.close()


def write_video(frames, title, path):
    pass


def numpy_to_torch(array):
    return torch.tensor(array)