"""
train_siamese_v2.py — 重新设计的 Siamese 预训练 (SimCLR 范式)

== 与 v1 (train_siamese.py) 的核心区别 ==
1. 训练目标修正:
   v1: 距离 < 5 网格 → 正样本   (本质是"判断空间是否接近", 与 env.py 用法不匹配!)
   v2: 同一图块 + 强增强 → 正样本 (判断"是不是同一个地点", 与 env.py 用法一致)

2. InfoNCE / NT-Xent 损失 + L2 归一化特征
   - 输出归一化到单位球面: df ∈ [0, 2]
   - 同 batch 内其他样本作为负样本 (隐式 hard negative mining)

3. 完整的航拍数据增强 (旋转/翻转/颜色/模糊/裁剪)

4. 验证集 + AUC 评估 + 自动建议 df_max

5. 现代训练配方: AdamW + cosine schedule + grad clip + 80 epoch

== env.py 必须配套的修改 ==
1. SiameseNetwork.forward_one 末尾加: x = F.normalize(x, p=2, dim=1)
2. df_max 从 5.0 改为脚本训练完后建议的值 (通常 0.6 ~ 1.0)

用法:
  python train_siamese_v2.py
  python train_siamese_v2.py --epochs 100 --batch-size 256
"""

import argparse
import gc
import os
import time
import pickle

# [抑制 C 层错误输出] OpenCV/libpng/libtiff 遇到坏图会往 stderr 直接写红字,
# Python try-except 接不到. 这里在 import cv2 之前关掉 OpenCV 日志.
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

import cv2
cv2.setLogLevel(0)   # 进一步确保静默

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import mysql.connector

# [抑制 PIL 坏图警告] 一些坏 PNG 只是部分损坏, PIL 能读一部分, 但会 warning 刷屏
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    raise ImportError("需要 scikit-learn: pip install scikit-learn")

from tqdm.auto import tqdm


# ============================================================================
#  Siamese Network — 与 env.py 完全兼容 (ResNet18 + Linear(512,128))
# ============================================================================

class SiameseNetwork(nn.Module):
    """
    与 env.py 中的 SiameseNetwork 同结构. state_dict 可互相加载.

    [关键改动] forward_one 输出 L2 归一化的 128 维特征.
    这样 df = ||f1 - f2||_2 ∈ [0, 2], 训练目标更稳定 (cosine 几何).
    """

    def __init__(self, feature_dim=128, normalize=True, pretrained=True):
        super().__init__()
        import torchvision.models as models
        # [关键] 用 ImageNet 预训练权重初始化, 比从零开始训练效率高 2-3x
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, feature_dim)
        self.normalize = normalize

    def forward_one(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)


# ============================================================================
#  数据增强 — 航拍/卫星图专用
# ============================================================================

def build_augmentation():
    """
    航拍图像的强增强 pipeline. 输入 HWC uint8, 输出 CHW float ∈ [0, 1].

    设计原则:
      - 旋转 360°: UAV 可任意朝向
      - 翻转: 俯视航拍图天然对称
      - 颜色抖动: 模拟不同时段/季节的光照
      - 模糊: 镜头失焦/低空气流抖动
      - 随机裁剪缩放: 模拟轻微高度变化
      - 不做颜色反转/通道乱序 (会破坏地物语义)
    """
    return T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(64, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        T.RandomRotation(degrees=180, fill=0),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.ToTensor(),  # → CHW float [0, 1]
    ])


# ============================================================================
#  Dataset: SimCLR 风格 — 每次返回同一图块的两个独立增强视图
# ============================================================================

class CellSimCLRDataset(Dataset):
    def __init__(self, images, augment):
        """
        images: list of np.uint8 (3, 64, 64)
        augment: torchvision augmentation pipeline
        """
        self.images = images
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].transpose(1, 2, 0)  # CHW → HWC for PIL
        v1 = self.augment(img)
        v2 = self.augment(img)
        return v1, v2


# ============================================================================
#  NT-Xent (SimCLR) Loss
# ============================================================================

class NTXentLoss(nn.Module):
    """
    SimCLR 风格的对称对比损失.

    输入: z1, z2 各 (N, D), 已 L2 归一化.
    正样本对: (z1[i], z2[i])
    负样本: 同 batch 内其他 2N-2 个样本 (隐式 hard negative)
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        N, D = z1.shape
        z = torch.cat([z1, z2], dim=0)                  # (2N, D)
        sim = torch.matmul(z, z.t()) / self.temperature # (2N, 2N), 余弦相似度

        # 排除对角线 (自身)
        mask_self = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask_self, float('-inf'))

        # 第 i 个的正样本: 第 (i+N) % (2N) 个
        labels = torch.cat([
            torch.arange(N, 2 * N),
            torch.arange(0, N),
        ]).to(z.device)

        return F.cross_entropy(sim, labels)


# ============================================================================
#  数据库 & 切图
# ============================================================================

def _decode_blob(blob):
    """
    多级 fallback 解码 BLOB → np.uint8 BGR 图像.

    顺序:
      1. pickle (原始脚本可能用 pickle 存 ndarray)
      2. OpenCV (快, 但 TIFF/特殊格式经常挂)
      3. Pillow (慢但兼容性极好, 覆盖 OpenCV 解不了的 TIFF/WebP)

    全部失败返回 None.
    """
    # 1. pickle
    try:
        obj = pickle.loads(blob)
        if isinstance(obj, np.ndarray):
            return obj
    except Exception:
        pass

    # 2. OpenCV (把 stderr 临时屏蔽, 不污染日志)
    arr = np.frombuffer(blob, np.uint8)
    try:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass

    # 3. Pillow fallback (对 TIFF 兼容好得多)
    try:
        from PIL import Image
        import io
        pil = Image.open(io.BytesIO(blob))
        pil.load()
        if pil.mode == 'RGBA':
            pil = pil.convert('RGB')
        elif pil.mode == 'L':
            pil = pil.convert('RGB')
        elif pil.mode != 'RGB':
            pil = pil.convert('RGB')
        # PIL 是 RGB, OpenCV 代码假设 BGR, 所以转一下保持一致
        arr = np.array(pil)[:, :, ::-1].copy()   # RGB → BGR
        return arr
    except Exception:
        return None


def load_single_map(conn, map_id, map_size, table='image_maps1'):
    """从数据库读取一张地图; 返回 np.uint8 (H, W, 3) 或 None."""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT image_data FROM `{table}` WHERE id = %s", (int(map_id),))
        row = cursor.fetchone()
        cursor.close()
        if row is None:
            return None
        blob = row[0]

        sat = _decode_blob(blob)

        if sat is None:
            return None
        if len(sat.shape) == 2:
            sat = cv2.cvtColor(sat, cv2.COLOR_GRAY2RGB)
        elif sat.shape[2] == 4:
            sat = cv2.cvtColor(sat, cv2.COLOR_BGRA2RGB)
        if sat.shape[0] != map_size or sat.shape[1] != map_size:
            sat = cv2.resize(sat, (map_size, map_size), interpolation=cv2.INTER_LINEAR)
        return sat.astype(np.uint8)
    except Exception as e:
        print(f"  [WARN] map {map_id} load failed: {e}")
        return None


def cut_into_cells(sat_map, map_size, grid_size, img_size=64):
    """切成 grid_size × grid_size 个 64×64 的图块 (CHW uint8)."""
    cell_size = map_size / grid_size
    cells = []
    for gx in range(grid_size):
        for gy in range(grid_size):
            x_s = int(gx * cell_size)
            y_s = int(gy * cell_size)
            x_e = min(int((gx + 1) * cell_size), map_size)
            y_e = min(int((gy + 1) * cell_size), map_size)
            cell = sat_map[x_s:x_e, y_s:y_e]
            cell = cv2.resize(cell, (img_size, img_size))
            cell = cell.transpose(2, 0, 1)  # HWC → CHW
            cells.append(cell)
    return cells


# ============================================================================
#  评估: 与 env.py 用法一致的"目标 vs 非目标"判别 AUC
# ============================================================================

@torch.no_grad()
def evaluate_auc(model, val_cells_per_map, device, n_pairs_per_map=200):
    """
    在留出地图上评估. 模拟 env.py 的实际用法:
      - 正样本: 同地点 (微小增强差) → df 应小
      - 负样本: 不同地点 → df 应大

    返回 dict: auc, df_pos_mean/std, df_neg_mean/std
    """
    model.eval()
    # 评估时用更弱的增强, 模拟"同一地点不同时刻拍摄"的差异
    eval_aug = T.Compose([
        T.ToPILImage(),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
    ])

    all_dist, all_label = [], []

    for cells in val_cells_per_map:
        n_cells = len(cells)
        if n_cells < 2:
            continue
        n_pos = n_pairs_per_map // 2
        n_neg = n_pairs_per_map - n_pos

        # ===== 正样本: 同 cell, 弱增强差 =====
        idxs = np.random.randint(0, n_cells, size=n_pos)
        v1s = torch.stack([eval_aug(cells[i].transpose(1, 2, 0)) for i in idxs]).to(device)
        v2s = torch.stack([eval_aug(cells[i].transpose(1, 2, 0)) for i in idxs]).to(device)
        f1 = model.forward_one(v1s)
        f2 = model.forward_one(v2s)
        df_pos = torch.norm(f1 - f2, p=2, dim=1).cpu().numpy()
        all_dist.extend(df_pos.tolist())
        all_label.extend([1] * n_pos)

        # ===== 负样本: 不同 cell =====
        ia = np.random.randint(0, n_cells, size=n_neg)
        ib = np.random.randint(0, n_cells, size=n_neg)
        for k in range(n_neg):
            while ib[k] == ia[k]:
                ib[k] = np.random.randint(0, n_cells)
        v1s = torch.stack([eval_aug(cells[i].transpose(1, 2, 0)) for i in ia]).to(device)
        v2s = torch.stack([eval_aug(cells[i].transpose(1, 2, 0)) for i in ib]).to(device)
        f1 = model.forward_one(v1s)
        f2 = model.forward_one(v2s)
        df_neg = torch.norm(f1 - f2, p=2, dim=1).cpu().numpy()
        all_dist.extend(df_neg.tolist())
        all_label.extend([0] * n_neg)

    # AUC: distance 越小越像 → score = -distance
    auc = roc_auc_score(all_label, [-d for d in all_dist])

    df_pos_arr = np.array([d for d, l in zip(all_dist, all_label) if l == 1])
    df_neg_arr = np.array([d for d, l in zip(all_dist, all_label) if l == 0])

    return {
        'auc': float(auc),
        'df_pos_mean': float(df_pos_arr.mean()),
        'df_pos_std': float(df_pos_arr.std()),
        'df_neg_mean': float(df_neg_arr.mean()),
        'df_neg_std': float(df_neg_arr.std()),
    }


# ============================================================================
#  训练主函数
# ============================================================================

def train(args):
    # --- 设备 ---
    if args.cpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"[Train] Device: {device}")
    if device.type == 'cuda':
        print(f"        GPU: {torch.cuda.get_device_name(0)}")

    # --- 数据库 ---
    db_config = {
        'user': args.db_user, 'password': args.db_password,
        'host': args.db_host, 'database': args.db_name,
        'raise_on_warnings': True,
    }
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id FROM {args.db_table}")
    map_ids = [x[0] for x in cursor.fetchall()]
    cursor.close()
    conn.close()
    print(f"[Train] 总地图数: {len(map_ids)}")
    if len(map_ids) == 0:
        print("[Train] ERROR: 数据库无地图")
        return

    # --- Train/Val 切分 ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    perm = np.random.permutation(len(map_ids))
    n_val = max(1, int(len(map_ids) * args.val_ratio))
    val_ids = [map_ids[i] for i in perm[:n_val]]
    train_ids = [map_ids[i] for i in perm[n_val:]]
    print(f"[Train] Train: {len(train_ids)} maps,  Val: {len(val_ids)} maps")

    # --- 模型 ---
    model = SiameseNetwork(feature_dim=128, normalize=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] 参数量: {n_params:,}")

    criterion = NTXentLoss(temperature=args.temperature)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # T_max 用粗略 step 估计
    est_steps_per_epoch = len(train_ids) * (900 // args.batch_size + 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * est_steps_per_epoch, eta_min=args.lr * 0.01
    )

    augment = build_augmentation()

    # --- 预加载验证集 (避免每个 epoch 重读) ---
    print("[Train] 预加载验证集...")
    val_cells_list = []
    conn = mysql.connector.connect(**db_config)
    for vid in val_ids:
        sat = load_single_map(conn, vid, args.map_size, table=args.db_table)
        if sat is None:
            continue
        val_cells_list.append(cut_into_cells(sat, args.map_size, args.grid_size))
        del sat
    conn.close()
    gc.collect()
    print(f"[Train] 验证集 cells/map: {len(val_cells_list[0]) if val_cells_list else 0}")

    # --- 训练循环 ---
    best_auc = 0.0
    train_start = time.time()
    last_eval = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_steps = 0.0, 0
        epoch_start = time.time()

        order = np.random.permutation(len(train_ids))

        map_pbar = tqdm(
            order,
            desc=f"Epoch {epoch}/{args.epochs}",
            dynamic_ncols=True,
            leave=True
        )

        for map_idx_i, oi in enumerate(map_pbar, 1):
            map_id = train_ids[oi]

            # 加载 → 切图 → 立即释放大图
            conn = mysql.connector.connect(**db_config)
            sat = load_single_map(conn, map_id, args.map_size, table=args.db_table)
            conn.close()
            if sat is None:
                continue
            cells = cut_into_cells(sat, args.map_size, args.grid_size)
            del sat
            gc.collect()

            dataset = CellSimCLRDataset(cells, augment)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=True,  # NT-Xent 要求每个 batch 大小一致
                pin_memory=(device.type == 'cuda'),
            )

            for v1, v2 in loader:
                v1 = v1.to(device, non_blocking=True)
                v2 = v2.to(device, non_blocking=True)

                f1 = model.forward_one(v1)
                f2 = model.forward_one(v2)
                loss = criterion(f1, f2)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                epoch_steps += 1

            del cells, dataset, loader
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

            avg_loss_so_far = epoch_loss / max(epoch_steps, 1)
            epoch_elapsed = time.time() - epoch_start
            total_elapsed = time.time() - train_start

            map_pbar.set_postfix(
                loss=f"{avg_loss_so_far:.4f}",
                maps=f"{map_idx_i}/{len(order)}",
                epoch_time=f"{epoch_elapsed:.0f}s",
                total_time=f"{total_elapsed/60:.1f}m"
            )

        # --- Epoch 结束: 评估 + 日志 ---
        avg_loss = epoch_loss / max(epoch_steps, 1)
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        eta = elapsed / epoch * (args.epochs - epoch)

        last_eval = evaluate_auc(model, val_cells_list, device,
                                 n_pairs_per_map=args.eval_pairs_per_map)

        tqdm.write(
            f"\n=== Epoch {epoch:3d}/{args.epochs} ===\n"
            f"  loss={avg_loss:.4f}  steps={epoch_steps}  "
            f"epoch_time={epoch_time:.0f}s  ETA={eta/60:.1f}min  "
            f"lr={scheduler.get_last_lr()[0]:.2e}\n"
            f"  [Val] AUC={last_eval['auc']:.4f}  "
            f"df_pos={last_eval['df_pos_mean']:.3f}±{last_eval['df_pos_std']:.3f}  "
            f"df_neg={last_eval['df_neg_mean']:.3f}±{last_eval['df_neg_std']:.3f}  "
            f"sep={last_eval['df_neg_mean'] - last_eval['df_pos_mean']:+.3f}"
        )

        # 保存最优 (按 AUC, 不按 loss)
        if last_eval['auc'] > best_auc:
            best_auc = last_eval['auc']
            torch.save(model.state_dict(), args.save_path)
            print(f"  ★ 新最优 AUC={best_auc:.4f} → {args.save_path}")

        # 定期 checkpoint
        if epoch % 20 == 0:
            ckpt = args.save_path.replace('.pth', f'_ep{epoch}.pth')
            torch.save(model.state_dict(), ckpt)

    # --- 训练完成: 给出 env.py 配套修改建议 ---
    print(f"\n{'='*70}")
    print(f"训练完成. 总耗时 {(time.time() - train_start)/60:.1f} 分钟. 最优 AUC: {best_auc:.4f}")
    print(f"{'='*70}")

    if last_eval is not None:
        df_pos = last_eval['df_pos_mean']
        df_neg = last_eval['df_neg_mean']
        # 推荐 df_max: 让正样本 vs ≈ 0.9, 负样本 vs ≈ 0
        # vs = 1 - df/df_max → 设 df_pos 时 vs=0.9 → df_max = df_pos / 0.1 太大
        # 改用: 取正负中点偏负, 让正样本 vs > 0.7, 负样本 vs < 0.1
        suggested = (df_pos + df_neg) / 2.0

        print("\n>>> env.py 需要做的两处修改 <<<\n")
        print("【1】SiameseNetwork.forward_one 末尾加一行 L2 归一化:")
        print("    def forward_one(self, x):")
        print("        x = self.backbone(x)")
        print("        x = x.view(x.size(0), -1)")
        print("        x = self.fc(x)")
        print("        x = F.normalize(x, p=2, dim=1)   # ← 新增")
        print("        return x")
        print("    (顶部需要 import torch.nn.functional as F)")
        print()
        print(f"【2】UAVNavigationEnv 构造函数中 df_max 改为:")
        print(f"     df_max = {suggested:.2f}   # 自动建议值")
        print(f"     (基于 val 集上 df_pos≈{df_pos:.3f}, df_neg≈{df_neg:.3f})")
        print()
        print("【3】(可选, 推荐) 在 env.py 用 vs 之前对 Mrel 做 Gaussian 模糊,")
        print("    给 RL 一个空间梯度信号:")
        print("    from scipy.ndimage import gaussian_filter")
        print("    self.semantic_map[CH_REL] = gaussian_filter(self.semantic_map[CH_REL], sigma=1.0)")

        if best_auc < 0.85:
            print("\n⚠️  警告: AUC < 0.85, Siamese 区分能力不足. 建议:")
            print("    - 增加 epochs (尝试 150)")
            print("    - 增大 batch_size (256 或 512, 给 InfoNCE 更多负样本)")
            print("    - 降低 temperature (0.05)")
            print("    - 检查地图数据质量 (是否大量纯绿地/纯水域?)")
        else:
            print(f"\n✓ AUC={best_auc:.4f} ≥ 0.85, 可放进 RL 训练.")


# ============================================================================
#  Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Siamese Pre-training v2 (SimCLR)')

    # 数据库
    parser.add_argument('--db-user', type=str, default='root')
    parser.add_argument('--db-password', type=str, default='Wqw030221')
    parser.add_argument('--db-host', type=str, default='localhost')
    parser.add_argument('--db-name', type=str, default='senmap')
    parser.add_argument('--db-table', type=str, default='image_maps2')

    # 数据
    parser.add_argument('--map-size', type=int, default=3000)
    parser.add_argument('--grid-size', type=int, default=30)
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='留作验证的地图比例')
    parser.add_argument('--eval-pairs-per-map', type=int, default=200)

    # 训练
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch 越大, NT-Xent 负样本越多, 一般效果越好')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='NT-Xent 温度, 通常 0.05~0.5')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--save-path', type=str, default='siamese_model.pth')

    args = parser.parse_args()
    train(args)