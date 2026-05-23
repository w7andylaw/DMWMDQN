# Author: Qiwei Wang
"""Training script for the Siamese visual similarity model."""

import argparse
import gc
import os
import time
import pickle

os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
import cv2
cv2.setLogLevel(0)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import mysql.connector
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    raise ImportError("需要 scikit-learn: pip install scikit-learn")
from tqdm.auto import tqdm






class SiameseNetwork(nn.Module):
    """Siamese visual feature network."""

    def __init__(self, feature_dim=128, normalize=True, pretrained=True):
        super().__init__()
        import torchvision.models as models
        
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






def build_augmentation():
    """Build augmentation."""
    return T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(64, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        T.RandomRotation(degrees=180, fill=0),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.ToTensor(),  
    ])






class CellSimCLRDataset(Dataset):
    def __init__(self, images, augment):
        """Init."""
        self.images = images
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].transpose(1, 2, 0)  
        v1 = self.augment(img)
        v2 = self.augment(img)
        return v1, v2






class NTXentLoss(nn.Module):
    """N T Xent Loss component."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        N, D = z1.shape
        z = torch.cat([z1, z2], dim=0)                  
        sim = torch.matmul(z, z.t()) / self.temperature 

        
        mask_self = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask_self, float('-inf'))

        
        labels = torch.cat([
            torch.arange(N, 2 * N),
            torch.arange(0, N),
        ]).to(z.device)

        return F.cross_entropy(sim, labels)






def _decode_blob(blob):
    """Decode blob."""
    
    try:
        obj = pickle.loads(blob)
        if isinstance(obj, np.ndarray):
            return obj
    except Exception:
        pass

    
    arr = np.frombuffer(blob, np.uint8)
    try:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass

    
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
        
        arr = np.array(pil)[:, :, ::-1].copy()   
        return arr
    except Exception:
        return None


def load_single_map(conn, map_id, map_size, table='image_maps1'):
    """Load single map."""
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
    """Cut into cells."""
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
            cell = cell.transpose(2, 0, 1)  
            cells.append(cell)
    return cells






@torch.no_grad()
def evaluate_auc(model, val_cells_per_map, device, n_pairs_per_map=200):
    """Evaluate auc."""
    model.eval()
    
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

        
        idxs = np.random.randint(0, n_cells, size=n_pos)
        v1s = torch.stack([eval_aug(cells[i].transpose(1, 2, 0)) for i in idxs]).to(device)
        v2s = torch.stack([eval_aug(cells[i].transpose(1, 2, 0)) for i in idxs]).to(device)
        f1 = model.forward_one(v1s)
        f2 = model.forward_one(v2s)
        df_pos = torch.norm(f1 - f2, p=2, dim=1).cpu().numpy()
        all_dist.extend(df_pos.tolist())
        all_label.extend([1] * n_pos)

        
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






def train(args):
    
    if args.cpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"[Train] Device: {device}")
    if device.type == 'cuda':
        print(f"        GPU: {torch.cuda.get_device_name(0)}")

    
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

    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    perm = np.random.permutation(len(map_ids))
    n_val = max(1, int(len(map_ids) * args.val_ratio))
    val_ids = [map_ids[i] for i in perm[:n_val]]
    train_ids = [map_ids[i] for i in perm[n_val:]]
    print(f"[Train] Train: {len(train_ids)} maps,  Val: {len(val_ids)} maps")

    
    model = SiameseNetwork(feature_dim=128, normalize=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] 参数量: {n_params:,}")

    criterion = NTXentLoss(temperature=args.temperature)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    est_steps_per_epoch = len(train_ids) * (900 // args.batch_size + 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * est_steps_per_epoch, eta_min=args.lr * 0.01
    )

    augment = build_augmentation()

    
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
                drop_last=True,  
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

        
        if last_eval['auc'] > best_auc:
            best_auc = last_eval['auc']
            torch.save(model.state_dict(), args.save_path)
            print(f"  ★ 新最优 AUC={best_auc:.4f} → {args.save_path}")

        
        if epoch % 20 == 0:
            ckpt = args.save_path.replace('.pth', f'_ep{epoch}.pth')
            torch.save(model.state_dict(), ckpt)

    
    print(f"\n{'='*70}")
    print(f"训练完成. 总耗时 {(time.time() - train_start)/60:.1f} 分钟. 最优 AUC: {best_auc:.4f}")
    print(f"{'='*70}")

    if last_eval is not None:
        df_pos = last_eval['df_pos_mean']
        df_neg = last_eval['df_neg_mean']
        
        
        
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






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Siamese Pre-training v2 (SimCLR)')

    
    parser.add_argument('--db-user', type=str, default='root')
    parser.add_argument('--db-password', type=str, default='Wqw030221')
    parser.add_argument('--db-host', type=str, default='localhost')
    parser.add_argument('--db-name', type=str, default='senmap')
    parser.add_argument('--db-table', type=str, default='image_maps2')

    
    parser.add_argument('--map-size', type=int, default=3000)
    parser.add_argument('--grid-size', type=int, default=30)
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='留作验证的地图比例')
    parser.add_argument('--eval-pairs-per-map', type=int, default=200)

    
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