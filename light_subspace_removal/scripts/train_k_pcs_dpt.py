#!/usr/bin/env python3
"""
Patch-agnostic dense canopy-height regression from pre-encoded DINOv2/DINOv3 patch tokens.

- Supports both dinov2_base and dinov3_sat configurations from HF dataset
- DPT-style decoder with soft discretization: logits over B bins; convert to cm via expectation
- ViT-pyramid neck constructs 4 scales from final-layer tokens
- Spatio-temporal CV: 5 spatial folds × 3 temporal holdouts = 15 validation metrics
- Trains 50 epochs with AdamW; reports validation RMSE history
- Writes JSON: results/{model_config}/dpt/cv_s{spatial}_t{temporal}_k{k}.json

Usage:
  python train_k_pcs_dpt.py --model_config dinov2_base --spatial_fold 0 --temporal_holdout 0 --k 3
"""

import argparse, json, os, random, math
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import KFold

# ---------------------------
# Repro
# ---------------------------
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------
# Grid/targets (patch-agnostic)
# ---------------------------
def infer_token_grid(X_patch: np.ndarray):
    assert X_patch.ndim == 3, f"Expected [M,Np,D], got {X_patch.shape}"
    _, Np, D = X_patch.shape
    side = int(round(Np ** 0.5))
    assert side * side == Np, f"Tokens not square: Np={Np}"
    return side, side, D  # H, W, D

def tokens_to_chw(tokens: np.ndarray, H: int, W: int):
    D = tokens.shape[-1]
    return tokens.reshape(H, W, D).transpose(2, 0, 1)  # [D,H,W]

def make_target(y, H_out: int, W_out: int):
    if np.isscalar(y):
        return torch.full((1, H_out, W_out), float(y), dtype=torch.float32)
    arr = np.array(y)
    t = torch.from_numpy(arr).float()
    if t.ndim == 2:
        t = t[None, None, ...]
    else:
        t = t.view(1, 1, *t.shape[-2:])
    return F.interpolate(t, size=(H_out, W_out), mode='bilinear', align_corners=False)[0]

# ---------------------------
# tcSVD lighting subspace (patch-agnostic)
# ---------------------------
@torch.no_grad()
def tall_skinny_svd(D: torch.Tensor, k: int):
    if k == 0: return None
    U, S, Vh = torch.linalg.svd(D.cpu(), full_matrices=False)   # Vh: [d,d]
    Vk = Vh[:k].T.contiguous()                                  # [d,k]
    Q, _ = torch.linalg.qr(Vk)                                  # [d,k]
    return Q

def project_out(Z: torch.Tensor, Q: torch.Tensor | None):
    if Q is None or Q.numel() == 0: return Z
    return Z - Z @ (Q @ Q.T)

def estimate_Q_train_only_patchwise(Xtr_patch: np.ndarray, tr_ids: list[str], T=3, k=0):
    if k == 0: return None
    groups = defaultdict(list)
    for i, tid in enumerate(tr_ids): groups[tid].append(i)
    diffs = []
    for tid, idxs in groups.items():
        if len(idxs) != T: continue
        Z = torch.tensor(Xtr_patch[idxs], dtype=torch.float32)  # [T,Np,D]
        mu = Z.mean(dim=0, keepdim=True)                        # [1,Np,D]
        diffs.append(Z - mu)
    if not diffs: return None
    D_mat = torch.cat(diffs, dim=0).reshape(-1, Xtr_patch.shape[-1])  # [T*Ntiles*Np, D]
    return tall_skinny_svd(D_mat, k)

def apply_projection_np(X_patch: np.ndarray, Q: torch.Tensor | None):
    X = torch.from_numpy(X_patch).float()                        # [M,Np,D]
    if (Q is None) or (Q.numel() == 0): return X.numpy().astype(np.float32)
    P = Q @ Q.T                                                  # [D,D]
    Xp = X - X @ P
    return Xp.numpy().astype(np.float32)

# ---------------------------
# Data (HF)
# ---------------------------
def load_arrays_from_hf(model_config: str, temporal_holdout: int, H_out: int, W_out: int):
    """
    Load data for spatio-temporal CV.
    temporal_holdout: 0=train on t1,t2 validate on t0; 1=train on t0,t2 validate on t1; 2=train on t0,t1 validate on t2
    """
    # Load embedding dataset and default dataset (for canopy height)
    ds_embed = load_dataset("mpg-ranch/light-stable-semantics", model_config, split='train')
    ds_default = load_dataset("mpg-ranch/light-stable-semantics", "default", split='train')

    # Create mapping from idx to canopy height
    canopy_map = {ex['idx']: ex['canopy_height'] for ex in ds_default}

    # Define temporal splits
    time_keys = ['t0', 't1', 't2']
    if temporal_holdout == 0:  # Validate on morning
        train_keys = ['t1', 't2']
        val_key = 't0'
    elif temporal_holdout == 1:  # Validate on noon
        train_keys = ['t0', 't2']
        val_key = 't1'
    else:  # temporal_holdout == 2, validate on afternoon
        train_keys = ['t0', 't1']
        val_key = 't2'

    Xp_train, ids_train, Y_train = [], [], []
    Xp_val, ids_val, Y_val = [], [], []

    for ex in ds_embed:
        idx = ex['idx']
        canopy_height = canopy_map[idx]
        target = make_target(canopy_height, H_out, W_out).numpy()  # [1,H_out,W_out]

        # Add training time points
        for key in train_keys:
            tokens = np.array(ex[f'patch_{key}'], dtype=np.float32)  # [Np,D]
            Xp_train.append(tokens)
            ids_train.append(idx)
            Y_train.append(target)

        # Add validation time point
        tokens = np.array(ex[f'patch_{val_key}'], dtype=np.float32)  # [Np,D]
        Xp_val.append(tokens)
        ids_val.append(idx)
        Y_val.append(target)

    Xp_train = np.stack(Xp_train, 0)  # [M_train,Np,D]
    Y_train = np.stack(Y_train, 0)    # [M_train,1,H_out,W_out]
    Xp_val = np.stack(Xp_val, 0)      # [M_val,Np,D]
    Y_val = np.stack(Y_val, 0)        # [M_val,1,H_out,W_out]

    return (Xp_train, ids_train, Y_train), (Xp_val, ids_val, Y_val)

class DenseSplit(Dataset):
    def __init__(self, X_patch, Y, H, W):
        self.Xp = X_patch
        self.Y  = Y
        self.H, self.W = H, W
    def __len__(self): return len(self.Xp)
    def __getitem__(self, i):
        x = tokens_to_chw(self.Xp[i], self.H, self.W)      # [D,H,W]
        y = torch.from_numpy(self.Y[i]).float()            # [1,H_out,W_out]
        return torch.from_numpy(x).float(), y

# ---------------------------
# Decoders
# ---------------------------
class UpBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.gn1   = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.gn2   = nn.GroupNorm(8, c_out)
    def forward(self, x):
        x = F.gelu(self.gn1(self.conv1(x)))
        x = F.gelu(self.gn2(self.conv2(x)))
        return x

class GenericDenseDecoder(nn.Module):
    """Simple zoom-and-refine: [B,D,H,W] -> [B,1,H_out,W_out] via power-of-two upsampling."""
    def __init__(self, c_in: int, H: int, W: int, H_out: int, W_out: int,
                 base: int = 256, dropout: float = 0.05):
        super().__init__()
        assert (H_out % H == 0) and (W_out % W == 0)
        sx = H_out // H; sy = W_out // W
        assert sx == sy and (sx & (sx - 1) == 0), "Scale must be equal and power of two"
        n_ups = int(math.log2(sx))
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, base, 1), nn.GELU(), nn.Dropout2d(dropout), UpBlock(base, base)
        )
        ups, blks = [], []
        c = base
        for _ in range(n_ups):
            ups.append(nn.ConvTranspose2d(c, c // 2, 2, 2))
            blks.append(UpBlock(c // 2, c // 2))
            c //= 2
        self.ups = nn.ModuleList(ups); self.blks = nn.ModuleList(blks)
        self.head = nn.Conv2d(c, 1, 1)
    def forward(self, x):
        x = self.stem(x)
        for up, blk in zip(self.ups, self.blks):
            x = blk(up(x))
        return self.head(x)

# ---- DPT-style blocks (decoder head) ----

class ConvBNAct(nn.Module):
    """Conv → BN → GELU (as used in DPT-style decoders)."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
    def forward(self, x): return self.block(x)

class FusionBlock(nn.Module):
    """RefineNet-like fusion: project both, upsample prev, add, refine."""
    def __init__(self, in_ch_cur, in_ch_prev, out_ch):
        super().__init__()
        self.proj_cur  = ConvBNAct(in_ch_cur, out_ch, k=1, s=1, p=0)
        self.proj_prev = ConvBNAct(in_ch_prev, out_ch, k=1, s=1, p=0)
        self.refine    = nn.Sequential(
            ConvBNAct(out_ch, out_ch, k=3, s=1, p=1),
            ConvBNAct(out_ch, out_ch, k=3, s=1, p=1),
        )
    def forward(self, cur, prev):
        prev_up = F.interpolate(prev, size=cur.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj_cur(cur) + self.proj_prev(prev_up)
        return self.refine(x)

class DPTDepthDecoder(nn.Module):
    """
    DPT-style depth decoder producing logits over discretized bins.
    We keep units in centimeters by default (min_cm..max_cm).
    """
    def __init__(self, in_channels, mid_channels=256, out_bins=256,
                 order="deep_to_shallow", dropout_p=0.05,
                 min_cm=0.0, max_cm=3000.0):
        super().__init__()
        assert len(in_channels) == 4
        assert order in {"deep_to_shallow", "shallow_to_deep"}
        if order == "shallow_to_deep":
            in_channels = list(reversed(in_channels))  # now deep→shallow
        c4, c3, c2, c1 = in_channels  # deep→shallow

        # 1) project each stage to common width
        self.proj4 = ConvBNAct(c4, mid_channels, k=1, s=1, p=0)
        self.proj3 = ConvBNAct(c3, mid_channels, k=1, s=1, p=0)
        self.proj2 = ConvBNAct(c2, mid_channels, k=1, s=1, p=0)
        self.proj1 = ConvBNAct(c1, mid_channels, k=1, s=1, p=0)

        # 2) top-down fusion
        self.fuse34 = FusionBlock(mid_channels, mid_channels, mid_channels)
        self.fuse23 = FusionBlock(mid_channels, mid_channels, mid_channels)
        self.fuse12 = FusionBlock(mid_channels, mid_channels, mid_channels)

        # 3) head to bin logits
        self.head = nn.Sequential(
            ConvBNAct(mid_channels, mid_channels, k=3, s=1, p=1),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(mid_channels, out_bins, kernel_size=1, bias=True),
        )
        # bin centers in cm
        self.register_buffer("min_cm", torch.tensor(float(min_cm)), persistent=False)
        self.register_buffer("max_cm", torch.tensor(float(max_cm)), persistent=False)
        self.out_bins = out_bins
        self.order = order

    @torch.no_grad()
    def bin_centers(self, device=None, dtype=None):
        edges = torch.linspace(self.min_cm, self.max_cm, self.out_bins + 1, device=device, dtype=dtype)
        return 0.5 * (edges[:-1] + edges[1:])  # [B]

    def forward(self, feats):
        if self.order == "shallow_to_deep":
            feats = feats[::-1]
        f4, f3, f2, f1 = feats
        p4 = self.proj4(f4); p3 = self.proj3(f3); p2 = self.proj2(f2); p1 = self.proj1(f1)
        x3 = self.fuse34(p3, p4)
        x2 = self.fuse23(p2, x3)
        x1 = self.fuse12(p1, x2)
        logits = self.head(x1)  # [B, out_bins, H1, W1] at finest scale
        return logits

# ---- ViT-pyramid neck: build 4 scales from final tokens ----
class ViTPyramidNeck(nn.Module):
    """
    Input:  [B, D, H, W] tokens (single-scale)
    Output: list of 4 feature maps [f4, f3, f2, f1] with strides x/2, x, 2x, 4x (relative to HxW),
            i.e., [H/2, H, 2H, 4H], capped so that finest matches target if needed.
    For H=W=14 and out_size=224, we produce sizes: 7, 14, 28, 56 (decoder will output at 56, and
    we upsample once more outside to hit 224).
    """
    def __init__(self, c_in: int, base: int = 256):
        super().__init__()
        self.reduce = nn.Conv2d(c_in, base, 1)
        self.refine = UpBlock(base, base)
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.up   = nn.ConvTranspose2d(base, base, 2, 2)

    def forward(self, x):
        # x: [B, D, H, W]
        x = self.refine(self.reduce(x))          # [B, C, H, W]
        f3 = x                                   # HxW
        f4 = self.down(x)                        # H/2 x W/2
        f2 = self.up(x)                          # 2H x 2W
        f1 = self.up(f2)                         # 4H x 4W
        return [f4, f3, f2, f1]                  # deep→shallow

# ---------------------------
# Train / Eval (50 epochs, best-epoch RMSE in cm)
# ---------------------------
def rmse_map(y_true, y_pred):  # y: [B,1,H,W]
    return torch.sqrt(torch.mean((y_true - y_pred)**2))

def train_epoch_simple(model, opt, loader, device):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        loss.backward(); opt.step()

@torch.no_grad()
def eval_epoch_simple(model, loader, device):
    model.eval()
    rmses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        rmses.append(rmse_map(yb, pred).cpu())
    return float(torch.stack(rmses).mean())

def train_epoch_dpt(neck, decoder, H_out, loader, device, min_cm, max_cm):
    neck.train(); decoder.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)           # yb: [B,1,H_out,W_out]
        opt = decoder.opt                                # optimizer attached (set below)
        opt.zero_grad(set_to_none=True)
        feats = neck(xb)                                 # 4 scales
        logits = decoder(feats)                          # [B,Bins,H1,W1]
        # ensure logits match target size
        logits = F.interpolate(logits, size=(H_out, H_out), mode="bilinear", align_corners=False)
        probs = logits.softmax(dim=1)                    # [B,Bins,H_out,W_out]
        centers = decoder.bin_centers(probs.device, probs.dtype).view(1, -1, 1, 1)  # [1,Bins,1,1]
        pred_cm = (probs * centers).sum(dim=1, keepdim=True)  # [B,1,H_out,W_out]
        loss = F.mse_loss(pred_cm, yb)                   # MSE in centimeters
        loss.backward(); opt.step()

@torch.no_grad()
def eval_epoch_dpt(neck, decoder, H_out, loader, device):
    neck.eval(); decoder.eval()
    rmses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        feats = neck(xb)
        logits = decoder(feats)
        logits = F.interpolate(logits, size=(H_out, H_out), mode="bilinear", align_corners=False)
        probs = logits.softmax(dim=1)
        centers = decoder.bin_centers(probs.device, probs.dtype).view(1, -1, 1, 1)
        pred_cm = (probs * centers).sum(dim=1, keepdim=True)
        rmses.append(rmse_map(yb, pred_cm).cpu())
    return float(torch.stack(rmses).mean())

# ---------------------------
# Main (one (fold,k) job)
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    # Legacy single-experiment arguments (for backward compatibility)
    ap.add_argument("--spatial_fold", type=int, default=None, help="spatial fold index in [0..4]")
    ap.add_argument("--temporal_holdout", type=int, default=None, help="temporal holdout in [0..2]")
    ap.add_argument("--k", type=int, default=None, help="tcSVD rank (0=baseline)")
    ap.add_argument("--model_config", type=str, default=None, help="dinov2_base or dinov3_sat")
    # New multi-experiment arguments
    ap.add_argument("--job_id", type=int, default=None)
    ap.add_argument("--total_jobs", type=int, default=8)
    ap.add_argument("--total_configs", type=int, default=330)
    # Other arguments
    ap.add_argument("--outdir", type=str, default="results_cv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--base", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--out_size", type=int, default=224)
    ap.add_argument("--decoder", type=str, default="dpt", choices=["simple","dpt"])
    # DPT discretization (centimeters)
    ap.add_argument("--bins", type=int, default=256)
    ap.add_argument("--min_cm", type=float, default=0.0)
    ap.add_argument("--max_cm", type=float, default=1500.0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # Determine which k/fold combinations to run
    if args.job_id is not None:
        # Multi-experiment mode: calculate which configs this job handles
        KS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        SPATIAL_FOLDS = [0, 1, 2, 3, 4]
        TEMPORAL_HOLDOUTS = [0, 1, 2]
        MODEL_CONFIGS = ['dinov2_base', 'dinov3_sat']

        # Generate all combinations: k × spatial_fold × temporal_holdout × model_config
        all_configs = [(k, spatial_fold, temporal_holdout, model_config)
                      for model_config in MODEL_CONFIGS
                      for spatial_fold in SPATIAL_FOLDS
                      for temporal_holdout in TEMPORAL_HOLDOUTS
                      for k in KS]
        assert len(all_configs) == args.total_configs, f"Expected {args.total_configs} configs, got {len(all_configs)}"

        # Distribute configs across jobs
        configs_per_job = args.total_configs // args.total_jobs
        extra_configs = args.total_configs % args.total_jobs

        # Jobs 0 through (extra_configs-1) get one extra config
        if args.job_id < extra_configs:
            start_idx = args.job_id * (configs_per_job + 1)
            end_idx = start_idx + configs_per_job + 1
        else:
            start_idx = extra_configs * (configs_per_job + 1) + (args.job_id - extra_configs) * configs_per_job
            end_idx = start_idx + configs_per_job

        job_configs = all_configs[start_idx:end_idx]
        print(f"Job {args.job_id} handling {len(job_configs)} configs")
    else:
        # Legacy single-experiment mode
        if args.spatial_fold is None or args.temporal_holdout is None or args.k is None or args.model_config is None:
            raise ValueError("Either provide --job_id for multi-experiment mode, or all single-experiment arguments")
        job_configs = [(args.k, args.spatial_fold, args.temporal_holdout, args.model_config)]

    # Loop over all configurations assigned to this job
    for k, spatial_fold, temporal_holdout, model_config in job_configs:
        print(f"\n=== Running k={k}, spatial_fold={spatial_fold}, temporal_holdout={temporal_holdout}, model_config={model_config} ===")

        # Check if result already exists (resume capability)
        model_outdir = os.path.join(args.outdir.replace('results_cv', 'results/light_subspace_removal'), model_config, 'dpt')
        os.makedirs(model_outdir, exist_ok=True)
        out_path = os.path.join(model_outdir, f"cv_s{spatial_fold}_t{temporal_holdout}_k{k}_{args.decoder}.json")
        if os.path.exists(out_path):
            print(f"Result already exists at {out_path}, skipping...")
            continue

        assert 0 <= spatial_fold < 5, "spatial_fold must be 0..4"
        assert 0 <= temporal_holdout < 3, "temporal_holdout must be 0..2"

        # Load data with spatio-temporal split
        (X_train, ids_train, Y_train), (X_val, ids_val, Y_val) = load_arrays_from_hf(
            model_config, temporal_holdout, args.out_size, args.out_size)

        # Infer grid from training data
        H, W, D = infer_token_grid(X_train)
        print(f"Inferred token grid: H={H} W={W} D={D}; supervising at {args.out_size}x{args.out_size}; decoder={args.decoder}")

        # Group training data by tile id for spatial CV
        train_groups = defaultdict(list)
        for i, tid in enumerate(ids_train):
            train_groups[tid].append(i)
        train_tiles = sorted(train_groups.keys())

        # Apply spatial fold to training data
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        spatial_folds = list(kf.split(train_tiles))

        tr_idx, va_idx = spatial_folds[spatial_fold]
        tr_tiles = [train_tiles[i] for i in tr_idx]
        va_tiles = [train_tiles[i] for i in va_idx]
        tr_rows = [j for t in tr_tiles for j in train_groups[t]]
        va_rows = [j for t in va_tiles for j in train_groups[t]]

        Xtr, Ytr = X_train[tr_rows], Y_train[tr_rows]
        Xva, Yva = X_train[va_rows], Y_train[va_rows]
        id_tr = [ids_train[j] for j in tr_rows]

        # tcSVD on TRAIN only
        Q = estimate_Q_train_only_patchwise(Xtr, id_tr, T=2, k=k)  # T=2 since we use 2 time points for training
        XtrP = apply_projection_np(Xtr, Q)
        XvaP = apply_projection_np(Xva, Q)

        # Datasets / loaders
        train_ds = DenseSplit(XtrP, Ytr, H, W)
        val_ds   = DenseSplit(XvaP, Yva, H, W)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Build model(s)
        val_rmse_history = []
        if args.decoder == "simple":
            model = GenericDenseDecoder(c_in=D, H=H, W=W, H_out=args.out_size, W_out=args.out_size,
                                        base=args.base, dropout=args.dropout).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            for epoch in range(1, args.epochs+1):
                # pack batch inputs
                # each xb is [B,D,H,W] already from dataset
                train_epoch_simple(model, opt, train_loader, device)
                rm = eval_epoch_simple(model, val_loader, device)
                val_rmse_history.append(rm)
                print(f"[s{spatial_fold}_t{temporal_holdout}_k{k}] epoch {epoch:03d}  val_RMSE={rm:.3f} cm")

        else:  # DPT-style
            neck = ViTPyramidNeck(c_in=D, base=args.base).to(device)
            dpt  = DPTDepthDecoder(
                in_channels=[args.base, args.base, args.base, args.base],
                mid_channels=args.base,
                out_bins=args.bins,
                order="deep_to_shallow",
                dropout_p=args.dropout,
                min_cm=args.min_cm,
                max_cm=args.max_cm,
            ).to(device)
            # attach optimizer to decoder (neck is lightweight; share optimizer)
            opt = torch.optim.AdamW(list(neck.parameters()) + list(dpt.parameters()), lr=1e-3)
            dpt.opt = opt  # small convenience for train function

            for epoch in range(1, args.epochs+1):
                train_epoch_dpt(neck, dpt, args.out_size, train_loader, device, args.min_cm, args.max_cm)
                rm = eval_epoch_dpt(neck, dpt, args.out_size, val_loader, device)
                val_rmse_history.append(rm)
                print(f"[s{spatial_fold}_t{temporal_holdout}_k{k}] epoch {epoch:03d}  val_RMSE={rm:.3f} cm  (DPT)")

        # Save metrics JSON
        result = {
            "spatial_fold": spatial_fold,
            "temporal_holdout": temporal_holdout,
            "k": k,
            "model_config": model_config,
            "seed": args.seed,
            "epochs": args.epochs,
            "val_rmse_history": [round(x, 6) for x in val_rmse_history],
            "token_grid": [H, W, D],
            "out_size": args.out_size,
            "bins": args.bins if args.decoder == "dpt" else None,
            "min_cm": args.min_cm if args.decoder == "dpt" else None,
            "max_cm": args.max_cm if args.decoder == "dpt" else None,
            "n_train_rows": len(tr_rows),
            "n_val_rows": len(va_rows),
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved {out_path}")

    print(f"Completed all {len(job_configs)} configurations for this job.")

if __name__ == "__main__":
    main()