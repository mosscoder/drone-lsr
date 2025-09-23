#!/usr/bin/env python3
"""
Patch-agnostic dense CHM regression from pre-encoded DINOv2/DINOv3 patch tokens.

- Supports both dinov2_base and dinov3_sat configurations from HF dataset
- Infers token grid H×W from X_patch shape [M, Np, D] (requires Np be a square).
- Decoder upsamples from [D, H, W] to [1, H_out, W_out] where H_out/W_out are args (default 224).
- tcSVD lighting subspace removal with rank k (0 = baseline).
- Spatio-temporal CV: 5 spatial folds × 3 temporal holdouts = 15 validation metrics.
- Trains 50 epochs (no early stopping) with AdamW; reports TEMPORAL-VAL RMSE history
  (evaluated on the held-out timepoint for the spatially held-out tiles).
- Writes JSON: results/{model_config}/simple_decoder/cv_s{spatial}_t{temporal}_k{k}.json

Usage:
  # Single experiment
  python train_k_pcs_simple_decoder.py --model_config dinov2_base --spatial_fold 0 --temporal_holdout 0 --k 3

  # Distributed grid (k × spatial_fold × temporal_holdout × model_config)
  python train_k_pcs_simple_decoder.py --job_id 0 --total_jobs 8 --total_configs 330
"""

import argparse, json, os, random
from collections import defaultdict
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import KFold

# ---------------------------
# Helper functions for power-of-2 handling
# ---------------------------
def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0

def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()

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
    """
    X_patch: [M, Np, D]
    Returns (H, W, D); asserts Np is a perfect square.
    """
    assert X_patch.ndim == 3, f"Expected [M,Np,D], got {X_patch.shape}"
    _, Np, D = X_patch.shape
    side = int(round(Np ** 0.5))
    assert side * side == Np, f"Tokens not square: Np={Np}"
    return side, side, D

def tokens_to_chw(tokens: np.ndarray, H: int, W: int):
    """[Np, D] -> [D, H, W]"""
    D = tokens.shape[-1]
    return tokens.reshape(H, W, D).transpose(2, 0, 1)

def make_target(y, H_out: int, W_out: int):
    """
    y scalar or HxW array -> [1,H_out,W_out].
    """
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
    """
    Xtr_patch: [M, Np, D]; residual per (tile, patch, time): z_{i,p,t} - mean_t z_{i,p,t}
    Returns Q: [D,k] or None.
    """
    if k == 0: return None
    groups = defaultdict(list)
    for i, tid in enumerate(tr_ids): groups[tid].append(i)

    diffs = []
    for tid, idxs in groups.items():
        if len(idxs) != T:  # skip incomplete tiles
            continue
        Z = torch.tensor(Xtr_patch[idxs], dtype=torch.float32)  # [T,Np,D]
        mu = Z.mean(dim=0, keepdim=True)                        # [1,Np,D]
        diffs.append(Z - mu)
    if not diffs: return None
    D_mat = torch.cat(diffs, dim=0).reshape(-1, Xtr_patch.shape[-1])  # [T*Ntiles*Np, D]
    return tall_skinny_svd(D_mat, k)

def apply_projection_np(X_patch: np.ndarray, Q: torch.Tensor | None):
    X = torch.from_numpy(X_patch).float()                        # [M,Np,D]
    if (Q is None) or (Q.numel() == 0):
        return X.numpy().astype(np.float32)
    P = Q @ Q.T                                                  # [D,D]
    Xp = X - X @ P                                               # matmul along last dim
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
    if temporal_holdout == 0:  # Validate on t0
        train_keys = ['t1', 't2']
        val_key = 't0'
    elif temporal_holdout == 1:  # Validate on t1
        train_keys = ['t0', 't2']
        val_key = 't1'
    else:  # temporal_holdout == 2, Validate on t2
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

        # Add validation time point (held-out time)
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
# Decoder (generic upsampler)
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
    """
    Input : [B, D, H, W]   (token grid)
    Output: [B, 1, H_out, W_out]

    - Builds a 2^n transposed-conv pyramid up to the *next power-of-two* multiple
      of (H,W), then antialiased bilinear-resizes to (H_out, W_out) if needed.
    """
    def __init__(self, c_in: int, H: int, W: int, H_out: int, W_out: int,
                 base: int = 256, dropout: float = 0.05):
        super().__init__()
        assert (H_out % H == 0) and (W_out % W == 0), "Output must be an integer multiple of token grid"
        sx = H_out // H
        sy = W_out // W
        assert sx == sy, f"Non-uniform scale not supported (sx={sx}, sy={sy})"
        self.H_out, self.W_out = H_out, W_out

        # 1) Stem at token resolution
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, base, 1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            UpBlock(base, base),
        )

        # 2) Decide how far the deconv pyramid goes (power of two)
        sx_p2 = sx if _is_pow2(sx) else _next_pow2(sx)  # e.g., 14 -> 16
        self.mid_scale = sx_p2
        n_ups = int(math.log2(sx_p2))
        ups, blks = [], []
        c = base
        for _ in range(n_ups):
            ups.append(nn.ConvTranspose2d(c, c // 2, 2, 2))
            blks.append(UpBlock(c // 2, c // 2))
            c //= 2
        self.ups = nn.ModuleList(ups)
        self.blks = nn.ModuleList(blks)

        # 3) Head at the mid (power-of-two) resolution
        self.head_mid = nn.Conv2d(c, 1, 1)

        # 4) If sx wasn't a power of two, we'll downsample to the exact target
        self.need_final_resize = (sx_p2 != sx)

    def forward(self, x):
        x = self.stem(x)
        for up, blk in zip(self.ups, self.blks):
            x = blk(up(x))
        x = self.head_mid(x)  # [B,1,H*2^n, W*2^n]

        if self.need_final_resize:
            # Antialiased downsample to the exact requested size (e.g., 256 -> 224)
            x = F.interpolate(x, size=(self.H_out, self.W_out),
                              mode='bilinear', align_corners=False, antialias=True)
        return x

# ---------------------------
# Train / Eval (50 epochs, best-epoch optional)
# ---------------------------
def rmse_map(y_true, y_pred):  # y: [B,1,H,W]
    return torch.sqrt(torch.mean((y_true - y_pred)**2))

def train_epoch(model, opt, loader, device):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        loss.backward(); opt.step()

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    rmses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        rmses.append(rmse_map(yb, pred).cpu())
    return float(torch.stack(rmses).mean())

# ---------------------------
# Main (one (fold,k,time,model) job)
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
    ap.add_argument("--outdir", type=str, default="results/light_subspace_removal")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)  # fixed per request
    ap.add_argument("--base", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--out_size", type=int, default=224, help="supervision size; must be multiple of token grid via power-of-two")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # Determine which configurations to run
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
        model_outdir = os.path.join(args.outdir.replace('simple_decoder', ''), model_config, 'simple_decoder')
        os.makedirs(model_outdir, exist_ok=True)
        out_path = os.path.join(model_outdir, f"cv_s{spatial_fold}_t{temporal_holdout}_k{k}.json")
        if os.path.exists(out_path):
            print(f"Result already exists at {out_path}, skipping...")
            continue

        assert 0 <= spatial_fold < 5, "spatial_fold must be 0..4"
        assert 0 <= temporal_holdout < 3, "temporal_holdout must be 0..2"

        # Load data with spatio-temporal split (train-times vs held-out time)
        (X_train, ids_train, Y_train), (X_val, ids_val, Y_val) = load_arrays_from_hf(
            model_config, temporal_holdout, args.out_size, args.out_size)

        # Infer grid from training data
        H, W, D = infer_token_grid(X_train)
        print(f"Inferred token grid: H={H}, W={W}, D={D}; supervising at {args.out_size}x{args.out_size}")

        # Group training data by tile id for spatial CV
        train_groups = defaultdict(list)
        for i, tid in enumerate(ids_train):
            train_groups[tid].append(i)
        train_tiles = sorted(train_groups.keys())

        # Group validation (held-out time) data by tile id
        val_groups = defaultdict(list)
        for i, tid in enumerate(ids_val):
            val_groups[tid].append(i)

        # Apply spatial fold to TRAIN TILES
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        spatial_folds = list(kf.split(train_tiles))
        tr_idx, va_idx = spatial_folds[spatial_fold]
        tr_tiles = [train_tiles[i] for i in tr_idx]
        va_tiles = [train_tiles[i] for i in va_idx]

        # Row indices for train (from train times) and temporal-val (held-out time)
        tr_rows = [j for t in tr_tiles for j in train_groups.get(t, [])]
        va_rows_temporal = [j for t in va_tiles for j in val_groups.get(t, [])]

        Xtr, Ytr = X_train[tr_rows], Y_train[tr_rows]
        Xval_time, Yval_time = X_val[va_rows_temporal], Y_val[va_rows_temporal]
        id_tr = [ids_train[j] for j in tr_rows]

        # Fit tcSVD on TRAIN ONLY (T=2 since we use two training timepoints)
        Q = estimate_Q_train_only_patchwise(Xtr, id_tr, T=2, k=k)
        XtrP       = apply_projection_np(Xtr, Q)
        Xval_timeP = apply_projection_np(Xval_time, Q)

        # Datasets / loaders: VALIDATION IS ON THE HELD-OUT TIME
        train_ds = DenseSplit(XtrP, Ytr, H, W)
        val_ds   = DenseSplit(Xval_timeP, Yval_time, H, W)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Model + optimizer (AdamW)
        model = GenericDenseDecoder(c_in=D, H=H, W=W, H_out=args.out_size, W_out=args.out_size,
                                    base=args.base, dropout=args.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Train for specified epochs; collect TEMPORAL-VAL RMSE history
        val_rmse_history = []
        for epoch in range(1, args.epochs+1):
            train_epoch(model, opt, train_loader, device)
            rm = eval_epoch(model, val_loader, device)
            val_rmse_history.append(rm)
            print(f"[s{spatial_fold}_t{temporal_holdout}_k{k}] epoch {epoch:03d}  TEMPORAL-VAL RMSE@{args.out_size} = {rm:.3f} cm")

        # Save metrics JSON
        result = {
            "spatial_fold": spatial_fold,
            "temporal_holdout": temporal_holdout,
            "k": k,
            "model_config": model_config,
            "seed": args.seed,
            "epochs": args.epochs,
            "val_rmse_history": [round(x, 6) for x in val_rmse_history],  # held-out time on held-out tiles
            "token_grid": [H, W, D],
            "out_size": args.out_size,
            "n_train_rows": len(tr_rows),
            "n_temporal_val_rows": len(va_rows_temporal),
            "train_tiles": tr_tiles,
            "val_tiles": va_tiles,  # spatially held-out tiles for this fold
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved {out_path}")

    print(f"Completed all {len(job_configs)} configurations for this job.")

if __name__ == "__main__":
    main()