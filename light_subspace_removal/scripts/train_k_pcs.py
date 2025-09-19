#!/usr/bin/env python3
"""
Patch-agnostic dense CHM regression from pre-encoded DINOv3 patch tokens.

- Infers token grid H×W from X_patch shape [M, Np, D] (requires Np be a square).
- Decoder upsamples from [D, H, W] to [1, H_out, W_out] where H_out/W_out are args (default 224).
- tcSVD lighting subspace removal with rank k (0 = baseline).
- 5-fold CV grouped by tile id, but THIS SCRIPT runs ONE (fold, k) job (for Slurm arrays).
- Trains 50 epochs (no early stopping) with default Adam; reports best-epoch RMSE@H_out×W_out.
- Writes JSON: results_cv/cv_fold{fold}_k{k}.json

Usage:
  python train_k_pcs.py --fold 0 --k 3 --outdir results_cv --out_size 224
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
def load_arrays_from_hf(H_out: int, W_out: int):
    ds = load_dataset("mpg-ranch/light-stable-semantics")
    Xp, ids, Y = [], [], []
    for ex in ds['train']:
        for key in ('t0', 't1', 't2'):
            tokens = np.array(ex[f'patch_{key}'], dtype=np.float32)  # [Np,D]
            Xp.append(tokens)
            ids.append(ex['idx'])
            Y.append(make_target(ex['canopy_height'], H_out, W_out).numpy())  # [1,H_out,W_out]
    Xp = np.stack(Xp, 0)     # [M,Np,D]
    Y  = np.stack(Y, 0)      # [M,1,H_out,W_out]
    return Xp, ids, Y

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
    Input:  [B, D, H, W]  (inferred from tokens)
    Output: [B, 1, H_out, W_out] where H_out/H == W_out/W == 2**n (power-of-two)
    """
    def __init__(self, c_in: int, H: int, W: int, H_out: int, W_out: int,
                 base: int = 256, dropout: float = 0.05):
        super().__init__()
        assert (H_out % H == 0) and (W_out % W == 0), "Output must be integer multiple of token grid"
        sx = H_out // H
        sy = W_out // W
        assert sx == sy, f"Non-uniform scale not supported (sx={sx}, sy={sy})"
        n_ups = int(math.log2(sx))
        assert 2 ** n_ups == sx, f"Scale {sx} must be power of two"

        self.stem = nn.Sequential(
            nn.Conv2d(c_in, base, 1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            UpBlock(base, base),
        )
        ups, blks = [], []
        c = base
        for _ in range(n_ups):
            ups.append(nn.ConvTranspose2d(c, c // 2, 2, 2))
            blks.append(UpBlock(c // 2, c // 2))
            c //= 2
        self.ups = nn.ModuleList(ups)
        self.blks = nn.ModuleList(blks)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        for up, blk in zip(self.ups, self.blks):
            x = blk(up(x))
        return self.head(x)

# ---------------------------
# Train / Eval (50 epochs, best-epoch RMSE)
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
# Main (one (fold,k) job)
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, help="fold index in [0..4]")
    ap.add_argument("--k", type=int, required=True, help="tcSVD rank (0=baseline)")
    ap.add_argument("--outdir", type=str, default="results/light_subspace_removal")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)  # fixed per request
    ap.add_argument("--base", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--out_size", type=int, default=224, help="supervision size; must be multiple of token grid via power-of-two")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # Load once to infer grid and build targets at requested size
    X_patch, ids, Y = load_arrays_from_hf(H_out=args.out_size, W_out=args.out_size)  # [M,Np,D], [M,1,O,O]
    H, W, D = infer_token_grid(X_patch)
    print(f"Inferred token grid: H={H}, W={W}, D={D}; supervising at {args.out_size}x{args.out_size}")

    # Group by tile id; build KFold (tile-level)
    groups = defaultdict(list)
    for i, tid in enumerate(ids): groups[tid].append(i)
    tiles = sorted(groups.keys())
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    folds = list(kf.split(tiles))
    assert 0 <= args.fold < 5, "fold must be 0..4"

    tr_idx, va_idx = folds[args.fold]
    tr_tiles = [tiles[i] for i in tr_idx]
    va_tiles = [tiles[i] for i in va_idx]
    tr_rows  = [j for t in tr_tiles for j in groups[t]]
    va_rows  = [j for t in va_tiles for j in groups[t]]

    Xtr, Ytr = X_patch[tr_rows], Y[tr_rows]
    Xva, Yva = X_patch[va_rows], Y[va_rows]
    id_tr = [ids[j] for j in tr_rows]

    # Fit tcSVD on TRAIN only; project train/val
    Q = estimate_Q_train_only_patchwise(Xtr, id_tr, T=3, k=args.k)
    XtrP = apply_projection_np(Xtr, Q)
    XvaP = apply_projection_np(Xva, Q)

    # Datasets / loaders
    train_ds = DenseSplit(XtrP, Ytr, H, W)
    val_ds   = DenseSplit(XvaP, Yva, H, W)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Model + optimizer (default Adam)
    model = GenericDenseDecoder(c_in=D, H=H, W=W, H_out=args.out_size, W_out=args.out_size,
                                base=args.base, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters())  # defaults: lr=1e-3, betas=(0.9,0.999), wd=0

    # Train fixed 50 epochs; record best-epoch RMSE
    best_rmse, best_epoch = float('inf'), -1
    for epoch in range(1, args.epochs+1):
        train_epoch(model, opt, train_loader, device)
        rm = eval_epoch(model, val_loader, device)
        if rm < best_rmse:
            best_rmse, best_epoch = rm, epoch
        print(f"[fold={args.fold} k={args.k}] epoch {epoch:03d}  val_RMSE@{args.out_size} = {rm:.3f} cm")

    # Save metrics JSON
    os.makedirs(args.outdir, exist_ok=True)
    result = {
        "fold": args.fold,
        "k": args.k,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_rmse_cm": round(best_rmse, 6),
        "token_grid": [H, W, D],
        "out_size": args.out_size,
        "n_train_rows": len(tr_rows),
        "n_val_rows": len(va_rows),
    }
    out_path = os.path.join(args.outdir, f"cv_fold{args.fold}_k{args.k}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()