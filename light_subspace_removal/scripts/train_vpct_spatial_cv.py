#!/usr/bin/env python3
"""
Patch-agnostic dense CHM regression from pre-encoded DINOv2/DINOv3 patch tokens.

Spatial-only 5-fold CV:
- Train: ~80% of tiles, all timepoints (t0,t1,t2)
- Val  : held-out ~20% tiles, all timepoints (t0,t1,t2)

Lighting subspace removal by TARGET VARIANCE EXPLAINED:
- --var_pct in {0,5,10,25,50,100}
- Chooses minimal k with cumulative EVR >= var_pct/100 on TRAIN-ONLY residuals
  (per-patch, per-tile residuals z_{i,p,t} - mean_t z_{i,p,·}, T=3).

Writes:
  results/{model_config}/simple_decoder/cv_s{fold}_v{vvv}.json
"""

import argparse, json, os, random, math
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import KFold

# ---------------------------
# Small helpers
# ---------------------------
def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0

def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------
# Grid / targets
# ---------------------------
def infer_token_grid(X_patch: np.ndarray):
    assert X_patch.ndim == 3, f"Expected [M,Np,D], got {X_patch.shape}"
    _, Np, D = X_patch.shape
    side = int(round(Np ** 0.5))
    assert side * side == Np, f"Tokens not square: Np={Np}"
    return side, side, D

def tokens_to_chw(tokens: np.ndarray, H: int, W: int):
    D = tokens.shape[-1]
    return tokens.reshape(H, W, D).transpose(2, 0, 1)

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
# tcSVD with variance target
# ---------------------------
@torch.no_grad()
def svd_rank_for_var_explained(D: torch.Tensor, target_pct: float):
    target_pct = float(target_pct)
    if target_pct <= 0: return None, 0, np.array([]), np.array([])
    U, S, Vh = torch.linalg.svd(D.cpu(), full_matrices=False)  # Vh: [r,d]
    var = S**2
    total = var.sum().item()
    if total <= 0: return None, 0, np.array([]), np.array([])
    evr = (var / total).cpu().numpy()
    cum = np.cumsum(evr)
    k = int(np.searchsorted(cum, target_pct/100.0) + 1)
    k = max(0, min(k, Vh.shape[0]))
    if k == 0: return None, 0, evr, cum
    Vk = Vh[:k].T.contiguous()
    Q, _ = torch.linalg.qr(Vk)  # [d,k]
    return Q, k, evr, cum

@torch.no_grad()
def estimate_Q_train_only_patchwise_vpct(X_patch: np.ndarray, ids: list[str], T=3, var_pct: float = 0.0):
    """Compute residual matrix D from TRAIN tiles only, across all patches/time,
       with z_{i,p,t} - mean_t z_{i,p,·}. Then pick k by target EVR."""
    if var_pct <= 0: return None, 0, np.array([]), np.array([])
    groups = defaultdict(list)
    for i, tid in enumerate(ids): groups[tid].append(i)
    diffs = []
    for tid, idxs in groups.items():
        if len(idxs) != T:  # expect all three times in train
            continue
        Z = torch.tensor(X_patch[idxs], dtype=torch.float32)  # [T,Np,D]
        mu = Z.mean(dim=0, keepdim=True)
        diffs.append(Z - mu)
    if not diffs: return None, 0, np.array([]), np.array([])
    D_mat = torch.cat(diffs, dim=0).reshape(-1, X_patch.shape[-1])  # [T*Ntiles*Np, D]
    return svd_rank_for_var_explained(D_mat, var_pct)

def apply_projection_np(X_patch: np.ndarray, Q: torch.Tensor | None):
    X = torch.from_numpy(X_patch).float()
    if (Q is None) or (Q.numel() == 0): return X.numpy().astype(np.float32)
    P = Q @ Q.T
    return (X - X @ P).numpy().astype(np.float32)

# ---------------------------
# Data (HF): load ALL times
# ---------------------------
def load_all_times_from_hf(model_config: str, H_out: int, W_out: int):
    """Return arrays containing ALL timepoints for every tile."""
    ds_embed = load_dataset("mpg-ranch/light-stable-semantics", model_config, split='train')
    ds_default = load_dataset("mpg-ranch/light-stable-semantics", "default", split='train')
    canopy_map = {ex['idx']: ex['canopy_height'] for ex in ds_default}

    X_all, ids_all, Y_all = [], [], []
    for ex in ds_embed:
        idx = ex['idx']
        target = make_target(canopy_map[idx], H_out, W_out).numpy()
        for key in ('t0', 't1', 't2'):
            tokens = np.array(ex[f'patch_{key}'], dtype=np.float32)  # [Np,D]
            X_all.append(tokens); ids_all.append(idx); Y_all.append(target)

    return np.stack(X_all, 0), ids_all, np.stack(Y_all, 0)  # [3*Ntiles, Np/D or 1/H/W]

class DenseSplit(Dataset):
    def __init__(self, X_patch, Y, H, W):
        self.Xp = X_patch; self.Y = Y; self.H, self.W = H, W
    def __len__(self): return len(self.Xp)
    def __getitem__(self, i):
        x = tokens_to_chw(self.Xp[i], self.H, self.W)      # [D,H,W]
        y = torch.from_numpy(self.Y[i]).float()            # [1,H_out,W_out]
        return torch.from_numpy(x).float(), y

# ---------------------------
# Decoder
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
    def __init__(self, c_in: int, H: int, W: int, H_out: int, W_out: int,
                 base: int = 256, dropout: float = 0.05):
        super().__init__()
        assert (H_out % H == 0) and (W_out % W == 0)
        sx = H_out // H; sy = W_out // W
        assert sx == sy
        self.H_out, self.W_out = H_out, W_out
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, base, 1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            UpBlock(base, base),
        )
        sx_p2 = sx if _is_pow2(sx) else _next_pow2(sx)
        n_ups = int(math.log2(sx_p2))
        ups, blks = [], []
        c = base
        for _ in range(n_ups):
            ups.append(nn.ConvTranspose2d(c, c // 2, 2, 2))
            blks.append(UpBlock(c // 2, c // 2))
            c //= 2
        self.ups = nn.ModuleList(ups)
        self.blks = nn.ModuleList(blks)
        self.head_mid = nn.Conv2d(c, 1, 1)
        self.need_final_resize = (sx_p2 != sx)

    def forward(self, x):
        x = self.stem(x)
        for up, blk in zip(self.ups, self.blks):
            x = blk(up(x))
        x = self.head_mid(x)
        if self.need_final_resize:
            x = F.interpolate(x, size=(self.H_out, self.W_out),
                              mode='bilinear', align_corners=False, antialias=True)
        return x

# ---------------------------
# Train / Eval
# ---------------------------
def rmse_map(y_true, y_pred):
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
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spatial_fold", type=int, default=None, help="fold index in [0..4] (single-exp mode)")
    ap.add_argument("--var_pct", type=float, default=None, help="target % variance explained to remove [0..100]")
    ap.add_argument("--model_config", type=str, default=None, help="dinov2_base or dinov3_sat")

    ap.add_argument("--job_id", type=int, default=None)
    ap.add_argument("--total_jobs", type=int, default=8)
    ap.add_argument("--total_configs", type=int, default=110)  # 11 var levels × 5 folds × 2 configs

    ap.add_argument("--outdir", type=str, default="results/light_subspace_removal")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--base", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--out_size", type=int, default=224)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    if args.job_id is not None:
        VAR_PCTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        FOLDS = [0,1,2,3,4]
        MODEL_CONFIGS = ['dinov2_base', 'dinov3_sat']
        all_configs = [(vp, f, mc) for mc in MODEL_CONFIGS for f in FOLDS for vp in VAR_PCTS]
        assert len(all_configs) == args.total_configs, f"Expected {args.total_configs}, got {len(all_configs)}"
        per_job = args.total_configs // args.total_jobs
        extra = args.total_configs % args.total_jobs
        if args.job_id < extra:
            start = args.job_id * (per_job + 1); end = start + per_job + 1
        else:
            start = extra * (per_job + 1) + (args.job_id - extra) * per_job
            end = start + per_job
        job_configs = all_configs[start:end]
        print(f"Job {args.job_id} handling {len(job_configs)} configs")
    else:
        if args.spatial_fold is None or args.var_pct is None or args.model_config is None:
            raise ValueError("Provide --job_id ... or all of: --spatial_fold --var_pct --model_config")
        job_configs = [(float(args.var_pct), int(args.spatial_fold), args.model_config)]

    for var_pct, spatial_fold, model_config in job_configs:
        vtag = f"{int(round(var_pct)):03d}"
        print(f"\n=== v={var_pct}%, spatial_fold={spatial_fold}, model_config={model_config} ===")

        model_outdir = os.path.join(args.outdir, model_config, 'simple_decoder')
        os.makedirs(model_outdir, exist_ok=True)
        out_path = os.path.join(model_outdir, f"cv_s{spatial_fold}_v{vtag}.json")
        if os.path.exists(out_path):
            print(f"Exists: {out_path} — skipping.")
            continue

        # Load all times, all tiles
        X_all, ids_all, Y_all = load_all_times_from_hf(model_config, args.out_size, args.out_size)
        H, W, D = infer_token_grid(X_all)
        print(f"Inferred token grid: H={H}, W={W}, D={D}; supervising at {args.out_size}x{args.out_size}")

        # Group by tile id (each tile should have exactly 3 rows: t0,t1,t2)
        groups = defaultdict(list)
        for i, tid in enumerate(ids_all):
            groups[tid].append(i)
        tiles = sorted(groups.keys())

        # Spatial 5-fold over tiles
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        folds = list(kf.split(tiles))
        tr_idx, va_idx = folds[spatial_fold]
        tr_tiles = [tiles[i] for i in tr_idx]
        va_tiles = [tiles[i] for i in va_idx]
        tr_rows = [j for t in tr_tiles for j in groups.get(t, [])]
        va_rows = [j for t in va_tiles for j in groups.get(t, [])]

        # Prepare arrays
        Xtr, Ytr = X_all[tr_rows], Y_all[tr_rows]
        Xva, Yva = X_all[va_rows], Y_all[va_rows]
        id_tr = [ids_all[j] for j in tr_rows]

        # Fit tcSVD on TRAIN-ONLY residuals (T=3)
        Q, k_chosen, evr, cum = estimate_Q_train_only_patchwise_vpct(Xtr, id_tr, T=3, var_pct=var_pct)

        # Project train/val
        XtrP = apply_projection_np(Xtr, Q)
        XvaP = apply_projection_np(Xva, Q)

        # Datasets / loaders
        train_ds = DenseSplit(XtrP, Ytr, H, W)
        val_ds   = DenseSplit(XvaP, Yva, H, W)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Model + optimizer
        model = GenericDenseDecoder(c_in=D, H=H, W=W, H_out=args.out_size, W_out=args.out_size,
                                    base=args.base, dropout=args.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Train
        val_rmse_history = []
        for epoch in range(1, args.epochs+1):
            train_epoch(model, opt, train_loader, device)
            rm = eval_epoch(model, val_loader, device)
            val_rmse_history.append(rm)
            print(f"[s{spatial_fold}_v{var_pct:.0f}% (k={k_chosen})] epoch {epoch:03d}  VAL RMSE@{args.out_size} = {rm:.3f} cm")

        # Save
        evr_head = [float(x) for x in evr[:10]] if evr.size else []
        cum_head = [float(x) for x in cum[:10]] if cum.size else []
        achieved_cum = float(cum[k_chosen-1]) if (k_chosen > 0 and cum.size >= k_chosen) else 0.0

        out = {
            "spatial_fold": spatial_fold,
            "var_pct_target": float(var_pct),
            "k_chosen": int(k_chosen),
            "cum_evr_at_k": achieved_cum,
            "evr_head": evr_head,
            "cum_evr_head": cum_head,
            "model_config": model_config,
            "seed": args.seed,
            "epochs": args.epochs,
            "val_rmse_history": [round(x, 6) for x in val_rmse_history],
            "token_grid": [H, W, D],
            "out_size": args.out_size,
            "n_train_rows": len(tr_rows),
            "n_val_rows": len(va_rows),
            "train_tiles": tr_tiles,
            "val_tiles": va_tiles,
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved {out_path}")

    print("Done.")

if __name__ == "__main__":
    main()