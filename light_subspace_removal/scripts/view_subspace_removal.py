#!/usr/bin/env python3
"""
Visualize a single tile across time with shadow-subspace removal + local PCA.

Layout (3 rows × 4 columns):
  - Column 1: RGB at t0, t1, t2 (rows: t0→t2)
  - Column 2: False color composite after 0% variance removal
  - Column 3: False color composite after 50% variance removal
  - Column 4: False color composite after 100% variance removal

Shadow/lighting basis (Q) is fit on TRAIN tiles only for the chosen spatial fold (T=3),
using variance thresholds of 0%, 50%, and 100%. False color composites map the top 3
PCA components to RGB channels for visualization.

Embeddings source:
  load_dataset("mpg-ranch/drone-lsr", {model_config}, split="train")

RGB source:
  - Tries to read from the "default" HF config using common key patterns
  - Or pass --rgb_template like "/path/to/rgb/{tile_id}_{time}.png" with time in {t0,t1,t2}

You can select the tile either by --tile_id or by --row_index into the HF split (row_index overrides).

Example:
  python view_subspace_removal.py \
    --model_config dinov3_sat --spatial_fold 0 \
    --row_index 137 \
    --out_png results/light_subspace_removal/viz/row137_fold0_false_color.png
"""

import argparse, os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from datasets import load_dataset
from sklearn.model_selection import KFold
from PIL import Image
from scipy.linalg import orthogonal_procrustes

# --- display prefs
mpl.rcParams["figure.dpi"] = 120
PRGN = "PRGn"  # Diverging for signed PCA scores (user preference)

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def infer_token_grid(X_patch: np.ndarray):
    """X_patch: [M, Np, D] -> returns (H, W, D), with Np = H*W a perfect square."""
    assert X_patch.ndim == 3, f"Expected [M,Np,D], got {X_patch.shape}"
    _, Np, D = X_patch.shape
    side = int(round(Np ** 0.5))
    assert side * side == Np, f"Tokens not square: Np={Np}"
    return side, side, D

@torch.no_grad()
def svd_rank_for_var_explained(D: torch.Tensor, target_pct: float):
    """
    SVD on residual matrix D [N, d]. Choose minimal k s.t. cumEVR >= target_pct/100.
    Returns (Q [d,k] or None, k, evr np.array, cum np.array).
    """
    target_pct = float(target_pct)
    if target_pct <= 0:
        return None, 0, np.array([]), np.array([])
    U, S, Vh = torch.linalg.svd(D.cpu(), full_matrices=False)  # Vh: [r,d]
    var = S ** 2
    total = var.sum().item()
    if total <= 0:
        return None, 0, np.array([]), np.array([])
    evr = (var / total).cpu().numpy()
    cum = np.cumsum(evr)
    k = int(np.searchsorted(cum, target_pct / 100.0) + 1)
    k = max(0, min(k, Vh.shape[0]))
    if k == 0:
        return None, 0, evr, cum
    Vk = Vh[:k].T.contiguous()
    Q, _ = torch.linalg.qr(Vk)  # [d,k]
    return Q, k, evr, cum

@torch.no_grad()
def estimate_Q_train_only_patchwise_vpct(Xtr_patch: np.ndarray, tr_ids: list[str], T=3, var_pct: float = 0.0):
    """
    Build residuals per (tile, patch, time): z_{i,p,t} - mean_t z_{i,p,·} for TRAIN tiles (expect T=3).
    Stack across tiles/patches -> D_mat [T*Ntiles*Np, D], then choose k by target EVR.
    """
    if var_pct <= 0:
        return None, 0, np.array([]), np.array([])
    groups = defaultdict(list)
    for i, tid in enumerate(tr_ids):
        groups[tid].append(i)
    diffs = []
    for tid, idxs in groups.items():
        if len(idxs) != T:
            continue
        Z = torch.tensor(Xtr_patch[idxs], dtype=torch.float32)  # [T,Np,D]
        mu = Z.mean(dim=0, keepdim=True)
        diffs.append(Z - mu)
    if not diffs:
        return None, 0, np.array([]), np.array([])
    D_mat = torch.cat(diffs, dim=0).reshape(-1, Xtr_patch.shape[-1])  # [T*Ntiles*Np, D]
    return svd_rank_for_var_explained(D_mat, var_pct)

def apply_projection_np(X_patch: np.ndarray, Q: torch.Tensor | None):
    """Project out columns of Q from last-dim of X_patch [M,Np,D]."""
    X = torch.from_numpy(X_patch).float()
    if (Q is None) or (Q.numel() == 0):
        return X.numpy().astype(np.float32)
    P = Q @ Q.T  # [D,D]
    return (X - X @ P).numpy().astype(np.float32)

def align_components(target_components: np.ndarray, reference_components: np.ndarray) -> np.ndarray:
    """Align target PCA components to reference using Procrustes rotation.

    Args:
        target_components: [n_components, n_features] array to be aligned
        reference_components: [n_components, n_features] reference array

    Returns:
        aligned_components: [n_components, n_features] rotated target components
    """
    # Components are [3, D]. We want to rotate in 3D component space, not D-dimensional feature space
    # Transpose to [D, 3] so we find 3x3 rotation matrix R
    target_T = target_components.T      # [D, 3]
    reference_T = reference_components.T # [D, 3]

    # Find 3x3 rotation R such that target_T @ R ≈ reference_T
    R, _ = orthogonal_procrustes(target_T, reference_T)

    # Apply rotation and transpose back to [3, D]
    aligned_components = (target_T @ R).T
    return aligned_components

def make_false_color_composite(Xt_proj: np.ndarray, H: int, W: int, target_height: int = 1024,
                             target_width: int = 1024, seed: int = 42,
                             reference_pca=None) -> tuple[np.ndarray, object]:
    """Convert projected embeddings [3, Np, D] to false color RGB [3, target_H, target_W, 3].
    Uses PCA to get top 3 components and maps them to RGB channels.

    Args:
        Xt_proj: Projected embeddings [3, Np, D]
        H, W: Patch grid dimensions
        target_height, target_width: Output image dimensions (default 1024x1024)
        seed: Random seed for PCA
        reference_pca: Optional reference PCA object for component alignment

    Returns:
        (false_colors, pca): False color images [3, target_H, target_W, 3] and fitted PCA object
    """
    # Flatten across time and patches for PCA
    D = Xt_proj.shape[-1]
    Xtile_cat = Xt_proj.reshape(-1, D)  # [3*Np, D]

    if float(Xtile_cat.var()) == 0.0:
        # If no variance, return black images
        return np.zeros((3, target_height, target_width, 3), dtype=np.uint8), None

    # PCA to get top 3 components
    pca = PCA(n_components=3, svd_solver="auto", random_state=seed)
    scores = pca.fit_transform(Xtile_cat)  # [3*Np, 3]

    # Align with reference PCA if provided
    if reference_pca is not None:
        # Align current PCA components to reference
        aligned_components = align_components(pca.components_, reference_pca.components_)
        # Re-orthonormalize to eliminate numerical drift
        U, _, Vt = np.linalg.svd(aligned_components, full_matrices=False)
        aligned_components = (U @ Vt)
        # Recompute scores using aligned components (must center first!)
        scores = (Xtile_cat - pca.mean_) @ aligned_components.T

    # Split back by time and reshape to spatial grid
    Np = H * W
    false_colors = []
    for t in range(3):
        pc_scores = scores[t*Np:(t+1)*Np, :].reshape(H, W, 3)  # [H, W, 3]

        # Create HSV image: PC1→Hue, PC2→Saturation, PC3→Value
        hsv_img = np.zeros((H, W, 3), dtype=np.float32)

        # PC1 → Hue (0-360°, but PIL uses 0-179 for H)
        hue_channel = pc_scores[:, :, 0]
        h_low, h_high = np.nanpercentile(hue_channel, [1, 99])
        if h_high > h_low:
            hue_norm = np.clip((hue_channel - h_low) / (h_high - h_low), 0, 1)
        else:
            hue_norm = np.zeros_like(hue_channel)
        hsv_img[:, :, 0] = hue_norm * 179  # Scale to PIL's hue range [0, 179]

        # PC2 → Saturation (0-1)
        sat_channel = pc_scores[:, :, 1]
        s_low, s_high = np.nanpercentile(sat_channel, [1, 99])
        if s_high > s_low:
            sat_norm = np.clip((sat_channel - s_low) / (s_high - s_low), 0, 1)
        else:
            sat_norm = np.zeros_like(sat_channel)
        hsv_img[:, :, 1] = sat_norm * 255  # Scale to [0, 255] for PIL

        # PC3 → Value/Brightness (0-1)
        val_channel = pc_scores[:, :, 2]
        v_low, v_high = np.nanpercentile(val_channel, [1, 99])
        if v_high > v_low:
            val_norm = np.clip((val_channel - v_low) / (v_high - v_low), 0, 1)
        else:
            val_norm = np.zeros_like(val_channel)
        hsv_img[:, :, 2] = val_norm * 255  # Scale to [0, 255] for PIL

        # Convert HSV to RGB
        hsv_img_uint8 = hsv_img.astype(np.uint8)
        hsv_pil = Image.fromarray(hsv_img_uint8, mode='HSV')
        rgb_img = np.array(hsv_pil.convert('RGB'))

        # Upsample to target resolution using PIL
        if (H, W) != (target_height, target_width):
            pil_img = Image.fromarray(rgb_img)
            rgb_img_upsampled = np.array(pil_img.resize((target_width, target_height), Image.LANCZOS))
        else:
            rgb_img_upsampled = rgb_img

        false_colors.append(rgb_img_upsampled)

    return np.stack(false_colors, 0), pca  # [3, target_H, target_W, 3]

def load_embeddings(model_config: str):
    """Return (tile_ids list, dict tile_id -> {'t0','t1','t2': np.ndarray[Np,D]})"""
    ds = load_dataset("mpg-ranch/drone-lsr", model_config, split="train")
    tiles, X_by_tile = [], {}
    for ex in ds:
        tid = ex["idx"]
        X_by_tile[tid] = {
            "t0": np.array(ex["patch_t0"], dtype=np.float32),
            "t1": np.array(ex["patch_t1"], dtype=np.float32),
            "t2": np.array(ex["patch_t2"], dtype=np.float32),
        }
        tiles.append(tid)
    return tiles, X_by_tile

def resolve_tile_id_from_row(model_config: str, row_index: int) -> str:
    ds = load_dataset("mpg-ranch/drone-lsr", model_config, split="train")
    if not (0 <= row_index < len(ds)):
        raise IndexError(f"--row_index {row_index} out of range [0, {len(ds)-1}]")
    return ds[row_index]["idx"]

def load_rgb_triplet(tile_id: str, rgb_template: str | None):
    # Try user template first
    if rgb_template:
        frames = []
        for tk in ("t0","t1","t2"):
            p = Path(rgb_template.format(tile_id=tile_id, time=tk))
            if not p.exists():
                raise FileNotFoundError(f"RGB template path not found: {p}")
            frames.append(Image.open(p).convert("RGB"))
        return frames

    # Try HF default config with common key patterns
    ds_def = load_dataset("mpg-ranch/drone-lsr", "default", split="train")
    by_id = {ex["idx"]: ex for ex in ds_def}
    if tile_id not in by_id:
        raise KeyError(f"Tile {tile_id} not found in default config")
    ex = by_id[tile_id]
    candidates = [
        ("rgb_t0","rgb_t1","rgb_t2"),
        ("image_t0","image_t1","image_t2"),
        ("t0","t1","t2"),
    ]
    for t0k,t1k,t2k in candidates:
        if t0k in ex and t1k in ex and t2k in ex:
            def to_img(x):
                return x if isinstance(x, Image.Image) else Image.fromarray(np.array(x))
            return [to_img(ex[t0k]).convert("RGB"), to_img(ex[t1k]).convert("RGB"), to_img(ex[t2k]).convert("RGB")]
    raise KeyError(f"No RGB keys found for tile {tile_id}. Provide --rgb_template.")

def build_folds(all_tile_ids, seed=42):
    tiles = sorted(set(all_tile_ids))
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    return tiles, list(kf.split(tiles))

def tile_rows_for_ids(X_by_tile, tile_ids):
    rows, ids = [], []
    for tid in tile_ids:
        for tk in ("t0","t1","t2"):
            rows.append(X_by_tile[tid][tk])
            ids.append(tid)
    return np.stack(rows, 0), ids  # [3*Nt, Np, D]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_config", type=str, required=True, choices=["dinov2_base","dinov3_sat"])
    ap.add_argument("--spatial_fold", type=int, required=True, help="fold in [0..4]")
    ap.add_argument("--tile_id", type=str, default=None, help="Tile identifier (overridden by --row_index if set)")
    ap.add_argument("--row_index", type=int, default=None, help="Index into HF dataset split='train' (overrides --tile_id)")
    ap.add_argument("--rgb_template", type=str, default=None,
                    help="e.g., '/data/rgb/{tile_id}_{time}.png' with time in {t0,t1,t2}'")
    ap.add_argument("--out_png", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    # Resolve tile_id by row_index if provided
    if args.row_index is not None:
        resolved = resolve_tile_id_from_row(args.model_config, args.row_index)
        print(f"[info] Resolved tile_id from row_index={args.row_index} -> '{resolved}'")
        args.tile_id = resolved
    if not args.tile_id:
        raise ValueError("Provide either --tile_id or --row_index")

    # Load embeddings for all tiles (needed to fit Q on train-only)
    all_tiles, X_by_tile = load_embeddings(args.model_config)
    if args.tile_id not in X_by_tile:
        raise KeyError(f"Tile '{args.tile_id}' not found in embeddings")

    # Shape info
    sample = X_by_tile[args.tile_id]["t0"]
    H, W, D = infer_token_grid(np.stack([sample], 0))
    Np = H * W

    # Spatial folds by tile-id; get TRAIN tiles for the requested fold
    tiles, folds = build_folds(all_tiles, seed=args.seed)
    assert 0 <= args.spatial_fold < 5
    tr_idx, va_idx = folds[args.spatial_fold]
    tr_tiles = [tiles[i] for i in tr_idx]
    va_tiles = [tiles[i] for i in va_idx]

    # Fit Q matrices for 0%, 50%, 100% variance removal
    Xtr, id_tr = tile_rows_for_ids(X_by_tile, tr_tiles)  # [3*Nt, Np, D]
    var_levels = [0.0, 50.0, 100.0]
    Q_matrices = []
    k_values = []

    for var_pct in var_levels:
        Q, k_chosen, evr, cum = estimate_Q_train_only_patchwise_vpct(Xtr, id_tr, T=3, var_pct=var_pct)
        Q_matrices.append(Q)
        k_values.append(k_chosen)
        if Q is None:
            print(f"[info] var_pct={var_pct}% → k=0 (no removal)")
        else:
            cumk = (cum[k_chosen-1] if len(cum) >= k_chosen and k_chosen > 0 else 0.0)
            print(f"[info] var_pct={var_pct}% → k={k_chosen}, cumEVR≈{cumk:.3f}")

    # Pull this tile's 3 timepoints
    Xt_tile = np.stack([X_by_tile[args.tile_id][tk] for tk in ("t0","t1","t2")], 0)  # [3, Np, D]

    # Generate false color composites for each variance level
    false_color_imgs = []
    reference_pca = None

    for i, Q in enumerate(Q_matrices):
        Xt_proj = apply_projection_np(Xt_tile, Q)  # [3, Np, D]

        if i == 0:
            # First level (0%) - establish reference
            fc_rgb, reference_pca = make_false_color_composite(
                Xt_proj, H, W, target_height=1024, target_width=1024,
                seed=args.seed, reference_pca=None)
        else:
            # Subsequent levels (50%, 100%) - align to reference
            fc_rgb, _ = make_false_color_composite(
                Xt_proj, H, W, target_height=1024, target_width=1024,
                seed=args.seed, reference_pca=reference_pca)

        false_color_imgs.append(fc_rgb)

    # Load RGB triplet
    rgb_imgs = load_rgb_triplet(args.tile_id, args.rgb_template)  # list[PIL] length 3

    # === Plot 3x4 grid: rows t0,t1,t2; col1 RGB, col2-4 false color at 0%,50%,100% ===
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    times = ["t0","t1","t2"]
    col_headers = ["RGB", "0%", "50%", "100%"]

    # Add column headers at the top
    for c, header in enumerate(col_headers):
        axes[0, c].text(0.5, 1.05, header, transform=axes[0, c].transAxes,
                       ha='center', va='bottom', fontsize=14, fontweight='bold')

    for r in range(3):
        # Column 1: RGB
        ax = axes[r,0]
        ax.imshow(rgb_imgs[r])
        ax.set_axis_off()

        # Add time label on the left
        ax.text(-0.1, 0.5, times[r], transform=ax.transAxes,
               ha='right', va='center', fontsize=12, rotation=90)

        # Columns 2-4: False color composites for 0%, 50%, 100%
        for c in range(1, 4):
            var_idx = c - 1  # 0, 1, 2 for 0%, 50%, 100%
            ax = axes[r, c]
            ax.imshow(false_color_imgs[var_idx][r])  # [time_idx][H, W, 3]
            ax.set_axis_off()

    plt.tight_layout()

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")

if __name__ == "__main__":
    main()