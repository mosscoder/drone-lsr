#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_shadow_axes_vs_rgb.py

Purpose
-------
Show the first three global "shadow" principal components (PC1, PC2, PC3) of the residual
embedding space, compared directly to the RGB tile at each timepoint (no projection/removal).

Layout: 3 rows (Morning/Noon/Afternoon) × 4 columns:
  [RGB] [PC1 score map] [PC2 score map] [PC3 score map]

Notes
-----
- Shadow PCs are computed by SVD on residuals: z_t - mean_t z (per tile, per patch).
- Patch-grid (H×W) is inferred from Np = number of patch tokens (sqrt).
- Each PC column uses a fixed color scale across the three times to make rows comparable.
- Assumes the HF dataset provides 'image_t0/t1/t2' (PIL-compatible) and 'patch_t0/t1/t2'.

Usage
-----
python visualize_shadow_axes_vs_rgb.py \
  --out pngs/shadow_axes_vs_rgb_ds6.png \
  --index 6 \
  --clip 1.0  # optional percentile clipping for robust color scaling
"""

import argparse
import math
import os
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image


# ---------------------------
# Helpers
# ---------------------------

def infer_token_grid(x_np: np.ndarray):
    """
    x_np: [M, Np, D] or [Np, D]
    returns (H, W, D)
    """
    if x_np.ndim == 2:
        Np, D = x_np.shape
    elif x_np.ndim == 3:
        _, Np, D = x_np.shape
    else:
        raise ValueError(f"Unexpected shape {x_np.shape}")
    side = int(round(math.sqrt(Np)))
    assert side * side == Np, f"Tokens must form a square grid (Np={Np})."
    return side, side, D


@torch.no_grad()
def compute_residual_matrix(X_patch: np.ndarray, ids: list[str], times_per_tile=3) -> torch.Tensor:
    """
    X_patch: [M, Np, D] stacked as ... t0, t1, t2 ... across all tiles.
    ids: list[str] length M (tile ids aligned with X_patch rows)
    Returns: torch.Tensor [T * N_tiles * Np, D] of residuals (z_t - mean_t z)
    """
    groups = defaultdict(list)
    for i, tid in enumerate(ids):
        groups[tid].append(i)

    diffs = []
    for tid, idxs in groups.items():
        if len(idxs) != times_per_tile:
            continue
        Z = torch.tensor(X_patch[idxs], dtype=torch.float32)  # [T, Np, D]
        mu = Z.mean(dim=0, keepdim=True)                      # [1, Np, D]
        diffs.append(Z - mu)                                  # [T, Np, D]

    if not diffs:
        raise RuntimeError("No complete tiles with exactly 3 timepoints found.")

    D_mat = torch.cat(diffs, dim=0).reshape(-1, X_patch.shape[-1])  # [*, D]
    return D_mat


@torch.no_grad()
def residual_pcs(D_mat: torch.Tensor, num_pcs: int = 3):
    """
    D_mat: [N, D] residual matrix
    Returns V (D, num_pcs) with columns = PC1..PCk
    """
    _U, _S, Vh = torch.linalg.svd(D_mat.cpu(), full_matrices=False)  # Vh: [D, D]
    V = Vh[:num_pcs].T.contiguous()  # [D, num_pcs]
    return V


def pc_score_map(tokens_np: np.ndarray, v: torch.Tensor, H: int, W: int):
    """
    tokens_np: [Np, D]
    v: [D] (one PC direction)
    Returns 2D array [H, W] of dot products z·v
    """
    Z = torch.from_numpy(tokens_np).float()   # [Np, D]
    scores = (Z @ v).cpu().numpy()           # [Np]
    return scores.reshape(H, W)


def get_tile_tokens_and_images(ds_train, index: int):
    """
    Returns:
      tile_id: str
      toks: list of 3 arrays [Np, D] for t0, t1, t2
      rgbs: list of 3 PIL Images for t0, t1, t2 (or None if not present)
      labels: ["Morning","Noon","Afternoon"]
    """
    ex = ds_train[index]
    tile_id = ex["idx"]

    toks = [
        np.array(ex["patch_t0"], dtype=np.float32),
        np.array(ex["patch_t1"], dtype=np.float32),
        np.array(ex["patch_t2"], dtype=np.float32),
    ]

    # Try to fetch images; if not present, keep None
    rgb_keys = ["image_t0", "image_t1", "image_t2"]
    rgbs = []
    for k in rgb_keys:
        im = ex.get(k, None)
        if im is None:
            rgbs.append(None)
        else:
            # make sure it's PIL
            if isinstance(im, Image.Image):
                rgbs.append(im)
            else:
                try:
                    rgbs.append(Image.fromarray(im))
                except Exception:
                    rgbs.append(None)

    return tile_id, toks, rgbs, ["Morning", "Noon", "Afternoon"]


def robust_min_max(arrays, clip_percent=0.0):
    """
    Compute shared min/max across a list of numpy arrays, with optional symmetric percentile clipping.
    If clip_percent > 0, we clip to the [p, 100-p] percentiles *per component set*.
    """
    concat = np.concatenate([a.reshape(-1) for a in arrays], axis=0)
    if clip_percent > 0:
        lo = np.percentile(concat, clip_percent)
        hi = np.percentile(concat, 100 - clip_percent)
        return float(lo), float(hi)
    return float(concat.min()), float(concat.max())


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=6, help="which ds['train'][index] to visualize")
    ap.add_argument("--out", type=str, default="shadow_axes_vs_rgb.png")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--figsize", type=float, nargs=2, default=(10, 8))
    ap.add_argument("--clip", type=float, default=1.0, help="percentile clipping for PC maps (e.g., 1.0)")
    args = ap.parse_args()

    # 1) Load dataset
    ds = load_dataset("mpg-ranch/light-stable-semantics")
    ds_train = ds["train"]

    # 2) Build residual matrix over TRAIN and compute first 3 PCs (global lighting axes)
    X_list, ids = [], []
    for ex in ds_train:
        for tkey in ("t0", "t1", "t2"):
            X_list.append(np.array(ex[f"patch_{tkey}"], dtype=np.float32))  # [Np, D]
            ids.append(ex["idx"])
    X_patch = np.stack(X_list, 0)  # [M, Np, D]
    H, W, D = infer_token_grid(X_patch)
    print(f"Inferred grid: H={H}, W={W}, D={D}, total rows={len(X_patch)}")

    D_mat = compute_residual_matrix(X_patch, ids, times_per_tile=3)
    print(f"Residual matrix shape: {tuple(D_mat.shape)}")

    V = residual_pcs(D_mat, num_pcs=3)   # [D, 3]
    v1, v2, v3 = V[:, 0].contiguous(), V[:, 1].contiguous(), V[:, 2].contiguous()

    # 3) Pull tokens and RGBs for one tile (t0, t1, t2)
    tile_id, toks_list, rgb_list, time_labels = get_tile_tokens_and_images(ds_train, args.index)
    print(f"Selected tile: {tile_id}")

    # 4) Precompute PC maps and shared color scales (per component across times)
    pc1_maps, pc2_maps, pc3_maps = [], [], []
    for toks in toks_list:
        pc1_maps.append(pc_score_map(toks, v1, H, W))
        pc2_maps.append(pc_score_map(toks, v2, H, W))
        pc3_maps.append(pc_score_map(toks, v3, H, W))

    # Robust min/max per component (same scale across morning/noon/afternoon)
    pc1_min, pc1_max = robust_min_max(pc1_maps, clip_percent=args.clip)
    pc2_min, pc2_max = robust_min_max(pc2_maps, clip_percent=args.clip)
    pc3_min, pc3_max = robust_min_max(pc3_maps, clip_percent=args.clip)

    # 5) Plot 3×4 grid: RGB | PC1 | PC2 | PC3
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=args.figsize, constrained_layout=True)

    for r in range(3):
        # Column 1: RGB (if available)
        ax_rgb = axes[r, 0]
        rgb = rgb_list[r]
        if rgb is None:
            # fallback: blank pane with note
            ax_rgb.imshow(np.zeros((H*16, W*16, 3), dtype=np.uint8))  # rough scaling
            ax_rgb.text(0.5, 0.5, "RGB not in dataset", ha="center", va="center", color="w")
        else:
            ax_rgb.imshow(rgb)
        if r == 0:
            ax_rgb.set_title("RGB", fontsize=10)
        ax_rgb.set_ylabel(time_labels[r], fontsize=10)
        ax_rgb.set_xticks([]); ax_rgb.set_yticks([])

        # Column 2: PC1 score map
        ax1 = axes[r, 1]
        im1 = ax1.imshow(pc1_maps[r], cmap="coolwarm", vmin=pc1_min, vmax=pc1_max, interpolation="nearest")
        if r == 0: ax1.set_title("PC1 (shadow axis 1)", fontsize=10)
        ax1.set_xticks([]); ax1.set_yticks([])

        # Column 3: PC2 score map
        ax2 = axes[r, 2]
        im2 = ax2.imshow(pc2_maps[r], cmap="coolwarm", vmin=pc2_min, vmax=pc2_max, interpolation="nearest")
        if r == 0: ax2.set_title("PC2 (shadow axis 2)", fontsize=10)
        ax2.set_xticks([]); ax2.set_yticks([])

        # Column 4: PC3 score map
        ax3 = axes[r, 3]
        im3 = ax3.imshow(pc3_maps[r], cmap="coolwarm", vmin=pc3_min, vmax=pc3_max, interpolation="nearest")
        if r == 0: ax3.set_title("PC3 (shadow axis 3)", fontsize=10)
        ax3.set_xticks([]); ax3.set_yticks([])

    fig.suptitle(
        f"First three residual PCs (shadow subspace) vs RGB — tile {tile_id}\n"
        f"Grid {H}×{W}, feature dim {D} | SVD on residuals: z_t − mean_t z (train split)",
        fontsize=11
    )

    # Optional shared colorbars (one per PC) on the right
    cbar_kw = dict(shrink=0.6, pad=0.02)
    # Create tiny axes for colorbars
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    for c_idx, (col, vmin, vmax, label) in enumerate([(1, pc1_min, pc1_max, "PC1 score"),
                                                      (2, pc2_min, pc2_max, "PC2 score"),
                                                      (3, pc3_min, pc3_max, "PC3 score")]):
        # attach colorbar to the top row plot of that column
        ax_ref = axes[0, col]
        divider = make_axes_locatable(ax_ref)
        cax = divider.append_axes("top", size="5%", pad="2%")
        plt.colorbar(ax_ref.images[0], cax=cax, orientation="horizontal")
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cax.set_xlabel(label, fontsize=9)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=args.dpi)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()