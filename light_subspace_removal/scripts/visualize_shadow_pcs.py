#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_shadow_pcs.py

Purpose
-------
Show temporal shadow principal components and their embedding correlates. Computes temporal
residuals (z_t - z_{t-1}) at the patch level, finds the first three PCs of this temporal
shadow subspace, and identifies which embedding dimensions correlate most strongly with each PC.

Layout: 3 rows (Morning/Noon/Afternoon) × 4 columns:
  [RGB] [PC1 Correlate] [PC2 Correlate] [PC3 Correlate]

Notes
-----
- Temporal shadow PCs are computed by SVD on temporal residuals: z_t - z_{t-1} (per patch, per tile).
- For each PC, we identify the embedding dimension with highest absolute magnitude.
- Correlate columns show the raw embedding values at those dimensions across patches.
- Patch-grid (H×W) is inferred from Np = number of patch tokens (sqrt).
- Each correlate column uses a fixed color scale across the three times to make rows comparable.
- Assumes the HF dataset provides 'image_t0/t1/t2' (PIL-compatible) and 'patch_t0/t1/t2'.

Usage
-----
# Select by tile ID with 10 PCs (recommended)
python visualize_shadow_pcs.py \
  --tile_id 22_19 \
  --out pngs/tile_22_19_10pcs.png \
  --model_config dinov3_sat \
  --num_pcs 10

# Or select by index with 3 PCs (fallback)
python visualize_shadow_pcs.py \
  --index 6 \
  --out pngs/temporal_shadow_correlates.png \
  --model_config dinov3_sat \
  --num_pcs 3 \
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
def compute_temporal_residual_matrix(X_patch: np.ndarray, ids: list[str], times_per_tile=3) -> torch.Tensor:
    """
    Compute temporal residuals at the patch level.
    X_patch: [M, Np, D] stacked as ... t0, t1, t2 ... across all tiles.
    ids: list[str] length M (tile ids aligned with X_patch rows)
    Returns: torch.Tensor [N_tiles * Np * (T-1), D] of temporal residuals
             (z_t - z_t-1 for each patch across time)
    """
    groups = defaultdict(list)
    for i, tid in enumerate(ids):
        groups[tid].append(i)

    temporal_diffs = []
    for tid, idxs in groups.items():
        if len(idxs) != times_per_tile:
            continue
        Z = torch.tensor(X_patch[idxs], dtype=torch.float32)  # [T, Np, D]

        # Compute temporal differences: t1-t0, t2-t1 for each patch
        for t in range(1, times_per_tile):
            diff = Z[t] - Z[t-1]  # [Np, D] - differences for all patches at this temporal step
            temporal_diffs.append(diff)

    if not temporal_diffs:
        raise RuntimeError("No complete tiles with exactly 3 timepoints found.")

    D_mat = torch.cat(temporal_diffs, dim=0)  # [N_tiles * Np * (T-1), D]
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


@torch.no_grad()
def find_embedding_correlates(V: torch.Tensor, num_pcs: int = 3):
    """
    Find which embedding dimensions have highest magnitude for each PC.
    V: [D, num_pcs] - PC directions
    Returns: list of (pc_idx, embed_dim, magnitude) tuples
    """
    correlates = []
    for pc_idx in range(num_pcs):
        pc_vector = V[:, pc_idx]
        # Find dimension with highest absolute magnitude
        abs_magnitudes = torch.abs(pc_vector)
        max_dim = torch.argmax(abs_magnitudes).item()
        max_magnitude = pc_vector[max_dim].item()
        correlates.append((pc_idx, max_dim, max_magnitude))
        print(f"PC{pc_idx+1}: highest correlate is embedding dim {max_dim} (magnitude: {max_magnitude:.4f})")
    return correlates


def pc_score_map(tokens_np: np.ndarray, v: torch.Tensor, H: int, W: int):
    """
    tokens_np: [Np, D]
    v: [D] (one PC direction)
    Returns 2D array [H, W] of dot products z·v
    """
    Z = torch.from_numpy(tokens_np).float()   # [Np, D]
    scores = (Z @ v).cpu().numpy()           # [Np]
    return scores.reshape(H, W)


def embedding_correlate_map(tokens_np: np.ndarray, embed_dim: int, H: int, W: int):
    """
    tokens_np: [Np, D]
    embed_dim: int (which embedding dimension to visualize)
    Returns 2D array [H, W] of raw embedding values at that dimension
    """
    values = tokens_np[:, embed_dim]  # [Np]
    return values.reshape(H, W)


def upsample_to_rgb_size(patch_map: np.ndarray, target_height: int, target_width: int):
    """
    Upsample patch-level map to RGB resolution using bilinear interpolation.
    patch_map: [H, W] patch-level map
    Returns: [target_height, target_width] upsampled map
    """
    import cv2
    return cv2.resize(patch_map, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def get_tile_tokens_and_images(ds_embed_train, default_by_idx, index: int):
    """
    Returns:
      tile_id: str
      toks: list of 3 arrays [Np, D] for t0, t1, t2 (from embedding dataset)
      rgbs: list of 3 PIL Images for t0, t1, t2 (from default dataset)
      labels: ["Morning","Noon","Afternoon"]
    """
    # Get embedding data
    ex_embed = ds_embed_train[index]
    tile_id = ex_embed["idx"]

    toks = [
        np.array(ex_embed["patch_t0"], dtype=np.float32),
        np.array(ex_embed["patch_t1"], dtype=np.float32),
        np.array(ex_embed["patch_t2"], dtype=np.float32),
    ]

    # Get RGB data from default dataset using tile_id
    ex_default = default_by_idx.get(tile_id, None)
    rgb_keys = ["image_t0", "image_t1", "image_t2"]
    rgbs = []

    if ex_default is None:
        print(f"Warning: No RGB data found for tile {tile_id}")
        rgbs = [None, None, None]
    else:
        for k in rgb_keys:
            im = ex_default.get(k, None)
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
    ap.add_argument("--index", type=int, default=6, help="which ds['train'][index] to visualize (ignored if --tile_id is provided)")
    ap.add_argument("--tile_id", type=str, default=None, help="specific tile ID to visualize (e.g., '22_19')")
    ap.add_argument("--out", type=str, default="shadow_axes_vs_rgb.png")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--figsize", type=float, nargs=2, default=(14, 6))
    ap.add_argument("--clip", type=float, default=1.0, help="percentile clipping for PC maps (e.g., 1.0)")
    ap.add_argument("--num_pcs", type=int, default=10, help="number of PCs to visualize (default: 10)")
    ap.add_argument("--model_config", type=str, default="dinov3_sat",
                   choices=["dinov2_base", "dinov3_sat"],
                   help="which model config to use for embeddings")
    args = ap.parse_args()

    # 1) Load datasets - both embedding config and default config for RGB
    print(f"Loading model config: {args.model_config}")
    ds_embed = load_dataset("mpg-ranch/light-stable-semantics", args.model_config)
    ds_default = load_dataset("mpg-ranch/light-stable-semantics", "default")

    ds_embed_train = ds_embed["train"]
    ds_default_train = ds_default["train"]

    # Create mapping from idx to default dataset entry for RGB lookup
    default_by_idx = {ex["idx"]: ex for ex in ds_default_train}
    print(f"Loaded {len(ds_embed_train)} embedding samples, {len(ds_default_train)} default samples")

    # Determine which tile to visualize
    if args.tile_id is not None:
        # Find tile by ID
        selected_index = None
        for i, ex in enumerate(ds_embed_train):
            if ex["idx"] == args.tile_id:
                selected_index = i
                break

        if selected_index is None:
            # Show available tile IDs for debugging
            available_ids = [ex["idx"] for ex in ds_embed_train[:10]]  # First 10 for brevity
            print(f"Error: Tile ID '{args.tile_id}' not found in dataset.")
            print(f"Available tile IDs (first 10): {available_ids}")
            return 1

        print(f"Found tile '{args.tile_id}' at index {selected_index}")
        tile_index = selected_index
    else:
        # Use provided index
        tile_index = args.index
        if tile_index >= len(ds_embed_train):
            print(f"Error: Index {tile_index} out of range (dataset has {len(ds_embed_train)} samples)")
            return 1
        selected_tile_id = ds_embed_train[tile_index]["idx"]
        print(f"Using index {tile_index}, tile ID: '{selected_tile_id}'")

    # 2) Build temporal residual matrix over TRAIN and compute first 3 PCs (temporal shadow axes)
    X_list, ids = [], []
    for ex in ds_embed_train:
        for tkey in ("t0", "t1", "t2"):
            X_list.append(np.array(ex[f"patch_{tkey}"], dtype=np.float32))  # [Np, D]
            ids.append(ex["idx"])
    X_patch = np.stack(X_list, 0)  # [M, Np, D]
    H, W, D = infer_token_grid(X_patch)
    print(f"Inferred grid: H={H}, W={W}, D={D}, total rows={len(X_patch)}")

    D_mat = compute_temporal_residual_matrix(X_patch, ids, times_per_tile=3)
    print(f"Temporal residual matrix shape: {tuple(D_mat.shape)}")

    V = residual_pcs(D_mat, num_pcs=args.num_pcs)   # [D, num_pcs]

    # Extract all PC vectors
    pc_vectors = [V[:, i].contiguous() for i in range(args.num_pcs)]

    # Find embedding correlates
    correlates = find_embedding_correlates(V, num_pcs=args.num_pcs)
    correlate_dims = [corr[1] for corr in correlates]  # Extract embedding dimensions

    # Group PCs by their embedding dimension to collapse duplicates
    embed_to_pcs = {}
    for pc_idx, (_, embed_dim, _) in enumerate(correlates):
        if embed_dim not in embed_to_pcs:
            embed_to_pcs[embed_dim] = []
        embed_to_pcs[embed_dim].append(pc_idx + 1)  # PC numbers are 1-indexed

    # Sort embedding dimensions by minimum PC number (greedy approach)
    embed_dim_order = []
    for embed_dim, pc_list in embed_to_pcs.items():
        min_pc = min(pc_list)
        embed_dim_order.append((min_pc, embed_dim))

    # Sort by minimum PC number to get intuitive left-to-right ordering
    embed_dim_order.sort(key=lambda x: x[0])
    unique_embed_dims = [dim for _, dim in embed_dim_order]

    print(f"Unique embedding dimensions: {len(unique_embed_dims)} (from {args.num_pcs} PCs)")
    for _, embed_dim in embed_dim_order:
        pc_list = embed_to_pcs[embed_dim]
        min_pc = min(pc_list)
        print(f"  Feat. dim {embed_dim}: PC{', PC'.join(map(str, pc_list))} (min: PC{min_pc})")

    # 3) Pull tokens and RGBs for one tile (t0, t1, t2)
    tile_id, toks_list, rgb_list, time_labels = get_tile_tokens_and_images(
        ds_embed_train, default_by_idx, tile_index)
    print(f"Selected tile: {tile_id}")

    # 4) Precompute correlate maps for unique embedding dimensions
    num_unique_dims = len(unique_embed_dims)
    correlate_maps = [[] for _ in range(num_unique_dims)]  # List for each unique embedding dim
    correlate_upsampled = [[] for _ in range(num_unique_dims)]

    # Get RGB dimensions for upsampling
    rgb_height, rgb_width = None, None
    if rgb_list[0] is not None:
        rgb_array = np.array(rgb_list[0])
        rgb_height, rgb_width = rgb_array.shape[:2]
    else:
        # Fallback if no RGB available
        rgb_height, rgb_width = 1024, 1024

    for toks in toks_list:
        # Create patch-level correlate maps for each unique embedding dimension
        for dim_idx, embed_dim in enumerate(unique_embed_dims):
            corr_map = embedding_correlate_map(toks, embed_dim, H, W)
            correlate_maps[dim_idx].append(corr_map)

            # Upsample to RGB resolution
            upsampled = upsample_to_rgb_size(corr_map, rgb_height, rgb_width)
            correlate_upsampled[dim_idx].append(upsampled)

    # Per-column scaling for independent visualization of each embedding dimension
    column_mins = []
    column_maxs = []
    for dim_idx in range(num_unique_dims):
        # Get all 3 time points for this embedding dimension
        column_data = correlate_maps[dim_idx]  # List of 3 arrays
        col_min, col_max = robust_min_max(column_data, clip_percent=args.clip)
        column_mins.append(col_min)
        column_maxs.append(col_max)
        embed_dim = unique_embed_dims[dim_idx]
        print(f"Feat. dim {embed_dim} range: [{col_min:.4f}, {col_max:.4f}]")

    # 5) Plot 3×(num_unique_dims+1) grid: RGB | RGB+correlates by unique embedding dim
    nrows, ncols = 3, num_unique_dims + 1  # +1 for RGB column
    fig, axes = plt.subplots(nrows, ncols, figsize=args.figsize,
                            gridspec_kw={'wspace': 0.02, 'hspace': 0.02})

    for r in range(3):
        rgb = rgb_list[r]

        # Column 1: RGB only (reference)
        ax_rgb = axes[r, 0]
        if rgb is None:
            # fallback: blank pane with note
            ax_rgb.imshow(np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8))
            ax_rgb.text(0.5, 0.5, "RGB not in dataset", ha="center", va="center",
                       color="w", transform=ax_rgb.transAxes)
        else:
            ax_rgb.imshow(rgb)

        # Add column title for RGB
        if r == 0:
            ax_rgb.set_title("RGB", fontsize=10)
        ax_rgb.set_ylabel(time_labels[r], fontsize=10)
        ax_rgb.set_xticks([]); ax_rgb.set_yticks([])

        # Columns for each unique embedding dimension
        for dim_idx, embed_dim in enumerate(unique_embed_dims):
            col_idx = dim_idx + 1
            ax = axes[r, col_idx]
            correlate_map = correlate_upsampled[dim_idx][r]

            # Show RGB as background
            if rgb is not None:
                ax.imshow(rgb)
            else:
                ax.imshow(np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8))

            # Overlay correlate with alpha blending using per-column scaling
            col_min = column_mins[dim_idx]
            col_max = column_maxs[dim_idx]
            im = ax.imshow(correlate_map, cmap="RdBu_r", alpha=0.7,
                          vmin=col_min, vmax=col_max, interpolation="bilinear")

            # Add column title showing PCs and embedding dimension
            if r == 0:
                pc_list = embed_to_pcs[embed_dim]
                pc_title = "PC" + ", PC".join(map(str, pc_list))
                title = f"{pc_title}\nEmbed dim {embed_dim}"
                ax.set_title(title, fontsize=10)

            ax.set_xticks([]); ax.set_yticks([])

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()