#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_light_projection.py

Diagnostics for lighting subspace removal on DINOv3 embeddings.

Checks per k in --ks:
  1) Residual spectrum (printed once): EVR head + cumulative.
  2) Fraction of feature energy removed (patch tokens, optional subset for speed).
  3) Orthogonality residual: ||(Z - Z QQ^T)Q||_F / ||Z||_F  (should be ~0).
  4) Time-of-day linear probe accuracy on CLS tokens before vs after projection.

Notes
-----
- Uses HF dataset: "mpg-ranch/light-stable-semantics".
- Residual matrix is built from PATCH tokens (z_t - mean_t z per tile/patch).
- Linear probe uses CLS tokens to keep dimensionality manageable.
- Projection is learned from residual SVD and applied to BOTH patch and CLS embeddings
  (they share feature dimension D=1024).
"""

import argparse
import json
import math
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------
# Helpers
# ---------------------------

def infer_token_grid(x_np: np.ndarray):
    """x_np: [M,Np,D] or [Np,D] -> (H,W,D)."""
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
def build_residual_matrix(X_patch: np.ndarray, ids: list[str], times_per_tile=3) -> torch.Tensor:
    """
    X_patch: [M, Np, D] (built by appending t0,t1,t2 for each tile example)
    ids: tile id per row (length M)
    Returns residual matrix [T * N_tiles * Np, D] as torch.float32
    """
    groups = defaultdict(list)
    for i, tid in enumerate(ids):
        groups[tid].append(i)
    diffs = []
    for tid, idxs in groups.items():
        if len(idxs) != times_per_tile:
            continue
        Z = torch.tensor(X_patch[idxs], dtype=torch.float32)  # [T,Np,D]
        mu = Z.mean(dim=0, keepdim=True)                      # [1,Np,D]
        diffs.append(Z - mu)
    if not diffs:
        raise RuntimeError("No complete tiles with exactly 3 timepoints found.")
    D_mat = torch.cat(diffs, dim=0).reshape(-1, X_patch.shape[-1])  # [*,D]
    return D_mat


@torch.no_grad()
def residual_svd(D_mat: torch.Tensor):
    """Return (S, Vh) from SVD of residual matrix (no need to return U)."""
    _U, S, Vh = torch.linalg.svd(D_mat.cpu(), full_matrices=False)  # shapes: [n],[D,D]
    return S, Vh  # Vh rows are PCs (right singular vectors)


def make_Q(Vh: torch.Tensor, k: int) -> torch.Tensor | None:
    """Form an orthonormal basis Q [D,k] from top-k rows of Vh (or None if k==0)."""
    if k <= 0:
        return None
    Vk = Vh[:k].T.contiguous()          # [D,k]
    Q, _ = torch.linalg.qr(Vk)          # orthonormalize for stability
    return Q


def project_np(X_np: np.ndarray, Q: torch.Tensor | None) -> np.ndarray:
    """Project out span(Q): X - X QQ^T. Supports [N,D] or [*,D] shapes flattened to [N,D]."""
    if (Q is None) or (Q.numel() == 0):
        return X_np
    X = torch.from_numpy(X_np).float()          # [...,D]
    X2 = X.reshape(-1, X.shape[-1])             # [N,D]
    P = Q @ Q.T                                 # [D,D]
    Xp = X2 - X2 @ P
    return Xp.reshape(X.shape).numpy().astype(np.float32)


def frob_energy(x: torch.Tensor) -> float:
    """Frobenius norm (squared) as scalar float."""
    return float(torch.sum(x * x).item())


# ---------------------------
# Linear probe on CLS (time-of-day)
# ---------------------------

def tod_probe_accuracy(X_cls: np.ndarray, y_tod: np.ndarray, random_state=1337) -> float:
    """
    Train/test split 80/20, standardize features, multinomial logistic regression (liblinear fallback).
    Returns test accuracy.
    """
    Xtr, Xte, ytr, yte = train_test_split(X_cls, y_tod, test_size=0.2, random_state=random_state, stratify=y_tod)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    # for 3-class small dim (1024), lbfgs works; if convergence issues, bump max_iter or switch saga
    clf = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='lbfgs')
    clf.fit(Xtr_s, ytr)
    return float(clf.score(Xte_s, yte))


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=str, default="0,1,2,3,4,5,8,12,16,24,32,64,128,256,512",
                    help="comma-separated k values to test")
    ap.add_argument("--sample_frac", type=float, default=0.25,
                    help="fraction of patch rows to sample for energy checks (speed)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # 1) Load dataset
    ds = load_dataset("mpg-ranch/light-stable-semantics")
    train = ds["train"]

    # Build stacked PATCH tokens and ids for residual SVD
    Xp_list, ids = [], []
    # Also collect CLS with time labels for probe
    Xcls_list, y_tod_list = [], []

    # time label mapping
    tlabel = {"t0": 0, "t1": 1, "t2": 2}

    for ex in train:
        tid = ex["idx"]
        # PATCH tokens
        for tkey in ("t0", "t1", "t2"):
            Xp_list.append(np.array(ex[f"patch_{tkey}"], dtype=np.float32))  # [Np,D]
            ids.append(tid)
        # CLS tokens for probe
        for tkey in ("t0", "t1", "t2"):
            Xcls_list.append(np.array(ex[f"cls_{tkey}"], dtype=np.float32))  # [D]
            y_tod_list.append(tlabel[tkey])

    X_patch = np.stack(Xp_list, 0)     # [M, Np, D]
    X_cls   = np.stack(Xcls_list, 0)   # [M, D]
    y_tod   = np.array(y_tod_list, dtype=np.int64)

    H, W, D = infer_token_grid(X_patch)
    print(f"Inferred grid: H={H} W={W} D={D}; total rows (per time): {len(X_patch)}")

    # 2) Residual SVD
    D_mat = build_residual_matrix(X_patch, ids, times_per_tile=3)  # [*,D]
    print(f"Residual matrix shape: {tuple(D_mat.shape)}")
    S, Vh = residual_svd(D_mat)  # S: [min(N,D)], Vh: [D,D]

    # Spectrum head
    ev = (S**2)
    evr = ev / ev.sum()
    evr_np = evr.cpu().numpy()
    print("\nResidual spectrum (top 10 EVR):")
    print("  evr[:10]   =", np.round(evr_np[:10], 4))
    print("  cumsum[:10]=", np.round(np.cumsum(evr_np[:10]), 4))

    # 3) Prepare a subset of patch tokens for energy/orthogonality checks
    M = X_patch.shape[0]
    n_sample = int(max(1, round(args.sample_frac * M)))
    idx_sample = rng.choice(M, size=n_sample, replace=False)
    Z_sample = torch.from_numpy(X_patch[idx_sample].reshape(-1, D)).float()  # [n_sample*Np, D]
    Z_energy_total = frob_energy(Z_sample)

    # 4) Baseline time-of-day probe (no projection)
    acc_baseline = tod_probe_accuracy(X_cls, y_tod, random_state=args.seed)
    print(f"\nTime-of-day probe accuracy (CLS) BEFORE projection: {acc_baseline:.3f}")

    # 5) Sweep k
    ks = [int(s) for s in args.ks.split(",") if s.strip() != ""]
    results = {
        "grid": {"H": int(H), "W": int(W), "D": int(D)},
        "spectrum_top10_evr": np.round(evr_np[:10], 6).tolist(),
        "spectrum_top10_evr_cumsum": np.round(np.cumsum(evr_np[:10]), 6).tolist(),
        "baseline_time_of_day_acc": round(acc_baseline, 6),
        "per_k": []
    }

    for k in ks:
        Q = make_Q(Vh, k)
        # Energy removed (patch subset)
        if Q is None:
            frac_removed = 0.0
            ortho_resid = 0.0
        else:
            QQ = Q @ Q.T  # [D,D]
            proj = Z_sample @ QQ
            removed_energy = frob_energy(proj)
            frac_removed = float(removed_energy / Z_energy_total)
            Zp = Z_sample - proj
            # orthogonality residual: norm((Zp @ Q)) / norm(Z_sample)
            ortho = torch.norm(Zp @ Q)
            ortho_resid = float((ortho**2).item() / Z_energy_total)

        # Time-of-day probe AFTER projection (CLS)
        X_cls_proj = project_np(X_cls, Q) if Q is not None else X_cls
        acc_after = tod_probe_accuracy(X_cls_proj, y_tod, random_state=args.seed)

        print(f"\n[k={k:>3}]  frac_removed={frac_removed:.3f}  "
              f"orth_resid={ortho_resid:.4e}  ToD_acc={acc_after:.3f}")

        results["per_k"].append({
            "k": k,
            "fraction_energy_removed_patch_subset": round(frac_removed, 6),
            "orthogonality_residual_over_total_energy": f"{ortho_resid:.6e}",
            "time_of_day_acc_after": round(acc_after, 6),
        })

    # 6) Save JSON (optional)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved diagnostics to {args.out_json}")


if __name__ == "__main__":
    main()