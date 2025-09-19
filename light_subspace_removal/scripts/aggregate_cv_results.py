#!/usr/bin/env python3
"""
Aggregate per-(fold,k) JSON results into CV summaries.
"""
import argparse, json, os, glob
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="results/light_subspace_removal")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.indir, "cv_fold*_k*.json")))
    if not paths:
        print("No result files found in", args.indir)
        return

    rows = []
    for p in paths:
        with open(p, "r") as f:
            rows.append(json.load(f))

    ks = sorted({r["k"] for r in rows})
    print("===== 5-FOLD CV (best-epoch RMSE, cm) =====")
    for k in ks:
        vals = [r["best_rmse_cm"] for r in rows if r["k"] == k]
        if len(vals) == 5:
            arr = np.array(vals, dtype=np.float32)
            print(f"k={k:>2}:  {arr.mean():.3f} ± {arr.std():.3f}  folds=({', '.join(f'{v:.3f}' for v in arr)})")
        else:
            print(f"k={k:>2}:  {len(vals)} / 5 folds present")

    # Optional: show token grid & out_size (assumes consistent across runs)
    try:
        g = rows[0]["token_grid"]; out = rows[0]["out_size"]
        print(f"\nToken grid inferred: H={g[0]} W={g[1]} D={g[2]}  |  Supervision size: {out}×{out}")
    except Exception:
        pass

if __name__ == "__main__":
    main()