#!/usr/bin/env python3
"""
Aggregate spatial 5-fold CV results into side-by-side model comparison plots.

Directory structure expected:
results/{model_config}/simple_decoder/cv_s{fold}_v{vvv}.json
where vvv is zero-padded integer percent (e.g., 000, 005, 010, 025, 050, 100)

Each JSON should include:
- val_rmse_history (list[float])
- var_pct_target (float)
- k_chosen (int)
- cum_evr_at_k (float)
"""

import argparse, json, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# --------------------------
# IO
# --------------------------
def _safe_load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def load_results_for_model(base_dir, model_config):
    """
    Returns: dict[var_pct] -> list[result_dict]
    """
    model_dir = os.path.join(base_dir, model_config, "simple_decoder")
    if not os.path.exists(model_dir):
        print(f"Directory not found: {model_dir}")
        return {}

    # New naming: cv_s{fold}_v{vvv}.json
    paths = sorted(glob.glob(os.path.join(model_dir, "cv_s*_v*.json")))
    if not paths:
        print(f"No result files found in {model_dir}")
        return {}

    print(f"Found {len(paths)} result files for {model_config}/simple_decoder")

    by_var = {}
    for p in paths:
        r = _safe_load_json(p)
        if not r: 
            continue
        if "val_rmse_history" not in r:
            print(f"Warning: {p} missing val_rmse_history, skipping")
            continue

        # Prefer explicit field; fallback to filename parse if needed
        if "var_pct_target" in r:
            vp = float(r["var_pct_target"])
        else:
            # Fallback: parse ..._v{vvv}.json
            stem = Path(p).stem
            try:
                vtag = stem.split("_v")[-1]
                vp = float(int(vtag))
            except Exception:
                print(f"Warning: could not infer var_pct from {p}; skipping")
                continue

        by_var.setdefault(vp, []).append(r)

    return by_var

# --------------------------
# CV metrics
# --------------------------
def compute_cv_metrics_with_epoch(fold_results):
    """
    Given a list of result dicts (one per fold), compute:
    mean RMSE at the epoch minimizing the mean RMSE across folds,
    plus 95% t-CI across folds at that epoch, and the best epoch number.
    """
    if not fold_results:
        return None, None, None, None

    histories = []
    for r in fold_results:
        h = r.get("val_rmse_history", [])
        if h:
            histories.append(h)

    if not histories:
        return None, None, None, None

    min_len = min(len(h) for h in histories)
    H = np.array([h[:min_len] for h in histories])  # [n_folds, n_epochs]
    n_folds = H.shape[0]

    mean_per_epoch = H.mean(axis=0)
    best_epoch = int(np.argmin(mean_per_epoch))

    fold_rmses = H[:, best_epoch]
    mean_rmse = float(fold_rmses.mean())
    std_rmse  = float(fold_rmses.std(ddof=1)) if n_folds > 1 else 0.0

    if n_folds > 1:
        t_crit = stats.t.ppf(0.975, df=n_folds - 1)
        margin = float(t_crit * std_rmse / np.sqrt(n_folds))
        ci_lower, ci_upper = mean_rmse - margin, mean_rmse + margin
    else:
        ci_lower = ci_upper = mean_rmse

    return mean_rmse, ci_lower, ci_upper, best_epoch

def compute_cv_metrics(fold_results):
    """
    Given a list of result dicts (one per fold), compute:
    mean RMSE at the epoch minimizing the mean RMSE across folds,
    plus 95% t-CI across folds at that epoch.
    """
    mean_rmse, ci_lower, ci_upper, _ = compute_cv_metrics_with_epoch(fold_results)
    return mean_rmse, ci_lower, ci_upper

# --------------------------
# Plot
# --------------------------
def create_comparison_plot(dinov2_results, dinov3_results, output_path):
    """
    Combined comparison of DINOv2 vs DINOv3 across % variance removed.
    X-axis: percent variance removed (0..100).
    """
    v2_keys = set(dinov2_results.keys()) if dinov2_results else set()
    v3_keys = set(dinov3_results.keys()) if dinov3_results else set()
    all_vps = sorted(v2_keys | v3_keys)

    if not all_vps:
        print("No variance-percent keys found; nothing to plot.")
        return

    vps = np.array(all_vps, dtype=float)

    def collect(model_dict):
        means, lows, ups = [], [], []
        for vp in all_vps:
            fr = model_dict.get(vp, [])
            m, lo, up = compute_cv_metrics(fr)
            means.append(np.nan if m is None else m)
            lows.append(np.nan if lo is None else lo)
            ups.append(np.nan if up is None else up)
        return np.array(means), np.array(lows), np.array(ups)

    v2_mean, v2_low, v2_up = collect(dinov2_results)
    v3_mean, v3_low, v3_up = collect(dinov3_results)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # DINOv2 - Python blue (C0)
    mask2 = ~np.isnan(v2_mean)
    if np.any(mask2):
        ax.plot(vps[mask2], v2_mean[mask2], color='C0', linewidth=2, marker='o',
                label='DINOv2-base')
        ax.fill_between(vps[mask2], v2_low[mask2], v2_up[mask2], color='C0', alpha=0.25)

    # DINOv3 - Python orange (C1)
    mask3 = ~np.isnan(v3_mean)
    if np.any(mask3):
        ax.plot(vps[mask3], v3_mean[mask3], color='C1', linewidth=2, marker='o',
                label='DINOv3-sat')
        ax.fill_between(vps[mask3], v3_low[mask3], v3_up[mask3], color='C1', alpha=0.25)

    ax.set_xlabel('Percent of lighting subspace variance removed')
    ax.set_ylabel('Plant canopy height RMSE (cm)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 102)

    # Set y-limits based on available data
    if np.any(mask2) or np.any(mask3):
        all_lows = []
        all_ups = []
        if np.any(mask2):
            all_lows.extend(v2_low[mask2])
            all_ups.extend(v2_up[mask2])
        if np.any(mask3):
            all_lows.extend(v3_low[mask3])
            all_ups.extend(v3_up[mask3])

        y_min = np.nanmin(all_lows)
        y_max = np.nanmax(all_ups)
        pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
        ax.set_ylim(y_min - pad, y_max + pad)

    # Legend in lower left with Model title
    legend = ax.legend(title='Model', loc='upper left', framealpha=0.9)
    legend.get_title().set_fontweight('bold')

    # Main title
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {output_path}")

# --------------------------
# Tabular outputs
# --------------------------
def create_results_dataframe(dinov2_results, dinov3_results):
    """
    Create a DataFrame with CV results for both models across all variance percentages.
    """
    v2_keys = set(dinov2_results.keys()) if dinov2_results else set()
    v3_keys = set(dinov3_results.keys()) if dinov3_results else set()
    all_vps = sorted(v2_keys | v3_keys)

    if not all_vps:
        return pd.DataFrame()

    rows = []
    for model_name, model_results in [("DINOv2-base", dinov2_results),
                                       ("DINOv3-sat", dinov3_results)]:
        if not model_results:
            continue

        for vp in all_vps:
            fold_results = model_results.get(vp, [])
            mean_rmse, ci_lower, ci_upper, best_epoch = compute_cv_metrics_with_epoch(fold_results)

            if mean_rmse is not None:
                rows.append({
                    'Model': model_name,
                    'Variance_Removed_Pct': int(vp),
                    'Mean_RMSE_cm': mean_rmse,
                    'CI_Lower_cm': ci_lower,
                    'CI_Upper_cm': ci_upper,
                    'Best_Epoch': best_epoch,
                    'N_Folds': len(fold_results)
                })

    df = pd.DataFrame(rows)
    return df

def save_csv_table(df, output_path):
    """Save results DataFrame as CSV."""
    if df.empty:
        print("Warning: Empty DataFrame, skipping CSV output")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"Saved CSV table: {output_path}")

def save_latex_table(df, output_path):
    """Save results DataFrame as LaTeX table."""
    if df.empty:
        print("Warning: Empty DataFrame, skipping LaTeX output")
        return

    # Format the DataFrame for LaTeX
    latex_str = df.to_latex(
        index=False,
        float_format='%.2f',
        caption='Cross-validation results showing mean RMSE at best epoch with 95\\% confidence intervals',
        label='tab:cv_results',
        column_format='l' + 'r' * (len(df.columns) - 1),
        escape=False
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex_str)
    print(f"Saved LaTeX table: {output_path}")

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Aggregate spatial CV results into model comparison plots")
    parser.add_argument("--results_dir", type=str, default="results/light_subspace_removal",
                        help="Base results directory")
    parser.add_argument("--output_dir", type=str, default="plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    model_configs = ["dinov2_base", "dinov3_sat"]

    # Load results
    dinov2_results = load_results_for_model(args.results_dir, "dinov2_base")
    dinov3_results = load_results_for_model(args.results_dir, "dinov3_sat")

    if not dinov2_results and not dinov3_results:
        print("No results found for either model; nothing to plot.")
        return

    # Plot
    out_path = os.path.join(args.output_dir, "cv_comparison_variance_pct.png")
    create_comparison_plot(dinov2_results, dinov3_results, out_path)

    # Generate tabular outputs
    df = create_results_dataframe(dinov2_results, dinov3_results)
    if not df.empty:
        csv_path = os.path.join(args.output_dir, "cv_results_table.csv")
        save_csv_table(df, csv_path)

        latex_path = os.path.join(args.output_dir, "cv_results_table.tex")
        save_latex_table(df, latex_path)

    # Console summary
    print("\nSummary by % variance removed:")
    for mc, res in zip(model_configs, [dinov2_results, dinov3_results]):
        if not res:
            print(f"  {mc}: No results")
            continue
        print(f"  {mc}:")
        for vp in sorted(res.keys()):
            m, lo, up = compute_cv_metrics(res[vp])
            if m is not None:
                n_folds = len(res[vp])
                ci = (up - lo) / 2.0 if up is not None and lo is not None else np.nan
                print(f"    v={int(vp):3d}%: {m:.2f} ± {ci:.2f} cm (n={n_folds})")
            else:
                print(f"    v={int(vp):3d}%: No valid results")

if __name__ == "__main__":
    main()