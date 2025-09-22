#!/usr/bin/env python3
"""
Aggregate per-(fold,k) JSON results into CV summaries.

Updated to work with val_rmse_history format where each result contains
the full validation RMSE history across all epochs.
"""
import argparse, json, os, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_results(indir):
    """Load all result files and group by k value."""
    pattern = os.path.join(indir, "cv_fold*_k*.json")
    paths = sorted(glob.glob(pattern))

    if not paths:
        print(f"No result files found in {indir}")
        return {}

    print(f"Found {len(paths)} result files")

    # Load all results
    results = []
    for p in paths:
        try:
            with open(p, "r") as f:
                result = json.load(f)
                # Validate required fields
                if "val_rmse_history" not in result:
                    print(f"Warning: {p} missing val_rmse_history, skipping")
                    continue
                results.append(result)
        except Exception as e:
            print(f"Error loading {p}: {e}")
            continue

    # Group by k value
    by_k = {}
    for r in results:
        k = r["k"]
        if k not in by_k:
            by_k[k] = []
        by_k[k].append(r)

    return by_k

def compute_cv_metrics(fold_results):
    """
    Compute cross-validated metrics from fold results.

    Returns:
        best_epoch: 0-indexed epoch with lowest mean CV RMSE
        best_mean_rmse: mean RMSE across folds at best epoch
        best_std_rmse: std RMSE across folds at best epoch
        fold_rmses: RMSE values for each fold at best epoch
        mean_history: mean RMSE across folds for each epoch
        std_history: std RMSE across folds for each epoch
    """
    # Stack histories into matrix [n_folds, n_epochs]
    histories = []
    for result in fold_results:
        hist = result["val_rmse_history"]
        histories.append(hist)

    hist_matrix = np.array(histories)  # [n_folds, n_epochs]

    # Compute mean and std across folds for each epoch
    mean_history = hist_matrix.mean(axis=0)  # [n_epochs]
    std_history = hist_matrix.std(axis=0)    # [n_epochs]

    # Find best CV epoch (lowest mean RMSE)
    best_epoch = np.argmin(mean_history)
    best_mean_rmse = mean_history[best_epoch]
    best_std_rmse = std_history[best_epoch]
    fold_rmses = hist_matrix[:, best_epoch]  # RMSE for each fold at best epoch

    return best_epoch, best_mean_rmse, best_std_rmse, fold_rmses, mean_history, std_history

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, required=True, help="Directory containing result files")
    ap.add_argument("--plot", action="store_true", help="Generate learning curve plots")
    ap.add_argument("--outdir", type=str, default="plots", help="Output directory for plots")
    args = ap.parse_args()

    by_k = load_results(args.indir)

    if not by_k:
        return

    # Print CV summary
    print(f"\n===== CROSS-VALIDATED RESULTS ({args.indir}) =====")
    print("k    | Best Epoch | CV RMSE (cm)    | Fold RMSEs")
    print("-" * 55)

    plot_data = {}  # For optional plotting

    for k in sorted(by_k.keys()):
        fold_results = by_k[k]
        n_folds = len(fold_results)

        if n_folds < 5:
            print(f"k={k:>2}: Only {n_folds}/5 folds present")
            continue

        # Check all folds have same number of epochs
        epochs = [len(r["val_rmse_history"]) for r in fold_results]
        if len(set(epochs)) > 1:
            print(f"k={k:>2}: Inconsistent epoch counts: {epochs}")
            continue

        best_epoch, best_mean, best_std, fold_rmses, mean_hist, std_hist = compute_cv_metrics(fold_results)

        # Format fold RMSEs
        fold_str = ", ".join(f"{x:.3f}" for x in fold_rmses)

        print(f"k={k:>2}: {best_epoch+1:>3}/50     | {best_mean:.3f} ± {best_std:.3f} | [{fold_str}]")

        # Store for plotting
        if args.plot:
            plot_data[k] = {
                'mean_history': mean_hist,
                'std_history': std_hist,
                'best_epoch': best_epoch,
                'fold_results': fold_results
            }

    # Optional plotting
    if args.plot and plot_data:
        os.makedirs(args.outdir, exist_ok=True)

        # Plot 1: Learning curves for all k values
        fig, ax = plt.subplots(figsize=(12, 8))

        for k in sorted(plot_data.keys()):
            data = plot_data[k]
            epochs = np.arange(1, len(data['mean_history']) + 1)
            mean_hist = data['mean_history']
            std_hist = data['std_history']
            best_epoch = data['best_epoch']

            # Plot mean with error bars
            ax.plot(epochs, mean_hist, label=f'k={k}', marker='o', markersize=3, alpha=0.8)
            ax.fill_between(epochs, mean_hist - std_hist, mean_hist + std_hist, alpha=0.2)

            # Mark best epoch
            ax.axvline(x=best_epoch+1, color=ax.lines[-1].get_color(),
                      linestyle='--', alpha=0.5, linewidth=1)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation RMSE (cm)')
        ax.set_title('Cross-Validated Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(args.outdir, 'cv_learning_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved learning curves to {plot_path}")

        # Plot 2: Best CV RMSE vs k with 95% CI ribbon
        fig, ax = plt.subplots(figsize=(10, 6))

        ks = sorted(plot_data.keys())
        best_rmses = []
        ci_95_lower = []
        ci_95_upper = []

        for k in ks:
            data = plot_data[k]
            best_epoch = data['best_epoch']

            # Get RMSE values for all folds at best epoch
            fold_rmses = np.array([r['val_rmse_history'][best_epoch] for r in data['fold_results']])

            mean_rmse = fold_rmses.mean()
            std_rmse = fold_rmses.std(ddof=1)  # Sample standard deviation
            n_folds = len(fold_rmses)

            # Calculate 95% confidence interval using t-distribution
            t_critical = stats.t.ppf(0.975, df=n_folds-1)  # 97.5th percentile for 95% CI
            margin_error = t_critical * std_rmse / np.sqrt(n_folds)

            best_rmses.append(mean_rmse)
            ci_95_lower.append(mean_rmse - margin_error)
            ci_95_upper.append(mean_rmse + margin_error)

        best_rmses = np.array(best_rmses)
        ci_95_lower = np.array(ci_95_lower)
        ci_95_upper = np.array(ci_95_upper)

        # Plot line with 95% CI ribbon
        ax.plot(ks, best_rmses, 'o-', color='steelblue', linewidth=2, markersize=6,
                label='Best CV RMSE')
        ax.fill_between(ks, ci_95_lower, ci_95_upper, color='steelblue', alpha=0.3,
                       label='95% Confidence Interval')

        ax.set_xlabel('k (Number of PCs Removed)', fontsize=12)
        ax.set_ylabel('Best CV RMSE (cm)', fontsize=12)
        ax.set_title('Cross-Validated Performance vs Lighting Subspace Removal', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Mark best k
        best_k_idx = np.argmin(best_rmses)
        best_k = ks[best_k_idx]
        ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.8, linewidth=2,
                  label=f'Best k = {best_k}')

        # Add point annotation for best k
        ax.annotate(f'k = {best_k}\nRMSE = {best_rmses[best_k_idx]:.3f}',
                   xy=(best_k, best_rmses[best_k_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        ax.legend(frameon=True, fancybox=True, shadow=True)

        plot_path = os.path.join(args.outdir, 'cv_performance_vs_k.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved performance plot to {plot_path}")

        plt.close('all')

    # Show dataset info
    if by_k:
        sample_result = list(by_k.values())[0][0]
        if "token_grid" in sample_result and "out_size" in sample_result:
            g = sample_result["token_grid"]
            out = sample_result["out_size"]
            print(f"\nToken grid: H={g[0]} W={g[1]} D={g[2]} | Supervision size: {out}×{out}")

if __name__ == "__main__":
    main()