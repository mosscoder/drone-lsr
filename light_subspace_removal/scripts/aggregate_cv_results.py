#!/usr/bin/env python3
"""
Aggregate spatio-temporal CV results into side-by-side model comparison plots.

Updated to work with new directory structure:
results/{model_config}/{decoder_type}/cv_s{spatial}_t{temporal}_k{k}.json

Creates side-by-side plots comparing DINOv2 vs DINOv3 models.
"""
import argparse, json, os, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

def load_results_for_model(base_dir, model_config, decoder_type):
    """Load all result files for a specific model config and decoder type."""
    model_dir = os.path.join(base_dir, model_config, decoder_type)
    if not os.path.exists(model_dir):
        print(f"Directory not found: {model_dir}")
        return {}

    pattern = os.path.join(model_dir, "cv_s*_t*_k*.json")
    paths = sorted(glob.glob(pattern))

    if not paths:
        print(f"No result files found in {model_dir}")
        return {}

    print(f"Found {len(paths)} result files for {model_config}/{decoder_type}")

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
    Compute cross-validation metrics from spatio-temporal fold results.
    Each fold_results entry contains val_rmse_history across epochs.
    """
    if not fold_results:
        return None, None, None

    # Extract validation RMSE histories
    histories = []
    for result in fold_results:
        history = result["val_rmse_history"]
        if not history:
            continue
        histories.append(history)

    if not histories:
        return None, None, None

    # Ensure all histories have the same length (should be 50 epochs)
    min_len = min(len(h) for h in histories)
    histories = [h[:min_len] for h in histories]

    # Convert to numpy array [n_folds, n_epochs]
    hist_matrix = np.array(histories)
    n_folds = hist_matrix.shape[0]

    # Take mean across folds for each epoch
    mean_history = hist_matrix.mean(axis=0)

    # Find best epoch based on mean validation RMSE
    best_epoch = np.argmin(mean_history)

    # Extract RMSE at best epoch for each fold
    fold_rmses = hist_matrix[:, best_epoch]

    # Compute statistics
    mean_rmse = fold_rmses.mean()
    std_rmse = fold_rmses.std()

    # 95% confidence interval using t-distribution
    if n_folds > 1:
        t_critical = stats.t.ppf(0.975, df=n_folds-1)
        margin_error = t_critical * std_rmse / np.sqrt(n_folds)
        ci_lower = mean_rmse - margin_error
        ci_upper = mean_rmse + margin_error
    else:
        ci_lower = ci_upper = mean_rmse

    return mean_rmse, ci_lower, ci_upper

def create_comparison_plot(dinov2_results, dinov3_results, decoder_type, output_path):
    """Create side-by-side comparison plot for DINOv2 vs DINOv3."""

    # Get all k values that appear in both models
    dinov2_ks = set(dinov2_results.keys()) if dinov2_results else set()
    dinov3_ks = set(dinov3_results.keys()) if dinov3_results else set()
    all_ks = sorted(dinov2_ks | dinov3_ks)

    if not all_ks:
        print(f"No k values found for {decoder_type}")
        return

    # Compute metrics for each k value
    dinov2_means, dinov2_lowers, dinov2_uppers = [], [], []
    dinov3_means, dinov3_lowers, dinov3_uppers = [], [], []

    for k in all_ks:
        # DINOv2 metrics
        if k in dinov2_results:
            mean, lower, upper = compute_cv_metrics(dinov2_results[k])
            dinov2_means.append(mean if mean is not None else np.nan)
            dinov2_lowers.append(lower if lower is not None else np.nan)
            dinov2_uppers.append(upper if upper is not None else np.nan)
        else:
            dinov2_means.append(np.nan)
            dinov2_lowers.append(np.nan)
            dinov2_uppers.append(np.nan)

        # DINOv3 metrics
        if k in dinov3_results:
            mean, lower, upper = compute_cv_metrics(dinov3_results[k])
            dinov3_means.append(mean if mean is not None else np.nan)
            dinov3_lowers.append(lower if lower is not None else np.nan)
            dinov3_uppers.append(upper if upper is not None else np.nan)
        else:
            dinov3_means.append(np.nan)
            dinov3_lowers.append(np.nan)
            dinov3_uppers.append(np.nan)

    # Convert to numpy arrays
    k_values = np.array(all_ks)
    dinov2_means = np.array(dinov2_means)
    dinov2_lowers = np.array(dinov2_lowers)
    dinov2_uppers = np.array(dinov2_uppers)
    dinov3_means = np.array(dinov3_means)
    dinov3_lowers = np.array(dinov3_lowers)
    dinov3_uppers = np.array(dinov3_uppers)

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: DINOv2 Base
    valid_dinov2 = ~np.isnan(dinov2_means)
    if np.any(valid_dinov2):
        ax1.plot(k_values[valid_dinov2], dinov2_means[valid_dinov2], 'b-', linewidth=2)
        ax1.fill_between(k_values[valid_dinov2],
                        dinov2_lowers[valid_dinov2],
                        dinov2_uppers[valid_dinov2],
                        alpha=0.3, color='blue')
    ax1.set_title('DINOv2 Base (ViT-B/14)', fontsize=14)
    ax1.set_xlabel('Number of PCs Removed (k)')
    ax1.set_ylabel('RMSE (cm)')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.8, max(all_ks) * 1.2)

    # Right panel: DINOv3 SAT
    valid_dinov3 = ~np.isnan(dinov3_means)
    if np.any(valid_dinov3):
        ax2.plot(k_values[valid_dinov3], dinov3_means[valid_dinov3], 'r-', linewidth=2)
        ax2.fill_between(k_values[valid_dinov3],
                        dinov3_lowers[valid_dinov3],
                        dinov3_uppers[valid_dinov3],
                        alpha=0.3, color='red')
    ax2.set_title('DINOv3 Large + SAT (ViT-L/16)', fontsize=14)
    ax2.set_xlabel('Number of PCs Removed (k)')
    ax2.set_ylabel('RMSE (cm)')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.8, max(all_ks) * 1.2)

    # Match y-axis ranges
    if np.any(valid_dinov2) and np.any(valid_dinov3):
        y_min = min(np.min(dinov2_lowers[valid_dinov2]), np.min(dinov3_lowers[valid_dinov3]))
        y_max = max(np.max(dinov2_uppers[valid_dinov2]), np.max(dinov3_uppers[valid_dinov3]))
        y_range = y_max - y_min
        ax1.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        ax2.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    # Add super title
    decoder_name = "DPT" if decoder_type == "dpt" else "Simple Decoder"
    plt.suptitle(f'Canopy Height Prediction: Effect of Lighting Subspace Removal ({decoder_name})',
                fontsize=16, y=0.98)
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Aggregate CV results into model comparison plots")
    parser.add_argument("--results_dir", type=str, default="results/light_subspace_removal",
                       help="Base results directory")
    parser.add_argument("--output_dir", type=str, default="plots",
                       help="Output directory for plots")
    args = parser.parse_args()

    # Process both decoder types
    decoder_types = ["simple_decoder", "dpt"]
    model_configs = ["dinov2_base", "dinov3_sat"]

    for decoder_type in decoder_types:
        print(f"\n=== Processing {decoder_type} ===")

        # Load results for both models
        dinov2_results = load_results_for_model(args.results_dir, "dinov2_base", decoder_type)
        dinov3_results = load_results_for_model(args.results_dir, "dinov3_sat", decoder_type)

        if not dinov2_results and not dinov3_results:
            print(f"No results found for {decoder_type}, skipping")
            continue

        # Create comparison plot
        output_path = os.path.join(args.output_dir, f"cv_comparison_{decoder_type}.png")
        create_comparison_plot(dinov2_results, dinov3_results, decoder_type, output_path)

        # Print summary statistics
        print(f"\n{decoder_type} Summary:")
        for model_config in model_configs:
            results = dinov2_results if model_config == "dinov2_base" else dinov3_results
            if not results:
                print(f"  {model_config}: No results")
                continue

            print(f"  {model_config}:")
            for k in sorted(results.keys()):
                mean, lower, upper = compute_cv_metrics(results[k])
                if mean is not None:
                    n_folds = len(results[k])
                    print(f"    k={k:3d}: {mean:.2f} ± {(upper-lower)/2:.2f} cm (n={n_folds})")
                else:
                    print(f"    k={k:3d}: No valid results")

if __name__ == "__main__":
    main()