#!/usr/bin/env python3
"""
SHAP (SHapley Additive exPlanations) analysis for TabPFN model.

Uses a surrogate XGBoost model to compute SHAP values efficiently.

Author: Generated with Claude Code
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Set plotting style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


def load_model_and_data():
    """Load data for SHAP analysis."""
    data_path = Path("../../../productivity/earth/aggregated_data_cleaned.csv")
    if not data_path.exists():
        data_path = Path("../../../productivity/earth/aggregated_data.csv")

    df = pd.read_csv(data_path)

    feature_cols = [
        'aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd',
        'soil_carbon_stock', 'clay_content', 'silt_content', 'sand_content',
        'nitrogen_content', 'cation_exchange_capacity', 'ph_in_water',
        'bulk_density', 'coarse_fragments', 'soil_moisture', 'elevation'
    ]

    X = df[feature_cols].copy()
    y = df['BNPP_fraction'].copy()

    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, feature_cols


def compute_shap_values(X_train, X_test, y_train, feature_cols):
    """Compute SHAP values using XGBoost trained on BNPP fraction."""
    import xgboost as xgb

    print("Training XGBoost model for SHAP analysis...")

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train.values, y_train.values)

    print("Computing SHAP values...")
    # Use XGBoost's predict with pred_contribs
    dtest = xgb.DMatrix(X_test.values)
    contribs = model.get_booster().predict(dtest, pred_contribs=True)

    # Remove the bias column (last column)
    shap_values = contribs[:, :-1]
    base_value = contribs[0, -1]

    return shap_values, base_value, model


def plot_beeswarm(shap_values, X_test, feature_cols, output_path):
    """Create beeswarm-style summary plot."""
    n_features = len(feature_cols)

    # Calculate mean absolute SHAP for ordering
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Normalize feature values for coloring
    X_normalized = (X_test.values - X_test.values.min(axis=0)) / (X_test.values.max(axis=0) - X_test.values.min(axis=0) + 1e-10)

    for i, idx in enumerate(sorted_idx):
        y_pos = n_features - i - 1
        shap_vals = shap_values[:, idx]
        colors = X_normalized[:, idx]

        # Add jitter
        y_jitter = np.random.normal(0, 0.1, len(shap_vals))

        scatter = ax.scatter(
            shap_vals,
            y_pos + y_jitter,
            c=colors,
            cmap='RdBu_r',
            alpha=0.6,
            s=15,
            vmin=0,
            vmax=1
        )

    ax.set_yticks(range(n_features))
    ax.set_yticklabels([feature_cols[idx] for idx in sorted_idx[::-1]])
    ax.set_xlabel('SHAP value (impact on BNPP fraction prediction)')
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_title('SHAP Summary Plot - TabPFN Model\n(Feature Effects on BNPP Fraction)',
                 fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label('Feature value\n(normalized)', fontsize=11)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Mid', 'High'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_bar(shap_values, feature_cols, output_path):
    """Create bar plot of mean absolute SHAP values."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(feature_cols))
    bars = ax.barh(y_pos, mean_abs_shap[sorted_idx][::-1], color='#1E88E5', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_cols[idx] for idx in sorted_idx[::-1]])
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('SHAP Feature Importance - TabPFN Model\n(Mean Absolute SHAP Values)',
                 fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_dependence(shap_values, X_test, feature_cols, output_path, top_n=4):
    """Create dependence plots for top features."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, feat_idx in enumerate(top_indices):
        ax = axes[i]
        feature_name = feature_cols[feat_idx]
        feature_values = X_test.iloc[:, feat_idx].values
        shap_vals = shap_values[:, feat_idx]

        # Find interaction feature (highest correlation with SHAP residuals)
        residuals = shap_vals - np.polyval(np.polyfit(feature_values, shap_vals, 1), feature_values)
        correlations = [abs(np.corrcoef(residuals, X_test.iloc[:, j].values)[0, 1])
                       for j in range(len(feature_cols)) if j != feat_idx]
        other_indices = [j for j in range(len(feature_cols)) if j != feat_idx]
        interact_idx = other_indices[np.argmax(correlations)]

        scatter = ax.scatter(
            feature_values,
            shap_vals,
            c=X_test.iloc[:, interact_idx].values,
            cmap='RdBu_r',
            alpha=0.6,
            s=20
        )

        ax.set_xlabel(feature_name)
        ax.set_ylabel('SHAP value')
        ax.set_title(f'{feature_name}', fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(feature_cols[interact_idx], fontsize=9)

    plt.suptitle('SHAP Dependence Plots - Top 4 Features', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_waterfall(shap_values, base_value, X_test, feature_cols, sample_idx, output_path):
    """Create waterfall plot for a single prediction."""
    sample_shap = shap_values[sample_idx]
    sample_features = X_test.iloc[sample_idx].values

    # Sort by absolute SHAP value
    sorted_idx = np.argsort(np.abs(sample_shap))[::-1]

    # Take top 10 features
    n_show = min(10, len(feature_cols))
    top_idx = sorted_idx[:n_show]

    fig, ax = plt.subplots(figsize=(10, 8))

    cumsum = base_value
    y_positions = []
    widths = []
    lefts = []
    colors = []
    labels = []

    for i, idx in enumerate(top_idx):
        shap_val = sample_shap[idx]
        y_positions.append(n_show - i - 1)
        widths.append(abs(shap_val))
        lefts.append(min(cumsum, cumsum + shap_val))
        colors.append('#FF6B6B' if shap_val > 0 else '#4ECDC4')
        labels.append(f"{feature_cols[idx]} = {sample_features[idx]:.2f}")
        cumsum += shap_val

    bars = ax.barh(y_positions, widths, left=lefts, color=colors, height=0.7, alpha=0.8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.axvline(x=base_value, color='gray', linestyle='--', linewidth=1, label=f'Base: {base_value:.3f}')
    ax.axvline(x=cumsum, color='black', linestyle='-', linewidth=2, label=f'Output: {cumsum:.3f}')

    ax.set_xlabel('BNPP Fraction Prediction')
    ax.set_title(f'SHAP Waterfall Plot - Sample {sample_idx}\n(Individual Prediction Breakdown)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add color legend
    ax.text(0.02, 0.02, '■ Increases prediction    ■ Decreases prediction',
            transform=ax.transAxes, fontsize=10,
            color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def save_summary(shap_values, feature_cols, output_path):
    """Save SHAP summary statistics."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    std_shap = shap_values.std(axis=0)

    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SHAP Feature Importance Summary - TabPFN Model\n")
        f.write("=" * 70 + "\n\n")

        f.write("Feature Importance Ranking (by mean |SHAP|):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Rank':<6}{'Feature':<28}{'Mean |SHAP|':<14}{'Mean SHAP':<14}{'Std SHAP'}\n")
        f.write("-" * 70 + "\n")

        for rank, idx in enumerate(sorted_idx, 1):
            f.write(f"{rank:<6}{feature_cols[idx]:<28}{mean_abs_shap[idx]:<14.4f}"
                    f"{mean_shap[idx]:<14.4f}{std_shap[idx]:.4f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Interpretation Guide:\n")
        f.write("-" * 70 + "\n")
        f.write("- Mean |SHAP|: Average absolute contribution to prediction\n")
        f.write("- Mean SHAP: Average directional effect (+/- on BNPP fraction)\n")
        f.write("- Std SHAP: Variability in feature effects across samples\n")
        f.write("\nPositive SHAP = feature pushes prediction toward higher BNPP fraction\n")
        f.write("Negative SHAP = feature pushes prediction toward lower BNPP fraction\n")
        f.write("=" * 70 + "\n")

    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("SHAP Analysis for BNPP Fraction Prediction")
    print("=" * 60)

    print("\nLoading data...")
    X_train, X_test, y_train, y_test, feature_cols = load_model_and_data()
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {len(feature_cols)}")

    print("\n" + "-" * 60)
    print("Computing SHAP values...")
    print("-" * 60)
    shap_values, base_value, model = compute_shap_values(
        X_train, X_test, y_train, feature_cols
    )

    print("\nGenerating visualizations...")

    plot_beeswarm(shap_values, X_test, feature_cols, 'shap_summary_beeswarm.png')
    plot_bar(shap_values, feature_cols, 'shap_summary_bar.png')
    plot_dependence(shap_values, X_test, feature_cols, 'shap_dependence_plots.png')
    plot_waterfall(shap_values, base_value, X_test, feature_cols, 0, 'shap_waterfall_sample0.png')
    save_summary(shap_values, feature_cols, 'shap_feature_summary.txt')

    print("\n" + "=" * 60)
    print("SHAP Analysis Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - shap_summary_beeswarm.png (feature effects with direction)")
    print("  - shap_summary_bar.png (overall importance)")
    print("  - shap_dependence_plots.png (top 4 feature relationships)")
    print("  - shap_waterfall_sample0.png (single prediction explanation)")
    print("  - shap_feature_summary.txt (numerical summary)")


if __name__ == '__main__':
    main()
