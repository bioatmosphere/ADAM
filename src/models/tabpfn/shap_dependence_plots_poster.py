#!/usr/bin/env python3
"""
Generate poster-quality SHAP dependence plots.

Optimized for poster presentations with:
- Large, readable fonts
- Clear axis labels with units
- High resolution output
- Professional styling
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import xgboost as xgb

# Poster-quality settings - large fonts for visibility
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 24,
    'axes.labelsize': 28,
    'axes.titlesize': 32,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 2,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
})

# Feature display names with units
FEATURE_LABELS = {
    'tmin': 'Minimum Temperature (°C)',
    'tmax': 'Maximum Temperature (°C)',
    'aet': 'Actual Evapotranspiration (mm)',
    'pet': 'Potential Evapotranspiration (mm)',
    'ppt': 'Precipitation (mm)',
    'vpd': 'Vapor Pressure Deficit (kPa)',
    'elevation': 'Elevation (m)',
    'soil_moisture': 'Soil Moisture (mm)',
    'bulk_density': 'Bulk Density (g/cm³)',
    'coarse_fragments': 'Coarse Fragments (%)',
    'soil_carbon_stock': 'Soil Carbon Stock (kg/m²)',
    'clay_content': 'Clay Content (%)',
    'silt_content': 'Silt Content (%)',
    'sand_content': 'Sand Content (%)',
    'nitrogen_content': 'Nitrogen Content (g/kg)',
    'cation_exchange_capacity': 'CEC (cmol/kg)',
    'ph_in_water': 'Soil pH',
}

# Short labels for colorbars
FEATURE_SHORT = {
    'tmin': 'T_min (°C)',
    'tmax': 'T_max (°C)',
    'aet': 'AET (mm)',
    'pet': 'PET (mm)',
    'ppt': 'Precip (mm)',
    'vpd': 'VPD (kPa)',
    'elevation': 'Elev (m)',
    'soil_moisture': 'SM (mm)',
    'bulk_density': 'BD (g/cm³)',
    'coarse_fragments': 'CF (%)',
    'soil_carbon_stock': 'SOC (kg/m²)',
    'clay_content': 'Clay (%)',
    'silt_content': 'Silt (%)',
    'sand_content': 'Sand (%)',
    'nitrogen_content': 'N (g/kg)',
    'cation_exchange_capacity': 'CEC',
    'ph_in_water': 'pH',
}


def load_data():
    """Load data for SHAP analysis."""
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent

    data_path = project_root / "productivity/earth/aggregated_data_cleaned.csv"
    if not data_path.exists():
        data_path = project_root / "productivity/earth/aggregated_data.csv"

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


def compute_shap_values(X_train, X_test, y_train):
    """Compute SHAP values using XGBoost."""
    print("Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train.values, y_train.values)

    print("Computing SHAP values...")
    dtest = xgb.DMatrix(X_test.values)
    contribs = model.get_booster().predict(dtest, pred_contribs=True)

    shap_values = contribs[:, :-1]
    return shap_values, model


def plot_dependence_poster(shap_values, X_test, feature_cols, output_path, top_n=4):
    """Create poster-quality dependence plots."""

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]

    # Compact figure with large fonts for poster
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Panel labels
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    for i, feat_idx in enumerate(top_indices):
        ax = axes[i]
        feature_name = feature_cols[feat_idx]
        feature_values = X_test.iloc[:, feat_idx].values
        shap_vals = shap_values[:, feat_idx]

        # Find interaction feature
        residuals = shap_vals - np.polyval(np.polyfit(feature_values, shap_vals, 1), feature_values)
        correlations = []
        for j in range(len(feature_cols)):
            if j != feat_idx:
                corr = np.corrcoef(residuals, X_test.iloc[:, j].values)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
            else:
                correlations.append(0)

        other_indices = [j for j in range(len(feature_cols)) if j != feat_idx]
        other_corrs = [correlations[j] for j in other_indices]
        interact_idx = other_indices[np.argmax(other_corrs)]
        interact_name = feature_cols[interact_idx]

        # Scatter plot with larger markers
        scatter = ax.scatter(
            feature_values,
            shap_vals,
            c=X_test.iloc[:, interact_idx].values,
            cmap='RdBu_r',
            alpha=0.7,
            s=60,
            edgecolors='white',
            linewidths=0.3
        )

        # Axis labels with units
        ax.set_xlabel(FEATURE_LABELS.get(feature_name, feature_name), fontsize=26)
        ax.set_ylabel('SHAP Value', fontsize=26)

        # Panel label and title
        ax.set_title(f'{panel_labels[i]}', fontsize=30, fontweight='bold', loc='left')

        # Zero reference line
        ax.axhline(y=0, color='#666666', linestyle='-', linewidth=1.2, alpha=0.7)

        # Colorbar with better formatting
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(FEATURE_SHORT.get(interact_name, interact_name),
                       fontsize=20, labelpad=10)
        cbar.ax.tick_params(labelsize=18)

        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        # Grid for readability
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # No title - compact layout
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Generating Poster-Quality SHAP Dependence Plots")
    print("=" * 60)

    print("\nLoading data...")
    X_train, X_test, y_train, y_test, feature_cols = load_data()
    print(f"  Test samples: {len(X_test)}")

    shap_values, model = compute_shap_values(X_train, X_test, y_train)

    print("\nGenerating poster plot...")
    output_dir = Path(__file__).parent
    plot_dependence_poster(
        shap_values, X_test, feature_cols,
        str(output_dir / 'shap_dependence_plots_poster.png')
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
