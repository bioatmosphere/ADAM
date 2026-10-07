#!/usr/bin/env python3
"""
Regenerate TabPFN predictions plot for presentation.
Smaller figure size with larger fonts.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set presentation-quality settings (larger fonts, smaller figure)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'lines.linewidth': 2.5,
})


def main():
    # Load the saved model
    model_path = Path("tabpfn_model.pkl")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    scaler = model_data['scaler']
    metrics = model_data['metrics']

    # Load data
    data_path = Path("../../../productivity/earth/aggregated_data_cleaned.csv")
    if not data_path.exists():
        data_path = Path("../../../productivity/earth/aggregated_data.csv")

    df = pd.read_csv(data_path)

    # Prepare features (same as in training)
    feature_cols = [
        'aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd',
        'soil_carbon_stock', 'clay_content', 'silt_content', 'sand_content',
        'nitrogen_content', 'cation_exchange_capacity', 'ph_in_water',
        'bulk_density', 'coarse_fragments', 'soil_moisture', 'elevation'
    ]

    X = df[feature_cols].copy()
    y = df['BNPP_fraction'].copy()

    # Remove rows with missing target
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    X = X.fillna(X.median())

    # Split data (same random state as training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test.values

    # Get predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    n = len(y_test)

    # Create smaller figure for presentations
    fig, ax = plt.subplots(figsize=(5.5, 5))

    ax.scatter(y_test, y_pred, alpha=0.6, s=60, color='purple',
               edgecolors='black', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], '--r', linewidth=2.5,
            label='1:1 line')

    ax.set_xlabel('Observed BNPP Fraction')
    ax.set_ylabel('Predicted BNPP Fraction')
    ax.legend(frameon=False, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add metrics text inside the plot (upper left)
    metrics_text = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nn = {n:,}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=16,
            verticalalignment='top')

    plt.tight_layout()

    # Save
    plt.savefig('tabpfn_predictions_plot_publication.png', dpi=600,
                bbox_inches='tight', facecolor='white')
    plt.savefig('tabpfn_predictions_plot_publication.pdf',
                bbox_inches='tight', facecolor='white')

    print(f"Publication plot saved:")
    print(f"  - tabpfn_predictions_plot_publication.png (600 dpi)")
    print(f"  - tabpfn_predictions_plot_publication.pdf (vector)")
    print(f"  R² = {r2:.4f}")


if __name__ == '__main__':
    main()
