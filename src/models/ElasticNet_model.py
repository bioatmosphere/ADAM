"""
A Belowground Productivity (BP) Elastic Net model for the ELM-TAM benchmark pipeline.

This module provides Elastic Net regression functionality for predicting belowground 
net primary productivity (BNPP) using environmental predictors from the TAM framework.

Key functions:
- train_elasticnet: Train Elastic Net model on integrated dataset
- apply_global_elasticnet: Apply trained model for global predictions
- evaluate_model: Comprehensive model evaluation with saved visualizations

Data sources integrated:
- ForC global forest carbon database (529 forest sites)
- Global grassland productivity database (953 grassland sites)  
- TerraClimate environmental variables (aet, pet, ppt, tmax, tmin, vpd)
- GLASS satellite data (yearly GPP)
- Unit conversions: Forest data converted from Mg C ha⁻¹ yr⁻¹ to g C m⁻² yr⁻¹

Model outputs:
- Trained model file (elasticnet_model.pkl)
- Feature coefficients plot (feature_coefficients.png)
- Predictions scatter plot (predictions_plot.png)
- Model summary text file (model_summary.txt)

Target variable: BNPP in standardized units (g C m⁻² yr⁻¹)
Total samples: 1,482 measurements from global ecosystems

Author: TAM Development Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
from typing import Tuple, Dict

from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_integrated_data(data_path: str = "../../productivity/earth/aggregated_data_cleaned.csv") -> pd.DataFrame:
    """
    Load the integrated dataset from the data aggregation pipeline.
    
    Args:
        data_path: Path to the integrated dataset CSV file
        
    Returns:
        DataFrame with integrated BNPP and environmental data
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If required columns are missing
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        # Try fallback to original uncleaned data
        fallback_path = Path("../../productivity/earth/aggregated_data.csv")
        if fallback_path.exists():
            print(f"Cleaned data not found, using original data from: {fallback_path}")
            data_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Integrated dataset not found at {data_path}. "
                "Please run the data aggregation pipeline first."
            )
    
    print(f"Loading integrated data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Validate required columns (case-insensitive)
    required_cols = ['bnpp', 'lat', 'lon']  # Minimum required columns
    df_cols_lower = [col.lower() for col in df.columns]
    missing_cols = [col for col in required_cols if col not in df_cols_lower]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Loaded {len(df)} records with {len(df.columns)} features")
    # Find the BNPP column (case-insensitive)
    bnpp_col = next((col for col in df.columns if col.lower() == 'bnpp'), None)
    if bnpp_col:
        print(f"Target variable ({bnpp_col}) range: {df[bnpp_col].min():.2f} to {df[bnpp_col].max():.2f}")
    
    return df


def prepare_features_target(df: pd.DataFrame, target_col: str = 'BNPP') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and target vector for machine learning.
    
    Args:
        df: Integrated dataset
        target_col: Name of target variable column
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Remove non-predictive columns (including lat/lon to avoid spatial overfitting)
    exclude_cols = [
        target_col, 'site_id', 'study_id', 'measurement_id', 'lat', 'lon',
        'BNPP_units', 'data_source', 'biome'  # New columns from updated aggregation
    ] 
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Handle missing values
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Remove rows with missing target values
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"Initial features: {list(X.columns)}")
    
    # Remove categorical variables (keep only numeric features)
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"Excluding categorical variables: {list(categorical_cols)}")
        X = X.select_dtypes(exclude=['object'])
    
    # Fill missing features with median values
    X = X.fillna(X.median())
    
    print(f"Features prepared: {X.shape[1]} variables, {X.shape[0]} samples")
    print(f"Final feature columns: {list(X.columns)}")
    print(f"Target variable ({target_col}) range: {y.min():.2f} to {y.max():.2f}")
    
    return X, y


def plot_data_distribution(df: pd.DataFrame, save_path: str = "elasticnet/en_data_distribution.png"):
    """Plot distribution of BNPP data by ecosystem type and data source."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('BNPP Data Distribution Analysis - Elastic Net', fontsize=16, fontweight='bold')
    
    # 1. Histogram of BNPP values
    ax1 = axes[0, 0]
    ax1.hist(df['BNPP'], bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax1.set_xlabel('BNPP (g C m⁻² yr⁻¹)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of BNPP Values')
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot by data source if available
    ax2 = axes[0, 1]
    if 'data_source' in df.columns:
        data_sources = df['data_source'].dropna().unique()
        if len(data_sources) > 1:
            import seaborn as sns
            sns.boxplot(data=df, x='data_source', y='BNPP', ax=ax2)
            ax2.set_xlabel('Data Source')
            ax2.set_ylabel('BNPP (g C m⁻² yr⁻¹)')
            ax2.set_title('BNPP by Data Source')
        else:
            ax2.text(0.5, 0.5, 'Single data source', ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'No data source info', ha='center', va='center', transform=ax2.transAxes)
    
    # 3. Scatter plot: BNPP vs minimum temperature
    ax3 = axes[1, 0]
    if 'tmin' in df.columns:
        ax3.scatter(df['tmin'], df['BNPP'], alpha=0.6, color='brown')
        ax3.set_xlabel('Minimum Temperature (°C)')
        ax3.set_ylabel('BNPP (g C m⁻² yr⁻¹)')
        ax3.set_title('BNPP vs Minimum Temperature')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No minimum temperature data', ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Feature correlation heatmap
    ax4 = axes[1, 1]
    numeric_cols = ['BNPP', 'aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) >= 3:
        import seaborn as sns
        corr_matrix = df[available_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax4,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        ax4.set_title('Feature Correlation Matrix')
    else:
        ax4.text(0.5, 0.5, 'Insufficient features for correlation', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Elastic Net data distribution plot saved to: {save_path}")
    plt.close()


def train_elasticnet(
    X: pd.DataFrame, 
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    tune_hyperparameters: bool = True,
    cv_folds: int = 5
) -> Tuple[ElasticNet, Dict[str, float], StandardScaler]:
    """
    Train Elastic Net model with optional hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Target vector  
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        tune_hyperparameters: Whether to perform cross-validation for optimal parameters
        cv_folds: Number of cross-validation folds
        
    Returns:
        Tuple of (trained model, performance metrics dict, scaler)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features (important for Elastic Net)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if tune_hyperparameters:
        print("Performing hyperparameter tuning with cross-validation...")
        
        # Use ElasticNetCV for automatic hyperparameter tuning
        # Test different l1_ratio values (0=Ridge, 1=Lasso, 0.5=Elastic Net)
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        en_cv = ElasticNetCV(
            l1_ratio=l1_ratios,
            alphas=None,  # Let it choose automatically
            cv=cv_folds,
            random_state=random_state,
            max_iter=10000,
            selection='cyclic'
        )
        
        en_cv.fit(X_train_scaled, y_train)
        
        # Get best model
        en_model = ElasticNet(
            alpha=en_cv.alpha_,
            l1_ratio=en_cv.l1_ratio_,
            random_state=random_state,
            max_iter=10000
        )
        
        en_model.fit(X_train_scaled, y_train)
        
        print(f"Best alpha: {en_cv.alpha_:.6f}")
        print(f"Best l1_ratio: {en_cv.l1_ratio_:.4f}")
        
    else:
        # Use default parameters
        en_model = ElasticNet(
            alpha=1.0,
            l1_ratio=0.5,
            random_state=random_state,
            max_iter=10000
        )
        
        print("Training Elastic Net with default parameters...")
        en_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    metrics = evaluate_model(en_model, X_train_scaled, X_test_scaled, y_train, y_test, X.columns)
    
    return en_model, metrics, scaler


def evaluate_model(
    model: ElasticNet,
    X_train: np.ndarray,
    X_test: np.ndarray, 
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list
) -> Dict[str, float]:
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Args:
        model: Trained Elastic Net model
        X_train, X_test: Training and test feature sets (scaled)
        y_train, y_test: Training and test target values
        feature_names: List of feature names
        
    Returns:
        Dictionary of performance metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'n_features': len(feature_names),
        'n_train': len(y_train),
        'n_test': len(y_test),
        'alpha': model.alpha,
        'l1_ratio': model.l1_ratio,
        'n_features_selected': np.sum(model.coef_ != 0)
    }
    
    # Print results
    print(f"\n{'='*50}")
    print("ELASTIC NET MODEL PERFORMANCE METRICS")
    print(f"{'='*50}")
    print(f"Training R²: {metrics['train_r2']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    print(f"Training RMSE: {metrics['train_rmse']:.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Training MAE: {metrics['train_mae']:.4f}")
    print(f"Test MAE: {metrics['test_mae']:.4f}")
    print(f"Alpha (regularization strength): {metrics['alpha']:.6f}")
    print(f"L1 ratio: {metrics['l1_ratio']:.4f}")
    print(f"Features selected: {metrics['n_features_selected']}/{metrics['n_features']}")
    
    # Feature coefficients
    plot_feature_coefficients(model, feature_names)
    
    # Actual vs predicted plot
    plot_predictions(y_test, y_test_pred, metrics['test_r2'])
    
    return metrics


def plot_feature_coefficients(model: ElasticNet, feature_names: list, top_n: int = 15, save_path: str = "elasticnet/en_feature_coefficients.png"):
    """Plot feature coefficients from trained Elastic Net model."""
    coefficients = model.coef_
    feature_coeff_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })
    
    # Get non-zero coefficients
    non_zero_coeff = feature_coeff_df[feature_coeff_df['coefficient'] != 0]
    
    if len(non_zero_coeff) == 0:
        print("All coefficients are zero (model is too regularized)")
        return
    
    # Sort by absolute coefficient value
    non_zero_coeff['abs_coeff'] = non_zero_coeff['coefficient'].abs()
    non_zero_coeff = non_zero_coeff.sort_values('abs_coeff', ascending=False)
    
    # Plot top features
    plot_data = non_zero_coeff.head(top_n)
    
    plt.figure(figsize=(10, 8))
    colors = ['red' if x < 0 else 'blue' for x in plot_data['coefficient']]
    bars = plt.barh(plot_data['feature'], plot_data['coefficient'], color=colors, alpha=0.7)
    
    plt.title(f'Top {min(top_n, len(non_zero_coeff))} Feature Coefficients - Elastic Net Model')
    plt.xlabel('Coefficient Value')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Feature coefficients plot saved to: {save_path}")
    plt.close()
    
    print(f"\nTop {min(top_n, len(non_zero_coeff))} non-zero feature coefficients:")
    for _, row in plot_data.iterrows():
        print(f"{row['feature']}: {row['coefficient']:.4f}")
    
    print(f"\nTotal features with non-zero coefficients: {len(non_zero_coeff)}")


def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, r2: float, save_path: str = "elasticnet/en_predictions_plot.png"):
    """Plot actual vs predicted values with perfect prediction line."""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, color='coral', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--r', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual BNPP (g C m⁻² yr⁻¹)')
    plt.ylabel('Predicted BNPP (g C m⁻² yr⁻¹)')
    plt.title(f'Actual vs Predicted BNPP - Elastic Net (R² = {r2:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Predictions plot saved to: {save_path}")
    plt.close()


def save_model(model: ElasticNet, metrics: Dict, scaler: StandardScaler, output_path: str = "elasticnet/en_model.pkl"):
    """Save trained model, scaler and metrics to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'feature_names': list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else None,
        'coefficients': model.coef_ if hasattr(model, 'coef_') else None
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {output_path}")
    
    # Save model summary as text file
    summary_path = output_path.parent / "en_model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Elastic Net Model Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model Performance:\n")
        f.write(f"Training R²: {metrics['train_r2']:.4f}\n")
        f.write(f"Test R²: {metrics['test_r2']:.4f}\n")
        f.write(f"Training RMSE: {metrics['train_rmse']:.4f}\n")
        f.write(f"Test RMSE: {metrics['test_rmse']:.4f}\n")
        f.write(f"Training MAE: {metrics['train_mae']:.4f}\n")
        f.write(f"Test MAE: {metrics['test_mae']:.4f}\n\n")
        
        f.write(f"Dataset Information:\n")
        f.write(f"Number of features: {metrics['n_features']}\n")
        f.write(f"Features selected: {metrics['n_features_selected']}\n")
        f.write(f"Training samples: {metrics['n_train']}\n")
        f.write(f"Test samples: {metrics['n_test']}\n\n")
        
        f.write(f"Model Parameters:\n")
        f.write(f"Alpha (regularization): {metrics['alpha']:.6f}\n")
        f.write(f"L1 ratio: {metrics['l1_ratio']:.4f}\n\n")
        
        if model_data['feature_names'] and model_data['coefficients'] is not None:
            f.write(f"Feature Coefficients:\n")
            feature_coeff = list(zip(model_data['feature_names'], model_data['coefficients']))
            # Sort by absolute coefficient value
            feature_coeff.sort(key=lambda x: abs(x[1]), reverse=True)
            for feature, coeff in feature_coeff:
                if coeff != 0:  # Only show non-zero coefficients
                    f.write(f"{feature}: {coeff:.6f}\n")
    
    print(f"Model summary saved to: {summary_path}")


def main():
    """
    Main function for standalone execution of Elastic Net model training.
    Loads data, trains model, and evaluates performance.
    """
    try:
        print("="*60)
        print("ELASTIC NET MODEL TRAINING")
        print("="*60)
        
        # Load integrated data
        df = load_integrated_data()
        
        # Plot data distribution
        plot_data_distribution(df)
        
        # Prepare features and target
        X, y = prepare_features_target(df)
        
        # Train model
        model, metrics, scaler = train_elasticnet(X, y, tune_hyperparameters=True)
        
        # Save model
        save_model(model, metrics, scaler)
        
        print(f"\n{'='*60}")
        print("ELASTIC NET TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"✓ Model trained on {len(df)} samples with {X.shape[1]} features")
        print(f"✓ Features selected: {metrics['n_features_selected']}/{metrics['n_features']}")
        print(f"✓ Test R²: {metrics['test_r2']:.4f}")
        print(f"✓ Test RMSE: {metrics['test_rmse']:.4f} g C m⁻² yr⁻¹")
        print(f"✓ Alpha: {metrics['alpha']:.6f}, L1 ratio: {metrics['l1_ratio']:.4f}")
        print(f"✓ Outputs saved to: elasticnet/ directory")
        
    except Exception as e:
        print(f"Error in Elastic Net model training: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()