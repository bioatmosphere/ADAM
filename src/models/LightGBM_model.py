"""
A Belowground Productivity (BP) LightGBM model for the ELM-TAM benchmark pipeline.

This module provides LightGBM regression functionality for predicting belowground 
net primary productivity (BNPP) using environmental predictors from the TAM framework.

Key functions:
- train_lightgbm: Train LightGBM model on integrated dataset
- apply_global_lightgbm: Apply trained model for global predictions
- evaluate_model: Comprehensive model evaluation with saved visualizations

Data sources integrated:
- ForC global forest carbon database (529 forest sites)
- Global grassland productivity database (953 grassland sites)  
- TerraClimate environmental variables (aet, pet, ppt, tmax, tmin, vpd)
- GLASS satellite data (yearly GPP)
- Unit conversions: Forest data converted from Mg C ha⁻¹ yr⁻¹ to g C m⁻² yr⁻¹

Model outputs:
- Trained model file (lightgbm_model.pkl)
- Feature importance plot (feature_importance.png)
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

import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
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


def plot_data_distribution(df: pd.DataFrame, save_path: str = "lightgbm/lgb_data_distribution.png"):
    """Plot distribution of BNPP data by ecosystem type and data source."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('BNPP Data Distribution Analysis - LightGBM', fontsize=16, fontweight='bold')
    
    # 1. Histogram of BNPP values
    ax1 = axes[0, 0]
    ax1.hist(df['BNPP'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
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
    
    # 3. Scatter plot: BNPP vs VPD
    ax3 = axes[1, 0]
    if 'vpd' in df.columns:
        ax3.scatter(df['vpd'], df['BNPP'], alpha=0.6, color='darkgreen')
        ax3.set_xlabel('Vapor Pressure Deficit (kPa)')
        ax3.set_ylabel('BNPP (g C m⁻² yr⁻¹)')
        ax3.set_title('BNPP vs Vapor Pressure Deficit')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No VPD data', ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Scatter plot: BNPP vs PET
    ax4 = axes[1, 1]
    if 'pet' in df.columns:
        ax4.scatter(df['pet'], df['BNPP'], alpha=0.6, color='purple')
        ax4.set_xlabel('Potential Evapotranspiration (mm)')
        ax4.set_ylabel('BNPP (g C m⁻² yr⁻¹)')
        ax4.set_title('BNPP vs Potential Evapotranspiration')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No PET data', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"LightGBM data distribution plot saved to: {save_path}")
    plt.close()


def train_lightgbm(
    X: pd.DataFrame, 
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    tune_hyperparameters: bool = True,
    cv_folds: int = 5
) -> Tuple[lgb.LGBMRegressor, Dict[str, float]]:
    """
    Train LightGBM model with optional hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Target vector  
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        tune_hyperparameters: Whether to perform grid search for optimal parameters
        cv_folds: Number of cross-validation folds
        
    Returns:
        Tuple of (trained model, performance metrics dict)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    if tune_hyperparameters:
        print("Performing hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, -1],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Grid search with cross-validation
        lgb_grid = GridSearchCV(
            lgb.LGBMRegressor(
                random_state=random_state,
                verbose=-1,
                n_jobs=-1
            ),
            param_grid,
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        lgb_grid.fit(X_train, y_train)
        lgb_model = lgb_grid.best_estimator_
        
        print(f"Best parameters: {lgb_grid.best_params_}")
        print(f"Best CV score: {lgb_grid.best_score_:.4f}")
        
    else:
        # Use default parameters with some optimization
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=20,
            learning_rate=0.1,
            num_leaves=50,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1
        )
        
        print("Training LightGBM with default parameters...")
        lgb_model.fit(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(lgb_model, X_train, X_test, y_train, y_test, X.columns)
    
    return lgb_model, metrics


def evaluate_model(
    model: lgb.LGBMRegressor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame, 
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list
) -> Dict[str, float]:
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Args:
        model: Trained LightGBM model
        X_train, X_test: Training and test feature sets
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
        'n_test': len(y_test)
    }
    
    # Print results
    print(f"\n{'='*50}")
    print("LIGHTGBM MODEL PERFORMANCE METRICS")
    print(f"{'='*50}")
    print(f"Training R²: {metrics['train_r2']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    print(f"Training RMSE: {metrics['train_rmse']:.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Training MAE: {metrics['train_mae']:.4f}")
    print(f"Test MAE: {metrics['test_mae']:.4f}")
    
    # Feature importances
    plot_feature_importance(model, feature_names)
    
    # Actual vs predicted plot
    plot_predictions(y_test, y_test_pred, metrics['test_r2'])
    
    return metrics


def plot_feature_importance(model: lgb.LGBMRegressor, feature_names: list, top_n: int = 15, save_path: str = "lightgbm/lgb_feature_importance.png"):
    """Plot feature importances from trained LightGBM model."""
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=feature_importance_df.head(top_n),
        x='importance',
        y='feature',
        palette='viridis'
    )
    plt.title(f'Top {top_n} Feature Importances - LightGBM Model')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Feature importance plot saved to: {save_path}")
    plt.close()
    
    print(f"\nTop {top_n} most important features:")
    for _, row in feature_importance_df.head(top_n).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")


def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, r2: float, save_path: str = "lightgbm/lgb_predictions_plot.png"):
    """Plot actual vs predicted values with perfect prediction line."""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, color='green', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--r', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual BNPP (g C m⁻² yr⁻¹)')
    plt.ylabel('Predicted BNPP (g C m⁻² yr⁻¹)')
    plt.title(f'Actual vs Predicted BNPP - LightGBM (R² = {r2:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Predictions plot saved to: {save_path}")
    plt.close()


def save_model(model: lgb.LGBMRegressor, metrics: Dict, output_path: str = "lightgbm/lgb_model.pkl"):
    """Save trained model and metrics to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'metrics': metrics,
        'feature_names': list(model.feature_name_) if hasattr(model, 'feature_name_') else None,
        'feature_importances': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {output_path}")
    
    # Save model summary as text file
    summary_path = output_path.parent / "lgb_model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("LightGBM Model Summary\n")
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
        f.write(f"Training samples: {metrics['n_train']}\n")
        f.write(f"Test samples: {metrics['n_test']}\n\n")
        
        if model_data['feature_names'] and model_data['feature_importances'] is not None:
            f.write(f"Feature Importances:\n")
            feature_imp = list(zip(model_data['feature_names'], model_data['feature_importances']))
            feature_imp.sort(key=lambda x: x[1], reverse=True)
            for feature, importance in feature_imp:
                f.write(f"{feature}: {importance:.4f}\n")
    
    print(f"Model summary saved to: {summary_path}")


def main():
    """
    Main function for standalone execution of LightGBM model training.
    Loads data, trains model, and evaluates performance.
    """
    try:
        print("="*60)
        print("LIGHTGBM MODEL TRAINING")
        print("="*60)
        
        # Load integrated data
        df = load_integrated_data()
        
        # Plot data distribution
        plot_data_distribution(df)
        
        # Prepare features and target
        X, y = prepare_features_target(df)
        
        # Train model
        model, metrics = train_lightgbm(X, y, tune_hyperparameters=False)
        
        # Save model
        save_model(model, metrics)
        
        print(f"\n{'='*60}")
        print("LIGHTGBM TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"✓ Model trained on {len(df)} samples with {X.shape[1]} features")
        print(f"✓ Test R²: {metrics['test_r2']:.4f}")
        print(f"✓ Test RMSE: {metrics['test_rmse']:.4f} g C m⁻² yr⁻¹")
        print(f"✓ Outputs saved to: lightgbm/ directory")
        
    except Exception as e:
        print(f"Error in LightGBM model training: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()