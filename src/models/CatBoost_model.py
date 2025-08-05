"""
CatBoost BNPP Prediction Model for the ADAM benchmark pipeline.

This module provides CatBoost (Categorical Boosting) functionality for predicting 
belowground net primary productivity (BNPP) using environmental predictors from 
the ADAM framework.

CatBoost is a gradient boosting library that excels at handling categorical features
and provides excellent performance with minimal hyperparameter tuning.
Key features:
- Built-in categorical feature handling
- Robust to overfitting
- Fast training and inference
- Excellent default parameters
- GPU support

Key functions:
- load_integrated_data: Load the aggregated dataset
- prepare_features_target: Prepare features and target variables
- train_catboost: Train CatBoost model with hyperparameter tuning
- evaluate_model: Comprehensive model evaluation with visualizations

Data sources integrated:
- ForC global forest carbon database (529 forest sites)
- Global grassland productivity database (953 grassland sites)  
- TerraClimate environmental variables (aet, pet, ppt, tmax, tmin, vpd)
- GLASS satellite data (yearly GPP)
- SoilGrids soil properties
- Elevation data

Model outputs:
- Trained CatBoost model (catboost_model.pkl)
- Feature importance plot (catboost_feature_importance.png)
- Predictions scatter plot (catboost_predictions_plot.png)
- Data distribution plot (catboost_data_distribution.png)
- Model summary text file (catboost_model_summary.txt)

Target variable: BNPP in standardized units (g C m⁻² yr⁻¹)
Total samples: 1,420 measurements from global ecosystems (after outlier removal)

Author: ADAM Development Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
from typing import Tuple, Dict
import time

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

def load_integrated_data() -> pd.DataFrame:
    """
    Load the integrated dataset from the data aggregation pipeline.
    
    Returns:
        pd.DataFrame: Integrated dataset with all features and target variable
    """
    # Try to load cleaned data first, fall back to original if not available
    cleaned_path = "../../productivity/earth/aggregated_data_cleaned.csv"
    original_path = "../../productivity/earth/aggregated_data.csv"
    
    try:
        df = pd.read_csv(cleaned_path)
        print(f"Loading integrated data from: {cleaned_path}")
    except FileNotFoundError:
        try:
            df = pd.read_csv(original_path)
            print(f"Loading integrated data from: {original_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Integrated dataset not found at {cleaned_path} or {original_path}. "
                "Please run the data aggregation pipeline first."
            )
    
    print(f"Loaded {len(df)} records with {df.shape[1]} features")
    print(f"Target variable (BNPP) range: {df['BNPP'].min():.2f} to {df['BNPP'].max():.2f}")
    
    return df

def plot_data_distribution(df: pd.DataFrame, save_path: str = "catboost/catboost_data_distribution.png"):
    """Plot the distribution of BNPP data."""
    # Create output directory
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histogram
    axes[0, 0].hist(df['BNPP'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('BNPP (g C m⁻² yr⁻¹)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of BNPP Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot
    axes[0, 1].boxplot(df['BNPP'])
    axes[0, 1].set_ylabel('BNPP (g C m⁻² yr⁻¹)')
    axes[0, 1].set_title('BNPP Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(df['BNPP'], dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Log-scale histogram
    axes[1, 1].hist(np.log1p(df['BNPP']), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('log(BNPP + 1)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of log(BNPP + 1)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"CatBoost data distribution plot saved to: {save_path}")
    plt.close()

def prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features (X) and target variable (y) for modeling.
    
    Args:
        df: Input dataframe with all variables
        
    Returns:
        Tuple of (X_features, y_target)
    """
    # Define feature columns (excluding geographic coordinates and metadata)
    feature_columns = [
        'aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd',
        'gpp_yearly', 'soil_carbon_stock', 'clay_content', 
        'silt_content', 'sand_content', 'nitrogen_content',
        'cation_exchange_capacity', 'ph_in_water', 'bulk_density',
        'coarse_fragments', 'soil_moisture', 'elevation'
    ]
    
    # Filter to only include columns that exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]
    
    print(f"Initial features: {available_features}")
    
    # Create feature matrix
    X = df[available_features].copy()
    
    # Handle missing values - CatBoost can handle them, but let's be explicit
    missing_info = X.isnull().sum()
    if missing_info.sum() > 0:
        print("Missing values per feature:")
        for feature, missing_count in missing_info[missing_info > 0].items():
            print(f"  {feature}: {missing_count} ({missing_count/len(X)*100:.1f}%)")
    
    # Target variable
    y = df['BNPP'].copy()
    
    # Remove any rows where target is missing
    valid_mask = ~y.isnull()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"Features prepared: {X.shape[1]} variables, {X.shape[0]} samples")
    print(f"Final feature columns: {list(X.columns)}")
    print(f"Target variable (BNPP) range: {y.min():.2f} to {y.max():.2f}")
    
    return X, y

def train_catboost(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2,
    random_state: int = 42,
    tune_hyperparameters: bool = True,
    verbose: bool = True
) -> Tuple[CatBoostRegressor, Dict[str, float]]:
    """
    Train CatBoost model with optional hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Target variable
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        tune_hyperparameters: Whether to perform hyperparameter tuning
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (trained_model, performance_metrics)
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    if tune_hyperparameters and verbose:
        print("Performing hyperparameter tuning...")
    
    start_time = time.time()
    
    if tune_hyperparameters:
        # Define parameter grid for tuning
        param_grid = {
            'iterations': [500, 1000, 1500],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5]
        }
        
        # Create base model
        base_model = CatBoostRegressor(
            random_state=random_state,
            verbose=False,
            early_stopping_rounds=50
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        catboost_model = grid_search.best_estimator_
        
        if verbose:
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {-grid_search.best_score_:.4f}")
        
    else:
        # Use default parameters with some optimization
        catboost_model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            random_state=random_state,
            verbose=False,
            early_stopping_rounds=50
        )
        
        if verbose:
            print("Training CatBoost with default optimized parameters...")
        
        # Train the model
        catboost_model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            verbose=False
        )
    
    training_time = time.time() - start_time
    
    # Make predictions
    y_train_pred = catboost_model.predict(X_train)
    y_test_pred = catboost_model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    metrics = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'training_time': training_time
    }
    
    if verbose:
        print(f"\nCatBoost training completed in {training_time:.2f} seconds")
        print("\n" + "="*50)
        print("CATBOOST MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
    
    return catboost_model, metrics

def plot_feature_importance(
    model: CatBoostRegressor, 
    feature_names: list, 
    save_path: str = "catboost/catboost_feature_importance.png",
    top_n: int = 15
):
    """Plot CatBoost feature importances."""
    # Get feature importances
    importances = model.get_feature_importance()
    
    # Create DataFrame for plotting
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    
    # Plot top features
    plot_data = feature_importance_df.head(top_n)
    bars = plt.barh(plot_data['feature'], plot_data['importance'], alpha=0.7, color='green')
    
    plt.title(f'Top {top_n} Feature Importances - CatBoost Model')
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"CatBoost feature importance plot saved to: {save_path}")
    plt.close()
    
    print(f"\nTop {top_n} most important features:")
    for _, row in plot_data.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")

def plot_predictions(
    y_true: pd.Series, 
    y_pred: np.ndarray, 
    r2_score: float,
    save_path: str = "catboost/catboost_predictions_plot.png"
):
    """Plot predicted vs actual values."""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, color='green')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Labels and title
    plt.xlabel('Actual BNPP (g C m⁻² yr⁻¹)')
    plt.ylabel('Predicted BNPP (g C m⁻² yr⁻¹)')
    plt.title(f'CatBoost Model: Predicted vs Actual BNPP\nR² = {r2_score:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'R² = {r2_score:.4f}\nn = {len(y_true)} samples'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top', fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Predictions plot saved to: {save_path}")
    plt.close()

def save_model_summary(
    model: CatBoostRegressor, 
    metrics: Dict[str, float], 
    feature_names: list,
    save_path: str = "catboost/catboost_model_summary.txt"
):
    """Save model summary to text file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("CatBoost Model Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Model Parameters:\n")
        f.write("-" * 20 + "\n")
        params = model.get_params()
        for param, value in params.items():
            f.write(f"{param}: {value}\n")
        f.write("\n")
        
        f.write("Performance Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Training R²: {metrics['train_r2']:.4f}\n")
        f.write(f"Test R²: {metrics['test_r2']:.4f}\n")
        f.write(f"Training RMSE: {metrics['train_rmse']:.4f} g C m⁻² yr⁻¹\n")
        f.write(f"Test RMSE: {metrics['test_rmse']:.4f} g C m⁻² yr⁻¹\n")
        f.write(f"Training MAE: {metrics['train_mae']:.4f} g C m⁻² yr⁻¹\n")
        f.write(f"Test MAE: {metrics['test_mae']:.4f} g C m⁻² yr⁻¹\n")
        f.write(f"Training Time: {metrics['training_time']:.2f} seconds\n\n")
        
        f.write("Feature Information:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Number of features: {len(feature_names)}\n")
        f.write("Features used:\n")
        for i, feature in enumerate(feature_names, 1):
            f.write(f"{i:2d}. {feature}\n")
        f.write("\n")
        
        f.write("Feature Importances:\n")
        f.write("-" * 20 + "\n")
        importances = model.get_feature_importance()
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        for _, row in feature_importance_df.iterrows():
            f.write(f"{row['feature']}: {row['importance']:.4f}\n")
    
    print(f"Model summary saved to: {save_path}")

def main():
    """
    Main function for standalone execution of CatBoost model training.
    Loads data, trains model, and evaluates performance.
    """
    try:
        print("="*60)
        print("CATBOOST MODEL TRAINING")
        print("="*60)
        
        # Load integrated data
        df = load_integrated_data()
        
        # Plot data distribution
        plot_data_distribution(df)
        
        # Prepare features and target
        X, y = prepare_features_target(df)
        
        # Train CatBoost model (skip hyperparameter tuning for speed)
        model, metrics = train_catboost(X, y, tune_hyperparameters=False)
        
        # Split data for plotting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Make predictions for plotting
        y_test_pred = model.predict(X_test)
        
        # Create visualizations
        plot_feature_importance(model, list(X.columns))
        plot_predictions(y_test, y_test_pred, metrics['test_r2'])
        
        # Save model and summary
        model_path = Path("catboost/catboost_model.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to: {model_path}")
        
        save_model_summary(model, metrics, list(X.columns))
        
        print(f"\n{'='*60}")
        print("CATBOOST TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"✓ Model trained on {len(X)} samples with {X.shape[1]} features")
        print(f"✓ Test R²: {metrics['test_r2']:.4f}")
        print(f"✓ Test RMSE: {metrics['test_rmse']:.4f} g C m⁻² yr⁻¹")
        print(f"✓ Training time: {metrics['training_time']:.2f} seconds")
        print(f"✓ Outputs saved to: catboost/ directory")
        
    except Exception as e:
        print(f"Error in CatBoost model training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()