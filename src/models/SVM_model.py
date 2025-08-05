"""
Support Vector Machine (SVM) BNPP Prediction Model for the ADAM benchmark pipeline.

This module provides Support Vector Machine functionality for predicting 
belowground net primary productivity (BNPP) using environmental predictors from 
the ADAM framework.

Support Vector Machines are powerful algorithms that work well for both linear 
and non-linear regression tasks by finding optimal hyperplanes in high-dimensional 
feature spaces.

Key features:
- Support Vector Regression (SVR) for continuous target prediction
- Multiple kernel options (RBF, polynomial, linear, sigmoid)
- Robust to outliers and high-dimensional data
- Feature scaling integration for optimal performance
- Hyperparameter tuning with cross-validation

Key functions:
- load_integrated_data: Load the aggregated dataset
- prepare_features_target: Prepare features and target variables with scaling
- train_svm: Train SVM model with hyperparameter tuning
- evaluate_model: Comprehensive model evaluation with visualizations

Data sources integrated:
- ForC global forest carbon database (529 forest sites)
- Global grassland productivity database (953 grassland sites)  
- TerraClimate environmental variables (aet, pet, ppt, tmax, tmin, vpd)
- GLASS satellite data (yearly GPP)
- SoilGrids soil properties
- Elevation data

Model outputs:
- Trained SVM model with scaler (svm_model.pkl)
- Feature importance plot (svm_feature_importance.png)
- Predictions scatter plot (svm_predictions_plot.png)
- Data distribution plot (svm_data_distribution.png)
- Model summary text file (svm_model_summary.txt)

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

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

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

def plot_data_distribution(df: pd.DataFrame, save_path: str = "svm/svm_data_distribution.png"):
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
    print(f"SVM data distribution plot saved to: {save_path}")
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
    
    # Handle missing values by forward filling and then backward filling
    X = X.fillna(method='ffill').fillna(method='bfill')
    
    # Check for any remaining missing values
    missing_info = X.isnull().sum()
    if missing_info.sum() > 0:
        print("Remaining missing values per feature:")
        for feature, missing_count in missing_info[missing_info > 0].items():
            print(f"  {feature}: {missing_count} ({missing_count/len(X)*100:.1f}%)")
        # Fill any remaining missing values with column means
        X = X.fillna(X.mean())
    
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

def train_svm(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2,
    random_state: int = 42,
    tune_hyperparameters: bool = True,
    verbose: bool = True
) -> Tuple[SVR, StandardScaler, Dict[str, float]]:
    """
    Train SVM model with feature scaling and optional hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Target variable
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        tune_hyperparameters: Whether to perform hyperparameter tuning
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (trained_model, scaler, performance_metrics)
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Feature scaling (essential for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if tune_hyperparameters and verbose:
        print("Performing hyperparameter tuning...")
    
    start_time = time.time()
    
    if tune_hyperparameters:
        # Define parameter grid for tuning (reduced for faster execution)
        param_grid = {
            'kernel': ['rbf', 'poly'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1],
            'epsilon': [0.01, 0.1, 0.2]
        }
        
        # Create base model
        base_model = SVR()
        
        # Perform grid search with reduced CV folds for speed
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=3,  # Reduced from 5 for speed
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        svm_model = grid_search.best_estimator_
        
        if verbose:
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {-grid_search.best_score_:.4f}")
        
    else:
        # Use optimized default parameters
        svm_model = SVR(
            kernel='rbf',
            C=10,
            gamma='scale',
            epsilon=0.1
        )
        
        if verbose:
            print("Training SVM with default optimized parameters...")
        
        # Train the model
        svm_model.fit(X_train_scaled, y_train)
    
    training_time = time.time() - start_time
    
    # Make predictions
    y_train_pred = svm_model.predict(X_train_scaled)
    y_test_pred = svm_model.predict(X_test_scaled)
    
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
        print(f"\nSVM training completed in {training_time:.2f} seconds")
        print("\n" + "="*50)
        print("SVM MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
    
    return svm_model, scaler, metrics

def plot_feature_importance(
    model: SVR, 
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list, 
    save_path: str = "svm/svm_feature_importance.png",
    top_n: int = 15
):
    """Plot SVM feature importances using permutation importance."""
    print("Calculating permutation importance for SVM...")
    
    # Scale the test data
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate permutation importance (reduced repeats for speed)
    perm_importance = permutation_importance(
        model, X_test_scaled, y_test, 
        n_repeats=3, random_state=42, scoring='r2'
    )
    
    # Create DataFrame for plotting
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    
    # Plot top features with error bars
    plot_data = feature_importance_df.head(top_n)
    bars = plt.barh(plot_data['feature'], plot_data['importance'], 
                   xerr=plot_data['importance_std'], alpha=0.7, color='red')
    
    plt.title(f'Top {top_n} Feature Importances - SVM Model\n(Permutation Importance)')
    plt.xlabel('Importance (R² decrease)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"SVM feature importance plot saved to: {save_path}")
    plt.close()
    
    print(f"\nTop {top_n} most important features:")
    for _, row in plot_data.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f} ± {row['importance_std']:.4f}")

def plot_predictions(
    y_true: pd.Series, 
    y_pred: np.ndarray, 
    r2_score: float,
    save_path: str = "svm/svm_predictions_plot.png"
):
    """Plot predicted vs actual values."""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, color='red')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    
    # Labels and title
    plt.xlabel('Actual BNPP (g C m⁻² yr⁻¹)')
    plt.ylabel('Predicted BNPP (g C m⁻² yr⁻¹)')
    plt.title(f'SVM Model: Predicted vs Actual BNPP\nR² = {r2_score:.4f}')
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
    model: SVR, 
    scaler: StandardScaler,
    metrics: Dict[str, float], 
    feature_names: list,
    save_path: str = "svm/svm_model_summary.txt"
):
    """Save model summary to text file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("Support Vector Machine (SVM) Model Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Model Parameters:\n")
        f.write("-" * 20 + "\n")
        params = model.get_params()
        for param, value in params.items():
            f.write(f"{param}: {value}\n")
        f.write("\n")
        
        f.write("Preprocessing:\n")
        f.write("-" * 20 + "\n")
        f.write("Feature scaling: StandardScaler applied\n")
        f.write(f"Scaler mean: {scaler.mean_}\n")
        f.write(f"Scaler scale: {scaler.scale_}\n\n")
        
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
    
    print(f"Model summary saved to: {save_path}")

def main():
    """
    Main function for standalone execution of SVM model training.
    Loads data, trains model, and evaluates performance.
    """
    try:
        print("="*60)
        print("SVM MODEL TRAINING")
        print("="*60)
        
        # Load integrated data
        df = load_integrated_data()
        
        # Plot data distribution
        plot_data_distribution(df)
        
        # Prepare features and target
        X, y = prepare_features_target(df)
        
        # Train SVM model (skip hyperparameter tuning for speed)
        model, scaler, metrics = train_svm(X, y, tune_hyperparameters=False)
        
        # Split data for plotting and feature importance
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Make predictions for plotting
        X_test_scaled = scaler.transform(X_test)
        y_test_pred = model.predict(X_test_scaled)
        
        # Create visualizations
        plot_feature_importance(model, scaler, X_test, y_test, list(X.columns))
        plot_predictions(y_test, y_test_pred, metrics['test_r2'])
        
        # Save model and summary
        model_path = Path("svm/svm_model.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save both model and scaler
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler}, f)
        print(f"Model and scaler saved to: {model_path}")
        
        save_model_summary(model, scaler, metrics, list(X.columns))
        
        print(f"\n{'='*60}")
        print("SVM TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"✓ Model trained on {len(X)} samples with {X.shape[1]} features")
        print(f"✓ Test R²: {metrics['test_r2']:.4f}")
        print(f"✓ Test RMSE: {metrics['test_rmse']:.4f} g C m⁻² yr⁻¹")
        print(f"✓ Training time: {metrics['training_time']:.2f} seconds")
        print(f"✓ Outputs saved to: svm/ directory")
        
    except Exception as e:
        print(f"Error in SVM model training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()