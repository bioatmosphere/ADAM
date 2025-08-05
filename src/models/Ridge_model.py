"""
Ridge Regression BNPP Prediction Model for the ADAM benchmark pipeline.

This module provides Ridge Regression functionality for predicting 
belowground net primary productivity (BNPP) using environmental predictors from 
the ADAM framework.

Ridge Regression is a linear regression technique with L2 regularization that 
helps prevent overfitting and handles multicollinearity among features.

Key features:
- L2 regularization to prevent overfitting
- Handles multicollinearity in environmental variables
- Provides interpretable linear coefficients
- Fast training and inference
- Built-in cross-validation for alpha selection
- Feature scaling integration

Key functions:
- load_integrated_data: Load the aggregated dataset
- prepare_features_target: Prepare features and target variables with scaling
- train_ridge: Train Ridge model with hyperparameter tuning
- evaluate_model: Comprehensive model evaluation with visualizations

Data sources integrated:
- ForC global forest carbon database (529 forest sites)
- Global grassland productivity database (953 grassland sites)  
- TerraClimate environmental variables (aet, pet, ppt, tmax, tmin, vpd)
- GLASS satellite data (yearly GPP)
- SoilGrids soil properties
- Elevation data

Model outputs:
- Trained Ridge model with scaler (ridge_model.pkl)
- Feature coefficients plot (ridge_feature_coefficients.png)
- Predictions scatter plot (ridge_predictions_plot.png)
- Data distribution plot (ridge_data_distribution.png)
- Model summary text file (ridge_model_summary.txt)

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

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
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

def plot_data_distribution(df: pd.DataFrame, save_path: str = "ridge/ridge_data_distribution.png"):
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
    print(f"Ridge data distribution plot saved to: {save_path}")
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

def train_ridge(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2,
    random_state: int = 42,
    tune_hyperparameters: bool = True,
    verbose: bool = True
) -> Tuple[Ridge, StandardScaler, Dict[str, float]]:
    """
    Train Ridge Regression model with feature scaling and optional hyperparameter tuning.
    
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
    
    # Feature scaling (important for Ridge regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if tune_hyperparameters and verbose:
        print("Performing hyperparameter tuning with cross-validation...")
    
    start_time = time.time()
    
    if tune_hyperparameters:
        # Use RidgeCV for built-in cross-validation
        alphas = np.logspace(-3, 3, 50)  # Test alphas from 0.001 to 1000
        ridge_model = RidgeCV(
            alphas=alphas,
            cv=5,
            scoring='neg_mean_squared_error'
        )
        
        ridge_model.fit(X_train_scaled, y_train)
        
        if verbose:
            print(f"Best alpha: {ridge_model.alpha_:.6f}")
            print(f"Best CV score: {ridge_model.best_score_:.4f}")
        
    else:
        # Use default alpha
        ridge_model = Ridge(alpha=1.0)
        
        if verbose:
            print("Training Ridge with default alpha=1.0...")
        
        # Train the model
        ridge_model.fit(X_train_scaled, y_train)
    
    training_time = time.time() - start_time
    
    # Make predictions
    y_train_pred = ridge_model.predict(X_train_scaled)
    y_test_pred = ridge_model.predict(X_test_scaled)
    
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
        'training_time': training_time,
        'alpha': ridge_model.alpha_ if hasattr(ridge_model, 'alpha_') else ridge_model.alpha
    }
    
    if verbose:
        print(f"\nRidge training completed in {training_time:.2f} seconds")
        print("\n" + "="*50)
        print("RIDGE REGRESSION MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Regularization strength (alpha): {metrics['alpha']:.6f}")
    
    return ridge_model, scaler, metrics

def plot_feature_coefficients(
    model: Ridge, 
    feature_names: list, 
    save_path: str = "ridge/ridge_feature_coefficients.png",
    top_n: int = 15
):
    """Plot Ridge regression coefficients."""
    # Get coefficients
    coefficients = model.coef_
    
    # Create DataFrame for plotting
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    plt.figure(figsize=(10, 8))
    
    # Plot top features by absolute coefficient value
    plot_data = coef_df.head(top_n)
    colors = ['red' if coef < 0 else 'blue' for coef in plot_data['coefficient']]
    bars = plt.barh(plot_data['feature'], plot_data['coefficient'], 
                   alpha=0.7, color=colors)
    
    plt.title(f'Top {top_n} Feature Coefficients - Ridge Regression Model')
    plt.xlabel('Coefficient Value')
    plt.gca().invert_yaxis()
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Positive'),
                      Patch(facecolor='red', alpha=0.7, label='Negative')]
    plt.legend(handles=legend_elements)
    
    # Save the plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Ridge feature coefficients plot saved to: {save_path}")
    plt.close()
    
    print(f"\nTop {top_n} features by absolute coefficient:")
    for _, row in plot_data.iterrows():
        print(f"{row['feature']}: {row['coefficient']:.4f}")

def plot_predictions(
    y_true: pd.Series, 
    y_pred: np.ndarray, 
    r2_score: float,
    save_path: str = "ridge/ridge_predictions_plot.png"
):
    """Plot predicted vs actual values."""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, color='purple')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    
    # Labels and title
    plt.xlabel('Actual BNPP (g C m⁻² yr⁻¹)')
    plt.ylabel('Predicted BNPP (g C m⁻² yr⁻¹)')
    plt.title(f'Ridge Regression Model: Predicted vs Actual BNPP\nR² = {r2_score:.4f}')
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
    model: Ridge, 
    scaler: StandardScaler,
    metrics: Dict[str, float], 
    feature_names: list,
    save_path: str = "ridge/ridge_model_summary.txt"
):
    """Save model summary to text file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("Ridge Regression Model Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Model Parameters:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Alpha (regularization): {metrics['alpha']:.6f}\n")
        f.write(f"Fit intercept: True\n")
        f.write(f"Normalize: False (using StandardScaler)\n\n")
        
        f.write("Preprocessing:\n")
        f.write("-" * 20 + "\n")
        f.write("Feature scaling: StandardScaler applied\n")
        f.write(f"Scaler mean shape: {scaler.mean_.shape}\n")
        f.write(f"Scaler scale shape: {scaler.scale_.shape}\n\n")
        
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
        
        f.write("Feature Coefficients:\n")
        f.write("-" * 20 + "\n")
        coefficients = model.coef_
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients
        }).sort_values('coefficient', key=abs, ascending=False)
        
        for _, row in coef_df.iterrows():
            f.write(f"{row['feature']}: {row['coefficient']:.4f}\n")
    
    print(f"Model summary saved to: {save_path}")

def main():
    """
    Main function for standalone execution of Ridge Regression model training.
    Loads data, trains model, and evaluates performance.
    """
    try:
        print("="*60)
        print("RIDGE REGRESSION MODEL TRAINING")
        print("="*60)
        
        # Load integrated data
        df = load_integrated_data()
        
        # Plot data distribution
        plot_data_distribution(df)
        
        # Prepare features and target
        X, y = prepare_features_target(df)
        
        # Train Ridge model
        model, scaler, metrics = train_ridge(X, y, tune_hyperparameters=True)
        
        # Split data for plotting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Make predictions for plotting
        X_test_scaled = scaler.transform(X_test)
        y_test_pred = model.predict(X_test_scaled)
        
        # Create visualizations
        plot_feature_coefficients(model, list(X.columns))
        plot_predictions(y_test, y_test_pred, metrics['test_r2'])
        
        # Save model and summary
        model_path = Path("ridge/ridge_model.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save both model and scaler
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler}, f)
        print(f"Model and scaler saved to: {model_path}")
        
        save_model_summary(model, scaler, metrics, list(X.columns))
        
        print(f"\n{'='*60}")
        print("RIDGE REGRESSION TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"✓ Model trained on {len(X)} samples with {X.shape[1]} features")
        print(f"✓ Test R²: {metrics['test_r2']:.4f}")
        print(f"✓ Test RMSE: {metrics['test_rmse']:.4f} g C m⁻² yr⁻¹")
        print(f"✓ Alpha: {metrics['alpha']:.6f}")
        print(f"✓ Training time: {metrics['training_time']:.2f} seconds")
        print(f"✓ Outputs saved to: ridge/ directory")
        
    except Exception as e:
        print(f"Error in Ridge Regression model training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()