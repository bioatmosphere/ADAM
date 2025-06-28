"""
A Belowground Productivity (BP) Random Forest model for the ELM-TAM benchmark pipeline.

This module provides Random Forest regression functionality for predicting belowground 
net primary productivity (BNPP) using environmental predictors from the TAM framework.

Key functions:
- train_random_forest: Train RF model on integrated dataset
- apply_global_rf: Apply trained model for global predictions
- evaluate_model: Comprehensive model evaluation

Data sources integrated:
- ForC global forest carbon database
- GherardiSala grassland productivity data  
- TerraClimate environmental variables
- SoilGrids soil properties

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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_integrated_data(data_path: str = "../../productivity/earth/aggregated_data.csv") -> pd.DataFrame:
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
        raise FileNotFoundError(
            f"Integrated dataset not found at {data_path}. "
            "Please run the data aggregation pipeline first."
        )
    
    print(f"Loading integrated data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Validate required columns
    required_cols = ['bnpp', 'lat', 'lon']  # Minimum required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Loaded {len(df)} records with {len(df.columns)} features")
    print(f"Target variable (BNPP) range: {df['bnpp'].min():.2f} to {df['bnpp'].max():.2f}")
    
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
    # Remove non-predictive columns
    exclude_cols = [target_col, 'site_id', 'study_id', 'measurement_id'] 
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Handle missing values
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Remove rows with missing target values
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Fill missing features with median values
    X = X.fillna(X.median())
    
    print(f"Features prepared: {X.shape[1]} variables, {X.shape[0]} samples")
    print(f"Feature columns: {list(X.columns)}")
    
    return X, y


def train_random_forest(
    X: pd.DataFrame, 
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    tune_hyperparameters: bool = True,
    cv_folds: int = 5
) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """
    Train Random Forest model with optional hyperparameter tuning.
    
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
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Grid search with cross-validation
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=random_state),
            param_grid,
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        rf_grid.fit(X_train, y_train)
        rf_model = rf_grid.best_estimator_
        
        print(f"Best parameters: {rf_grid.best_params_}")
        print(f"Best CV score: {rf_grid.best_score_:.4f}")
        
    else:
        # Use default parameters with some optimization
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1
        )
        
        print("Training Random Forest with default parameters...")
        rf_model.fit(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(rf_model, X_train, X_test, y_train, y_test, X.columns)
    
    return rf_model, metrics


def evaluate_model(
    model: RandomForestRegressor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame, 
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list
) -> Dict[str, float]:
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Args:
        model: Trained Random Forest model
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
    print("MODEL PERFORMANCE METRICS")
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


def plot_feature_importance(model: RandomForestRegressor, feature_names: list, top_n: int = 15):
    """Plot feature importances from trained Random Forest model."""
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
    plt.title(f'Top {top_n} Feature Importances - Random Forest Model')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    print(f"\nTop {top_n} most important features:")
    for _, row in feature_importance_df.head(top_n).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")


def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, r2: float):
    """Plot actual vs predicted values with perfect prediction line."""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--r', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual BNPP')
    plt.ylabel('Predicted BNPP')
    plt.title(f'Actual vs Predicted BNPP (R² = {r2:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_model(model: RandomForestRegressor, metrics: Dict, output_path: str = "../models/rf_model.pkl"):
    """Save trained model and metrics to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'metrics': metrics,
        'feature_names': list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else None
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {output_path}")


def apply_global_rf(model: RandomForestRegressor, global_features: pd.DataFrame) -> np.ndarray:
    """
    Apply trained Random Forest model for global predictions.
    
    Args:
        model: Trained Random Forest model
        global_features: Global feature dataset for prediction
        
    Returns:
        Array of predicted BNPP values
    """
    print(f"Applying RF model to {len(global_features)} global grid points...")
    
    # Ensure features match training data
    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
    if model_features is not None:
        # Reorder columns to match training data
        global_features = global_features[model_features]
    
    # Handle missing values
    global_features_clean = global_features.fillna(global_features.median())
    
    # Make predictions
    predictions = model.predict(global_features_clean)
    
    print(f"Global predictions complete. Range: {predictions.min():.2f} to {predictions.max():.2f}")
    
    return predictions


def main():
    """
    Main function for standalone execution of RF model training.
    Loads data, trains model, and evaluates performance.
    """
    try:
        # Load integrated data
        df = load_integrated_data()
        
        # Prepare features and target
        X, y = prepare_features_target(df)
        
        # Train model
        model, metrics = train_random_forest(X, y, tune_hyperparameters=False)
        
        # Save model
        save_model(model, metrics)
        
        print("\nRandom Forest training completed successfully!")
        
    except Exception as e:
        print(f"Error in RF model training: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()