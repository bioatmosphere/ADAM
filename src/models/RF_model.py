"""
BNPP Fraction Random Forest Prediction Model for the ELM-TAM benchmark pipeline.

This module provides Random Forest regression functionality for predicting the
BNPP fraction (BNPP/TNPP ratio) using environmental predictors from the TAM framework.

Key functions:
- train_random_forest: Train Random Forest model on integrated dataset
- apply_global_rf: Apply trained model for global predictions
- evaluate_model: Comprehensive model evaluation with saved visualizations

Data sources integrated:
- ForC global forest carbon database (529 forest sites)
- Global grassland productivity database (953 grassland sites)
- TerraClimate environmental variables (aet, pet, ppt, tmax, tmin, vpd)
- SoilGrids soil properties (carbon, texture, nutrients, pH, bulk density)
- Soil moisture and elevation data
- Unit conversions: Forest data converted from Mg C ha⁻¹ yr⁻¹ to g C m⁻² yr⁻¹

Model outputs:
- Trained model file (rf_bnpp_fraction_model.pkl)
- Feature importance plot (rf_bnpp_fraction_feature_importance.png)
- Predictions scatter plot (rf_bnpp_fraction_predictions_plot.png)
- Model summary text file (rf_bnpp_fraction_model_summary.txt)

Target variable: BNPP_fraction (ratio of BNPP to TNPP, 0-1 scale)
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

from sklearn.ensemble import RandomForestRegressor
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
    required_cols = ['bnpp_fraction', 'lat', 'lon']  # Minimum required columns
    df_cols_lower = [col.lower() for col in df.columns]
    missing_cols = [col for col in required_cols if col not in df_cols_lower]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"Loaded {len(df)} records with {len(df.columns)} features")
    # Find the BNPP_fraction column (case-insensitive)
    bnpp_frac_col = next((col for col in df.columns if col.lower() == 'bnpp_fraction'), None)
    if bnpp_frac_col:
        print(f"Target variable ({bnpp_frac_col}) range: {df[bnpp_frac_col].min():.4f} to {df[bnpp_frac_col].max():.4f}")
    
    return df


def prepare_features_target(df: pd.DataFrame, target_col: str = 'BNPP_fraction', use_geographic: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and target vector for machine learning.

    Args:
        df: Integrated dataset
        target_col: Name of target variable column (default: BNPP_fraction)
        use_geographic: Whether to include lat/lon coordinates as features

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Define feature columns for BNPP_fraction prediction
    feature_columns = [
        'aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd',  # Climate variables
        'soil_carbon_stock', 'clay_content', 'silt_content', 'sand_content',  # Soil properties
        'nitrogen_content', 'cation_exchange_capacity', 'ph_in_water',
        'bulk_density', 'coarse_fragments', 'soil_moisture',  # Additional soil variables
        'elevation'  # Elevation data
    ]
    
    # Add geographic coordinates if requested
    if use_geographic:
        feature_columns = ['lat', 'lon'] + feature_columns
    
    # Select available feature columns
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")
    
    # Create feature matrix
    X = df[available_features].copy()
    
    # Handle missing values
    print(f"Missing values per feature:")
    missing_counts = X.isnull().sum()
    for feature, count in missing_counts.items():
        if count > 0:
            print(f"  {feature}: {count} ({count/len(X)*100:.1f}%)")
    
    # Fill missing values with median for numeric columns
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X[col].fillna(X[col].median(), inplace=True)
    
    # Create target vector
    y = df[target_col].copy()
    
    # Remove rows with missing target values
    valid_indices = ~y.isnull()
    X = X[valid_indices]
    y = y[valid_indices]
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Target variable statistics:")
    print(f"  Mean: {y.mean():.2f}")
    print(f"  Std: {y.std():.2f}")
    print(f"  Min: {y.min():.2f}")
    print(f"  Max: {y.max():.2f}")
    
    return X, y


def train_random_forest(X: pd.DataFrame, y: pd.Series, 
                       test_size: float = 0.2, 
                       random_state: int = 42,
                       tune_hyperparameters: bool = True) -> Dict:
    """
    Train a Random Forest model with optional hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        tune_hyperparameters: Whether to perform hyperparameter tuning
        
    Returns:
        Dictionary containing trained model and evaluation results
    """
    print("Training Random Forest model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    if tune_hyperparameters:
        print("Performing hyperparameter tuning...")
        
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Create base model
        rf = RandomForestRegressor(random_state=random_state)
        
        # Perform grid search
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='neg_mean_squared_error', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Best parameters: {best_params}")
        print(f"Best CV score: {-grid_search.best_score_:.4f}")
        
    else:
        print("Using default hyperparameters...")
        # Use default parameters with some sensible adjustments
        best_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1
        )
        best_params = best_model.get_params()
        best_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Print results
    print("\n" + "="*50)
    print("RANDOM FOREST MODEL RESULTS")
    print("="*50)
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Package results
    results = {
        'model': best_model,
        'best_params': best_params,
        'feature_names': list(X.columns),
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'feature_importance': feature_importance,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test
    }
    
    return results


def save_model_and_results(results: Dict, output_dir: str = "random_forest", suffix: str = "") -> None:
    """
    Save the trained model and generate evaluation plots.
    
    Args:
        results: Dictionary containing model and evaluation results
        output_dir: Directory to save outputs
        suffix: Suffix to add to output filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / f"rf_model{suffix}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(results['model'], f)
    print(f"Model saved to: {model_path}")
    
    # Save feature importance plot
    plt.figure(figsize=(12, 8))
    top_features = results['feature_importance'].head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    importance_path = output_dir / f"rf_feature_importance{suffix}.png"
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to: {importance_path}")
    plt.close()
    
    # Save predictions plot
    plt.figure(figsize=(12, 5))
    
    # Training predictions
    plt.subplot(1, 2, 1)
    plt.scatter(results['y_train'], results['y_pred_train'], alpha=0.6)
    plt.plot([results['y_train'].min(), results['y_train'].max()],
             [results['y_train'].min(), results['y_train'].max()], 'r--', lw=2)
    plt.xlabel('Actual BNPP Fraction')
    plt.ylabel('Predicted BNPP Fraction')
    plt.title(f'Training Set (R² = {results["train_r2"]:.3f})')
    plt.grid(True, alpha=0.3)

    # Test predictions
    plt.subplot(1, 2, 2)
    plt.scatter(results['y_test'], results['y_pred_test'], alpha=0.6)
    plt.plot([results['y_test'].min(), results['y_test'].max()],
             [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
    plt.xlabel('Actual BNPP Fraction')
    plt.ylabel('Predicted BNPP Fraction')
    plt.title(f'Test Set (R² = {results["test_r2"]:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pred_path = output_dir / f"rf_predictions_plot{suffix}.png"
    plt.savefig(pred_path, dpi=300, bbox_inches='tight')
    print(f"Predictions plot saved to: {pred_path}")
    plt.close()
    
    # Save model summary
    summary_path = output_dir / f"rf_model_summary{suffix}.txt"
    with open(summary_path, 'w') as f:
        f.write("Random Forest Model Summary\n")
        f.write("="*40 + "\n\n")
        f.write(f"Dataset size: {len(results['X_train']) + len(results['X_test'])} samples\n")
        f.write(f"Features: {len(results['feature_names'])}\n")
        f.write("Geographic features (lat, lon) excluded from training\n")
        f.write(f"Training samples: {len(results['X_train'])}\n")
        f.write(f"Test samples: {len(results['X_test'])}\n\n")
        
        f.write("Model Performance:\n")
        f.write(f"Training R²: {results['train_r2']:.4f}\n")
        f.write(f"Test R²: {results['test_r2']:.4f}\n")
        f.write(f"Training RMSE: {results['train_rmse']:.4f}\n")
        f.write(f"Test RMSE: {results['test_rmse']:.4f}\n")
        f.write(f"Training MAE: {results['train_mae']:.4f}\n")
        f.write(f"Test MAE: {results['test_mae']:.4f}\n\n")
        
        f.write("Best Hyperparameters:\n")
        for param, value in results['best_params'].items():
            f.write(f"{param}: {value}\n")
        f.write("\n")
        
        f.write("Feature Importance (Top 10):\n")
        for _, row in results['feature_importance'].head(10).iterrows():
            f.write(f"{row['feature']}: {row['importance']:.4f}\n")
    
    print(f"Model summary saved to: {summary_path}")


def main():
    """Main function to train and evaluate Random Forest model for BNPP_fraction prediction."""
    print("Random Forest BNPP Fraction Prediction Model (No Geographic Features)")
    print("="*70)

    try:
        # Load data
        df = load_integrated_data()

        # Prepare features and target (without geographic coordinates)
        X, y = prepare_features_target(df, target_col='BNPP_fraction', use_geographic=False)

        # Train model
        results = train_random_forest(X, y, tune_hyperparameters=True)

        # Save results with updated naming
        save_model_and_results(results, output_dir="random_forest_bnpp_fraction", suffix="")

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print("Model files saved in: random_forest_bnpp_fraction/")
        print("- rf_model.pkl: Trained Random Forest model for BNPP_fraction")
        print("- rf_feature_importance.png: Feature importance plot")
        print("- rf_predictions_plot.png: Predictions scatter plot")
        print("- rf_model_summary.txt: Model performance summary")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()