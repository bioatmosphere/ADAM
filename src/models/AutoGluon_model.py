"""
AutoGluon AutoML BNPP Prediction Model for the ADAM benchmark pipeline.

This module provides AutoGluon AutoML functionality for predicting 
belowground net primary productivity (BNPP) using environmental predictors from 
the ADAM framework.

AutoGluon is a state-of-the-art AutoML framework that automatically trains
and ensembles multiple machine learning models to achieve optimal performance
with minimal manual tuning.

Key features:
- Automatic model selection and hyperparameter tuning
- Multi-layer ensembling of diverse algorithms
- Handles missing values automatically
- Feature engineering and preprocessing
- Model interpretability and feature importance
- Time-efficient training with quality presets

Key functions:
- load_integrated_data: Load the aggregated dataset
- prepare_features_target: Prepare features and target variables
- train_autogluon: Train AutoGluon AutoML ensemble
- evaluate_model: Comprehensive model evaluation with visualizations

Data sources integrated:
- ForC global forest carbon database (529 forest sites)
- Global grassland productivity database (953 grassland sites)  
- TerraClimate environmental variables (aet, pet, ppt, tmax, tmin, vpd)
- GLASS satellite data (yearly GPP)
- SoilGrids soil properties
- Elevation data

Model outputs:
- Trained AutoGluon predictor (autogluon_model/)
- Feature importance plot (autogluon_feature_importance.png)
- Model leaderboard (autogluon_model_leaderboard.txt)
- Predictions scatter plot (autogluon_predictions_plot.png)
- Data distribution plot (autogluon_data_distribution.png)
- Model summary text file (autogluon_model_summary.txt)

Target variable: BNPP in standardized units (g C m⁻² yr⁻¹)
Total samples: 1,420 measurements from global ecosystems (after outlier removal)

Author: ADAM Development Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Tuple, Dict
import time
import shutil

from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
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

def plot_data_distribution(df: pd.DataFrame, save_path: str = "autogluon/autogluon_data_distribution.png"):
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
    print(f"AutoGluon data distribution plot saved to: {save_path}")
    plt.close()

def prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Prepare features and target for AutoGluon (which handles missing values automatically).
    
    Args:
        df: Input dataframe with all variables
        
    Returns:
        Tuple of (prepared_dataframe, target_column_name)
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
    
    # Create DataFrame with features and target
    target_col = 'BNPP'
    columns_to_keep = available_features + [target_col]
    
    prepared_df = df[columns_to_keep].copy()
    
    # Remove rows where target is missing (AutoGluon can handle missing features)
    prepared_df = prepared_df.dropna(subset=[target_col])
    
    print(f"Features prepared: {len(available_features)} variables, {len(prepared_df)} samples")
    print(f"Target variable (BNPP) range: {prepared_df[target_col].min():.2f} to {prepared_df[target_col].max():.2f}")
    
    # Check missing values in features
    missing_info = prepared_df[available_features].isnull().sum()
    if missing_info.sum() > 0:
        print("Missing values per feature (AutoGluon will handle these):")
        for feature, missing_count in missing_info[missing_info > 0].items():
            print(f"  {feature}: {missing_count} ({missing_count/len(prepared_df)*100:.1f}%)")
    
    return prepared_df, target_col

def train_autogluon(
    df: pd.DataFrame, 
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    time_limit: int = 600,  # 10 minutes
    quality: str = 'good_quality',
    verbose: bool = True
) -> Tuple[TabularPredictor, Dict[str, float]]:
    """
    Train AutoGluon AutoML model.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        time_limit: Training time limit in seconds
        quality: Quality preset ('best_quality', 'high_quality', 'good_quality', 'good_quality_faster_inference')
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (trained_predictor, performance_metrics)
    """
    # Split the data
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    if verbose:
        print(f"Training AutoGluon with quality preset: {quality}")
        print(f"Time limit: {time_limit} seconds")
    
    start_time = time.time()
    
    # Clean up any existing model directory
    model_path = "autogluon_model"
    if Path(model_path).exists():
        shutil.rmtree(model_path)
    
    # Create and train AutoGluon predictor
    predictor = TabularPredictor(
        label=target_col,
        path=model_path,
        eval_metric='root_mean_squared_error',
        problem_type='regression',
        verbosity=2 if verbose else 0
    )
    
    predictor.fit(
        train_data=train_df,
        time_limit=time_limit,
        presets=quality,
        auto_stack=True,  # Enable automatic ensembling
        num_stack_levels=1,  # One level of stacking for faster training
        hyperparameters='default',
    )
    
    training_time = time.time() - start_time
    
    # Make predictions
    train_pred = predictor.predict(train_df.drop(columns=[target_col]))
    test_pred = predictor.predict(test_df.drop(columns=[target_col]))
    
    # Calculate metrics
    train_r2 = r2_score(train_df[target_col], train_pred)
    test_r2 = r2_score(test_df[target_col], test_pred)
    train_rmse = np.sqrt(mean_squared_error(train_df[target_col], train_pred))
    test_rmse = np.sqrt(mean_squared_error(test_df[target_col], test_pred))
    train_mae = mean_absolute_error(train_df[target_col], train_pred)
    test_mae = mean_absolute_error(test_df[target_col], test_pred)
    
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
        print(f"\nAutoGluon training completed in {training_time:.2f} seconds")
        print("\n" + "="*50)
        print("AUTOGLUON MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
    
    return predictor, metrics

def plot_feature_importance(
    predictor: TabularPredictor, 
    save_path: str = "autogluon/autogluon_feature_importance.png",
    top_n: int = 15
):
    """Plot AutoGluon feature importances."""
    try:
        # Get feature importance
        importance = predictor.feature_importance(data=None, feature_stage='transformed', silent=True)
        
        # Create DataFrame for plotting
        feature_importance_df = pd.DataFrame({
            'feature': importance.index,
            'importance': importance.values
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        
        # Plot top features
        plot_data = feature_importance_df.head(top_n)
        bars = plt.barh(plot_data['feature'], plot_data['importance'], 
                       alpha=0.7, color='orange')
        
        plt.title(f'Top {top_n} Feature Importances - AutoGluon Model')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Save the plot
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"AutoGluon feature importance plot saved to: {save_path}")
        plt.close()
        
        print(f"\nTop {top_n} most important features:")
        for _, row in plot_data.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
            
    except Exception as e:
        print(f"Warning: Could not generate feature importance plot: {e}")

def plot_predictions(
    y_true: pd.Series, 
    y_pred: np.ndarray, 
    r2_score_val: float,
    save_path: str = "autogluon/autogluon_predictions_plot.png"
):
    """Plot predicted vs actual values."""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, color='orange')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    
    # Labels and title
    plt.xlabel('Actual BNPP (g C m⁻² yr⁻¹)')
    plt.ylabel('Predicted BNPP (g C m⁻² yr⁻¹)')
    plt.title(f'AutoGluon Model: Predicted vs Actual BNPP\nR² = {r2_score_val:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'R² = {r2_score_val:.4f}\nn = {len(y_true)} samples'
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

def save_model_leaderboard_and_summary(
    predictor: TabularPredictor, 
    metrics: Dict[str, float],
    save_dir: str = "autogluon"
):
    """Save model leaderboard and summary to text files."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save leaderboard
    leaderboard_path = save_dir / "autogluon_model_leaderboard.txt"
    with open(leaderboard_path, 'w') as f:
        f.write("AutoGluon Model Leaderboard\n")
        f.write("=" * 50 + "\n\n")
        
        try:
            leaderboard = predictor.leaderboard(silent=True)
            f.write(str(leaderboard))
            f.write("\n\n")
        except Exception as e:
            f.write(f"Could not generate leaderboard: {e}\n\n")
    
    # Save summary
    summary_path = save_dir / "autogluon_model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("AutoGluon AutoML Model Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Performance Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Training R²: {metrics['train_r2']:.4f}\n")
        f.write(f"Test R²: {metrics['test_r2']:.4f}\n")
        f.write(f"Training RMSE: {metrics['train_rmse']:.4f} g C m⁻² yr⁻¹\n")
        f.write(f"Test RMSE: {metrics['test_rmse']:.4f} g C m⁻² yr⁻¹\n")
        f.write(f"Training MAE: {metrics['train_mae']:.4f} g C m⁻² yr⁻¹\n")
        f.write(f"Test MAE: {metrics['test_mae']:.4f} g C m⁻² yr⁻¹\n")
        f.write(f"Training Time: {metrics['training_time']:.2f} seconds\n\n")
        
        f.write("Model Information:\n")
        f.write("-" * 20 + "\n")
        f.write("AutoML Framework: AutoGluon\n")
        f.write("Problem Type: Regression\n")
        f.write("Metric: Root Mean Squared Error\n")
        f.write("Auto-stacking: Enabled\n")
        f.write("Quality Preset: good_quality\n\n")
        
        try:
            # Get model info
            f.write("Best Model Info:\n")
            f.write("-" * 20 + "\n")
            leaderboard = predictor.leaderboard(silent=True)
            best_model = leaderboard.iloc[0]
            f.write(f"Best Model: {best_model['model']}\n")
            f.write(f"Score: {best_model['score_val']:.4f}\n")
            f.write(f"Fit Time: {best_model['fit_time']:.2f}s\n")
            f.write(f"Predict Time: {best_model['pred_time_val']:.4f}s\n")
        except Exception as e:
            f.write(f"Could not retrieve model info: {e}\n")
    
    print(f"Model leaderboard saved to: {leaderboard_path}")
    print(f"Model summary saved to: {summary_path}")

def main():
    """
    Main function for standalone execution of AutoGluon AutoML model training.
    Loads data, trains model, and evaluates performance.
    """
    try:
        print("="*60)
        print("AUTOGLUON AUTOML MODEL TRAINING")
        print("="*60)
        
        # Load integrated data
        df = load_integrated_data()
        
        # Plot data distribution
        plot_data_distribution(df)
        
        # Prepare features and target
        prepared_df, target_col = prepare_features_target(df)
        
        # Train AutoGluon model
        predictor, metrics = train_autogluon(
            prepared_df, 
            target_col, 
            time_limit=300,  # 5 minutes for faster training
            quality='good_quality'
        )
        
        # Split data for plotting
        train_df, test_df = train_test_split(prepared_df, test_size=0.2, random_state=42)
        
        # Make predictions for plotting
        test_pred = predictor.predict(test_df.drop(columns=[target_col]))
        
        # Create visualizations
        plot_feature_importance(predictor)
        plot_predictions(test_df[target_col], test_pred, metrics['test_r2'])
        
        # Save leaderboard and summary
        save_model_leaderboard_and_summary(predictor, metrics)
        
        print(f"\n{'='*60}")
        print("AUTOGLUON TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"✓ Model trained on {len(prepared_df)} samples with {len(prepared_df.columns)-1} features")
        print(f"✓ Test R²: {metrics['test_r2']:.4f}")
        print(f"✓ Test RMSE: {metrics['test_rmse']:.4f} g C m⁻² yr⁻¹")
        print(f"✓ Training time: {metrics['training_time']:.2f} seconds")
        print(f"✓ Model saved to: autogluon_model/")
        print(f"✓ Outputs saved to: autogluon/ directory")
        
    except Exception as e:
        print(f"Error in AutoGluon model training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()