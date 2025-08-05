"""
A Belowground Productivity (BP) TabNet model for the ELM-TAM benchmark pipeline.

This module provides TabNet (Attentive Interpretable Tabular Learning) functionality 
for predicting belowground net primary productivity (BNPP) using environmental 
predictors from the TAM framework.

TabNet is Google's transformer-based neural network specifically designed for 
tabular data, featuring:
- Sequential attention mechanism
- Feature selection and interpretability
- Efficient training on structured data

Key functions:
- train_tabnet: Train TabNet model on integrated dataset
- train_tabnet_with_hyperparameter_search: Test multiple configurations
- evaluate_model: Comprehensive model evaluation with saved visualizations

Data sources integrated:
- ForC global forest carbon database (529 forest sites)
- Global grassland productivity database (953 grassland sites)  
- TerraClimate environmental variables (aet, pet, ppt, tmax, tmin, vpd)
- GLASS satellite data (yearly GPP)
- Unit conversions: Forest data converted from Mg C ha⁻¹ yr⁻¹ to g C m⁻² yr⁻¹

Model outputs:
- Trained model file (tabnet_model.pkl)
- Feature importance plot (feature_importance.png)
- Attention masks visualization (attention_masks.png)
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
from typing import Tuple, Dict, List

import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


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


def plot_data_distribution(df: pd.DataFrame, save_path: str = "tabnet/tabnet_data_distribution.png"):
    """Plot distribution of BNPP data by ecosystem type and data source."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('BNPP Data Distribution Analysis - TabNet', fontsize=16, fontweight='bold')
    
    # 1. Histogram of BNPP values
    ax1 = axes[0, 0]
    ax1.hist(df['BNPP'], bins=30, alpha=0.7, color='darkblue', edgecolor='black')
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
    
    # 3. Scatter plot: BNPP vs GPP
    ax3 = axes[1, 0]
    if 'gpp_yearly' in df.columns:
        ax3.scatter(df['gpp_yearly'], df['BNPP'], alpha=0.6, color='darkgreen')
        ax3.set_xlabel('GPP (g C m⁻² yr⁻¹)')
        ax3.set_ylabel('BNPP (g C m⁻² yr⁻¹)')
        ax3.set_title('BNPP vs Gross Primary Productivity')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No GPP data', ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Feature correlation heatmap
    ax4 = axes[1, 1]
    numeric_cols = ['BNPP', 'aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd', 'gpp_yearly']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) >= 4:
        import seaborn as sns
        corr_matrix = df[available_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax4,
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
    print(f"TabNet data distribution plot saved to: {save_path}")
    plt.close()


def train_tabnet(
    X: pd.DataFrame, 
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    n_d: int = 64,
    n_a: int = 64,
    n_steps: int = 5,
    gamma: float = 1.5,
    lambda_sparse: float = 1e-3,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 256,
    virtual_batch_size: int = 128,
    verbose: bool = True
) -> Tuple[TabNetRegressor, Dict[str, float]]:
    """
    Train TabNet model on the dataset.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        n_d: Width of the decision prediction layer
        n_a: Width of the attention embedding for each mask
        n_steps: Number of steps in the architecture
        gamma: Coefficient for feature reusage in the masks
        lambda_sparse: Sparsity regularization coefficient
        max_epochs: Maximum number of training epochs
        patience: Early stopping patience
        batch_size: Batch size for training
        virtual_batch_size: Virtual batch size for ghost batch normalization
        verbose: Whether to print training progress
        
    Returns:
        Tuple of (trained model, performance metrics dict)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Further split training into train/validation for early stopping
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )
    
    if verbose:
        print(f"Training set: {X_train_split.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
    
    # Convert to numpy arrays
    X_train_np = X_train_split.values.astype(np.float32)
    X_val_np = X_val.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)
    y_train_np = y_train_split.values.astype(np.float32).reshape(-1, 1)
    y_val_np = y_val.values.astype(np.float32).reshape(-1, 1)
    y_test_np = y_test.values.astype(np.float32).reshape(-1, 1)
    
    # Initialize TabNet model
    tabnet_model = TabNetRegressor(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type='entmax',
        scheduler_params=dict(step_size=50, gamma=0.9),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=random_state,
        verbose=1 if verbose else 0
    )
    
    if verbose:
        print("Training TabNet model...")
    
    # Train the model
    tabnet_model.fit(
        X_train=X_train_np,
        y_train=y_train_np,
        eval_set=[(X_val_np, y_val_np)],
        eval_name=['validation'],
        eval_metric=['rmse'],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        num_workers=0,
        drop_last=False
    )
    
    # Evaluate model
    metrics = evaluate_model(
        tabnet_model, X_train_split, X_val, X_test, 
        y_train_split, y_val, y_test, X.columns, verbose=verbose
    )
    
    return tabnet_model, metrics


def train_tabnet_with_hyperparameter_search(X: pd.DataFrame, y: pd.Series) -> Tuple[TabNetRegressor, Dict[str, float]]:
    """
    Train multiple TabNet models with different hyperparameters and select the best one.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Best model and metrics
    """
    # Define hyperparameter configurations to test
    hyperparameter_configs = [
        # Default configuration
        {
            'n_d': 64, 'n_a': 64, 'n_steps': 5, 'gamma': 1.5, 
            'lambda_sparse': 1e-3, 'max_epochs': 200, 'name': 'Default'
        },
        # Deeper network
        {
            'n_d': 128, 'n_a': 128, 'n_steps': 7, 'gamma': 1.3, 
            'lambda_sparse': 1e-3, 'max_epochs': 300, 'name': 'Deep'
        },
        # Wider network
        {
            'n_d': 256, 'n_a': 256, 'n_steps': 5, 'gamma': 1.5, 
            'lambda_sparse': 1e-3, 'max_epochs': 200, 'name': 'Wide'
        },
        # More sparse
        {
            'n_d': 64, 'n_a': 64, 'n_steps': 6, 'gamma': 2.0, 
            'lambda_sparse': 1e-2, 'max_epochs': 250, 'name': 'Sparse'
        },
        # Conservative
        {
            'n_d': 32, 'n_a': 32, 'n_steps': 3, 'gamma': 1.2, 
            'lambda_sparse': 1e-4, 'max_epochs': 150, 'name': 'Conservative'
        }
    ]
    
    best_model = None
    best_metrics = None
    best_score = -float('inf')
    
    results_summary = []
    
    for i, config in enumerate(hyperparameter_configs):
        print(f"\n{'='*60}")
        print(f"Training TabNet Configuration {i+1}/5: {config['name']}")
        print(f"n_d: {config['n_d']}, n_a: {config['n_a']}, n_steps: {config['n_steps']}")
        print(f"gamma: {config['gamma']}, lambda_sparse: {config['lambda_sparse']}")
        print(f"max_epochs: {config['max_epochs']}")
        print(f"{'='*60}")
        
        # Train model with current configuration
        model, metrics = train_tabnet(
            X, y,
            n_d=config['n_d'],
            n_a=config['n_a'],
            n_steps=config['n_steps'],
            gamma=config['gamma'],
            lambda_sparse=config['lambda_sparse'],
            max_epochs=config['max_epochs'],
            verbose=False  # Reduce output during search
        )
        
        # Track results
        results_summary.append({
            'name': config['name'],
            'test_r2': metrics['test_r2'],
            'test_rmse': metrics['test_rmse'],
            'test_mae': metrics['test_mae'],
            'train_r2': metrics['train_r2']
        })
        
        print(f"Results - Test R²: {metrics['test_r2']:.4f}, Test RMSE: {metrics['test_rmse']:.2f}")
        
        # Check if this is the best model so far
        if metrics['test_r2'] > best_score:
            best_score = metrics['test_r2']
            best_model = model
            best_metrics = metrics
            print(f"✓ New best model found! Test R²: {best_score:.4f}")
    
    # Print comparison of all configurations
    print(f"\n{'='*80}")
    print("TABNET HYPERPARAMETER SEARCH RESULTS")
    print(f"{'='*80}")
    print(f"{'Configuration':<15} {'Test R²':<10} {'Test RMSE':<12} {'Test MAE':<10} {'Train R²':<10}")
    print(f"{'-'*80}")
    
    for result in results_summary:
        print(f"{result['name']:<15} {result['test_r2']:<10.4f} {result['test_rmse']:<12.2f} "
              f"{result['test_mae']:<10.2f} {result['train_r2']:<10.4f}")
    
    best_config_name = max(results_summary, key=lambda x: x['test_r2'])['name']
    print(f"\nBest configuration: {best_config_name}")
    print(f"Best Test R²: {best_score:.4f}")
    
    # Generate final plots for best model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make predictions for plotting
    y_test_pred = best_model.predict(X_test.values.astype(np.float32))
    
    plot_predictions(y_test, y_test_pred, best_metrics['test_r2'])
    plot_feature_importance(best_model, X.columns)
    
    return best_model, best_metrics


def evaluate_model(
    model: TabNetRegressor,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame, 
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    feature_names: List[str],
    verbose: bool = True
) -> Dict[str, float]:
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Args:
        model: Trained TabNet model
        X_train, X_val, X_test: Training, validation, and test feature sets
        y_train, y_val, y_test: Training, validation, and test target values
        feature_names: List of feature names
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary of performance metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train.values.astype(np.float32))
    y_val_pred = model.predict(X_val.values.astype(np.float32))
    y_test_pred = model.predict(X_test.values.astype(np.float32))
    
    # Calculate metrics
    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'val_r2': r2_score(y_val, y_val_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'val_mae': mean_absolute_error(y_val, y_val_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'n_features': len(feature_names),
        'n_train': len(y_train),
        'n_val': len(y_val),
        'n_test': len(y_test)
    }
    
    # Print results
    if verbose:
        print(f"\n{'='*50}")
        print("TABNET MODEL PERFORMANCE METRICS")
        print(f"{'='*50}")
        print(f"Training R²: {metrics['train_r2']:.4f}")
        print(f"Validation R²: {metrics['val_r2']:.4f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        print(f"Training RMSE: {metrics['train_rmse']:.4f}")
        print(f"Validation RMSE: {metrics['val_rmse']:.4f}")
        print(f"Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"Training MAE: {metrics['train_mae']:.4f}")
        print(f"Validation MAE: {metrics['val_mae']:.4f}")
        print(f"Test MAE: {metrics['test_mae']:.4f}")
    
    return metrics


def plot_feature_importance(model: TabNetRegressor, feature_names: List[str], top_n: int = 15, save_path: str = "tabnet/tabnet_feature_importance.png"):
    """Plot feature importances from trained TabNet model."""
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
    plt.title(f'Top {top_n} Feature Importances - TabNet Model')
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


def plot_attention_masks(model: TabNetRegressor, X_sample: pd.DataFrame, save_path: str = "tabnet/tabnet_attention_masks.png"):
    """Plot attention masks from TabNet model to show feature selection at each step."""
    # Get attention masks for a sample
    explain_matrix, masks = model.explain(X_sample.values.astype(np.float32))
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('TabNet Attention Masks by Step', fontsize=16, fontweight='bold')
    
    # Plot attention masks for each step
    for i, mask in enumerate(masks):
        if i >= 6:  # Only show first 6 steps
            break
        
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Average attention across samples
        avg_mask = np.mean(mask, axis=0)
        
        # Create barplot
        feature_names = X_sample.columns
        mask_df = pd.DataFrame({
            'feature': feature_names,
            'attention': avg_mask
        }).sort_values('attention', ascending=False)
        
        sns.barplot(data=mask_df.head(10), x='attention', y='feature', ax=ax, palette='plasma')
        ax.set_title(f'Step {i+1}')
        ax.set_xlabel('Attention Weight')
        
        if col > 0:
            ax.set_ylabel('')
    
    # Remove empty subplots
    for i in range(len(masks), 6):
        row = i // 3
        col = i % 3
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Attention masks plot saved to: {save_path}")
    plt.close()


def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, r2: float, save_path: str = "tabnet/tabnet_predictions_plot.png"):
    """Plot actual vs predicted values with perfect prediction line."""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, color='darkblue', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--r', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual BNPP (g C m⁻² yr⁻¹)')
    plt.ylabel('Predicted BNPP (g C m⁻² yr⁻¹)')
    plt.title(f'Actual vs Predicted BNPP - TabNet (R² = {r2:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Predictions plot saved to: {save_path}")
    plt.close()


def save_model(model: TabNetRegressor, metrics: Dict, output_path: str = "tabnet/tabnet_model.pkl"):
    """Save trained model and metrics to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save TabNet model
    model.save_model(str(output_path.parent / "tabnet_model"))
    
    # Save metrics and additional info
    model_data = {
        'metrics': metrics,
        'feature_names': list(model.feature_importances_),
        'feature_importances': model.feature_importances_
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {output_path}")
    
    # Save model summary as text file
    summary_path = output_path.parent / "tabnet_model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("TabNet Model Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model Performance:\n")
        f.write(f"Training R²: {metrics['train_r2']:.4f}\n")
        f.write(f"Validation R²: {metrics['val_r2']:.4f}\n")
        f.write(f"Test R²: {metrics['test_r2']:.4f}\n")
        f.write(f"Training RMSE: {metrics['train_rmse']:.4f}\n")
        f.write(f"Validation RMSE: {metrics['val_rmse']:.4f}\n")
        f.write(f"Test RMSE: {metrics['test_rmse']:.4f}\n")
        f.write(f"Training MAE: {metrics['train_mae']:.4f}\n")
        f.write(f"Validation MAE: {metrics['val_mae']:.4f}\n")
        f.write(f"Test MAE: {metrics['test_mae']:.4f}\n\n")
        
        f.write(f"Dataset Information:\n")
        f.write(f"Number of features: {metrics['n_features']}\n")
        f.write(f"Training samples: {metrics['n_train']}\n")
        f.write(f"Validation samples: {metrics['n_val']}\n")
        f.write(f"Test samples: {metrics['n_test']}\n\n")
        
        f.write(f"Feature Importances:\n")
        if 'feature_importances' in model_data:
            feature_imp = list(zip(model_data['feature_names'], model_data['feature_importances']))
            feature_imp.sort(key=lambda x: x[1], reverse=True)
            for feature, importance in feature_imp:
                f.write(f"{feature}: {importance:.4f}\n")
    
    print(f"Model summary saved to: {summary_path}")


def main():
    """
    Main function for standalone execution of TabNet model training.
    Loads data, trains model, and evaluates performance.
    """
    try:
        print("="*60)
        print("TABNET MODEL TRAINING")
        print("="*60)
        
        # Load integrated data
        df = load_integrated_data()
        
        # Plot data distribution
        plot_data_distribution(df)
        
        # Prepare features and target
        X, y = prepare_features_target(df)
        
        # Train multiple models with hyperparameter search
        print("Starting TabNet hyperparameter search...")
        model, metrics = train_tabnet_with_hyperparameter_search(X, y)
        
        # Plot attention masks for a sample (skip if there are issues)
        try:
            X_sample = X.sample(n=min(100, len(X)), random_state=42)
            plot_attention_masks(model, X_sample)
        except Exception as e:
            print(f"Warning: Could not plot attention masks: {e}")
        
        # Save the best model
        save_model(model, metrics)
        
        print(f"TabNet hyperparameter search completed successfully!")
        print(f"Best model saved with Test R²: {metrics['test_r2']:.4f}")
        
        print(f"\n{'='*60}")
        print("TABNET TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"✓ Model trained on {len(df)} samples with {X.shape[1]} features")
        print(f"✓ Test R²: {metrics['test_r2']:.4f}")
        print(f"✓ Test RMSE: {metrics['test_rmse']:.4f} g C m⁻² yr⁻¹")
        print(f"✓ Outputs saved to: tabnet/ directory")
        
    except Exception as e:
        print(f"Error in TabNet model training: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()