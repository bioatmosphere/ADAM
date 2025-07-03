"""
A Belowground Productivity (BP) Multi-Layer Perceptron (MLP) model for the ELM-TAM benchmark pipeline.

This module provides PyTorch-based neural network functionality for predicting belowground 
net primary productivity (BNPP) using environmental predictors from the TAM framework.

Key functions:
- train_mlp: Train MLP model on integrated dataset
- train_mlp_with_hyperparameter_search: Test multiple architectures and find best one
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class BP_MLP(nn.Module):
    """Belowground Productivity (BP) Multi-Layer Perceptron model."""
    
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        """
        Initialize an instance of the MLP model.
        
        Args:
            input_size (int): Size of the input layer.
            hidden_sizes (list): List of sizes for the hidden layers.
            output_size (int): Size of the output layer.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(BP_MLP, self).__init__()
        
        layers = []
        in_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            if i < len(hidden_sizes) - 1:  # Don't add dropout before output layer
                layers.append(nn.Dropout(dropout_rate))
            in_size = hidden_size
            
        # Output layer
        layers.append(nn.Linear(in_size, output_size))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


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
    
    # Validate required columns based on the known structure
    required_cols = ['BNPP', 'lat', 'lon']  # Minimum required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Loaded {len(df)} records with {len(df.columns)} features")
    print(f"Target variable (BNPP) range: {df['BNPP'].min():.2f} to {df['BNPP'].max():.2f}")
    
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
    exclude_cols = [target_col, 'site_id', 'study_id', 'measurement_id', 'lat', 'lon'] 
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Handle missing values
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Remove rows with missing target values
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Remove categorical variables (keep only numeric features)
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"Excluding categorical variables: {list(categorical_cols)}")
        X = X.select_dtypes(exclude=['object'])
    
    # Fill missing features with median values
    X = X.fillna(X.median())
    
    print(f"Features prepared: {X.shape[1]} variables, {X.shape[0]} samples")
    print(f"Feature columns: {list(X.columns)}")
    
    return X, y


def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=32):
    """Create PyTorch data loaders for training and testing."""
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).to(device)
    y_test_tensor = torch.FloatTensor(y_test.values).to(device)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_mlp(
    X: pd.DataFrame, 
    y: pd.Series,
    hidden_sizes: list = [128, 64, 32],
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    dropout_rate: float = 0.2,
    verbose: bool = True,
    save_training_curves: bool = False
) -> Tuple[BP_MLP, Dict[str, float], StandardScaler]:
    """
    Train MLP model with PyTorch.
    
    Args:
        X: Feature matrix
        y: Target vector  
        hidden_sizes: List of hidden layer sizes
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        dropout_rate: Dropout rate for regularization
        verbose: Whether to print training progress
        
    Returns:
        Tuple of (trained model, performance metrics dict, scaler)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    if verbose:
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        X_train_scaled, X_test_scaled, y_train, y_test, batch_size
    )
    
    # Initialize model
    input_size = X_train.shape[1]
    model = BP_MLP(input_size, hidden_sizes, output_size=1, dropout_rate=dropout_rate)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    if verbose:
        print("Training MLP model...")
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
    
    # Save training curves if requested
    if save_training_curves:
        plot_training_curves(train_losses, test_losses)
    
    # Evaluate model
    metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, X.columns, verbose=verbose)
    
    return model, metrics, scaler


def train_mlp_with_hyperparameter_search(X: pd.DataFrame, y: pd.Series):
    """
    Train multiple MLP models with different hyperparameters and select the best one.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Best model, metrics, and scaler
    """
    # Define hyperparameter combinations to test
    hyperparameter_configs = [
        # Original configuration
        {
            'hidden_sizes': [128, 64, 32],
            'learning_rate': 0.001,
            'epochs': 150,
            'batch_size': 32,
            'dropout_rate': 0.2,
            'name': 'Original'
        },
        # Deeper network
        {
            'hidden_sizes': [256, 128, 64, 32],
            'learning_rate': 0.001,
            'epochs': 200,
            'batch_size': 32,
            'dropout_rate': 0.3,
            'name': 'Deeper'
        },
        # Wider network
        {
            'hidden_sizes': [512, 256, 128],
            'learning_rate': 0.0005,
            'epochs': 150,
            'batch_size': 64,
            'dropout_rate': 0.4,
            'name': 'Wider'
        },
        # Smaller network with higher learning rate
        {
            'hidden_sizes': [64, 32],
            'learning_rate': 0.01,
            'epochs': 100,
            'batch_size': 16,
            'dropout_rate': 0.1,
            'name': 'Compact'
        },
        # Regularized network
        {
            'hidden_sizes': [128, 64, 32, 16],
            'learning_rate': 0.0001,
            'epochs': 300,
            'batch_size': 32,
            'dropout_rate': 0.5,
            'name': 'Regularized'
        }
    ]
    
    best_model = None
    best_metrics = None
    best_scaler = None
    best_score = -float('inf')
    
    results_summary = []
    
    for i, config in enumerate(hyperparameter_configs):
        print(f"\n{'='*60}")
        print(f"Training MLP Configuration {i+1}/5: {config['name']}")
        print(f"Hidden layers: {config['hidden_sizes']}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Epochs: {config['epochs']}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Dropout rate: {config['dropout_rate']}")
        print(f"{'='*60}")
        
        # Train model with current configuration
        model, metrics, scaler = train_mlp(
            X, y,
            hidden_sizes=config['hidden_sizes'],
            learning_rate=config['learning_rate'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            dropout_rate=config['dropout_rate'],
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
            best_scaler = scaler
            print(f"✓ New best model found! Test R²: {best_score:.4f}")
    
    # Print comparison of all configurations
    print(f"\n{'='*80}")
    print("MLP HYPERPARAMETER SEARCH RESULTS")
    print(f"{'='*80}")
    print(f"{'Configuration':<15} {'Test R²':<10} {'Test RMSE':<12} {'Test MAE':<10} {'Train R²':<10}")
    print(f"{'-'*80}")
    
    for result in results_summary:
        print(f"{result['name']:<15} {result['test_r2']:<10.4f} {result['test_rmse']:<12.2f} "
              f"{result['test_mae']:<10.2f} {result['train_r2']:<10.4f}")
    
    best_config_name = max(results_summary, key=lambda x: x['test_r2'])['name']
    print(f"\nBest configuration: {best_config_name}")
    print(f"Best Test R²: {best_score:.4f}")
    
    # Retrain best model to get training curves
    print(f"\nRetraining best model ({best_config_name}) to generate training curves...")
    best_config = next(config for config in hyperparameter_configs if config['name'] == best_config_name)
    
    final_model, final_metrics, final_scaler = train_mlp(
        X, y,
        hidden_sizes=best_config['hidden_sizes'],
        learning_rate=best_config['learning_rate'],
        epochs=best_config['epochs'],
        batch_size=best_config['batch_size'],
        dropout_rate=best_config['dropout_rate'],
        verbose=True,
        save_training_curves=True
    )
    
    # Generate final plots for best model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = pd.DataFrame(final_scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(final_scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    # Make predictions for plotting
    final_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled.values).to(device)
        y_test_pred = final_model(X_test_tensor).squeeze().cpu().numpy()
    
    plot_predictions(y_test, y_test_pred, final_metrics['test_r2'])
    
    # Update best model with the retrained one
    best_model = final_model
    best_metrics = final_metrics
    best_scaler = final_scaler
    
    return best_model, best_metrics, best_scaler


def evaluate_model(
    model: BP_MLP,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame, 
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Args:
        model: Trained MLP model
        X_train, X_test: Training and test feature sets
        y_train, y_test: Training and test target values
        feature_names: List of feature names
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary of performance metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values).to(device)
        X_test_tensor = torch.FloatTensor(X_test.values).to(device)
        
        # Predictions
        y_train_pred = model(X_train_tensor).squeeze().cpu().numpy()
        y_test_pred = model(X_test_tensor).squeeze().cpu().numpy()
    
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
    if verbose:
        print(f"\n{'='*50}")
        print("MLP MODEL PERFORMANCE METRICS")
        print(f"{'='*50}")
        print(f"Training R²: {metrics['train_r2']:.4f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        print(f"Training RMSE: {metrics['train_rmse']:.4f}")
        print(f"Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"Training MAE: {metrics['train_mae']:.4f}")
        print(f"Test MAE: {metrics['test_mae']:.4f}")
    
    return metrics


def plot_training_curves(train_losses, test_losses, save_path: str = "../models/mlp_training_curves_best.png"):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, test_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Best MLP Model - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Training curves plot saved to: {save_path}")
    plt.close()


def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, r2: float, save_path: str = "../models/mlp_predictions_plot_best.png"):
    """Plot actual vs predicted values with perfect prediction line."""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, color='blue', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--r', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual BNPP (gC m⁻² year⁻¹)')
    plt.ylabel('Predicted BNPP (gC m⁻² year⁻¹)')
    plt.title(f'Actual vs Predicted BNPP - Best MLP Model (R² = {r2:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Best MLP predictions plot saved to: {save_path}")
    plt.close()


def save_model(model: BP_MLP, metrics: Dict, scaler: StandardScaler, output_path: str = "../models/mlp_model_best.pkl"):
    """Save trained model, scaler and metrics to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'input_size': model.model[0].in_features,
            'hidden_sizes': [layer.out_features for layer in model.model if isinstance(layer, nn.Linear)][:-1],
            'output_size': 1,
            'dropout_rate': 0.2  # Store the dropout rate used
        },
        'scaler': scaler,
        'metrics': metrics
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {output_path}")
    
    # Save model summary as text file
    summary_path = output_path.parent / "mlp_model_best_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Best Multi-Layer Perceptron (MLP) Model Summary\n")
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
        
        f.write(f"Model Architecture:\n")
        arch = model_data['model_architecture']
        f.write(f"Input features: {arch['input_size']}\n")
        f.write(f"Hidden layers: {arch['hidden_sizes']}\n")
        f.write(f"Output size: {arch['output_size']}\n")
        f.write(f"Dropout rate: {arch['dropout_rate']}\n")
    
    print(f"Best MLP model summary saved to: {summary_path}")


def main():
    """
    Main function for standalone execution of MLP model training.
    Loads data, trains model, and evaluates performance.
    """
    try:
        # Load integrated data
        df = load_integrated_data()
        
        # Prepare features and target
        X, y = prepare_features_target(df)
        
        # Train multiple models with hyperparameter search
        print("Starting MLP hyperparameter search...")
        model, metrics, scaler = train_mlp_with_hyperparameter_search(X, y)
        
        # Save the best model
        save_model(model, metrics, scaler)
        
        print("\nMLP hyperparameter search completed successfully!")
        print(f"Best model saved with Test R²: {metrics['test_r2']:.4f}")
        
    except Exception as e:
        print(f"Error in MLP model training: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()