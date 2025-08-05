"""
A Belowground Productivity (BP) Deep Ensemble model for the ELM-TAM benchmark pipeline.

This module provides Deep Ensemble functionality for predicting belowground 
net primary productivity (BNPP) using environmental predictors from the TAM framework.

Deep Ensemble combines multiple neural networks with different architectures
to improve prediction accuracy and provide uncertainty estimates.
Key features:
- Multiple diverse neural network architectures
- Ensemble prediction by averaging
- Uncertainty quantification through prediction variance
- Improved robustness and generalization

Key functions:
- train_deep_ensemble: Train ensemble of neural networks
- evaluate_ensemble: Comprehensive ensemble evaluation
- predict_with_uncertainty: Predictions with uncertainty estimates

Data sources integrated:
- ForC global forest carbon database (529 forest sites)
- Global grassland productivity database (953 grassland sites)  
- TerraClimate environmental variables (aet, pet, ppt, tmax, tmin, vpd)
- GLASS satellite data (yearly GPP)
- Unit conversions: Forest data converted from Mg C ha⁻¹ yr⁻¹ to g C m⁻² yr⁻¹

Model outputs:
- Trained ensemble models (deep_ensemble_models.pkl)
- Ensemble predictions with uncertainty (predictions_with_uncertainty.png)
- Individual model performance (individual_models_performance.png)
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
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class BaseNeuralNetwork(nn.Module):
    """Base class for neural network architectures in the ensemble."""
    
    def __init__(self, input_size, output_size=1):
        super(BaseNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
    
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")


class DeepNetwork(BaseNeuralNetwork):
    """Deep neural network with multiple hidden layers."""
    
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], dropout_rate=0.3):
        super(DeepNetwork, self).__init__(input_size)
        
        layers = []
        in_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_size = hidden_size
        
        layers.append(nn.Linear(in_size, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class WideNetwork(BaseNeuralNetwork):
    """Wide neural network with fewer layers but more neurons."""
    
    def __init__(self, input_size, hidden_sizes=[512, 256], dropout_rate=0.2):
        super(WideNetwork, self).__init__(input_size)
        
        layers = []
        in_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_size = hidden_size
        
        layers.append(nn.Linear(in_size, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class ResidualNetwork(BaseNeuralNetwork):
    """Neural network with residual connections."""
    
    def __init__(self, input_size, hidden_size=128, num_blocks=3, dropout_rate=0.25):
        super(ResidualNetwork, self).__init__(input_size)
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList([
            self._make_residual_block(hidden_size, dropout_rate) 
            for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
    
    def _make_residual_block(self, hidden_size, dropout_rate):
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection
        
        x = self.dropout(x)
        return self.output_layer(x)


class AttentionNetwork(BaseNeuralNetwork):
    """Neural network with attention mechanism."""
    
    def __init__(self, input_size, hidden_size=128, num_heads=4, dropout_rate=0.2):
        super(AttentionNetwork, self).__init__(input_size)
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Input projection
        x = self.input_layer(x)
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # Self-attention
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        # Remove sequence dimension and output
        x = x.squeeze(1)
        x = self.dropout(x)
        return self.output_layer(x)


class SimpleNetwork(BaseNeuralNetwork):
    """Simple neural network with fewer parameters."""
    
    def __init__(self, input_size, hidden_sizes=[64, 32], dropout_rate=0.1):
        super(SimpleNetwork, self).__init__(input_size)
        
        layers = []
        in_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_size = hidden_size
        
        layers.append(nn.Linear(in_size, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


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


def plot_data_distribution(df: pd.DataFrame, save_path: str = "deep_ensemble/de_data_distribution.png"):
    """Plot distribution of BNPP data by ecosystem type and data source."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('BNPP Data Distribution Analysis - Deep Ensemble', fontsize=16, fontweight='bold')
    
    # 1. Histogram of BNPP values
    ax1 = axes[0, 0]
    ax1.hist(df['BNPP'], bins=30, alpha=0.7, color='navy', edgecolor='black')
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
    
    # 3. Ensemble architecture diagram
    ax3 = axes[1, 0]
    architectures = ['Deep\nNetwork', 'Wide\nNetwork', 'Residual\nNetwork', 'Attention\nNetwork', 'Simple\nNetwork']
    y_pos = np.arange(len(architectures))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    bars = ax3.barh(y_pos, [1]*len(architectures), color=colors, alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(architectures)
    ax3.set_xlabel('Ensemble Components')
    ax3.set_title('Deep Ensemble Architecture')
    ax3.text(0.5, -0.5, 'Ensemble = Average of 5 Models', ha='center', transform=ax3.transAxes, 
             fontsize=10, style='italic')
    
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
    print(f"Deep Ensemble data distribution plot saved to: {save_path}")
    plt.close()


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


def train_single_model(
    model: BaseNeuralNetwork,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 0.001,
    optimizer_type: str = 'adam',
    verbose: bool = False
) -> Tuple[BaseNeuralNetwork, List[float]]:
    """
    Train a single neural network model.
    
    Args:
        model: Neural network model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
        verbose: Whether to print training progress
        
    Returns:
        Tuple of (trained model, validation losses)
    """
    # Select optimizer
    if optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return model, val_losses


def train_deep_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 100,
    batch_size: int = 32,
    verbose: bool = True
) -> Tuple[List[BaseNeuralNetwork], Dict[str, float], StandardScaler]:
    """
    Train a deep ensemble of neural networks.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Fraction of data for testing
        val_size: Fraction of training data for validation
        random_state: Random seed for reproducibility
        epochs: Number of training epochs
        batch_size: Batch size for training
        verbose: Whether to print training progress
        
    Returns:
        Tuple of (ensemble models, performance metrics, scaler)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Further split training into train/validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )
    
    if verbose:
        print(f"Training set: {X_train_split.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_split), 
        columns=X_train_split.columns, 
        index=X_train_split.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), 
        columns=X_val.columns, 
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )
    
    # Create data loaders
    train_loader, _ = create_data_loaders(X_train_scaled, X_test_scaled, y_train_split, y_test, batch_size)
    val_loader, test_loader = create_data_loaders(X_val_scaled, X_test_scaled, y_val, y_test, batch_size)
    
    # Define ensemble architectures
    input_size = X_train_split.shape[1]
    ensemble_configs = [
        {
            'name': 'Deep Network',
            'model': DeepNetwork(input_size, hidden_sizes=[256, 128, 64, 32], dropout_rate=0.3),
            'optimizer': 'adam',
            'learning_rate': 0.001
        },
        {
            'name': 'Wide Network',
            'model': WideNetwork(input_size, hidden_sizes=[512, 256], dropout_rate=0.2),
            'optimizer': 'adam',
            'learning_rate': 0.0005
        },
        {
            'name': 'Residual Network',
            'model': ResidualNetwork(input_size, hidden_size=128, num_blocks=3, dropout_rate=0.25),
            'optimizer': 'adam',
            'learning_rate': 0.0005
        },
        {
            'name': 'Attention Network',
            'model': AttentionNetwork(input_size, hidden_size=128, num_heads=4, dropout_rate=0.2),
            'optimizer': 'adam',
            'learning_rate': 0.001
        },
        {
            'name': 'Simple Network',
            'model': SimpleNetwork(input_size, hidden_sizes=[64, 32], dropout_rate=0.1),
            'optimizer': 'rmsprop',
            'learning_rate': 0.001
        }
    ]
    
    # Train ensemble models
    ensemble_models = []
    individual_metrics = {}
    
    if verbose:
        print("\nTraining Deep Ensemble...")
    
    for i, config in enumerate(ensemble_configs):
        if verbose:
            print(f"\nTraining {config['name']} ({i+1}/{len(ensemble_configs)})...")
        
        model = config['model'].to(device)
        
        # Train the model
        trained_model, val_losses = train_single_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            learning_rate=config['learning_rate'],
            optimizer_type=config['optimizer'],
            verbose=verbose
        )
        
        ensemble_models.append(trained_model)
        
        # Evaluate individual model
        individual_metrics[config['name']] = evaluate_single_model(
            trained_model, X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_split, y_val, y_test, verbose=False
        )
        
        if verbose:
            metrics = individual_metrics[config['name']]
            print(f"  Test R²: {metrics['test_r2']:.4f}, Test RMSE: {metrics['test_rmse']:.2f}")
    
    # Evaluate ensemble
    if verbose:
        print(f"\nEvaluating ensemble performance...")
    
    ensemble_metrics = evaluate_ensemble(
        ensemble_models, X_train_scaled, X_val_scaled, X_test_scaled,
        y_train_split, y_val, y_test, X.columns, verbose=verbose
    )
    
    # Add individual model metrics
    ensemble_metrics['individual_models'] = individual_metrics
    
    return ensemble_models, ensemble_metrics, scaler


def evaluate_single_model(
    model: BaseNeuralNetwork,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    verbose: bool = True
) -> Dict[str, float]:
    """Evaluate a single model's performance."""
    model.eval()
    
    with torch.no_grad():
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values).to(device)
        X_val_tensor = torch.FloatTensor(X_val.values).to(device)
        X_test_tensor = torch.FloatTensor(X_test.values).to(device)
        
        # Predictions
        y_train_pred = model(X_train_tensor).squeeze().cpu().numpy()
        y_val_pred = model(X_val_tensor).squeeze().cpu().numpy()
        y_test_pred = model(X_test_tensor).squeeze().cpu().numpy()
    
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
        'test_mae': mean_absolute_error(y_test, y_test_pred)
    }
    
    return metrics


def evaluate_ensemble(
    models: List[BaseNeuralNetwork],
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
    Evaluate the ensemble performance.
    
    Args:
        models: List of trained models
        X_train, X_val, X_test: Feature sets
        y_train, y_val, y_test: Target values
        feature_names: List of feature names
        verbose: Whether to print results
        
    Returns:
        Dictionary of ensemble performance metrics
    """
    # Get ensemble predictions
    train_preds, _ = predict_with_uncertainty(models, X_train)
    val_preds, _ = predict_with_uncertainty(models, X_val)
    test_preds, test_uncertainty = predict_with_uncertainty(models, X_test)
    
    # Calculate metrics
    metrics = {
        'train_r2': r2_score(y_train, train_preds),
        'val_r2': r2_score(y_val, val_preds),
        'test_r2': r2_score(y_test, test_preds),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_preds)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_preds)),
        'train_mae': mean_absolute_error(y_train, train_preds),
        'val_mae': mean_absolute_error(y_val, val_preds),
        'test_mae': mean_absolute_error(y_test, test_preds),
        'n_features': len(feature_names),
        'n_train': len(y_train),
        'n_val': len(y_val),
        'n_test': len(y_test),
        'n_models': len(models),
        'mean_uncertainty': np.mean(test_uncertainty),
        'std_uncertainty': np.std(test_uncertainty)
    }
    
    # Print results
    if verbose:
        print(f"\n{'='*60}")
        print("DEEP ENSEMBLE PERFORMANCE METRICS")
        print(f"{'='*60}")
        print(f"Training R²: {metrics['train_r2']:.4f}")
        print(f"Validation R²: {metrics['val_r2']:.4f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        print(f"Training RMSE: {metrics['train_rmse']:.4f}")
        print(f"Validation RMSE: {metrics['val_rmse']:.4f}")
        print(f"Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"Training MAE: {metrics['train_mae']:.4f}")
        print(f"Validation MAE: {metrics['val_mae']:.4f}")
        print(f"Test MAE: {metrics['test_mae']:.4f}")
        print(f"Number of models: {metrics['n_models']}")
        print(f"Mean uncertainty: {metrics['mean_uncertainty']:.4f}")
        print(f"Std uncertainty: {metrics['std_uncertainty']:.4f}")
    
    return metrics


def predict_with_uncertainty(models: List[BaseNeuralNetwork], X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with uncertainty estimates using the ensemble.
    
    Args:
        models: List of trained models
        X: Feature matrix
        
    Returns:
        Tuple of (mean predictions, uncertainty estimates)
    """
    predictions = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values).to(device)
            pred = model(X_tensor).squeeze().cpu().numpy()
            predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Mean prediction
    mean_pred = np.mean(predictions, axis=0)
    
    # Uncertainty as standard deviation
    uncertainty = np.std(predictions, axis=0)
    
    return mean_pred, uncertainty


def plot_individual_models_performance(metrics: Dict, save_path: str = "deep_ensemble/individual_models_performance.png"):
    """Plot performance of individual models in the ensemble."""
    individual_metrics = metrics['individual_models']
    
    model_names = list(individual_metrics.keys())
    test_r2_scores = [individual_metrics[name]['test_r2'] for name in model_names]
    test_rmse_scores = [individual_metrics[name]['test_rmse'] for name in model_names]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # R² scores
    ax1 = axes[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars1 = ax1.bar(model_names, test_r2_scores, color=colors, alpha=0.7)
    ax1.set_ylabel('Test R²')
    ax1.set_title('Individual Model Performance - R²')
    ax1.set_ylim(0, max(test_r2_scores) * 1.1)
    ax1.grid(True, alpha=0.3)
    
    # Add ensemble performance line
    ensemble_r2 = metrics['test_r2']
    ax1.axhline(y=ensemble_r2, color='red', linestyle='--', linewidth=2, label=f'Ensemble R²: {ensemble_r2:.4f}')
    ax1.legend()
    
    # Add value labels on bars
    for bar, score in zip(bars1, test_r2_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # RMSE scores
    ax2 = axes[1]
    bars2 = ax2.bar(model_names, test_rmse_scores, color=colors, alpha=0.7)
    ax2.set_ylabel('Test RMSE')
    ax2.set_title('Individual Model Performance - RMSE')
    ax2.grid(True, alpha=0.3)
    
    # Add ensemble performance line
    ensemble_rmse = metrics['test_rmse']
    ax2.axhline(y=ensemble_rmse, color='red', linestyle='--', linewidth=2, label=f'Ensemble RMSE: {ensemble_rmse:.2f}')
    ax2.legend()
    
    # Add value labels on bars
    for bar, score in zip(bars2, test_rmse_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{score:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Individual models performance plot saved to: {save_path}")
    plt.close()


def plot_predictions_with_uncertainty(y_true: pd.Series, y_pred: np.ndarray, uncertainty: np.ndarray, r2: float, save_path: str = "deep_ensemble/predictions_with_uncertainty.png"):
    """Plot predictions with uncertainty estimates."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Main predictions plot
    ax1 = axes[0]
    scatter = ax1.scatter(y_true, y_pred, c=uncertainty, cmap='viridis', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], '--r', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual BNPP (g C m⁻² yr⁻¹)')
    ax1.set_ylabel('Predicted BNPP (g C m⁻² yr⁻¹)')
    ax1.set_title(f'Ensemble Predictions with Uncertainty (R² = {r2:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Colorbar for uncertainty
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Prediction Uncertainty (σ)')
    
    # Uncertainty distribution
    ax2 = axes[1]
    ax2.hist(uncertainty, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Prediction Uncertainty (σ)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Prediction Uncertainty')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    ax2.axvline(np.mean(uncertainty), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(uncertainty):.2f}')
    ax2.axvline(np.median(uncertainty), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(uncertainty):.2f}')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Predictions with uncertainty plot saved to: {save_path}")
    plt.close()


def save_ensemble(models: List[BaseNeuralNetwork], metrics: Dict, scaler: StandardScaler, output_path: str = "deep_ensemble/deep_ensemble_models.pkl"):
    """Save trained ensemble models, scaler, and metrics to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state dicts instead of full models for better compatibility
    model_states = []
    for i, model in enumerate(models):
        model_state = {
            'state_dict': model.state_dict(),
            'model_type': type(model).__name__,
            'model_config': getattr(model, 'config', {})
        }
        model_states.append(model_state)
    
    ensemble_data = {
        'model_states': model_states,
        'scaler': scaler,
        'metrics': metrics
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(ensemble_data, f)
    
    print(f"Ensemble models saved to: {output_path}")
    
    # Save model summary as text file
    summary_path = output_path.parent / "deep_ensemble_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Deep Ensemble Model Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write("Ensemble Architecture:\n")
        f.write("- Deep Network (4 layers, 256→128→64→32)\n")
        f.write("- Wide Network (2 layers, 512→256)\n")
        f.write("- Residual Network (3 residual blocks, 128 units)\n")
        f.write("- Attention Network (Multi-head attention, 4 heads)\n")
        f.write("- Simple Network (2 layers, 64→32)\n\n")
        
        f.write(f"Ensemble Performance:\n")
        f.write(f"Training R²: {metrics['train_r2']:.4f}\n")
        f.write(f"Validation R²: {metrics['val_r2']:.4f}\n")
        f.write(f"Test R²: {metrics['test_r2']:.4f}\n")
        f.write(f"Training RMSE: {metrics['train_rmse']:.4f}\n")
        f.write(f"Validation RMSE: {metrics['val_rmse']:.4f}\n")
        f.write(f"Test RMSE: {metrics['test_rmse']:.4f}\n")
        f.write(f"Training MAE: {metrics['train_mae']:.4f}\n")
        f.write(f"Validation MAE: {metrics['val_mae']:.4f}\n")
        f.write(f"Test MAE: {metrics['test_mae']:.4f}\n\n")
        
        f.write(f"Uncertainty Statistics:\n")
        f.write(f"Mean uncertainty: {metrics['mean_uncertainty']:.4f}\n")
        f.write(f"Std uncertainty: {metrics['std_uncertainty']:.4f}\n\n")
        
        f.write(f"Dataset Information:\n")
        f.write(f"Number of features: {metrics['n_features']}\n")
        f.write(f"Training samples: {metrics['n_train']}\n")
        f.write(f"Validation samples: {metrics['n_val']}\n")
        f.write(f"Test samples: {metrics['n_test']}\n")
        f.write(f"Number of models: {metrics['n_models']}\n\n")
        
        f.write(f"Individual Model Performance:\n")
        if 'individual_models' in metrics:
            for name, model_metrics in metrics['individual_models'].items():
                f.write(f"{name}:\n")
                f.write(f"  Test R²: {model_metrics['test_r2']:.4f}\n")
                f.write(f"  Test RMSE: {model_metrics['test_rmse']:.4f}\n")
                f.write(f"  Test MAE: {model_metrics['test_mae']:.4f}\n")
    
    print(f"Ensemble summary saved to: {summary_path}")


def main():
    """
    Main function for standalone execution of Deep Ensemble model training.
    Loads data, trains ensemble, and evaluates performance.
    """
    try:
        print("="*60)
        print("DEEP ENSEMBLE MODEL TRAINING")
        print("="*60)
        
        # Load integrated data
        df = load_integrated_data()
        
        # Plot data distribution
        plot_data_distribution(df)
        
        # Prepare features and target
        X, y = prepare_features_target(df)
        
        # Train ensemble
        models, metrics, scaler = train_deep_ensemble(X, y, epochs=50, verbose=True)
        
        # Split data for plotting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        # Plot individual model performance
        plot_individual_models_performance(metrics)
        
        # Plot predictions with uncertainty
        test_preds, test_uncertainty = predict_with_uncertainty(models, X_test_scaled)
        plot_predictions_with_uncertainty(y_test, test_preds, test_uncertainty, metrics['test_r2'])
        
        # Save ensemble
        save_ensemble(models, metrics, scaler)
        
        print(f"\n{'='*60}")
        print("DEEP ENSEMBLE TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"✓ Ensemble trained on {len(df)} samples with {X.shape[1]} features")
        print(f"✓ Number of models: {metrics['n_models']}")
        print(f"✓ Test R²: {metrics['test_r2']:.4f}")
        print(f"✓ Test RMSE: {metrics['test_rmse']:.4f} g C m⁻² yr⁻¹")
        print(f"✓ Mean uncertainty: {metrics['mean_uncertainty']:.4f}")
        print(f"✓ Outputs saved to: deep_ensemble/ directory")
        
    except Exception as e:
        print(f"Error in Deep Ensemble training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()