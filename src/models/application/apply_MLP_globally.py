"""
Apply trained MLP model globally using global climate and satellite data.

This script loads the trained MLP model and applies it to global gridded data to produce
worldwide BNPP predictions at 0.5-degree resolution.

Data sources for global application:
- TerraClimate: Global climate variables (aet, pet, ppt, tmax, tmin, vpd)
- GLASS: Global GPP satellite data from HDF tiles
- SoilGrids: Mean soil property values from training data
- Output: Global BNPP predictions at 0.5-degree resolution

Author: TAM Development Team
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.features as cfeatures
from pathlib import Path
import pickle
import warnings
from typing import Tuple, Dict
# import rasterio
# from rasterio.transform import from_bounds
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class BP_MLP(nn.Module):
    """
    Multi-Layer Perceptron for Below-ground Primary Productivity prediction.
    Dynamically builds architecture to match the saved model state dict.
    """
    def __init__(self, input_size: int, hidden_sizes: list, dropout_rate: float = 0.2):
        super(BP_MLP, self).__init__()

        # Build the model structure dynamically based on hidden_sizes
        # Match the exact structure: Linear, ReLU, Dropout for each hidden layer
        # EXCEPT the last hidden layer which has NO dropout
        layers = []
        prev_size = input_size

        # Add hidden layers dynamically
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            # Only add dropout if this is NOT the last hidden layer
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer (single neuron for regression)
        layers.append(nn.Linear(prev_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def load_trained_mlp_model(model_path: str = "/Users/6lw/Desktop/2_models/ADAM/src/models/mlp/mlp_model_best.pkl") -> Tuple[nn.Module, Dict]:
    """
    Load the trained MLP model and metadata.
    
    Args:
        model_path: Path to the saved MLP model file
        
    Returns:
        Tuple of (model, model metadata)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Trained MLP model not found at {model_path}")
    
    print(f"Loading trained MLP model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Reconstruct model from architecture
    arch = model_data['model_architecture']
    model = BP_MLP(
        input_size=arch['input_size'],
        hidden_sizes=arch['hidden_sizes'],
        dropout_rate=arch.get('dropout_rate', 0.2)  # Use the actual dropout rate from saved config
    )
    
    # Load state dict
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    
    scaler = model_data['scaler']
    metrics = model_data['metrics']

    # Get feature names from the scaler (most reliable source)
    if hasattr(scaler, 'feature_names_in_'):
        feature_names = list(scaler.feature_names_in_)
    else:
        # Fallback to BNPP_fraction training features
        feature_names = ['aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd',
                        'soil_carbon_stock', 'clay_content', 'silt_content', 'sand_content',
                        'nitrogen_content', 'cation_exchange_capacity', 'ph_in_water',
                        'bulk_density', 'coarse_fragments', 'soil_moisture', 'elevation']

    print(f"Model loaded successfully!")
    print(f"Model architecture: {arch['hidden_sizes']}")
    print(f"Model performance - Test R²: {metrics['test_r2']:.4f}")
    print(f"Required features ({len(feature_names)}): {feature_names}")

    return model, model_data


def load_global_terraclimate_data(data_dir: str = "/Users/6lw/Desktop/2_models/ADAM/ancillary/terraclimate", year: int = 2010) -> xr.Dataset:
    """
    Load global TerraClimate data for specified year.
    
    Args:
        data_dir: Directory containing TerraClimate netCDF files
        year: Year to load data for
        
    Returns:
        xarray Dataset with all climate variables
    """
    data_dir = Path(data_dir)
    
    print(f"Loading TerraClimate data for {year}...")
    
    # Define variables to load
    variables = ['aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd']
    
    datasets = {}
    for var in variables:
        file_path = data_dir / f"TerraClimate_{var}_{year}.nc"
        if file_path.exists():
            print(f"  Loading {var}...")
            ds = xr.open_dataset(file_path)
            # Take annual mean for temperature and vpd, annual sum for precipitation and ET
            if var in ['ppt', 'aet', 'pet']:
                datasets[var] = ds[var].sum(dim='time')
            else:  # tmax, tmin, vpd
                datasets[var] = ds[var].mean(dim='time')
        else:
            print(f"  Warning: {file_path} not found, skipping {var}")
    
    if not datasets:
        raise FileNotFoundError(f"No TerraClimate files found for {year} in {data_dir}")
    
    # Combine into single dataset
    combined_ds = xr.Dataset(datasets)
    
    print(f"TerraClimate data loaded: {list(combined_ds.data_vars)} variables")
    print(f"Spatial resolution: {len(combined_ds.lat)} x {len(combined_ds.lon)} grid points")
    
    return combined_ds


def load_glass_gpp_data(gpp_file: str = "/Users/6lw/Desktop/2_models/ADAM/ancillary/glass/global_gpp_yearly_2010_0.5deg_FIXED.nc") -> xr.DataArray:
    """
    Load GLASS GPP data for ecological filtering (not used as model feature).

    Args:
        gpp_file: Path to GLASS GPP NetCDF file

    Returns:
        Global GPP DataArray
    """
    gpp_path = Path(gpp_file)

    if not gpp_path.exists():
        print(f"\n⚠️  GLASS GPP file not found at: {gpp_path}")
        print("=" * 60)
        print("To create the GLASS GPP NetCDF file, run:")
        print("cd ../../../ancillary")
        print("python export_glass_to_netcdf.py --year 2010")
        print("=" * 60)
        raise FileNotFoundError(f"GLASS GPP NetCDF not found. Please run export_glass_to_netcdf.py first.")

    print(f"\nLoading GLASS GPP data from: {gpp_path}")
    gpp_da = xr.open_dataarray(gpp_path)
    print(f"  GPP data shape: {gpp_da.shape}")
    print(f"  GPP range: {gpp_da.min().values:.1f} to {gpp_da.max().values:.1f} gC m⁻² yr⁻¹")
    print(f"  GPP mean: {gpp_da.mean().values:.1f} gC m⁻² yr⁻¹")

    return gpp_da


def interpolate_to_common_grid(climate_ds: xr.Dataset, gpp_da: xr.DataArray, 
                              target_resolution: float = 0.5) -> xr.Dataset:
    """
    Interpolate all datasets to a common grid.
    
    Args:
        climate_ds: TerraClimate dataset
        gpp_da: GPP DataArray
        target_resolution: Target resolution in degrees
        
    Returns:
        Combined dataset on common grid
    """
    print(f"Interpolating to common {target_resolution}° grid...")
    
    # Define target grid
    target_lat = np.arange(-89.75, 90, target_resolution)
    target_lon = np.arange(-179.75, 180, target_resolution)
    
    # Interpolate climate data
    climate_interp = climate_ds.interp(lat=target_lat, lon=target_lon, method='linear')
    
    # Interpolate GPP data
    gpp_interp = gpp_da.interp(lat=target_lat, lon=target_lon, method='linear')
    
    # Combine datasets
    combined = climate_interp.copy()
    combined['gpp_yearly'] = gpp_interp
    
    print(f"Combined dataset shape: {len(combined.lat)} x {len(combined.lon)}")
    print(f"Variables: {list(combined.data_vars)}")
    
    return combined


def get_training_data_soil_means() -> dict:
    """
    Load training data to get mean soil property values for global application.
    
    Returns:
        Dictionary with mean soil property values
    """
    # Load the aggregated training data
    training_path = "/Users/6lw/Desktop/2_models/ADAM/productivity/earth/aggregated_data.csv"
    
    if not Path(training_path).exists():
        raise FileNotFoundError(f"Training data not found at {training_path}")
    
    df = pd.read_csv(training_path)
    
    # Calculate mean values for all non-climate properties (soil + topography)
    # These match what was used in BNPP_fraction training
    soil_properties = [
        'soil_carbon_stock', 'clay_content', 'silt_content', 'sand_content',
        'nitrogen_content', 'cation_exchange_capacity', 'ph_in_water',
        'bulk_density', 'coarse_fragments', 'soil_moisture', 'elevation'
    ]
    
    soil_means = {}
    for prop in soil_properties:
        if prop in df.columns:
            soil_means[prop] = df[prop].mean(skipna=True)
    
    print("Using mean soil property values from training data:")
    for prop, value in soil_means.items():
        print(f"  {prop}: {value:.3f}")
    
    return soil_means

def prepare_global_features(dataset: xr.Dataset, required_features: list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert global xarray dataset to DataFrame for model prediction.
    For missing soil data, use mean values from training data.
    GPP is kept separate for filtering only (NOT used as model feature).

    Args:
        dataset: Combined global dataset (includes GPP for filtering)
        required_features: List of features required by the model (excludes GPP)

    Returns:
        Tuple of (features DataFrame, coordinates DataFrame, GPP DataFrame)
    """
    print("\nPreparing global features for model application...")

    # Create meshgrid of coordinates
    lat_vals, lon_vals = np.meshgrid(dataset.lat.values, dataset.lon.values, indexing='ij')

    # Create DataFrame manually to avoid duplicate column issues
    data_dict = {}
    data_dict['lat'] = lat_vals.flatten()
    data_dict['lon'] = lon_vals.flatten()

    # Add each variable from dataset
    for var in dataset.data_vars:
        values = dataset[var].values.flatten()
        data_dict[var] = values

    df = pd.DataFrame(data_dict)

    # Remove NaN values (ocean, invalid data)
    df_clean = df.dropna()

    print(f"  Valid land grid points: {len(df_clean):,}")

    # Extract GPP for filtering (do this BEFORE selecting model features)
    if 'gpp_yearly' in df_clean.columns:
        gpp_df = df_clean[['gpp_yearly']].copy()
        print(f"  GPP data extracted for filtering (NOT used as model feature)")
    else:
        gpp_df = None
        print(f"  Warning: GPP data not found for filtering")

    # Check for missing features and add soil data means
    # NOTE: required_features should NOT include gpp_yearly
    missing_features = [f for f in required_features if f not in df_clean.columns]

    if missing_features:
        print(f"  Missing features detected: {missing_features}")
        print("  Adding mean values from training data...")

        # Get soil means from training data
        soil_means = get_training_data_soil_means()

        # Add missing soil features using mean values
        for feature in missing_features:
            if feature in soil_means:
                df_clean[feature] = soil_means[feature]
                print(f"    Added {feature} = {soil_means[feature]:.3f}")
            else:
                raise ValueError(f"Cannot find mean value for missing feature: {feature}")

    # Select only required features (excludes GPP)
    feature_df = df_clean[required_features].copy()

    # Get coordinates
    coord_cols = ['lat', 'lon']
    coords_df = df_clean[coord_cols].copy()

    print(f"  Global features prepared: {len(feature_df):,} grid points")
    print(f"  Model feature columns: {list(feature_df.columns)}")
    if gpp_df is not None:
        print(f"  ✓ GPP available for ecological filtering")

    return feature_df, coords_df, gpp_df


def apply_mlp_model_globally(model: nn.Module, features_df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """
    Apply MLP model to global features.
    
    Args:
        model: Trained MLP model
        features_df: Global features DataFrame
        scaler: Fitted scaler for feature normalization
        
    Returns:
        Array of global BNPP predictions
    """
    print("Applying MLP model globally...")
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features_scaled)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions_tensor = model(features_tensor)
        predictions = predictions_tensor.numpy().flatten()

    # Clip predictions to valid range [0, 1] for BNPP fraction
    # Neural networks can extrapolate beyond training range, so we constrain to physical bounds
    predictions_clipped = np.clip(predictions, 0.0, 1.0)
    n_clipped = np.sum((predictions < 0) | (predictions > 1))

    print(f"  Global predictions complete (before ecological filtering)!")
    print(f"  BNPP_fraction range: {predictions_clipped.min():.4f} to {predictions_clipped.max():.4f}")
    print(f"  Mean BNPP_fraction: {predictions_clipped.mean():.4f}")
    print(f"  Median BNPP_fraction: {np.median(predictions_clipped):.4f}")
    if n_clipped > 0:
        print(f"  Note: {n_clipped} predictions ({n_clipped/len(predictions)*100:.1f}%) were clipped to [0, 1] range")

    return predictions_clipped


def apply_ecological_constraints(predictions: np.ndarray,
                                 gpp_df: pd.DataFrame,
                                 gpp_threshold: float = 100.0) -> Tuple[np.ndarray, dict]:
    """
    Apply ecological constraints using GPP to filter out non-productive areas.

    Masks predictions in areas where GPP is very low, indicating:
    - Antarctica, Greenland, ice sheets
    - Extreme deserts (Sahara, Arabian, Gobi)
    - Barren/rock areas with minimal vegetation

    Args:
        predictions: Raw BNPP_fraction predictions
        gpp_df: DataFrame with GPP values (gC m⁻² yr⁻¹)
        gpp_threshold: Minimum GPP for valid prediction (default: 100 gC m⁻² yr⁻¹)

    Returns:
        Tuple of (constrained predictions, statistics dict)
    """
    print(f"\nApplying GPP-based ecological constraints...")
    print(f"  GPP threshold: > {gpp_threshold} gC m⁻² yr⁻¹")
    print(f"  Rationale: BNPP_fraction is only meaningful where vegetation productivity exists")

    # Get GPP values
    gpp_values = gpp_df['gpp_yearly'].values

    # Create mask for valid predictions (GPP above threshold)
    valid_mask = gpp_values > gpp_threshold

    # Count filtered pixels
    n_total = len(predictions)
    n_filtered = (~valid_mask).sum()
    n_kept = valid_mask.sum()

    print(f"\n  Filtering results:")
    print(f"    Total predictions: {n_total:,}")
    print(f"    Filtered (GPP ≤ {gpp_threshold}): {n_filtered:,} ({n_filtered/n_total*100:.1f}%)")
    print(f"    Valid predictions (GPP > {gpp_threshold}): {n_kept:,} ({n_kept/n_total*100:.1f}%)")

    # Create constrained predictions
    constrained_predictions = predictions.copy()
    constrained_predictions[~valid_mask] = np.nan

    # Calculate statistics on valid predictions only
    valid_predictions = constrained_predictions[~np.isnan(constrained_predictions)]
    valid_gpp = gpp_values[valid_mask]

    stats = {
        'n_total': n_total,
        'n_filtered': n_filtered,
        'n_kept': len(valid_predictions),
        'filter_percentage': (n_filtered / n_total) * 100,
        'min': valid_predictions.min() if len(valid_predictions) > 0 else np.nan,
        'max': valid_predictions.max() if len(valid_predictions) > 0 else np.nan,
        'mean': valid_predictions.mean() if len(valid_predictions) > 0 else np.nan,
        'median': np.median(valid_predictions) if len(valid_predictions) > 0 else np.nan,
        'std': valid_predictions.std() if len(valid_predictions) > 0 else np.nan,
        'gpp_min': valid_gpp.min() if len(valid_gpp) > 0 else np.nan,
        'gpp_max': valid_gpp.max() if len(valid_gpp) > 0 else np.nan,
        'gpp_mean': valid_gpp.mean() if len(valid_gpp) > 0 else np.nan
    }

    print(f"\n  Ecologically constrained BNPP_fraction statistics:")
    print(f"    Valid predictions: {stats['n_kept']:,}")
    print(f"    BNPP_fraction range: {stats['min']:.4f} - {stats['max']:.4f}")
    print(f"    BNPP_fraction mean: {stats['mean']:.4f}")
    print(f"    BNPP_fraction median: {stats['median']:.4f}")
    print(f"\n  GPP statistics for valid areas:")
    print(f"    GPP range: {stats['gpp_min']:.1f} - {stats['gpp_max']:.1f} gC m⁻² yr⁻¹")
    print(f"    GPP mean: {stats['gpp_mean']:.1f} gC m⁻² yr⁻¹")

    return constrained_predictions, stats


def create_global_prediction_map(predictions: np.ndarray, coords_df: pd.DataFrame,
                                output_path: str = "/Users/6lw/Desktop/2_models/ADAM/productivity/earth/global_bnpp_fraction_predictions_mlp_gpp_constrained.nc") -> xr.DataArray:
    """
    Create global map of BNPP fraction predictions.

    Args:
        predictions: Array of BNPP fraction predictions (0-1 scale)
        coords_df: DataFrame with lat/lon coordinates
        output_path: Path to save output file

    Returns:
        Global BNPP fraction DataArray
    """
    print("Creating global BNPP fraction prediction map...")

    # Create DataFrame with predictions and coordinates
    result_df = coords_df.copy()
    result_df['bnpp_fraction_predicted'] = predictions
    
    # Define target grid
    lat_bins = np.arange(-90, 90.5, 0.5)
    lon_bins = np.arange(-180, 180.5, 0.5)
    
    # Create empty grid
    bnpp_fraction_grid = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan)

    # Fill grid with predictions
    for _, row in result_df.iterrows():
        lat_idx = np.digitize(row['lat'], lat_bins) - 1
        lon_idx = np.digitize(row['lon'], lon_bins) - 1

        if 0 <= lat_idx < len(lat_bins)-1 and 0 <= lon_idx < len(lon_bins)-1:
            bnpp_fraction_grid[lat_idx, lon_idx] = row['bnpp_fraction_predicted']
    
    # Create DataArray
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2

    bnpp_fraction_da = xr.DataArray(
        bnpp_fraction_grid,
        dims=['lat', 'lon'],
        coords={'lat': lat_centers, 'lon': lon_centers},
        name='bnpp_fraction',
        attrs={
            'units': 'dimensionless (0-1 scale)',
            'long_name': 'BNPP Fraction (BNPP/TNPP)',
            'description': 'Global BNPP fraction predictions from MLP model with GPP-based ecological constraints',
            'model': 'Multi-Layer Perceptron (4 layers: 256-128-64-32)',
            'features': 'TerraClimate climate (aet, pet, ppt, tmax, tmin, vpd) + SoilGrids soil properties (training means)',
            'constraints': 'Filtered areas with GPP ≤ 0 gC m⁻² yr⁻¹ (GLASS satellite data)',
            'constraint_rationale': 'GPP used for filtering only, NOT as model predictor. Removes non-vegetated areas only.',
            'note': 'Values represent the fraction of total NPP allocated belowground. NaN indicates areas with no vegetation (GPP = 0).',
            'valid_range': '0.0 to 1.0'
        }
    )

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bnpp_fraction_da.to_netcdf(output_path)
    print(f"Global BNPP fraction predictions saved to: {output_path}")

    return bnpp_fraction_da


def plot_global_bnpp_map(bnpp_fraction_da: xr.DataArray, save_path: str = "/Users/6lw/Desktop/2_models/ADAM/productivity/earth/global_bnpp_fraction_map_mlp_gpp_constrained.png"):
    """
    Create global map visualization of BNPP fraction predictions.

    Args:
        bnpp_fraction_da: Global BNPP fraction DataArray (0-1 scale)
        save_path: Path to save the plot
    """
    print("\nCreating global BNPP fraction map visualization...")

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    # Plot BNPP fraction data
    im = bnpp_fraction_da.plot(
        ax=ax,
        cmap='RdYlGn',  # Red-Yellow-Green colormap
        vmin=0.0,
        vmax=1.0,  # Fixed range for fractions
        add_colorbar=False
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('BNPP Fraction (BNPP/TNPP)', fontsize=12)

    plt.title('Global BNPP Fraction\nMLP Model - GPP-Based Ecologically Constrained',
              fontsize=14, pad=20)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.tight_layout()

    # Save plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Global BNPP fraction map saved to: {save_path}")
    plt.close()


def main():
    """
    Main function to apply MLP model globally for BNPP fraction prediction.
    """
    try:
        print("="*60)
        print("GLOBAL MLP BNPP FRACTION PREDICTION")
        print("="*60)
        
        # 1. Load trained MLP model
        model, model_data = load_trained_mlp_model()
        # Get feature names from the scaler (the actual features the model was trained with)
        if hasattr(model_data['scaler'], 'feature_names_in_'):
            required_features = list(model_data['scaler'].feature_names_in_)
        else:
            # Fallback: BNPP_fraction model uses these 17 features (not gpp_yearly!)
            required_features = ['aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd',
                                'soil_carbon_stock', 'clay_content', 'silt_content', 'sand_content',
                                'nitrogen_content', 'cation_exchange_capacity', 'ph_in_water',
                                'bulk_density', 'coarse_fragments', 'soil_moisture', 'elevation']
        scaler = model_data['scaler']
        
        # 2. Load global climate data
        climate_ds = load_global_terraclimate_data(year=2010)
        
        # 3. Load GLASS GPP data for ecological filtering
        print("\n[3/9] Loading GLASS GPP data...")
        gpp_da = load_glass_gpp_data()
        
        # 4. Interpolate to common grid
        combined_ds = interpolate_to_common_grid(climate_ds, gpp_da)
        
        # 5. Prepare features for model
        features_df, coords_df, gpp_df = prepare_global_features(combined_ds, required_features)
        
        # 6. Apply MLP model globally
        predictions = apply_mlp_model_globally(model, features_df, scaler)
        
        # 7. Apply GPP-based ecological constraints
        print("\n[7/9] Applying GPP-based ecological constraints...")
        constrained_predictions, stats = apply_ecological_constraints(
            predictions,
            gpp_df,
            gpp_threshold=0.0
        )

        # 8. Create global prediction map
        print("\n[8/9] Creating outputs...")
        bnpp_da = create_global_prediction_map(constrained_predictions, coords_df)

        # 9. Plot global map
        plot_global_bnpp_map(bnpp_da)
        
        # 9. Print summary statistics
        print("\n" + "="*60)
        print("GLOBAL BNPP FRACTION PREDICTION SUMMARY")
        print("="*60)
        print(f"Total valid grid points: {len(constrained_predictions):,}")
        print(f"Global BNPP Fraction statistics:")
        print(f"  Minimum: {constrained_predictions.min():.4f}")
        print(f"  Maximum: {constrained_predictions.max():.4f}")
        print(f"  Mean: {constrained_predictions.mean():.4f}")
        print(f"  Median: {np.median(constrained_predictions):.4f}")
        print(f"  Standard deviation: {constrained_predictions.std():.4f}")

        print("\nGlobal MLP BNPP fraction application completed successfully!")
        print("\nNote: Predictions represent the fraction of total NPP allocated belowground (BNPP/TNPP)")
        print("      Values are constrained to [0, 1] range for physical validity")
        
    except Exception as e:
        print(f"Error in global MLP application: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()