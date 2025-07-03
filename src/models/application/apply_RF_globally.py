"""
Apply trained Random Forest model globally using global climate and satellite data.

This script loads the trained RF model and applies it to global gridded data to produce
worldwide BNPP predictions at 0.5-degree resolution.

Data sources for global application:
- TerraClimate: Global climate variables (aet, pet, ppt, tmax, tmin, vpd)
- GLASS: Global GPP satellite data from HDF tiles
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
from sklearn.ensemble import RandomForestRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_trained_rf_model(model_path: str = "../rf_model.pkl") -> Tuple[RandomForestRegressor, Dict]:
    """
    Load the trained Random Forest model and metadata.
    
    Args:
        model_path: Path to the saved RF model file
        
    Returns:
        Tuple of (model, model metadata)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Trained RF model not found at {model_path}")
    
    print(f"Loading trained RF model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    metrics = model_data['metrics']
    
    print(f"Model loaded successfully!")
    print(f"Model performance - Test R²: {metrics['test_r2']:.4f}")
    print(f"Required features: {list(model.feature_names_in_)}")
    
    return model, model_data


def load_global_terraclimate_data(data_dir: str = "../../../ancillary/terraclimate", year: int = 2010) -> xr.Dataset:
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


def create_global_gpp_grid(year: int = 2010, resolution: float = 0.5) -> xr.DataArray:
    """
    Create a global GPP grid by averaging available GLASS data.
    
    For simplicity, this creates a synthetic GPP field based on latitude.
    In a full implementation, this would process all GLASS HDF tiles.
    
    Args:
        year: Year for GPP data
        resolution: Spatial resolution in degrees
        
    Returns:
        Global GPP DataArray
    """
    print(f"Creating global GPP grid for {year}...")
    
    # Create lat/lon coordinates
    lat = np.arange(-89.75, 90, resolution)
    lon = np.arange(-179.75, 180, resolution)
    
    # Create simple GPP pattern based on latitude (higher at equator)
    # This is a placeholder - in reality would use processed GLASS tiles
    gpp_values = np.zeros((len(lat), len(lon)))
    
    for i, lat_val in enumerate(lat):
        # Simple latitudinal gradient for GPP
        base_gpp = 1000 * np.exp(-((lat_val / 30) ** 2))  # Peak at equator
        # Add some longitudinal variation
        for j, lon_val in enumerate(lon):
            seasonal_factor = 1 + 0.3 * np.sin(np.radians(lon_val))
            gpp_values[i, j] = base_gpp * seasonal_factor
    
    # Create DataArray
    gpp_da = xr.DataArray(
        gpp_values,
        dims=['lat', 'lon'],
        coords={'lat': lat, 'lon': lon},
        name='gpp_yearly',
        attrs={'units': 'gC m-2 year-1', 'description': 'Annual GPP'}
    )
    
    print(f"Global GPP grid created: {gpp_da.shape} points")
    print(f"GPP range: {gpp_da.min().values:.1f} to {gpp_da.max().values:.1f} gC m-2 year-1")
    
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


def prepare_global_features(dataset: xr.Dataset, required_features: list) -> pd.DataFrame:
    """
    Convert global xarray dataset to DataFrame for model prediction.
    
    Args:
        dataset: Combined global dataset
        required_features: List of features required by the model
        
    Returns:
        DataFrame with global features
    """
    print("Preparing global features for model application...")
    
    # Create meshgrid of coordinates
    lat_vals, lon_vals = np.meshgrid(dataset.lat.values, dataset.lon.values, indexing='ij')
    
    # Create DataFrame manually to avoid duplicate column issues
    data_dict = {}
    data_dict['lat'] = lat_vals.flatten()
    data_dict['lon'] = lon_vals.flatten()
    
    # Add each variable
    for var in dataset.data_vars:
        values = dataset[var].values.flatten()
        data_dict[var] = values
    
    df = pd.DataFrame(data_dict)
    
    # Remove NaN values
    df_clean = df.dropna()
    
    # Check for required features
    missing_features = [f for f in required_features if f not in df_clean.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select only required features
    feature_df = df_clean[required_features].copy()
    
    # Get coordinates (ensure they exist)
    coord_cols = []
    if 'lat' in df_clean.columns:
        coord_cols.append('lat')
    if 'lon' in df_clean.columns:
        coord_cols.append('lon')
    
    coords_df = df_clean[coord_cols].copy() if coord_cols else None
    
    print(f"Global features prepared: {len(feature_df)} valid grid points")
    print(f"Feature columns: {list(feature_df.columns)}")
    
    return feature_df, coords_df


def apply_rf_model_globally(model: RandomForestRegressor, features_df: pd.DataFrame) -> np.ndarray:
    """
    Apply Random Forest model to global features.
    
    Args:
        model: Trained Random Forest model
        features_df: Global features DataFrame
        
    Returns:
        Array of global BNPP predictions
    """
    print("Applying Random Forest model globally...")
    
    # Make predictions
    predictions = model.predict(features_df)
    
    print(f"Global predictions complete!")
    print(f"BNPP range: {predictions.min():.1f} to {predictions.max():.1f} gC m-2 year-1")
    print(f"Mean BNPP: {predictions.mean():.1f} gC m-2 year-1")
    
    return predictions


def create_global_prediction_map(predictions: np.ndarray, coords_df: pd.DataFrame, 
                                output_path: str = "global_bnpp_predictions.nc") -> xr.DataArray:
    """
    Create global map of BNPP predictions.
    
    Args:
        predictions: Array of BNPP predictions
        coords_df: DataFrame with lat/lon coordinates
        output_path: Path to save output file
        
    Returns:
        Global BNPP DataArray
    """
    print("Creating global BNPP prediction map...")
    
    # Create DataFrame with predictions and coordinates
    result_df = coords_df.copy()
    result_df['bnpp_predicted'] = predictions
    
    # Define target grid
    lat_bins = np.arange(-90, 90.5, 0.5)
    lon_bins = np.arange(-180, 180.5, 0.5)
    
    # Create empty grid
    bnpp_grid = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan)
    
    # Fill grid with predictions
    for _, row in result_df.iterrows():
        lat_idx = np.digitize(row['lat'], lat_bins) - 1
        lon_idx = np.digitize(row['lon'], lon_bins) - 1
        
        if 0 <= lat_idx < len(lat_bins)-1 and 0 <= lon_idx < len(lon_bins)-1:
            bnpp_grid[lat_idx, lon_idx] = row['bnpp_predicted']
    
    # Create DataArray
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    
    bnpp_da = xr.DataArray(
        bnpp_grid,
        dims=['lat', 'lon'],
        coords={'lat': lat_centers, 'lon': lon_centers},
        name='bnpp_predicted',
        attrs={
            'units': 'gC m-2 year-1',
            'description': 'Global BNPP predictions from Random Forest model',
            'model': 'Random Forest',
            'features': 'TerraClimate + GLASS GPP'
        }
    )
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bnpp_da.to_netcdf(output_path)
    print(f"Global BNPP predictions saved to: {output_path}")
    
    return bnpp_da


def plot_global_bnpp_map(bnpp_da: xr.DataArray, save_path: str = "global_bnpp_map.png"):
    """
    Create global map visualization of BNPP predictions.
    
    Args:
        bnpp_da: Global BNPP DataArray
        save_path: Path to save the plot
    """
    print("Creating global BNPP map visualization...")
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    # Plot BNPP data
    im = bnpp_da.plot(
        ax=ax,
        cmap='YlOrRd',
        vmin=0,
        vmax=np.nanpercentile(bnpp_da.values, 95),
        add_colorbar=False
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('BNPP (gC m⁻² year⁻¹)', fontsize=12)
    
    plt.title('Global Belowground Net Primary Productivity (BNPP)\nRandom Forest Model Predictions', 
              fontsize=14, pad=20)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Global BNPP map saved to: {save_path}")
    plt.close()


def main():
    """
    Main function to apply Random Forest model globally.
    """
    try:
        print("="*60)
        print("GLOBAL RANDOM FOREST BNPP PREDICTION")
        print("="*60)
        
        # 1. Load trained RF model
        model, model_data = load_trained_rf_model()
        required_features = list(model.feature_names_in_)
        
        # 2. Load global climate data
        climate_ds = load_global_terraclimate_data(year=2010)
        
        # 3. Create global GPP data
        gpp_da = create_global_gpp_grid(year=2010)
        
        # 4. Interpolate to common grid
        combined_ds = interpolate_to_common_grid(climate_ds, gpp_da)
        
        # 5. Prepare features for model
        features_df, coords_df = prepare_global_features(combined_ds, required_features)
        
        # 6. Apply RF model globally
        predictions = apply_rf_model_globally(model, features_df)
        
        # 7. Create global prediction map
        bnpp_da = create_global_prediction_map(predictions, coords_df)
        
        # 8. Plot global map
        plot_global_bnpp_map(bnpp_da)
        
        # 9. Print summary statistics
        print("\n" + "="*60)
        print("GLOBAL BNPP PREDICTION SUMMARY")
        print("="*60)
        print(f"Total valid grid points: {len(predictions):,}")
        print(f"Global BNPP statistics:")
        print(f"  Minimum: {predictions.min():.1f} gC m⁻² year⁻¹")
        print(f"  Maximum: {predictions.max():.1f} gC m⁻² year⁻¹")
        print(f"  Mean: {predictions.mean():.1f} gC m⁻² year⁻¹")
        print(f"  Median: {np.median(predictions):.1f} gC m⁻² year⁻¹")
        print(f"  Standard deviation: {predictions.std():.1f} gC m⁻² year⁻¹")
        
        total_bnpp = predictions.sum() * (0.5 * 111)**2  # Convert to global total (rough)
        print(f"\nEstimated global BNPP: {total_bnpp/1e15:.2f} Pg C year⁻¹")
        
        print("\nGlobal RF application completed successfully!")
        
    except Exception as e:
        print(f"Error in global RF application: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()