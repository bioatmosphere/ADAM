"""
Apply trained TabPFN model globally using global climate and satellite data.

This script loads the trained TabPFN model and applies it to global gridded data to produce
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
from pathlib import Path
import pickle
import warnings
from typing import Tuple, Dict
try:
    from tabpfn import TabPFNRegressor
except ImportError:
    print("TabPFN not available, using pickle to load model")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_trained_tabpfn_model(model_path: str = "/Users/6lw/Desktop/2_models/ADAM/src/models/tabpfn/tabpfn_model.pkl") -> Tuple[object, Dict]:
    """
    Load the trained TabPFN model.
    
    Args:
        model_path: Path to the saved TabPFN model file
        
    Returns:
        Tuple of (model, model metadata)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        # Try alternative paths
        fallback_paths = [
            Path("../tabpfn/tabpfn_model.pkl"),
            Path("../../models/tabpfn/tabpfn_model.pkl"), 
            Path("../../../src/models/tabpfn/tabpfn_model.pkl"),
            Path("../../../productivity/earth/tabpfn_model.pkl")
        ]
        
        for fallback_path in fallback_paths:
            if fallback_path.exists():
                print(f"Model not found at {model_path}, using: {fallback_path}")
                model_path = fallback_path
                break
        else:
            raise FileNotFoundError(f"Trained TabPFN model not found at {model_path} or fallback locations")
    
    print(f"Loading trained TabPFN model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Model loaded successfully!")
    
    # Extract the actual model from the dictionary
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
        print(f"Extracted model from dictionary. Model type: {type(model)}")
    else:
        model = model_data
        print(f"Model type: {type(model)}")
    
    # TabPFN models may not have feature_names_in_, so we'll use a default set
    # Note: 17 features used for BNPP_fraction prediction (no gpp_yearly)
    expected_features = [
        'aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd',
        'soil_carbon_stock', 'clay_content', 'silt_content', 'sand_content',
        'nitrogen_content', 'cation_exchange_capacity', 'ph_in_water',
        'bulk_density', 'coarse_fragments', 'soil_moisture', 'elevation'
    ]
    
    if hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
    else:
        # Use expected features based on training data
        features = expected_features
        print(f"Using expected feature set: {len(features)} features")
    
    return model, {"features": features}


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


def load_global_soil_data(soil_dir: str = "/Users/6lw/Desktop/2_models/ADAM/ancillary/soilgrids/global",
                          sm_path: str = "/Users/6lw/Desktop/2_models/ADAM/ancillary/soilmoisture/olc_ors.nc",
                          target_lat: np.ndarray = None,
                          target_lon: np.ndarray = None) -> xr.Dataset:
    """
    Load global soil data from GEE exports (OpenLandMap) at 0.5 degree resolution.

    Args:
        soil_dir: Directory containing soil GeoTIFFs from GEE export
        sm_path: Path to soil moisture NetCDF
        target_lat: Target latitude coordinates for interpolation
        target_lon: Target longitude coordinates for interpolation

    Returns:
        xarray Dataset with all soil properties interpolated to target grid
    """
    import rasterio

    soil_dir = Path(soil_dir)
    print("Loading global soil data from GEE exports...")

    # Default target grid if not provided
    if target_lat is None:
        target_lat = np.arange(-89.75, 90, 0.5)
    if target_lon is None:
        target_lon = np.arange(-179.75, 180, 0.5)

    # GEE export files (OpenLandMap format)
    gee_files = {
        'clay_content': 'clay_content_0p5deg.tif',
        'sand_content': 'sand_content_0p5deg.tif',
        'soil_carbon_stock': 'soil_carbon_stock_0p5deg.tif',
        'ph_in_water': 'ph_in_water_0p5deg.tif',
        'bulk_density': 'bulk_density_0p5deg.tif',
    }

    datasets = {}

    for var_name, file_name in gee_files.items():
        file_path = soil_dir / file_name
        if file_path.exists():
            with rasterio.open(file_path) as src:
                data = src.read(1).astype(np.float32)

                # Get geotransform and create source coordinates
                transform = src.transform
                height, width = data.shape
                src_lon = np.array([transform.c + (i + 0.5) * transform.a for i in range(width)])
                src_lat = np.array([transform.f + (j + 0.5) * transform.e for j in range(height)])

                # Handle zero as nodata for some properties
                if var_name in ['bulk_density']:
                    data[data == 0] = np.nan

            # Create DataArray with source coordinates
            da = xr.DataArray(data, dims=['lat', 'lon'],
                             coords={'lat': src_lat, 'lon': src_lon}, name=var_name)

            # Interpolate to target grid (will be NaN outside source bounds)
            da_interp = da.interp(lat=target_lat, lon=target_lon, method='linear')
            datasets[var_name] = da_interp

            valid_data = da_interp.values[~np.isnan(da_interp.values)]
            if len(valid_data) > 0:
                print(f"  Loaded {var_name}: range {valid_data.min():.2f} - {valid_data.max():.2f}, coverage {100*len(valid_data)/da_interp.size:.1f}%")
            else:
                print(f"  Loaded {var_name}: no valid data after interpolation")
        else:
            print(f"  Not found: {file_name}")

    # Calculate silt_content from clay and sand
    if 'clay_content' in datasets and 'sand_content' in datasets:
        silt = 100.0 - datasets['clay_content'] - datasets['sand_content']
        silt = silt.clip(min=0, max=100)
        datasets['silt_content'] = silt
        valid_silt = silt.values[~np.isnan(silt.values)]
        if len(valid_silt) > 0:
            print(f"  Calculated silt_content: range {valid_silt.min():.2f} - {valid_silt.max():.2f}")

    # Load soil moisture
    sm_path = Path(sm_path)
    if sm_path.exists():
        sm_ds = xr.open_dataset(sm_path)
        sm = sm_ds['sm'].isel(depth=0).mean(dim='time')
        # Interpolate to target grid
        sm_interp = sm.interp(lat=target_lat, lon=target_lon, method='linear')
        datasets['soil_moisture'] = sm_interp
        valid_sm = sm_interp.values[~np.isnan(sm_interp.values)]
        if len(valid_sm) > 0:
            print(f"  Loaded soil_moisture: range {valid_sm.min():.2f} - {valid_sm.max():.2f}")
    else:
        print(f"  Not found: soil moisture file")

    missing_props = ['nitrogen_content', 'cation_exchange_capacity', 'coarse_fragments']
    print(f"  Missing (will use training means): {missing_props}")

    combined = xr.Dataset(datasets)
    print(f"Loaded {len(datasets)} spatial soil properties")

    return combined


def load_global_elevation(data_path: str = "/Users/6lw/Desktop/2_models/ADAM/ancillary/elevation/global_elevation_0.5deg.nc") -> xr.DataArray:
    """
    Load global elevation data at 0.5 degree resolution.

    Args:
        data_path: Path to elevation NetCDF file

    Returns:
        Elevation DataArray
    """
    data_path = Path(data_path)

    if not data_path.exists():
        print(f"Warning: Global elevation file not found at {data_path}")
        print("Run: uv run python src/ancillary/download_global_elevation.py")
        return None

    print(f"Loading global elevation data from: {data_path}")
    ds = xr.open_dataset(data_path)

    elev = ds['elevation']

    # Set ocean/negative values to NaN for land-only analysis
    elev = elev.where(elev > 0)

    print(f"Elevation data loaded: {elev.shape}")
    print(f"Elevation range: {float(elev.min()):.1f} to {float(elev.max()):.1f} m")

    return elev


def get_training_data_soil_means() -> dict:
    """
    Load training data to get mean soil property values for global application.
    
    Returns:
        Dictionary with mean soil property values
    """
    # Load the aggregated training data (try cleaned data first)
    training_path_cleaned = "/Users/6lw/Desktop/2_models/ADAM/productivity/earth/aggregated_data_cleaned.csv"
    training_path_original = "/Users/6lw/Desktop/2_models/ADAM/productivity/earth/aggregated_data.csv"
    
    training_path = Path(training_path_cleaned)
    if not training_path.exists():
        training_path = Path(training_path_original)
        if not training_path.exists():
            raise FileNotFoundError(f"Training data not found at {training_path}")
    
    print(f"Loading training data from: {training_path}")
    df = pd.read_csv(training_path)
    
    # Calculate mean values for soil properties (elevation loaded separately as spatial data)
    soil_properties = [
        'soil_carbon_stock', 'clay_content', 'silt_content', 'sand_content',
        'nitrogen_content', 'cation_exchange_capacity', 'ph_in_water',
        'bulk_density', 'coarse_fragments', 'soil_moisture'
    ]
    
    soil_means = {}
    for prop in soil_properties:
        if prop in df.columns:
            soil_means[prop] = df[prop].mean(skipna=True)
    
    print("Using mean soil property values from training data:")
    for prop, value in soil_means.items():
        print(f"  {prop}: {value:.3f}")
    
    return soil_means


def prepare_global_features(dataset: xr.Dataset, required_features: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert global xarray dataset to DataFrame for model prediction.
    For missing soil data, use mean values from training data.
    Applies vegetation mask to exclude non-vegetated areas.

    Args:
        dataset: Combined global dataset
        required_features: List of features required by the model

    Returns:
        Tuple of (features DataFrame, coordinates DataFrame)
    """
    print("Preparing global features for TabPFN model application...")

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

    # Debug: Check for NaN counts per variable
    print(f"  Total grid points: {len(df):,}")
    nan_counts = df.isna().sum()
    for var in dataset.data_vars:
        if nan_counts[var] > 0:
            pct = 100 * nan_counts[var] / len(df)
            print(f"    {var}: {nan_counts[var]:,} NaN ({pct:.1f}%)")

    # Remove rows with NaN in climate variables only (soil NaN will be filled later)
    climate_vars = ['aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd']
    df_clean = df.dropna(subset=climate_vars)

    # Apply vegetation mask to exclude non-vegetated regions and water bodies
    print("Applying vegetation and water mask...")
    initial_count = len(df_clean)

    if initial_count == 0:
        raise ValueError("No valid data points after removing climate NaN values")

    # Mask 1: Non-vegetated/desert areas
    # AET < 100 mm/year typically indicates desert/barren land with minimal vegetation
    # This masks Sahara, Arabian, Gobi, Atacama, and other major deserts
    aet_threshold = 100  # mm/year
    vegetation_mask = df_clean['aet'] > aet_threshold
    print(f"  Using AET threshold: {aet_threshold} mm/year to mask deserts")

    # Mask 2: Water bodies - use soil data as land indicator
    # Lakes/water have no soil data (clay_content is NaN over water)
    if 'clay_content' in df_clean.columns:
        land_mask = df_clean['clay_content'].notna()
    else:
        land_mask = df_clean['elevation'].notna()

    # Combine masks
    combined_mask = vegetation_mask & land_mask
    df_clean = df_clean[combined_mask]

    masked_count = initial_count - len(df_clean)
    print(f"  Masked {masked_count:,} non-vegetated/water grid cells ({100*masked_count/initial_count:.1f}%)")
    print(f"  Remaining land vegetated grid cells: {len(df_clean):,}")

    # Get soil means for filling missing values
    soil_means = get_training_data_soil_means()

    # Check for missing features (not in dataset at all)
    missing_features = [f for f in required_features if f not in df_clean.columns]

    if missing_features:
        print(f"Missing features (adding from training means): {missing_features}")
        for feature in missing_features:
            if feature in soil_means:
                df_clean[feature] = soil_means[feature]
                print(f"  Added {feature} = {soil_means[feature]:.3f}")

    # Fill NaN values in soil properties with training means
    soil_features = [f for f in required_features if f in soil_means and f in df_clean.columns]
    for feature in soil_features:
        nan_count = df_clean[feature].isna().sum()
        if nan_count > 0:
            df_clean[feature] = df_clean[feature].fillna(soil_means[feature])
            print(f"  Filled {nan_count:,} NaN in {feature} with mean {soil_means[feature]:.3f}")

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


def apply_tabpfn_model_globally(model, features_df: pd.DataFrame, batch_size: int = 10000) -> np.ndarray:
    """
    Apply TabPFN model to global features with batch processing for memory efficiency.
    
    Args:
        model: Trained TabPFN model
        features_df: Global features DataFrame
        batch_size: Number of samples to process at once
        
    Returns:
        Array of global BNPP predictions
    """
    print("Applying TabPFN model globally...")
    print(f"Processing {len(features_df)} points in batches of {batch_size}")
    
    # Process in batches to avoid memory issues
    predictions = []
    
    for i in range(0, len(features_df), batch_size):
        batch_end = min(i + batch_size, len(features_df))
        batch_features = features_df.iloc[i:batch_end]
        
        print(f"  Processing batch {i//batch_size + 1}: samples {i} to {batch_end-1}")
        
        # Make predictions for this batch
        batch_predictions = model.predict(batch_features.values)
        predictions.extend(batch_predictions)
    
    predictions = np.array(predictions)

    # Clip predictions to valid range [0, 1] for BNPP_fraction
    predictions = np.clip(predictions, 0, 1)

    print(f"Global predictions complete!")
    print(f"BNPP_fraction range: {predictions.min():.4f} to {predictions.max():.4f}")
    print(f"Mean BNPP_fraction: {predictions.mean():.4f}")

    return predictions


def create_global_prediction_map(predictions: np.ndarray, coords_df: pd.DataFrame,
                                output_path: str = "global_bnpp_fraction_tabpfn.nc") -> xr.DataArray:
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
        name='bnpp_fraction',
        attrs={
            'units': 'fraction (0-1)',
            'long_name': 'BNPP fraction (BNPP/TNPP)',
            'description': 'Global BNPP fraction predictions from TabPFN model',
            'model': 'TabPFN v6.0.5',
            'features': 'TerraClimate + SoilGrids + Elevation (17 features)'
        }
    )
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bnpp_da.to_netcdf(output_path)
    print(f"Global BNPP predictions saved to: {output_path}")
    
    return bnpp_da


def plot_global_bnpp_map(bnpp_da: xr.DataArray, save_path: str = "global_bnpp_fraction_map_tabpfn.png"):
    """
    Create global map visualization of BNPP fraction predictions.

    Args:
        bnpp_da: Global BNPP fraction DataArray
        save_path: Path to save the plot
    """
    print("Creating global BNPP fraction map visualization...")

    # Try to use cartopy for better map visualization
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        use_cartopy = True
    except ImportError:
        use_cartopy = False
        print("  Cartopy not available, using basic plot")

    if use_cartopy:
        # Smaller figure for poster with larger fonts
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

        # Plot BNPP fraction data
        im = ax.pcolormesh(
            bnpp_da.lon, bnpp_da.lat, bnpp_da.values,
            transform=ccrs.PlateCarree(),
            cmap='YlGn',
            vmin=0.2,
            vmax=0.7,
            shading='auto'
        )

        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray')
        ax.set_global()

        # Add colorbar with larger font
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7, aspect=40)
        cbar.set_label('BNPP Fraction', fontsize=20)
        cbar.ax.tick_params(labelsize=16)

        plt.title('Global Belowground NPP Fraction', fontsize=24, pad=10)

    else:
        # Smaller figure for poster with larger fonts
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot BNPP fraction data
        im = bnpp_da.plot(
            ax=ax,
            cmap='YlGn',
            vmin=0.2,
            vmax=0.7,
            add_colorbar=False
        )

        # Add colorbar with larger font
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('BNPP Fraction', fontsize=20)
        cbar.ax.tick_params(labelsize=16)

        plt.title('Global Belowground NPP Fraction', fontsize=24, pad=10)

        ax.set_xlabel('Longitude', fontsize=16)
        ax.set_ylabel('Latitude', fontsize=16)
        ax.tick_params(labelsize=14)

    plt.tight_layout()

    # Save plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Global BNPP map saved to: {save_path}")
    plt.close()


def main():
    """
    Main function to apply TabPFN model globally.
    """
    try:
        print("="*60)
        print("GLOBAL TABPFN BNPP PREDICTION")
        print("="*60)

        # 1. Load trained TabPFN model
        model, model_data = load_trained_tabpfn_model()
        required_features = model_data["features"]

        # 2. Define target grid
        target_lat = np.arange(-89.75, 90, 0.5)
        target_lon = np.arange(-179.75, 180, 0.5)

        # 3. Load global climate data
        climate_ds = load_global_terraclimate_data(year=2010)

        # 4. Load global elevation data
        elevation_da = load_global_elevation()

        # 5. Load global soil data (from GEE exports) - pre-interpolated to target grid
        soil_ds = load_global_soil_data(target_lat=target_lat, target_lon=target_lon)

        # 6. Interpolate climate to common 0.5° grid
        print("Interpolating climate to common 0.5° grid...")
        combined_ds = climate_ds.interp(lat=target_lat, lon=target_lon, method='linear')

        # Add elevation
        if elevation_da is not None:
            elev_interp = elevation_da.interp(lat=target_lat, lon=target_lon, method='linear')
            combined_ds['elevation'] = elev_interp
            print(f"Added spatial elevation data")

        # Add soil properties (already on target grid)
        soil_vars_added = 0
        for var in soil_ds.data_vars:
            combined_ds[var] = soil_ds[var]
            soil_vars_added += 1
        print(f"Added {soil_vars_added} spatial soil properties")

        # Note: Missing soil properties (nitrogen, cec, coarse_fragments) will use training means

        print(f"Combined dataset shape: {len(combined_ds.lat)} x {len(combined_ds.lon)}")
        print(f"Variables: {list(combined_ds.data_vars)}")

        # 5. Prepare features for model
        features_df, coords_df = prepare_global_features(combined_ds, required_features)
        
        # 6. Apply TabPFN model globally (with batch processing)
        # Process full global dataset
        print(f"Processing full global dataset: {len(features_df)} points...")
        predictions = apply_tabpfn_model_globally(model, features_df, batch_size=5000)
        
        # 7. Create global prediction map
        output_dir = Path(__file__).parent
        bnpp_da = create_global_prediction_map(
            predictions, coords_df,
            output_path=output_dir / "global_bnpp_fraction_tabpfn.nc"
        )

        # 8. Plot global map
        plot_global_bnpp_map(bnpp_da, save_path=output_dir / "global_bnpp_fraction_map_tabpfn.png")
        
        # 9. Print summary statistics
        print("\n" + "="*60)
        print("GLOBAL TABPFN BNPP FRACTION PREDICTION SUMMARY")
        print("="*60)
        print(f"Total valid grid points: {len(predictions):,}")
        print(f"Global BNPP fraction statistics:")
        print(f"  Minimum: {predictions.min():.4f}")
        print(f"  Maximum: {predictions.max():.4f}")
        print(f"  Mean: {predictions.mean():.4f}")
        print(f"  Median: {np.median(predictions):.4f}")
        print(f"  Standard deviation: {predictions.std():.4f}")

        print("\nGlobal TabPFN application completed successfully!")
        
    except Exception as e:
        print(f"Error in global TabPFN application: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()