"""
Apply trained Random Forest model globally to predict BNPP fraction.

This script applies the BNPP_fraction (BNPP/TNPP ratio) prediction model globally
using environmental predictors from TerraClimate and SoilGrids.

Data sources for global application:
- TerraClimate: Global climate variables (aet, pet, ppt, tmax, tmin, vpd)
- SoilGrids: Training data means for soil properties
- Output: Global BNPP_fraction predictions at 0.5-degree resolution

Target: BNPP_fraction (0-1 scale, representing fraction of NPP allocated belowground)

Author: TAM Development Team
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import pickle
import warnings
from typing import Tuple, Dict
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')


def load_trained_rf_model(model_path: str = "../random_forest_bnpp_fraction/rf_model.pkl") -> RandomForestRegressor:
    """
    Load the trained Random Forest BNPP_fraction model.

    Args:
        model_path: Path to the saved RF model file

    Returns:
        Trained Random Forest model
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Trained RF model not found at {model_path}")

    print(f"Loading trained RF BNPP_fraction model from: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Required features: {list(model.feature_names_in_)}")

    return model


def load_global_terraclimate_data(data_dir: str = "../../../ancillary/terraclimate",
                                 year: int = 2010) -> xr.Dataset:
    """
    Load global TerraClimate data for specified year.

    Args:
        data_dir: Directory containing TerraClimate netCDF files
        year: Year to load data for

    Returns:
        xarray Dataset with all climate variables
    """
    data_dir = Path(data_dir)

    print(f"\nLoading TerraClimate data for {year}...")

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


def load_glass_gpp_data(gpp_file: str = "../../../ancillary/glass/global_gpp_yearly_2010_0.5deg_FIXED.nc") -> xr.DataArray:
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
        print("cd ../../ancillary")
        print("python export_glass_to_netcdf.py --year 2010")
        print("=" * 60)
        raise FileNotFoundError(f"GLASS GPP NetCDF not found. Please run export_glass_to_netcdf.py first.")

    print(f"\nLoading GLASS GPP data from: {gpp_path}")
    gpp_da = xr.open_dataarray(gpp_path)
    print(f"  GPP data shape: {gpp_da.shape}")
    print(f"  GPP range: {gpp_da.min().values:.1f} to {gpp_da.max().values:.1f} gC m⁻² yr⁻¹")
    print(f"  GPP mean: {gpp_da.mean().values:.1f} gC m⁻² yr⁻¹")

    return gpp_da


def interpolate_to_common_grid(climate_ds: xr.Dataset,
                               gpp_da: xr.DataArray,
                               target_resolution: float = 0.5) -> xr.Dataset:
    """
    Interpolate climate and GPP data to a common grid.

    Args:
        climate_ds: TerraClimate dataset
        gpp_da: GLASS GPP DataArray
        target_resolution: Target resolution in degrees

    Returns:
        Combined dataset on common grid
    """
    print(f"\nInterpolating to common {target_resolution}° grid...")

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

    print(f"  Combined dataset shape: {len(combined.lat)} x {len(combined.lon)}")
    print(f"  Variables: {list(combined.data_vars)}")

    return combined


def get_training_data_soil_means() -> dict:
    """
    Load training data to get mean soil property values for global application.

    Returns:
        Dictionary with mean soil property values
    """
    # Load the aggregated training data
    training_path = Path("../../../productivity/earth/aggregated_data.csv")

    if not training_path.exists():
        raise FileNotFoundError(f"Training data not found at {training_path}")

    print(f"\nLoading training data from: {training_path}")
    df = pd.read_csv(training_path)

    # Calculate mean values for soil properties and other variables
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

    # Create DataFrame
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


def apply_rf_model_globally(model: RandomForestRegressor, features_df: pd.DataFrame) -> np.ndarray:
    """
    Apply Random Forest model to global features.

    Args:
        model: Trained Random Forest model
        features_df: Global features DataFrame

    Returns:
        Array of global BNPP_fraction predictions
    """
    print("\nApplying Random Forest model globally...")

    # Make predictions
    predictions = model.predict(features_df)

    # Clip predictions to valid range [0, 1]
    predictions = np.clip(predictions, 0, 1)

    print(f"  Global predictions complete (before ecological filtering)!")
    print(f"  BNPP_fraction range: {predictions.min():.4f} to {predictions.max():.4f}")
    print(f"  Mean BNPP_fraction: {predictions.mean():.4f}")
    print(f"  Median BNPP_fraction: {np.median(predictions):.4f}")

    return predictions


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
                                output_path: str = "../../../productivity/earth/global_bnpp_fraction_predictions_rf.nc") -> xr.DataArray:
    """
    Create global map of BNPP_fraction predictions.

    Args:
        predictions: Array of BNPP_fraction predictions
        coords_df: DataFrame with lat/lon coordinates
        output_path: Path to save output file

    Returns:
        Global BNPP_fraction DataArray
    """
    print("\nCreating global BNPP_fraction prediction map...")

    # Create DataFrame with predictions and coordinates
    result_df = coords_df.copy()
    result_df['bnpp_fraction'] = predictions

    # Define target grid
    lat_bins = np.arange(-90, 90.5, 0.5)
    lon_bins = np.arange(-180, 180.5, 0.5)

    # Create empty grid
    bnpp_frac_grid = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan)

    # Fill grid with predictions
    for _, row in result_df.iterrows():
        lat_idx = np.digitize(row['lat'], lat_bins) - 1
        lon_idx = np.digitize(row['lon'], lon_bins) - 1

        if 0 <= lat_idx < len(lat_bins)-1 and 0 <= lon_idx < len(lon_bins)-1:
            bnpp_frac_grid[lat_idx, lon_idx] = row['bnpp_fraction']

    # Create DataArray
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2

    bnpp_frac_da = xr.DataArray(
        bnpp_frac_grid,
        dims=['lat', 'lon'],
        coords={'lat': lat_centers, 'lon': lon_centers},
        name='bnpp_fraction',
        attrs={
            'units': 'dimensionless (0-1)',
            'description': 'Global BNPP fraction (BNPP/TNPP) predictions from Random Forest model with GPP-based ecological constraints',
            'model': 'Random Forest BNPP_fraction',
            'features': 'TerraClimate climate (aet, pet, ppt, tmax, tmin, vpd) + SoilGrids soil properties (training means)',
            'constraints': 'Filtered areas with GPP ≤ 0 gC m⁻² yr⁻¹ (GLASS satellite data)',
            'constraint_rationale': 'GPP used for filtering only, NOT as model predictor. Removes non-vegetated areas only.',
            'note': 'Values represent the fraction of total NPP allocated belowground. NaN indicates areas with no vegetation (GPP = 0).'
        }
    )

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bnpp_frac_da.to_netcdf(output_path)
    print(f"  Global BNPP_fraction predictions saved to: {output_path}")

    return bnpp_frac_da


def plot_global_bnpp_fraction_map(bnpp_frac_da: xr.DataArray,
                                 save_path: str = "../../../productivity/earth/global_bnpp_fraction_map_rf.png"):
    """
    Create global map visualization of BNPP_fraction predictions.

    Args:
        bnpp_frac_da: Global BNPP_fraction DataArray
        save_path: Path to save the plot
    """
    print("\nCreating global BNPP_fraction map visualization...")

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', alpha=0.5)
    ax.set_global()

    # Plot BNPP_fraction data
    im = bnpp_frac_da.plot(
        ax=ax,
        cmap='RdYlGn_r',  # Reversed: higher fractions (more belowground allocation) in red
        vmin=0,
        vmax=1,
        transform=ccrs.PlateCarree(),
        add_colorbar=False
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7, aspect=40)
    cbar.set_label('BNPP Fraction', fontsize=12, fontweight='bold')

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    plt.title('Global BNPP Fraction\nRandom Forest Model - Ecologically Constrained',
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Global BNPP_fraction map saved to: {save_path}")
    plt.close()


def main():
    """
    Main function to apply Random Forest BNPP_fraction model globally with GPP-based ecological constraints.
    """
    try:
        print("="*70)
        print("GLOBAL RANDOM FOREST BNPP_FRACTION PREDICTION")
        print("WITH GPP-BASED ECOLOGICAL CONSTRAINTS")
        print("="*70)

        # Configuration
        GPP_THRESHOLD = 0.0  # Minimum GPP for valid prediction (gC m⁻² yr⁻¹) - exclude only non-vegetated areas
        YEAR = 2010  # Year for climate and GPP data

        # 1. Load trained RF model
        print("\n[1/8] Loading trained model...")
        model = load_trained_rf_model()
        required_features = list(model.feature_names_in_)
        print(f"  Note: GPP is used for filtering only, NOT as a model feature")

        # 2. Load global climate data
        print("\n[2/8] Loading global climate data...")
        climate_ds = load_global_terraclimate_data(year=YEAR)

        # 3. Load GLASS GPP data for ecological filtering
        print("\n[3/8] Loading GLASS GPP data...")
        gpp_da = load_glass_gpp_data()

        # 4. Interpolate to common grid
        print("\n[4/8] Interpolating data to common grid...")
        combined_ds = interpolate_to_common_grid(climate_ds, gpp_da)

        # 5. Prepare features for model (GPP kept separate for filtering)
        print("\n[5/8] Preparing global features...")
        features_df, coords_df, gpp_df = prepare_global_features(combined_ds, required_features)

        # 6. Apply RF model globally
        print("\n[6/8] Applying Random Forest model...")
        predictions = apply_rf_model_globally(model, features_df)

        # 7. Apply GPP-based ecological constraints
        print("\n[7/8] Applying GPP-based ecological constraints...")
        constrained_predictions, stats = apply_ecological_constraints(
            predictions,
            gpp_df,
            gpp_threshold=GPP_THRESHOLD
        )

        # 8. Create global prediction map
        print("\n[8/8] Creating outputs...")
        bnpp_frac_da = create_global_prediction_map(
            constrained_predictions,
            coords_df,
            output_path="../../../productivity/earth/global_bnpp_fraction_predictions_rf_gpp_constrained.nc"
        )

        # Plot global map
        plot_global_bnpp_fraction_map(
            bnpp_frac_da,
            save_path="../../../productivity/earth/global_bnpp_fraction_map_rf_gpp_constrained.png"
        )

        # Print summary statistics
        print("\n" + "="*70)
        print("GLOBAL BNPP_FRACTION PREDICTION SUMMARY (GPP-CONSTRAINED)")
        print("="*70)
        print(f"Total grid points processed: {stats['n_total']:,}")
        print(f"Valid predictions (GPP > {GPP_THRESHOLD}): {stats['n_kept']:,} ({stats['n_kept']/stats['n_total']*100:.1f}%)")
        print(f"Filtered (GPP ≤ {GPP_THRESHOLD}): {stats['n_filtered']:,} ({stats['filter_percentage']:.1f}%)")

        print(f"\nGlobal BNPP_fraction statistics (productive areas only):")
        print(f"  Minimum: {stats['min']:.4f}")
        print(f"  Maximum: {stats['max']:.4f}")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Standard deviation: {stats['std']:.4f}")

        # Percentiles on valid data
        valid_predictions = constrained_predictions[~np.isnan(constrained_predictions)]
        print(f"\nPercentiles (productive areas only):")
        print(f"  25th: {np.percentile(valid_predictions, 25):.4f}")
        print(f"  50th (median): {np.percentile(valid_predictions, 50):.4f}")
        print(f"  75th: {np.percentile(valid_predictions, 75):.4f}")
        print(f"  95th: {np.percentile(valid_predictions, 95):.4f}")

        print(f"\nGPP range in productive areas:")
        print(f"  Minimum: {stats['gpp_min']:.1f} gC m⁻² yr⁻¹")
        print(f"  Maximum: {stats['gpp_max']:.1f} gC m⁻² yr⁻¹")
        print(f"  Mean: {stats['gpp_mean']:.1f} gC m⁻² yr⁻¹")

        print("\n" + "="*70)
        print("✓ Global RF BNPP_fraction application completed successfully!")
        print("="*70)

        print("\nOutput files created:")
        print("  - productivity/earth/global_bnpp_fraction_predictions_rf_gpp_constrained.nc")
        print("  - productivity/earth/global_bnpp_fraction_map_rf_gpp_constrained.png")
        print("\nEcological filter applied:")
        print(f"  - Excluded areas with GPP ≤ {GPP_THRESHOLD} gC m⁻² yr⁻¹ (non-vegetated areas only)")
        print(f"  - This removes: Ice sheets, permanent snow, barren rock, water bodies")
        print(f"  - Includes: All vegetated areas including low-productivity ecosystems")
        print(f"  - GPP used for filtering only, NOT as a model predictor")

    except Exception as e:
        print(f"\n✗ Error in global RF application: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
