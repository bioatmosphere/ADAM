#!/usr/bin/env python
"""
Compare CMIP6 nppRoot data across multiple models.

This script loads data from CanESM5, CESM2, and NorESM2-LM and creates
comprehensive comparison plots.

Usage:
    cd src/cmip
    python compare_models.py
"""

import warnings
warnings.filterwarnings('ignore')

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from datetime import datetime
import seaborn as sns
from scipy import interpolate

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelData:
    """Container for model data and metadata."""
    def __init__(self, name, file_pattern, color):
        self.name = name
        self.file_pattern = file_pattern
        self.color = color
        self.ds = None
        self.ds_regridded = None  # For common grid comparison
        self.var_name = None
        self.original_land_fraction = None
        self.common_masked_land_fraction = None

    def load(self, cmip_dir):
        """Load the model data."""
        from glob import glob

        files = sorted(glob(str(cmip_dir / self.file_pattern)))
        if not files:
            raise FileNotFoundError(f"No files found for {self.name}: {self.file_pattern}")

        print(f"Loading {self.name}...")
        if len(files) > 1:
            self.ds = xr.open_mfdataset(files, decode_times=True, combine='by_coords')
        else:
            self.ds = xr.open_dataset(files[0], decode_times=True)

        # Get variable name (exclude bounds)
        self.var_name = [v for v in self.ds.data_vars if not v.endswith('_bnds')][0]

        # Calculate original land fraction
        data_sample = self.ds[self.var_name].isel(time=0)
        self.original_land_fraction = (~np.isnan(data_sample)).sum().values / data_sample.size

        print(f"  ✓ Loaded {len(self.ds.time)} time steps")
        print(f"  ✓ Native grid: {len(self.ds.lat)}×{len(self.ds.lon)}")
        print(f"  ✓ Land coverage: {self.original_land_fraction*100:.1f}%")

    def get_data_annual(self):
        """Get data converted to g C m-2 year-1."""
        return self.ds[self.var_name] * 86400 * 365 * 1000

    def get_temporal_mean(self):
        """Get temporal mean in annual units."""
        return self.get_data_annual().mean(dim='time')

    def get_global_mean_timeseries(self):
        """Get area-weighted global mean time series."""
        weights = np.cos(np.deg2rad(self.ds.lat))
        data_weighted = self.ds[self.var_name].weighted(weights)
        return data_weighted.mean(dim=['lat', 'lon']) * 86400 * 30 * 1000  # monthly

    def get_seasonal_cycle(self):
        """Get monthly climatology."""
        global_mean = self.get_global_mean_timeseries()
        return global_mean.groupby('time.month').mean()

    def get_zonal_mean(self):
        """Get zonal mean."""
        return self.ds[self.var_name].mean(dim=['time', 'lon']) * 86400 * 365 * 1000

    def get_data_annual_regridded(self):
        """Get regridded data converted to g C m-2 year-1."""
        if self.ds_regridded is None:
            raise ValueError(f"Model {self.name} has not been regridded yet")
        return self.ds_regridded[self.var_name] * 86400 * 365 * 1000

    def get_temporal_mean_regridded(self):
        """Get temporal mean from regridded data in annual units."""
        return self.get_data_annual_regridded().mean(dim='time')

    def get_global_mean_timeseries_regridded(self):
        """Get area-weighted global mean time series from regridded data."""
        weights = np.cos(np.deg2rad(self.ds_regridded.lat))
        data_weighted = self.ds_regridded[self.var_name].weighted(weights)
        return data_weighted.mean(dim=['lat', 'lon']) * 86400 * 30 * 1000

    def get_seasonal_cycle_regridded(self):
        """Get monthly climatology from regridded data."""
        global_mean = self.get_global_mean_timeseries_regridded()
        return global_mean.groupby('time.month').mean()

    def get_zonal_mean_regridded(self):
        """Get zonal mean from regridded data."""
        return self.ds_regridded[self.var_name].mean(dim=['time', 'lon']) * 86400 * 365 * 1000


def regrid_to_common_grid(models):
    """Regrid all models to a common grid.

    Uses the coarsest model's grid as the target to minimize interpolation errors.
    """
    print("\n" + "="*80)
    print("REGRIDDING TO COMMON GRID")
    print("="*80)

    # Find coarsest model (fewest grid cells)
    grid_sizes = [(len(m.ds.lat) * len(m.ds.lon), i, m) for i, m in enumerate(models)]
    grid_sizes.sort()
    coarsest_idx = grid_sizes[0][1]
    target_model = models[coarsest_idx]

    target_lat = target_model.ds.lat.values
    target_lon = target_model.ds.lon.values

    print(f"\nTarget grid: {target_model.name} ({len(target_lat)}×{len(target_lon)})")
    print(f"  Latitude range: {target_lat.min():.2f}° to {target_lat.max():.2f}°")
    print(f"  Longitude range: {target_lon.min():.2f}° to {target_lon.max():.2f}°\n")

    for model in models:
        if model.name == target_model.name:
            # No regridding needed for target model
            print(f"{model.name}: Using native grid (no regridding)")
            model.ds_regridded = model.ds.copy()
        else:
            print(f"{model.name}: Regridding from {len(model.ds.lat)}×{len(model.ds.lon)} to {len(target_lat)}×{len(target_lon)}...", end=' ')

            # Regrid using linear interpolation
            model.ds_regridded = model.ds.interp(
                lat=target_lat,
                lon=target_lon,
                method='linear'
            )
            print("✓ Done")

    return target_lat, target_lon


def create_common_mask(models):
    """Create a common land mask as the intersection of all models.

    Returns a boolean mask where True = land in ALL models.
    """
    print("\n" + "="*80)
    print("CREATING COMMON LAND MASK")
    print("="*80)

    # Initialize mask with all True
    first_model = models[0]
    common_mask = xr.ones_like(first_model.ds_regridded[first_model.var_name].isel(time=0), dtype=bool)

    # Take intersection: land must be valid in ALL models
    for model in models:
        # Get land mask for this model (any non-NaN time step indicates land)
        model_mask = ~np.isnan(model.ds_regridded[model.var_name].isel(time=0))
        common_mask = common_mask & model_mask

        land_cells = model_mask.sum().values
        print(f"  {model.name:15s}: {land_cells:5d} land cells ({land_cells/model_mask.size*100:5.1f}%)")

    common_land_cells = common_mask.sum().values
    total_cells = common_mask.size

    print(f"\n  Common mask: {common_land_cells:5d} land cells ({common_land_cells/total_cells*100:5.1f}%)")
    print(f"  Excluded: {total_cells - common_land_cells:5d} cells")

    return common_mask


def apply_common_mask(models, common_mask):
    """Apply common mask to all regridded models."""
    print("\n" + "="*80)
    print("APPLYING COMMON MASK")
    print("="*80)

    for model in models:
        print(f"\n{model.name}:")

        # Apply mask to data (set non-common-land to NaN)
        masked_data = model.ds_regridded[model.var_name].where(common_mask)

        # Count valid data points before and after
        original_valid = (~np.isnan(model.ds_regridded[model.var_name].isel(time=0))).sum().values
        masked_valid = (~np.isnan(masked_data.isel(time=0))).sum().values
        excluded = original_valid - masked_valid

        print(f"  Original valid cells: {original_valid:5d}")
        print(f"  After masking: {masked_valid:5d}")
        print(f"  Excluded: {excluded:5d} cells ({excluded/original_valid*100:5.1f}%)")

        # Update dataset with masked data
        model.ds_regridded[model.var_name] = masked_data

        # Store land fraction after common masking
        model.common_masked_land_fraction = masked_valid / masked_data.isel(time=0).size


def plot_masking_comparison(models, common_mask, output_dir):
    """Visualize the effect of common masking."""
    print("\nGenerating masking comparison plot...")

    fig = plt.figure(figsize=(18, 12))

    n_models = len(models)

    # Plot original masks
    for i, model in enumerate(models, 1):
        ax = plt.subplot(3, n_models, i, projection=ccrs.Robinson())

        # Original land mask
        data = model.ds[model.var_name].isel(time=0)
        mask = ~np.isnan(data)

        im = ax.pcolormesh(
            model.ds.lon, model.ds.lat, mask.astype(float),
            transform=ccrs.PlateCarree(),
            cmap='RdYlGn',
            vmin=0, vmax=1
        )

        ax.coastlines(linewidth=0.5)
        ax.set_title(f'{model.name}\nOriginal Land Mask\n{mask.sum().values} cells',
                    fontsize=10, fontweight='bold')

    # Plot regridded masks
    for i, model in enumerate(models, 1):
        ax = plt.subplot(3, n_models, n_models + i, projection=ccrs.Robinson())

        # Regridded land mask
        data_regrid = model.ds_regridded[model.var_name].isel(time=0)
        mask_regrid = ~np.isnan(data_regrid)

        im = ax.pcolormesh(
            model.ds_regridded.lon, model.ds_regridded.lat, mask_regrid.astype(float),
            transform=ccrs.PlateCarree(),
            cmap='RdYlGn',
            vmin=0, vmax=1
        )

        ax.coastlines(linewidth=0.5)
        ax.set_title(f'Regridded to Common Grid\n{mask_regrid.sum().values} cells',
                    fontsize=10, fontweight='bold')

    # Plot common mask
    ax = plt.subplot(3, n_models, 2*n_models + 2, projection=ccrs.Robinson())

    im = ax.pcolormesh(
        models[0].ds_regridded.lon, models[0].ds_regridded.lat, common_mask.astype(float),
        transform=ccrs.PlateCarree(),
        cmap='RdYlGn',
        vmin=0, vmax=1
    )

    ax.coastlines(linewidth=1.5, edgecolor='blue')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
    ax.set_title(f'Common Land Mask\n(Intersection of All Models)\n{common_mask.sum().values} cells',
                fontsize=11, fontweight='bold')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Land (1 = land, 0 = ocean/excluded)', fontsize=10)

    plt.suptitle('Common Masking Procedure\nTop: Original Native Grids | Middle: Regridded | Bottom: Common Mask',
                 fontsize=14, fontweight='bold', y=0.98)

    output_path = output_dir / 'model_comparison_masking.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_spatial_comparison(models, output_dir, use_common_mask=False):
    """Create side-by-side spatial comparison."""
    print(f"\nGenerating spatial comparison (common mask: {use_common_mask})...")

    fig = plt.figure(figsize=(18, 5))

    # Find common color scale
    if use_common_mask:
        vmax = max([np.nanpercentile(m.get_temporal_mean_regridded(), 95) for m in models])
    else:
        vmax = max([np.nanpercentile(m.get_temporal_mean(), 95) for m in models])

    for i, model in enumerate(models, 1):
        ax = plt.subplot(1, 3, i, projection=ccrs.Robinson())

        if use_common_mask:
            data = model.get_temporal_mean_regridded()
            lon = model.ds_regridded.lon
            lat = model.ds_regridded.lat
        else:
            data = model.get_temporal_mean()
            lon = model.ds.lon
            lat = model.ds.lat

        im = ax.pcolormesh(
            lon, lat, data,
            transform=ccrs.PlateCarree(),
            cmap='YlGn',
            vmin=0,
            vmax=vmax
        )

        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)

        # Get global mean
        if use_common_mask:
            weights = np.cos(np.deg2rad(model.ds_regridded.lat))
            data_weighted = model.ds_regridded[model.var_name].weighted(weights)
            resolution = f"{len(model.ds_regridded.lat)}×{len(model.ds_regridded.lon)} (common)"
        else:
            weights = np.cos(np.deg2rad(model.ds.lat))
            data_weighted = model.ds[model.var_name].weighted(weights)
            resolution = f"{len(model.ds.lat)}×{len(model.ds.lon)} (native)"

        global_mean = float(data_weighted.mean() * 86400 * 365 * 1000)

        ax.set_title(f'{model.name}\nGlobal Mean: {global_mean:.1f} g C m$^{{-2}}$ yr$^{{-1}}$\n{resolution}',
                    fontsize=11, fontweight='bold')

    # Add colorbar
    cbar_ax = fig.add_axes([0.2, 0.12, 0.6, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'Temporal Mean Root NPP (g C m$^{-2}$ year$^{-1}$)', fontsize=12, fontweight='bold')

    mask_status = " (Common Mask Applied)" if use_common_mask else " (Native Grids)"
    plt.suptitle(f'CMIP6 Historical Root NPP: Model Comparison (1850-2014){mask_status}',
                 fontsize=14, fontweight='bold', y=0.98)

    suffix = "_common_mask" if use_common_mask else ""
    output_path = output_dir / f'model_comparison_spatial{suffix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_seasonal_comparison(models, output_dir, use_common_mask=False):
    """Compare seasonal cycles."""
    print(f"\nGenerating seasonal cycle comparison (common mask: {use_common_mask})...")

    fig, ax = plt.subplots(figsize=(12, 7))

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for model in models:
        if use_common_mask:
            seasonal = model.get_seasonal_cycle_regridded()
        else:
            seasonal = model.get_seasonal_cycle()

        ax.plot(seasonal.month, seasonal.values,
               marker='o', linewidth=2.5, markersize=8,
               label=model.name, color=model.color, alpha=0.8)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Root NPP (g C m$^{-2}$ month$^{-1}$)', fontsize=12, fontweight='bold')

    mask_status = " (Common Mask Applied)" if use_common_mask else " (Native Grids)"
    ax.set_title(f'Global Mean Seasonal Cycle Comparison{mask_status}',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    suffix = "_common_mask" if use_common_mask else ""
    output_path = output_dir / f'model_comparison_seasonal{suffix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_timeseries_comparison(models, output_dir):
    """Compare global mean time series."""
    print("\nGenerating time series comparison...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Full time series
    for model in models:
        ts = model.get_global_mean_timeseries()
        time_vals = np.arange(len(ts))
        ax1.plot(time_vals, ts.values, linewidth=1.0,
                label=model.name, color=model.color, alpha=0.7)

        # Add 12-year rolling mean
        rolling = ts.rolling(time=144, center=True).mean()
        ax1.plot(time_vals, rolling.values, linewidth=2.5,
                color=model.color, alpha=1.0)

    ax1.set_xlabel('Time Step (months since 1850-01)', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'Root NPP (g C m$^{-2}$ month$^{-1}$)', fontsize=11, fontweight='bold')
    ax1.set_title('Global Mean Root NPP Time Series (1850-2014)\nThin lines: monthly values, Thick lines: 12-year rolling mean',
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)

    # Decadal means
    for model in models:
        ts = model.get_global_mean_timeseries()
        # Compute decadal means
        years = np.arange(1850, 2015, 10)
        decadal_means = []
        for year in years:
            if year < 2014:
                start_idx = (year - 1850) * 12
                end_idx = start_idx + 120  # 10 years
                decadal_means.append(ts[start_idx:end_idx].mean().values)
            else:
                start_idx = (year - 1850) * 12
                decadal_means.append(ts[start_idx:].mean().values)

        ax2.plot(years, decadal_means, marker='o', markersize=8,
                linewidth=2.5, label=model.name, color=model.color, alpha=0.8)

    ax2.set_xlabel('Decade', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'Root NPP (g C m$^{-2}$ month$^{-1}$)', fontsize=11, fontweight='bold')
    ax2.set_title('Decadal Mean Root NPP', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / 'model_comparison_timeseries.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_zonal_comparison(models, output_dir, use_common_mask=False):
    """Compare zonal means."""
    print(f"\nGenerating zonal mean comparison (common mask: {use_common_mask})...")

    fig, ax = plt.subplots(figsize=(10, 8))

    for model in models:
        if use_common_mask:
            zonal = model.get_zonal_mean_regridded()
            lat = model.ds_regridded.lat
        else:
            zonal = model.get_zonal_mean()
            lat = model.ds.lat

        ax.plot(zonal, lat, linewidth=2.5,
               label=model.name, color=model.color, alpha=0.8)

    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=23.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, label='Tropics')
    ax.axhline(y=-23.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhline(y=66.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, label='Polar circles')
    ax.axhline(y=-66.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    ax.set_xlabel(r'Root NPP (g C m$^{-2}$ year$^{-1}$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')

    mask_status = " (Common Mask Applied)" if use_common_mask else " (Native Grids)"
    ax.set_title(f'Zonal Mean Profile Comparison{mask_status}',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-90, 90)

    suffix = "_common_mask" if use_common_mask else ""
    output_path = output_dir / f'model_comparison_zonal{suffix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_distribution_comparison(models, output_dir):
    """Compare statistical distributions."""
    print("\nGenerating distribution comparison...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    for model in models:
        data_annual = model.get_data_annual()
        valid_data = data_annual.values[~np.isnan(data_annual.values)]

        ax1.hist(valid_data, bins=100, alpha=0.5, label=model.name,
                color=model.color, edgecolor='none', density=True)

    ax1.set_xlabel(r'Root NPP (g C m$^{-2}$ year$^{-1}$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Root NPP Values', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 500)

    # Box plot
    box_data = []
    labels = []
    colors = []
    for model in models:
        data_annual = model.get_data_annual()
        valid_data = data_annual.values[~np.isnan(data_annual.values)]
        # Sample to avoid memory issues
        if len(valid_data) > 100000:
            valid_data = np.random.choice(valid_data, 100000, replace=False)
        box_data.append(valid_data)
        labels.append(model.name)
        colors.append(model.color)

    bp = ax2.boxplot(box_data, labels=labels, patch_artist=True,
                     showfliers=False, widths=0.6)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel(r'Root NPP (g C m$^{-2}$ year$^{-1}$)', fontsize=12, fontweight='bold')
    ax2.set_title('Statistical Distribution Comparison', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 500)

    plt.tight_layout()

    output_path = output_dir / 'model_comparison_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def write_comparison_summary(models, output_dir):
    """Write text summary of comparison."""
    print("\nGenerating comparison summary...")

    output_path = output_dir / 'model_comparison_summary.txt'

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CMIP6 NPPROOT MODEL COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Models compared: {', '.join([m.name for m in models])}\n")
        f.write(f"Time period: 1850-2014 (165 years, 1980 months)\n\n")

        f.write("="*80 + "\n")
        f.write("MODEL CHARACTERISTICS\n")
        f.write("="*80 + "\n\n")

        for model in models:
            f.write(f"{model.name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Institution: {model.ds.attrs.get('institution', 'N/A')}\n")
            f.write(f"Variant: {model.ds.attrs.get('variant_label', 'N/A')}\n")
            f.write(f"Grid: {len(model.ds.lat)} × {len(model.ds.lon)} ({model.ds.attrs.get('nominal_resolution', 'N/A')})\n")
            f.write(f"Time steps: {len(model.ds.time)}\n")

            # Global statistics
            weights = np.cos(np.deg2rad(model.ds.lat))
            data_weighted = model.ds[model.var_name].weighted(weights)
            global_mean = float(data_weighted.mean() * 86400 * 365 * 1000)
            global_std = float(data_weighted.std() * 86400 * 365 * 1000)

            f.write(f"Global mean (area-weighted): {global_mean:.2f} g C m⁻² year⁻¹\n")
            f.write(f"Global std dev: {global_std:.2f} g C m⁻² year⁻¹\n")

            # Temporal trend
            global_ts = data_weighted.mean(dim=['lat', 'lon'])
            x = np.arange(len(global_ts))
            y = global_ts.values
            slope = np.polyfit(x, y, 1)[0]
            trend_annual = slope * 12 * 86400 * 365 * 1000
            f.write(f"Temporal trend: {trend_annual:+.3f} g C m⁻² year⁻²\n")

            f.write("\n")

        f.write("="*80 + "\n")
        f.write("RELATIVE COMPARISONS\n")
        f.write("="*80 + "\n\n")

        # Get all global means
        means = []
        for model in models:
            weights = np.cos(np.deg2rad(model.ds.lat))
            data_weighted = model.ds[model.var_name].weighted(weights)
            global_mean = float(data_weighted.mean() * 86400 * 365 * 1000)
            means.append(global_mean)

        f.write("Global Mean Root NPP (g C m⁻² year⁻¹):\n")
        for model, mean in zip(models, means):
            f.write(f"  {model.name:20s}: {mean:8.2f}\n")

        f.write(f"\nRange: {min(means):.2f} - {max(means):.2f} g C m⁻² year⁻¹\n")
        f.write(f"Spread: {max(means) - min(means):.2f} g C m⁻² year⁻¹ ({(max(means)/min(means) - 1)*100:.1f}% variation)\n")
        f.write(f"Mean across models: {np.mean(means):.2f} ± {np.std(means):.2f} g C m⁻² year⁻¹\n")

        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*80 + "\n\n")

        ratio = max(means) / min(means)
        if ratio > 2:
            f.write(f"⚠ LARGE MODEL SPREAD: The highest estimate is {ratio:.2f}× the lowest.\n")
            f.write("This indicates substantial uncertainty in root NPP representation across models.\n")
        elif ratio > 1.5:
            f.write(f"⚠ MODERATE MODEL SPREAD: The highest estimate is {ratio:.2f}× the lowest.\n")
            f.write("This suggests considerable differences in root allocation schemes.\n")
        else:
            f.write(f"✓ GOOD MODEL AGREEMENT: The highest estimate is {ratio:.2f}× the lowest.\n")
            f.write("Models show reasonable agreement in root NPP estimates.\n")

        f.write("\n")
        f.write("Key differences likely stem from:\n")
        f.write("  1. Different land surface model formulations (CLM5, CLASS-CTEM)\n")
        f.write("  2. Carbon allocation schemes (fixed vs dynamic root allocation)\n")
        f.write("  3. Spatial resolution and representation of vegetation\n")
        f.write("  4. Climate forcing and CO2 response parameterizations\n")

        f.write("\n" + "="*80 + "\n")

    print(f"Saved: {output_path}")


def main():
    print("="*80)
    print("CMIP6 MODEL COMPARISON TOOL")
    print("="*80)

    # Define models
    models = [
        ModelData('CanESM5',
                 'nppRoot_Lmon_CanESM5_historical_r9i1p2f1_gn_185001-201412.nc',
                 '#2E86AB'),  # Blue
        ModelData('CESM2',
                 'nppRoot_Lmon_CESM2_historical_r1i1p1f1_gn_*.nc',
                 '#A23B72'),  # Purple
        ModelData('NorESM2-LM',
                 'nppRoot_Lmon_NorESM2-LM_historical_r1i1p1f1_gn_*.nc',
                 '#F18F01'),  # Orange
    ]

    # Setup paths
    cmip_dir = Path('../../cmip')
    output_dir = cmip_dir / 'figures' / 'model_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all models
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    for model in models:
        try:
            model.load(cmip_dir)
        except Exception as e:
            print(f"ERROR loading {model.name}: {e}")
            return

    # Apply common masking
    regrid_to_common_grid(models)
    common_mask = create_common_mask(models)
    apply_common_mask(models, common_mask)

    # Generate comparison plots
    print("\n" + "="*80)
    print("GENERATING COMPARISONS")
    print("="*80)

    # Visualize masking procedure
    plot_masking_comparison(models, common_mask, output_dir)

    # Generate both native and common-masked comparisons
    plot_spatial_comparison(models, output_dir, use_common_mask=False)
    plot_spatial_comparison(models, output_dir, use_common_mask=True)
    plot_seasonal_comparison(models, output_dir, use_common_mask=False)
    plot_seasonal_comparison(models, output_dir, use_common_mask=True)
    plot_timeseries_comparison(models, output_dir)
    plot_zonal_comparison(models, output_dir, use_common_mask=False)
    plot_zonal_comparison(models, output_dir, use_common_mask=True)
    plot_distribution_comparison(models, output_dir)
    write_comparison_summary(models, output_dir)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nAll figures saved to: {output_dir}")
    print("\nGenerated files:")
    print("  • model_comparison_masking.png               - Masking visualization")
    print("  • model_comparison_spatial.png               - Side-by-side (native grids)")
    print("  • model_comparison_spatial_common_mask.png   - Side-by-side (common mask)")
    print("  • model_comparison_seasonal.png              - Seasonal cycle (native grids)")
    print("  • model_comparison_seasonal_common_mask.png  - Seasonal cycle (common mask)")
    print("  • model_comparison_timeseries.png            - Time series comparison")
    print("  • model_comparison_zonal.png                 - Zonal mean profiles (native grids)")
    print("  • model_comparison_zonal_common_mask.png     - Zonal mean profiles (common mask)")
    print("  • model_comparison_distribution.png          - Statistical distributions")
    print("  • model_comparison_summary.txt               - Text summary")
    print("="*80)


if __name__ == '__main__':
    main()
