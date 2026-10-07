#!/usr/bin/env python
"""
Visualize CMIP6 NetCDF data from the cmip directory.

This script provides comprehensive visualization of CMIP6 climate model outputs,
specifically designed for root NPP (Net Primary Production) data.

Usage:
    cd src/cmip
    python visualize_cmip.py --help
    python visualize_cmip.py --all
    python visualize_cmip.py --temporal-mean
    python visualize_cmip.py --time-series --lat 45 --lon -75
    python visualize_cmip.py --animation
"""

import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation, PillowWriter
from datetime import datetime


def load_cmip_data(file_path):
    """Load CMIP6 NetCDF file(s) and return dataset.

    Supports both single files and wildcards for multiple files.
    Multiple files will be concatenated along the time dimension.
    """
    from glob import glob

    print(f"Loading {file_path}...")

    # Check if file_path contains wildcards or multiple files
    if '*' in str(file_path) or '?' in str(file_path):
        files = sorted(glob(str(file_path)))
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {file_path}")
        print(f"Found {len(files)} files to process")
        # Use open_mfdataset for multiple files
        ds = xr.open_mfdataset(files, decode_times=True, combine='by_coords')
    else:
        # Single file
        ds = xr.open_dataset(file_path, decode_times=True)

    print(f"Dataset loaded successfully!")
    print(f"\nDataset info:")
    print(f"  Variable: {list(ds.data_vars)}")
    print(f"  Time steps: {len(ds.time)}")
    print(f"  Spatial grid: {len(ds.lat)} x {len(ds.lon)}")
    print(f"  Date range: {ds.time[0].dt.strftime('%Y-%m').values} to {ds.time[-1].dt.strftime('%Y-%m').values}")
    return ds


def plot_temporal_mean(ds, var_name, output_dir):
    """Plot temporal mean of the variable."""
    print("\nGenerating temporal mean map...")

    # Calculate temporal mean
    data_mean = ds[var_name].mean(dim='time')

    # Convert units from kg/m2/s to g/m2/year for better readability
    data_mean_annual = data_mean * 86400 * 365 * 1000  # seconds/day * days/year * g/kg

    # Create figure
    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.Robinson())

    # Plot data
    im = ax.pcolormesh(
        ds.lon, ds.lat, data_mean_annual,
        transform=ccrs.PlateCarree(),
        cmap='YlGn',
        vmin=0,
        vmax=np.nanpercentile(data_mean_annual, 95)
    )

    # Add features
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    cbar.set_label(r'Root NPP (g C m$^{-2}$ year$^{-1}$)', fontsize=12)

    # Title
    model_info = ds.attrs.get('source_id', 'Unknown')
    experiment = ds.attrs.get('experiment_id', 'Unknown')
    plt.title(f'Temporal Mean Root NPP\n{model_info} - {experiment}',
              fontsize=14, fontweight='bold', pad=20)

    # Save figure
    output_path = output_dir / 'cmip_temporal_mean.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_seasonal_cycle(ds, var_name, output_dir):
    """Plot seasonal cycle (global mean by month)."""
    print("\nGenerating seasonal cycle plot...")

    # Calculate global mean for each time step
    data = ds[var_name]

    # Weight by latitude (cos(lat) for area weighting)
    weights = np.cos(np.deg2rad(ds.lat))
    data_weighted = data.weighted(weights)
    global_mean = data_weighted.mean(dim=['lat', 'lon'])

    # Convert to g/m2/month
    global_mean_monthly = global_mean * 86400 * 30 * 1000

    # Group by month
    monthly_climatology = global_mean_monthly.groupby('time.month').mean()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    ax.plot(monthly_climatology.month, monthly_climatology.values,
            marker='o', linewidth=2, markersize=8, color='forestgreen')
    ax.fill_between(monthly_climatology.month, monthly_climatology.values,
                     alpha=0.3, color='forestgreen')

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Root NPP (g C m$^{-2}$ month$^{-1}$)', fontsize=12, fontweight='bold')
    ax.set_title('Global Mean Seasonal Cycle of Root NPP',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Save figure
    output_path = output_dir / 'cmip_seasonal_cycle.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_snapshot_grid(ds, var_name, output_dir, n_snapshots=6):
    """Plot a grid of snapshots at different time steps."""
    print(f"\nGenerating {n_snapshots}-panel snapshot grid...")

    # Select evenly spaced time steps
    time_indices = np.linspace(0, len(ds.time)-1, n_snapshots, dtype=int)

    # Create figure
    fig = plt.figure(figsize=(18, 12))

    for idx, time_idx in enumerate(time_indices):
        ax = plt.subplot(2, 3, idx+1, projection=ccrs.Robinson())

        # Get data for this time step
        data = ds[var_name].isel(time=time_idx)
        data_annual = data * 86400 * 365 * 1000

        # Plot
        im = ax.pcolormesh(
            ds.lon, ds.lat, data_annual,
            transform=ccrs.PlateCarree(),
            cmap='YlGn',
            vmin=0,
            vmax=np.nanpercentile(ds[var_name].values * 86400 * 365 * 1000, 95)
        )

        # Add features
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)

        # Title with date
        date_str = ds.time[time_idx].dt.strftime('%Y-%m').values
        ax.set_title(date_str, fontsize=12, fontweight='bold')

    # Add colorbar
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'Root NPP (g C m$^{-2}$ year$^{-1}$)', fontsize=12, fontweight='bold')

    # Main title
    model_info = ds.attrs.get('source_id', 'Unknown')
    fig.suptitle(f'Root NPP Temporal Evolution - {model_info}',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    output_path = output_dir / 'cmip_snapshot_grid.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_time_series(ds, var_name, lat, lon, output_dir):
    """Plot time series at a specific location."""
    print(f"\nGenerating time series at lat={lat}, lon={lon}...")

    # Find nearest grid point
    data_point = ds[var_name].sel(lat=lat, lon=lon, method='nearest')
    actual_lat = ds.lat.sel(lat=lat, method='nearest').values
    actual_lon = ds.lon.sel(lon=lon, method='nearest').values

    # Convert to g/m2/month
    data_monthly = data_point * 86400 * 30 * 1000

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Convert time to numeric index for plotting
    time_vals = np.arange(len(data_point.time))
    ax.plot(time_vals, data_monthly.values,
            linewidth=1.5, color='forestgreen', label='Monthly values')

    # Add rolling mean
    rolling_mean = data_monthly.rolling(time=12, center=True).mean()
    ax.plot(time_vals, rolling_mean.values,
            linewidth=2.5, color='darkgreen', label='12-month moving average')

    ax.set_xlabel('Time Step (months)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Root NPP (g C m$^{-2}$ month$^{-1}$)', fontsize=12, fontweight='bold')
    ax.set_title(f'Root NPP Time Series\nLocation: {actual_lat:.2f}°N, {actual_lon:.2f}°E',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Save figure
    output_path = output_dir / f'cmip_timeseries_lat{lat}_lon{lon}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_animation(ds, var_name, output_dir):
    """Create an animated GIF of the temporal evolution."""
    print("\nCreating animation (this may take a while)...")

    # Sample every 3rd time step to reduce file size
    time_indices = range(0, len(ds.time), 3)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.Robinson())

    # Get global vmin/vmax for consistent scale
    vmax = np.nanpercentile(ds[var_name].values * 86400 * 365 * 1000, 95)

    def update(frame_idx):
        ax.clear()
        time_idx = time_indices[frame_idx]

        # Get data
        data = ds[var_name].isel(time=time_idx)
        data_annual = data * 86400 * 365 * 1000

        # Plot
        im = ax.pcolormesh(
            ds.lon, ds.lat, data_annual,
            transform=ccrs.PlateCarree(),
            cmap='YlGn',
            vmin=0,
            vmax=vmax
        )

        # Add features
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)

        # Title
        date_str = ds.time[time_idx].dt.strftime('%Y-%m').values
        model_info = ds.attrs.get('source_id', 'Unknown')
        ax.set_title(f'Root NPP - {model_info}\n{date_str}',
                     fontsize=14, fontweight='bold')

        return im,

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(time_indices),
                         interval=200, blit=False)

    # Save animation
    output_path = output_dir / 'cmip_animation.gif'
    writer = PillowWriter(fps=5)
    anim.save(output_path, writer=writer, dpi=100)
    print(f"Saved: {output_path}")
    plt.close()


def write_data_summary(ds, var_name, output_dir):
    """Write comprehensive data summary to text file."""
    print("\nGenerating data summary report...")

    output_path = output_dir / 'cmip_data_summary.txt'

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CMIP6 DATA SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        # Dataset Information
        f.write("DATASET INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model: {ds.attrs.get('source_id', 'Unknown')}\n")
        f.write(f"Institution: {ds.attrs.get('institution', 'Unknown')}\n")
        f.write(f"Experiment: {ds.attrs.get('experiment_id', 'Unknown')}\n")
        f.write(f"Variant Label: {ds.attrs.get('variant_label', 'Unknown')}\n")
        f.write(f"Grid Label: {ds.attrs.get('grid_label', 'Unknown')}\n")
        f.write(f"Nominal Resolution: {ds.attrs.get('nominal_resolution', 'Unknown')}\n")
        f.write(f"Frequency: {ds.attrs.get('frequency', 'Unknown')}\n")
        f.write(f"Realm: {ds.attrs.get('realm', 'Unknown')}\n")
        f.write(f"Creation Date: {ds.attrs.get('creation_date', 'Unknown')}\n")
        f.write(f"\n")

        # Variable Information
        f.write("VARIABLE INFORMATION\n")
        f.write("-" * 80 + "\n")
        var = ds[var_name]
        f.write(f"Variable Name: {var_name}\n")
        f.write(f"Standard Name: {var.attrs.get('standard_name', 'N/A')}\n")
        f.write(f"Long Name: {var.attrs.get('long_name', 'N/A')}\n")
        f.write(f"Units: {var.attrs.get('units', 'N/A')}\n")
        f.write(f"Cell Methods: {var.attrs.get('cell_methods', 'N/A')}\n")
        f.write(f"\n")

        # Dimensions
        f.write("DIMENSIONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Time steps: {len(ds.time)}\n")
        f.write(f"  Range: {ds.time[0].dt.strftime('%Y-%m-%d').values} to {ds.time[-1].dt.strftime('%Y-%m-%d').values}\n")
        f.write(f"Latitude points: {len(ds.lat)}\n")
        f.write(f"  Range: {float(ds.lat.min()):.2f}° to {float(ds.lat.max()):.2f}°\n")
        f.write(f"Longitude points: {len(ds.lon)}\n")
        f.write(f"  Range: {float(ds.lon.min()):.2f}° to {float(ds.lon.max()):.2f}°\n")
        f.write(f"Total grid cells: {len(ds.lat) * len(ds.lon)}\n")
        f.write(f"\n")

        # Statistical Summary (original units: kg m-2 s-1)
        f.write("STATISTICAL SUMMARY (Original Units: kg m⁻² s⁻¹)\n")
        f.write("-" * 80 + "\n")
        data_values = var.values[~np.isnan(var.values)]
        f.write(f"Count: {len(data_values):,}\n")
        f.write(f"Mean: {np.mean(data_values):.6e}\n")
        f.write(f"Std Dev: {np.std(data_values):.6e}\n")
        f.write(f"Min: {np.min(data_values):.6e}\n")
        f.write(f"25th percentile: {np.percentile(data_values, 25):.6e}\n")
        f.write(f"Median: {np.median(data_values):.6e}\n")
        f.write(f"75th percentile: {np.percentile(data_values, 75):.6e}\n")
        f.write(f"Max: {np.max(data_values):.6e}\n")
        f.write(f"Missing values: {np.isnan(var.values).sum():,}\n")
        f.write(f"\n")

        # Statistical Summary (converted to g m-2 year-1)
        f.write("STATISTICAL SUMMARY (Converted Units: g C m⁻² year⁻¹)\n")
        f.write("-" * 80 + "\n")
        data_annual = data_values * 86400 * 365 * 1000
        f.write(f"Mean: {np.mean(data_annual):.2f}\n")
        f.write(f"Std Dev: {np.std(data_annual):.2f}\n")
        f.write(f"Min: {np.min(data_annual):.2f}\n")
        f.write(f"25th percentile: {np.percentile(data_annual, 25):.2f}\n")
        f.write(f"Median: {np.median(data_annual):.2f}\n")
        f.write(f"75th percentile: {np.percentile(data_annual, 75):.2f}\n")
        f.write(f"Max: {np.max(data_annual):.2f}\n")
        f.write(f"\n")

        # Temporal Statistics
        f.write("TEMPORAL STATISTICS\n")
        f.write("-" * 80 + "\n")
        temporal_mean = var.mean(dim=['lat', 'lon'])
        weights = np.cos(np.deg2rad(ds.lat))
        data_weighted = var.weighted(weights)
        global_mean = data_weighted.mean(dim=['lat', 'lon'])

        f.write(f"Global mean (area-weighted): {float(global_mean.mean()) * 86400 * 365 * 1000:.2f} g C m⁻² year⁻¹\n")
        f.write(f"Global std dev (area-weighted): {float(global_mean.std()) * 86400 * 365 * 1000:.2f} g C m⁻² year⁻¹\n")

        # Calculate trend
        x = np.arange(len(global_mean))
        y = global_mean.values
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            trend_annual = slope * 12 * 86400 * 365 * 1000  # per year
            trend_str = f"{trend_annual:+.3f} g C m⁻² year⁻² (over {len(x)} months)"
        else:
            trend_str = "N/A"
        f.write(f"Temporal trend: {trend_str}\n")
        f.write(f"\n")

        # Spatial Statistics
        f.write("SPATIAL STATISTICS\n")
        f.write("-" * 80 + "\n")
        spatial_mean = var.mean(dim='time')
        spatial_mean_annual = spatial_mean * 86400 * 365 * 1000
        f.write(f"Spatial mean: {float(spatial_mean_annual.mean()):.2f} g C m⁻² year⁻¹\n")
        f.write(f"Spatial std dev: {float(spatial_mean_annual.std()):.2f} g C m⁻² year⁻¹\n")
        f.write(f"Spatial min: {float(spatial_mean_annual.min()):.2f} g C m⁻² year⁻¹\n")
        f.write(f"Spatial max: {float(spatial_mean_annual.max()):.2f} g C m⁻² year⁻¹\n")

        # Find location of max and min
        max_idx = spatial_mean_annual.argmax(dim=['lat', 'lon'])
        min_idx = spatial_mean_annual.argmin(dim=['lat', 'lon'])
        max_lat = float(ds.lat[max_idx['lat'].values])
        max_lon = float(ds.lon[max_idx['lon'].values])
        min_lat = float(ds.lat[min_idx['lat'].values])
        min_lon = float(ds.lon[min_idx['lon'].values])

        f.write(f"Location of maximum: {max_lat:.2f}°N, {max_lon:.2f}°E\n")
        f.write(f"Location of minimum: {min_lat:.2f}°N, {min_lon:.2f}°E\n")
        f.write(f"\n")

        # Zonal Statistics
        f.write("ZONAL STATISTICS (by latitude band)\n")
        f.write("-" * 80 + "\n")
        zonal_mean = var.mean(dim=['time', 'lon']) * 86400 * 365 * 1000

        lat_bands = [
            ("Northern High (60°N - 90°N)", 60, 90),
            ("Northern Mid (30°N - 60°N)", 30, 60),
            ("Northern Tropical (0°N - 30°N)", 0, 30),
            ("Southern Tropical (30°S - 0°S)", -30, 0),
            ("Southern Mid (60°S - 30°S)", -60, -30),
            ("Southern High (90°S - 60°S)", -90, -60),
        ]

        for band_name, lat_min, lat_max in lat_bands:
            mask = (ds.lat >= lat_min) & (ds.lat < lat_max)
            if mask.sum() > 0:
                band_mean = float(zonal_mean.where(mask).mean())
                f.write(f"{band_name:30s}: {band_mean:8.2f} g C m⁻² year⁻¹\n")
        f.write(f"\n")

        # Global Attributes
        f.write("ADDITIONAL METADATA\n")
        f.write("-" * 80 + "\n")
        important_attrs = [
            'experiment', 'sub_experiment', 'branch_method',
            'parent_experiment_id', 'parent_variant_label',
            'grid', 'further_info_url', 'references', 'license'
        ]
        for attr in important_attrs:
            if attr in ds.attrs:
                value = ds.attrs[attr]
                # Wrap long lines
                if isinstance(value, str) and len(value) > 60:
                    f.write(f"{attr}:\n")
                    words = value.split()
                    line = "  "
                    for word in words:
                        if len(line) + len(word) + 1 > 78:
                            f.write(line + "\n")
                            line = "  " + word
                        else:
                            line += " " + word if line != "  " else word
                    f.write(line + "\n")
                else:
                    f.write(f"{attr}: {value}\n")

        f.write("\n" + "="*80 + "\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")

    print(f"Saved: {output_path}")


def plot_statistics_summary(ds, var_name, output_dir):
    """Create a summary panel with various statistics."""
    print("\nGenerating statistics summary...")

    fig = plt.figure(figsize=(16, 10))

    # 1. Temporal mean map
    ax1 = plt.subplot(2, 3, 1, projection=ccrs.Robinson())
    data_mean = ds[var_name].mean(dim='time') * 86400 * 365 * 1000
    im1 = ax1.pcolormesh(ds.lon, ds.lat, data_mean,
                         transform=ccrs.PlateCarree(),
                         cmap='YlGn', vmin=0, vmax=np.nanpercentile(data_mean, 95))
    ax1.coastlines(linewidth=0.5)
    ax1.set_title('Temporal Mean', fontweight='bold')
    plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.7)

    # 2. Temporal standard deviation
    ax2 = plt.subplot(2, 3, 2, projection=ccrs.Robinson())
    data_std = ds[var_name].std(dim='time') * 86400 * 365 * 1000
    im2 = ax2.pcolormesh(ds.lon, ds.lat, data_std,
                         transform=ccrs.PlateCarree(),
                         cmap='Reds', vmin=0)
    ax2.coastlines(linewidth=0.5)
    ax2.set_title('Temporal Std Dev', fontweight='bold')
    plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.7)

    # 3. Coefficient of variation
    ax3 = plt.subplot(2, 3, 3, projection=ccrs.Robinson())
    data_cv = (data_std / data_mean) * 100
    im3 = ax3.pcolormesh(ds.lon, ds.lat, data_cv,
                         transform=ccrs.PlateCarree(),
                         cmap='viridis', vmin=0, vmax=50)
    ax3.coastlines(linewidth=0.5)
    ax3.set_title('Coef. of Variation (%)', fontweight='bold')
    plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.7)

    # 4. Global mean time series
    ax4 = plt.subplot(2, 3, 4)
    weights = np.cos(np.deg2rad(ds.lat))
    data_weighted = ds[var_name].weighted(weights)
    global_mean = data_weighted.mean(dim=['lat', 'lon']) * 86400 * 30 * 1000
    # Convert time to numeric index for plotting
    time_vals = np.arange(len(ds.time))
    ax4.plot(time_vals, global_mean, linewidth=1.5, color='forestgreen')
    ax4.set_xlabel('Time Step (months)')
    ax4.set_ylabel(r'Root NPP (g C m$^{-2}$ month$^{-1}$)')
    ax4.set_title('Global Mean Time Series', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Latitude profile
    ax5 = plt.subplot(2, 3, 5)
    lat_mean = ds[var_name].mean(dim=['time', 'lon']) * 86400 * 365 * 1000
    ax5.plot(lat_mean, ds.lat, linewidth=2, color='forestgreen')
    ax5.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax5.set_xlabel(r'Root NPP (g C m$^{-2}$ year$^{-1}$)')
    ax5.set_ylabel('Latitude')
    ax5.set_title('Zonal Mean Profile', fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Histogram
    ax6 = plt.subplot(2, 3, 6)
    valid_data = ds[var_name].values[~np.isnan(ds[var_name].values)] * 86400 * 365 * 1000
    ax6.hist(valid_data, bins=50, color='forestgreen', alpha=0.7, edgecolor='black')
    ax6.set_xlabel(r'Root NPP (g C m$^{-2}$ year$^{-1}$)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distribution', fontweight='bold')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)

    # Main title
    model_info = ds.attrs.get('source_id', 'Unknown')
    experiment = ds.attrs.get('experiment_id', 'Unknown')
    fig.suptitle(f'Root NPP Statistics Summary - {model_info} ({experiment})',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'cmip_statistics_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize CMIP6 NetCDF data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all standard visualizations
  python visualize_cmip.py --all

  # Generate specific visualizations
  python visualize_cmip.py --temporal-mean --seasonal-cycle

  # Time series at a specific location
  python visualize_cmip.py --time-series --lat 45 --lon -75

  # Create animation
  python visualize_cmip.py --animation
        """
    )

    parser.add_argument('--file', type=str,
                        default='../../cmip/nppRoot_Lmon_CanESM5_dcppB-forecast_s2024-r9i1p2f1_gn_202501-203412.nc',
                        help='Path to NetCDF file')
    parser.add_argument('--output-dir', type=str, default='../../cmip/figures',
                        help='Output directory for figures')

    # Visualization options
    parser.add_argument('--all', action='store_true',
                        help='Generate all visualizations (except animation)')
    parser.add_argument('--temporal-mean', action='store_true',
                        help='Plot temporal mean')
    parser.add_argument('--seasonal-cycle', action='store_true',
                        help='Plot seasonal cycle')
    parser.add_argument('--snapshots', action='store_true',
                        help='Plot snapshot grid')
    parser.add_argument('--time-series', action='store_true',
                        help='Plot time series at specific location')
    parser.add_argument('--animation', action='store_true',
                        help='Create animated GIF')
    parser.add_argument('--statistics', action='store_true',
                        help='Generate statistics summary')
    parser.add_argument('--summary', action='store_true',
                        help='Write data summary to text file')

    # Time series options
    parser.add_argument('--lat', type=float, default=45.0,
                        help='Latitude for time series plot')
    parser.add_argument('--lon', type=float, default=-75.0,
                        help='Longitude for time series plot')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    ds = load_cmip_data(args.file)
    # Get the main variable name (exclude bounds variables)
    var_name = [v for v in ds.data_vars if not v.endswith('_bnds')][0]

    print(f"\nVisualization options:")
    print(f"  Output directory: {output_dir}")

    # Generate visualizations based on arguments
    if args.all:
        write_data_summary(ds, var_name, output_dir)
        plot_temporal_mean(ds, var_name, output_dir)
        plot_seasonal_cycle(ds, var_name, output_dir)
        plot_snapshot_grid(ds, var_name, output_dir)
        plot_statistics_summary(ds, var_name, output_dir)
    else:
        if args.summary:
            write_data_summary(ds, var_name, output_dir)
        if args.temporal_mean:
            plot_temporal_mean(ds, var_name, output_dir)
        if args.seasonal_cycle:
            plot_seasonal_cycle(ds, var_name, output_dir)
        if args.snapshots:
            plot_snapshot_grid(ds, var_name, output_dir)
        if args.statistics:
            plot_statistics_summary(ds, var_name, output_dir)

    if args.time_series:
        plot_time_series(ds, var_name, args.lat, args.lon, output_dir)

    if args.animation:
        create_animation(ds, var_name, output_dir)

    # If no options specified, show help
    if not any([args.all, args.temporal_mean, args.seasonal_cycle,
                args.snapshots, args.time_series, args.animation, args.statistics, args.summary]):
        parser.print_help()
        print("\nNo visualization options specified. Use --all for all standard plots.")
        return

    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"All figures saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
