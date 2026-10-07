#!/usr/bin/env python3
"""
Export processed GLASS GPP data to NetCDF for global model application.

This script reads your already-processed GLASS yearly GPP tiles and creates
a global 0.5° NetCDF grid that can be directly used in the RF global application.

Input: GLASS12E11 HDF yearly tiles (GPP_YEARLY/2010_001/*.hdf)
Output: global_gpp_yearly_2010_0.5deg.nc

Author: ADAM Development Team
"""

import numpy as np
import xarray as xr
from pathlib import Path
from pyhdf import SD
import warnings
from tqdm import tqdm

try:
    from pyproj import Proj, transform
    PYPROJ_AVAILABLE = True
except ImportError:
    print("WARNING: pyproj not available - install with: pip install pyproj")
    PYPROJ_AVAILABLE = False

warnings.filterwarnings('ignore')


def get_modis_projection():
    """Get the MODIS sinusoidal projection."""
    if PYPROJ_AVAILABLE:
        modis_proj = Proj(proj='sinu',
                         R=6371007.181,
                         x_0=0, y_0=0, lon_0=0)
        return modis_proj
    return None


def get_tile_bounds_from_name(filename):
    """Extract tile bounds from filename."""
    import re
    match = re.search(r'h(\d+)v(\d+)', filename)
    if not match:
        return None

    h_tile = int(match.group(1))
    v_tile = int(match.group(2))

    # MODIS tile grid parameters
    tile_size = 1111950.5196666666
    ul_x = -20015109.354
    ul_y = 10007554.677

    west_x = ul_x + h_tile * tile_size
    east_x = west_x + tile_size
    north_y = ul_y - v_tile * tile_size
    south_y = north_y - tile_size

    if PYPROJ_AVAILABLE:
        modis_proj = get_modis_projection()
        wgs84_proj = Proj(proj='latlong', datum='WGS84')

        west_lon, north_lat = transform(modis_proj, wgs84_proj, west_x, north_y)
        east_lon, south_lat = transform(modis_proj, wgs84_proj, east_x, south_y)

        return {
            'bounds': (west_lon, south_lat, east_lon, north_lat),
            'h': h_tile,
            'v': v_tile
        }
    return None


def read_gpp_from_hdf(hdf_file):
    """Read GPP data from HDF file."""
    try:
        hdf = SD.SD(str(hdf_file), SD.SDC.READ)
        datasets = hdf.datasets()

        # Find GPP dataset
        gpp_name = None
        for name in datasets.keys():
            if 'GPP' in name or 'gpp' in name.lower():
                gpp_name = name
                break

        if not gpp_name:
            gpp_name = list(datasets.keys())[0]

        # Read data
        gpp_dataset = hdf.select(gpp_name)
        gpp_data = gpp_dataset.get()

        # Get attributes
        attrs = gpp_dataset.attributes()
        scale_factor = attrs.get('scale_factor', 1.0)
        add_offset = attrs.get('add_offset', 0.0)
        fill_value = attrs.get('_FillValue', -9999)

        # Apply scaling
        gpp_data = gpp_data.astype(np.float32)
        gpp_data[gpp_data == fill_value] = np.nan
        gpp_data = gpp_data * scale_factor + add_offset

        # Filter invalid values
        gpp_data[gpp_data < 0] = np.nan
        gpp_data[gpp_data > 5000] = np.nan

        hdf.end()
        return gpp_data

    except Exception as e:
        print(f"Error reading {hdf_file.name}: {e}")
        return None


def aggregate_tiles_to_grid(glass_dir, year=2010, resolution=0.5):
    """
    Read all GLASS tiles and aggregate to global 0.5° grid.

    Args:
        glass_dir: Path to GLASS GPP_YEARLY directory
        year: Year to process
        resolution: Output resolution in degrees

    Returns:
        xr.DataArray: Global GPP grid
    """
    tile_dir = Path(glass_dir) / f"{year}_001"

    if not tile_dir.exists():
        raise FileNotFoundError(f"Tile directory not found: {tile_dir}")

    # Get all HDF files
    hdf_files = list(tile_dir.glob("GLASS12E11.*.hdf"))
    print(f"Found {len(hdf_files)} GLASS GPP tiles for {year}")

    if len(hdf_files) == 0:
        raise FileNotFoundError(f"No HDF files found in {tile_dir}")

    # Create output grid
    lat_edges = np.arange(-90, 90 + resolution, resolution)
    lon_edges = np.arange(-180, 180 + resolution, resolution)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

    gpp_grid = np.full((len(lat_centers), len(lon_centers)), np.nan, dtype=np.float32)
    count_grid = np.zeros_like(gpp_grid, dtype=np.int32)

    print(f"Output grid: {gpp_grid.shape[0]} x {gpp_grid.shape[1]}")
    print("\nProcessing tiles...")

    for hdf_file in tqdm(hdf_files):
        # Get tile bounds
        bounds_info = get_tile_bounds_from_name(hdf_file.name)
        if not bounds_info:
            continue

        # Read GPP data
        gpp_data = read_gpp_from_hdf(hdf_file)
        if gpp_data is None:
            continue

        # Get tile bounds
        west, south, east, north = bounds_info['bounds']

        # Calculate which grid cells this tile covers
        lat_start = np.searchsorted(lat_centers, south)
        lat_end = np.searchsorted(lat_centers, north)
        lon_start = np.searchsorted(lon_centers, west)
        lon_end = np.searchsorted(lon_centers, east)

        # Sample GPP data to match grid cells
        tile_lats = np.linspace(north, south, gpp_data.shape[0])
        tile_lons = np.linspace(west, east, gpp_data.shape[1])

        # For each grid cell in the tile's extent
        for i in range(max(0, lat_start), min(len(lat_centers), lat_end + 1)):
            for j in range(max(0, lon_start), min(len(lon_centers), lon_end + 1)):
                # Find nearest tile pixel
                lat_dist = np.abs(tile_lats - lat_centers[i])
                lon_dist = np.abs(tile_lons - lon_centers[j])
                tile_i = np.argmin(lat_dist)
                tile_j = np.argmin(lon_dist)

                gpp_value = gpp_data[tile_i, tile_j]
                if not np.isnan(gpp_value):
                    if np.isnan(gpp_grid[i, j]):
                        gpp_grid[i, j] = gpp_value
                        count_grid[i, j] = 1
                    else:
                        # Average overlapping values
                        gpp_grid[i, j] += gpp_value
                        count_grid[i, j] += 1

    # Compute averages for overlapping cells
    mask = count_grid > 1
    gpp_grid[mask] = gpp_grid[mask] / count_grid[mask]

    # Create DataArray
    gpp_da = xr.DataArray(
        gpp_grid,
        dims=['lat', 'lon'],
        coords={'lat': lat_centers, 'lon': lon_centers},
        name='gpp_yearly',
        attrs={
            'units': 'gC m-2 year-1',
            'long_name': 'Gross Primary Production (Yearly)',
            'source': 'GLASS GPP_YEARLY V60',
            'product': 'GLASS12E11',
            'resolution': f'{resolution} degrees',
            'year': str(year),
            'processing_method': 'Aggregated from MODIS sinusoidal tiles',
            'tiles_processed': len(hdf_files),
            'processing_date': str(np.datetime64('today'))
        }
    )

    # Print statistics
    valid_data = gpp_da.values[~np.isnan(gpp_da.values)]
    print(f"\n✓ Global GPP Grid Created:")
    print(f"  Valid pixels: {len(valid_data):,} / {gpp_grid.size:,}")
    print(f"  Coverage: {len(valid_data) / gpp_grid.size * 100:.1f}%")
    print(f"  Min: {valid_data.min():.1f} gC m⁻² yr⁻¹")
    print(f"  Max: {valid_data.max():.1f} gC m⁻² yr⁻¹")
    print(f"  Mean: {valid_data.mean():.1f} gC m⁻² yr⁻¹")
    print(f"  Median: {np.median(valid_data):.1f} gC m⁻² yr⁻¹")

    return gpp_da


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Export GLASS GPP to NetCDF for model application'
    )
    parser.add_argument(
        '--glass-dir',
        type=str,
        default='../../ancillary/glass/GPP_YEARLY',
        help='GLASS GPP_YEARLY directory'
    )
    parser.add_argument(
        '--year',
        type=int,
        default=2010,
        help='Year to process'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../../ancillary/glass/global_gpp_yearly_2010_0.5deg.nc',
        help='Output NetCDF file'
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=0.5,
        help='Output resolution in degrees'
    )

    args = parser.parse_args()

    print("="*70)
    print("GLASS GPP TO NETCDF EXPORT")
    print("="*70)
    print()

    try:
        # Process tiles
        gpp_da = aggregate_tiles_to_grid(
            args.glass_dir,
            year=args.year,
            resolution=args.resolution
        )

        # Save NetCDF
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gpp_da.to_netcdf(output_path)
        print(f"\n✓ NetCDF saved to: {output_path}")

        # Create visualization
        try:
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs

            fig, ax = plt.subplots(
                figsize=(15, 8),
                subplot_kw={'projection': ccrs.PlateCarree()}
            )

            gpp_da.plot(
                ax=ax,
                cmap='YlGn',
                vmin=0,
                vmax=np.nanpercentile(gpp_da.values, 98),
                transform=ccrs.PlateCarree(),
                cbar_kwargs={'label': 'GPP (gC m⁻² year⁻¹)', 'shrink': 0.8}
            )

            ax.coastlines(linewidth=0.5)
            ax.set_title(f'Global GLASS GPP - {args.year}', fontsize=14)

            fig_path = output_path.with_suffix('.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Map saved to: {fig_path}")
            plt.close()
        except Exception as e:
            print(f"Note: Could not create visualization: {e}")

        print("\n" + "="*70)
        print("✓ Export complete!")
        print("="*70)
        print(f"\nYou can now use this file in apply_RF_globally_v2.py:")
        print(f"  python apply_RF_globally_v2.py")

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
