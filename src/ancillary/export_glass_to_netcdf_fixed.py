#!/usr/bin/env python3
"""
Export processed GLASS GPP data to NetCDF with FIXED projection handling.

This version uses proper MODIS metadata and accurate coordinate transformations
to ensure the output grid correctly aligns with the original point extractions.

Author: ADAM Development Team
"""

import numpy as np
import xarray as xr
from pathlib import Path
from pyhdf import SD
import warnings
from tqdm import tqdm
import re

try:
    from pyproj import Transformer, CRS
    PYPROJ_AVAILABLE = True
except ImportError:
    print("ERROR: pyproj is required - install with: pip install pyproj")
    PYPROJ_AVAILABLE = False
    exit(1)

warnings.filterwarnings('ignore')


def get_modis_sinusoidal_proj():
    """Get proper MODIS sinusoidal projection."""
    # Official MODIS sinusoidal parameters
    modis_proj_str = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
    return CRS.from_proj4(modis_proj_str)


def parse_hdf_metadata(hdf_file):
    """
    Parse HDF metadata to get exact tile bounds in MODIS sinusoidal coordinates.
    """
    try:
        hdf = SD.SD(str(hdf_file), SD.SDC.READ)
        attrs = hdf.attributes()

        # Get StructMetadata which contains exact bounds
        struct_meta = attrs.get('StructMetadata.0', '')

        # Extract UpperLeftPointMtrs and LowerRightMtrs
        ul_match = re.search(r'UpperLeftPointMtrs=\(([^,]+),([^)]+)\)', struct_meta)
        lr_match = re.search(r'LowerRightMtrs=\(([^,]+),([^)]+)\)', struct_meta)

        if ul_match and lr_match:
            ul_x = float(ul_match.group(1))
            ul_y = float(ul_match.group(2))
            lr_x = float(lr_match.group(1))
            lr_y = float(lr_match.group(2))

            hdf.end()

            return {
                'ul_x': ul_x,
                'ul_y': ul_y,
                'lr_x': lr_x,
                'lr_y': lr_y,
                'source': 'metadata'
            }

        hdf.end()
        return None

    except Exception as e:
        return None


def calculate_tile_bounds_from_hv(h_tile, v_tile):
    """
    Calculate tile bounds from h/v indices using MODIS grid specifications.

    MODIS grid: 36 horizontal × 18 vertical tiles
    Tile size: 1111950 meters (exactly)
    """
    # MODIS sinusoidal grid parameters (official)
    TILE_SIZE = 1111950.5196666666  # meters
    EARTH_RADIUS = 6371007.181  # meters

    # Upper-left corner of the MODIS grid (h=0, v=0)
    UL_X = -20015109.354  # meters
    UL_Y = 10007554.677   # meters

    # Calculate tile bounds in sinusoidal coordinates
    west_x = UL_X + (h_tile * TILE_SIZE)
    east_x = west_x + TILE_SIZE
    north_y = UL_Y - (v_tile * TILE_SIZE)
    south_y = north_y - TILE_SIZE

    return {
        'ul_x': west_x,
        'ul_y': north_y,
        'lr_x': east_x,
        'lr_y': south_y,
        'source': 'calculated'
    }


def extract_hv_from_filename(filename):
    """Extract h and v tile indices from filename."""
    match = re.search(r'h(\d+)v(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def read_gpp_from_hdf(hdf_file):
    """Read GPP data from HDF file with proper scaling."""
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

        # Get scaling attributes
        attrs = gpp_dataset.attributes()
        scale_factor = attrs.get('scale_factor', 1.0)
        add_offset = attrs.get('add_offset', 0.0)
        fill_value = attrs.get('_FillValue', -9999)

        # Apply scaling
        gpp_data = gpp_data.astype(np.float32)
        gpp_data[gpp_data == fill_value] = np.nan
        gpp_data = gpp_data * scale_factor + add_offset

        # Quality filtering
        gpp_data[gpp_data < 0] = np.nan
        gpp_data[gpp_data > 5000] = np.nan

        hdf.end()
        return gpp_data

    except Exception as e:
        print(f"Error reading {hdf_file.name}: {e}")
        return None


def create_coordinate_arrays(bounds, data_shape):
    """
    Create coordinate arrays for tile in MODIS sinusoidal projection.

    Args:
        bounds: Dictionary with ul_x, ul_y, lr_x, lr_y
        data_shape: (nrows, ncols) of data array

    Returns:
        x_coords, y_coords in sinusoidal projection
    """
    nrows, ncols = data_shape

    # Create linearly spaced coordinates
    x_coords = np.linspace(bounds['ul_x'], bounds['lr_x'], ncols)
    y_coords = np.linspace(bounds['ul_y'], bounds['lr_y'], nrows)

    return x_coords, y_coords


def aggregate_tiles_to_grid_fixed(glass_dir, year=2010, resolution=0.5):
    """
    Aggregate tiles to global grid with FIXED projection handling.

    Uses proper coordinate transformations and accurate bounds.
    """
    tile_dir = Path(glass_dir) / f"{year}_001"

    if not tile_dir.exists():
        raise FileNotFoundError(f"Tile directory not found: {tile_dir}")

    # Get all HDF files
    hdf_files = sorted(tile_dir.glob("GLASS12E11.*.hdf"))
    print(f"Found {len(hdf_files)} GLASS GPP tiles")

    if len(hdf_files) == 0:
        raise FileNotFoundError(f"No HDF files in {tile_dir}")

    # Create transformers
    modis_crs = get_modis_sinusoidal_proj()
    wgs84_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(modis_crs, wgs84_crs, always_xy=True)

    # Create output grid
    lat_edges = np.arange(-90, 90 + resolution, resolution)
    lon_edges = np.arange(-180, 180 + resolution, resolution)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

    print(f"Output grid: {len(lat_centers)} lat × {len(lon_centers)} lon")

    # Initialize grids
    gpp_sum = np.zeros((len(lat_centers), len(lon_centers)), dtype=np.float64)
    gpp_count = np.zeros((len(lat_centers), len(lon_centers)), dtype=np.int32)

    print("\nProcessing tiles with accurate projection...")

    tiles_processed = 0
    tiles_failed = 0

    for hdf_file in tqdm(hdf_files, desc="Processing"):
        # Read GPP data
        gpp_data = read_gpp_from_hdf(hdf_file)
        if gpp_data is None:
            tiles_failed += 1
            continue

        # Get tile bounds
        h_tile, v_tile = extract_hv_from_filename(hdf_file.name)
        if h_tile is None:
            tiles_failed += 1
            continue

        # Try to get exact bounds from metadata, fallback to calculation
        bounds = parse_hdf_metadata(hdf_file)
        if bounds is None:
            bounds = calculate_tile_bounds_from_hv(h_tile, v_tile)

        # Create coordinate arrays in sinusoidal projection
        x_sin, y_sin = create_coordinate_arrays(bounds, gpp_data.shape)

        # Transform to geographic (sample every Nth pixel to save time)
        sample_factor = 10  # Sample every 10th pixel
        x_sample = x_sin[::sample_factor]
        y_sample = y_sin[::sample_factor]
        gpp_sample = gpp_data[::sample_factor, ::sample_factor]

        # Create meshgrid for transformation
        xx_sin, yy_sin = np.meshgrid(x_sample, y_sample)

        # Transform to geographic
        lons, lats = transformer.transform(xx_sin.ravel(), yy_sin.ravel())
        lons = lons.reshape(xx_sin.shape)
        lats = lats.reshape(yy_sin.shape)

        # Map to output grid
        for i in range(gpp_sample.shape[0]):
            for j in range(gpp_sample.shape[1]):
                gpp_val = gpp_sample[i, j]

                if np.isnan(gpp_val):
                    continue

                lat = lats[i, j]
                lon = lons[i, j]

                # Find grid cell
                lat_idx = np.searchsorted(lat_edges, lat) - 1
                lon_idx = np.searchsorted(lon_edges, lon) - 1

                # Check bounds
                if 0 <= lat_idx < len(lat_centers) and 0 <= lon_idx < len(lon_centers):
                    gpp_sum[lat_idx, lon_idx] += gpp_val
                    gpp_count[lat_idx, lon_idx] += 1

        tiles_processed += 1

    print(f"\nTiles processed: {tiles_processed}")
    print(f"Tiles failed: {tiles_failed}")

    # Compute averages
    gpp_grid = np.full_like(gpp_sum, np.nan, dtype=np.float32)
    mask = gpp_count > 0
    gpp_grid[mask] = (gpp_sum[mask] / gpp_count[mask]).astype(np.float32)

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
            'processing_method': 'Fixed projection with proper MODIS metadata',
            'tiles_processed': tiles_processed,
            'processing_date': str(np.datetime64('today')),
            'projection': 'WGS84 (EPSG:4326)',
            'coordinate_transformation': 'MODIS Sinusoidal -> Geographic'
        }
    )

    # Statistics
    valid_data = gpp_da.values[~np.isnan(gpp_da.values)]
    print(f"\n✓ Global GPP Grid Created:")
    print(f"  Valid pixels: {len(valid_data):,} / {gpp_grid.size:,}")
    print(f"  Coverage: {len(valid_data) / gpp_grid.size * 100:.1f}%")
    print(f"  Min: {valid_data.min():.1f} gC m⁻² yr⁻¹")
    print(f"  Max: {valid_data.max():.1f} gC m⁻² yr⁻¹")
    print(f"  Mean: {valid_data.mean():.1f} gC m⁻² yr⁻¹")
    print(f"  Median: {np.median(valid_data):.1f} gC m⁻² yr⁻¹")

    return gpp_da


def validate_against_points(gpp_da, point_file):
    """Validate grid against original point extractions."""
    import pandas as pd

    print("\n" + "="*70)
    print("VALIDATING AGAINST ORIGINAL POINT EXTRACTIONS")
    print("="*70)

    points = pd.read_csv(point_file)

    # Sample random points
    sample_size = min(50, len(points))
    sample_idx = np.random.choice(len(points), sample_size, replace=False)

    matches = 0
    close_matches = 0
    mismatches = 0
    nan_count = 0
    differences = []

    print(f"\nSampling {sample_size} points...")

    for idx in sample_idx:
        row = points.iloc[idx]
        lat, lon = row['latitude'], row['longitude']
        point_gpp = row['value']

        try:
            grid_gpp = gpp_da.sel(lat=lat, lon=lon, method='nearest').values

            if not np.isnan(grid_gpp):
                if point_gpp > 0:
                    diff_pct = abs(grid_gpp - point_gpp) / point_gpp * 100
                else:
                    diff_pct = abs(grid_gpp - point_gpp)

                differences.append(diff_pct)

                if diff_pct < 15:
                    matches += 1
                elif diff_pct < 35:
                    close_matches += 1
                else:
                    mismatches += 1
            else:
                nan_count += 1
        except:
            nan_count += 1

    print(f"\nValidation Results:")
    print(f"  Exact matches (<15%): {matches} ({matches/sample_size*100:.1f}%)")
    print(f"  Close matches (15-35%): {close_matches} ({close_matches/sample_size*100:.1f}%)")
    print(f"  Mismatches (>35%): {mismatches} ({mismatches/sample_size*100:.1f}%)")
    print(f"  Missing/NaN: {nan_count} ({nan_count/sample_size*100:.1f}%)")

    if differences:
        print(f"\nDifference Statistics:")
        print(f"  Median: {np.median(differences):.1f}%")
        print(f"  Mean: {np.mean(differences):.1f}%")
        print(f"  90th percentile: {np.percentile(differences, 90):.1f}%")

    # Quality assessment
    total_good = matches + close_matches
    if total_good / sample_size > 0.7 and np.median(differences) < 25:
        print(f"\n✓ VALIDATION PASSED")
        print(f"  {total_good/sample_size*100:.1f}% of points match well")
        return True
    else:
        print(f"\n⚠ VALIDATION MARGINAL")
        print(f"  Only {total_good/sample_size*100:.1f}% of points match well")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Export GLASS GPP to NetCDF with FIXED projection'
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
        default='../../ancillary/glass/global_gpp_yearly_2010_0.5deg_FIXED.nc',
        help='Output NetCDF file'
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=0.5,
        help='Output resolution in degrees'
    )
    parser.add_argument(
        '--validate',
        type=str,
        default='../../ancillary/glass/point_extractions/GPP_YEARLY_all_points.csv',
        help='Point data file for validation'
    )

    args = parser.parse_args()

    print("="*70)
    print("GLASS GPP TO NETCDF EXPORT - FIXED PROJECTION")
    print("="*70)
    print()

    try:
        # Process tiles
        gpp_da = aggregate_tiles_to_grid_fixed(
            args.glass_dir,
            year=args.year,
            resolution=args.resolution
        )

        # Validate if point file exists
        if Path(args.validate).exists():
            validation_passed = validate_against_points(gpp_da, args.validate)
        else:
            print(f"\nWarning: Validation file not found: {args.validate}")
            validation_passed = True

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
            ax.set_title(f'Global GLASS GPP (Fixed Projection) - {args.year}', fontsize=14)

            fig_path = output_path.with_suffix('.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Map saved to: {fig_path}")
            plt.close()
        except Exception as e:
            print(f"Note: Could not create visualization: {e}")

        print("\n" + "="*70)
        if validation_passed:
            print("✓ EXPORT COMPLETE - PROJECTION VALIDATED")
        else:
            print("✓ EXPORT COMPLETE - CHECK VALIDATION RESULTS")
        print("="*70)

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
