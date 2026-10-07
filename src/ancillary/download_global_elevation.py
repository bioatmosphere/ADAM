#!/usr/bin/env python3
"""
Download global elevation data for the TAM pipeline.

Downloads GEBCO 2023 global elevation grid (includes bathymetry).
Alternative: ETOPO 2022 from NOAA.

Author: TAM Development Team
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import xarray as xr
import numpy as np


def download_file(url: str, output_path: Path, description: str = "Downloading") -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def download_etopo_2022(output_dir: Path) -> Path:
    """
    Download ETOPO 2022 60 arc-second (~2km) global elevation.

    Source: NOAA NCEI
    Size: ~500MB
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ETOPO_2022_v1_60s_surface.nc"

    if output_file.exists():
        print(f"File already exists: {output_file}")
        return output_file

    # ETOPO 2022 60 arc-second Ice Surface
    url = "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"

    print(f"Downloading ETOPO 2022 (60 arc-second, ~2km resolution)...")
    print(f"Source: NOAA NCEI")
    print(f"Output: {output_file}")

    if download_file(url, output_file, "ETOPO 2022"):
        print(f"Successfully downloaded: {output_file}")
        return output_file
    else:
        raise RuntimeError("Failed to download ETOPO 2022")


def create_coarse_elevation_grid(input_file: Path, output_file: Path,
                                  target_resolution: float = 0.5) -> Path:
    """
    Resample elevation to coarser resolution for global application.

    Args:
        input_file: Path to high-res elevation NetCDF
        output_file: Path for output coarse grid
        target_resolution: Target resolution in degrees (default 0.5°)

    Returns:
        Path to output file
    """
    print(f"Resampling elevation to {target_resolution}° resolution...")

    ds = xr.open_dataset(input_file)

    # Get the elevation variable (ETOPO uses 'z')
    if 'z' in ds.data_vars:
        elev_var = 'z'
    elif 'elevation' in ds.data_vars:
        elev_var = 'elevation'
    else:
        elev_var = list(ds.data_vars)[0]
        print(f"Using variable: {elev_var}")

    elev = ds[elev_var]

    # Create target grid
    target_lat = np.arange(-89.75, 90, target_resolution)
    target_lon = np.arange(-179.75, 180, target_resolution)

    # Coarsen by averaging (for 60 arc-sec to 0.5 deg, factor of 30)
    # Use interpolation for flexibility
    elev_coarse = elev.interp(lat=target_lat, lon=target_lon, method='linear')

    # Create output dataset
    out_ds = xr.Dataset({
        'elevation': elev_coarse
    })
    out_ds['elevation'].attrs = {
        'units': 'm',
        'long_name': 'Elevation above sea level',
        'source': 'ETOPO 2022 resampled to 0.5 degree'
    }

    # Save
    out_ds.to_netcdf(output_file)
    print(f"Saved coarse elevation grid: {output_file}")
    print(f"Shape: {elev_coarse.shape}")
    print(f"Elevation range: {float(elev_coarse.min()):.1f} to {float(elev_coarse.max()):.1f} m")

    ds.close()
    return output_file


def main():
    """Download and process global elevation data."""
    print("=" * 60)
    print("GLOBAL ELEVATION DATA DOWNLOAD")
    print("=" * 60)

    # Set output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / "ancillary" / "elevation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download ETOPO 2022
    print("\n1. Downloading ETOPO 2022...")
    etopo_file = download_etopo_2022(output_dir)

    # Create coarse grid for global application
    print("\n2. Creating 0.5° resolution grid...")
    coarse_file = output_dir / "global_elevation_0.5deg.nc"
    create_coarse_elevation_grid(etopo_file, coarse_file, target_resolution=0.5)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {etopo_file} (full resolution)")
    print(f"  - {coarse_file} (0.5° for global application)")

    return coarse_file


if __name__ == "__main__":
    main()
