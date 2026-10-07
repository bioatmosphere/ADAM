"""
GLASS GPP Global Processor for ADAM Model Application

This script processes GLASS GPP_YEARLY HDF tiles (500m resolution) and creates
a global GPP grid at 0.5° resolution for use in BNPP prediction models.

Features:
- Reads GLASS HDF4 tiles in MODIS sinusoidal projection
- Mosaics all tiles into a global grid
- Reprojects from MODIS sinusoidal to geographic coordinates
- Aggregates from 500m to 0.5° resolution (~55km)
- Handles data quality flags and scaling factors
- Outputs NetCDF file compatible with model application pipeline

GLASS GPP Data Specifications:
- Product: GLASS12E11 (Yearly GPP)
- Resolution: 500m (native)
- Projection: MODIS Sinusoidal
- Units: gC m⁻² year⁻¹
- Tiles: ~288 HDF files covering global land areas
- Quality: Includes QC flags

Dependencies:
    pip install pyhdf numpy xarray rasterio pyproj tqdm pandas

Author: ADAM Development Team
Date: 2025
"""

import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import warnings
from typing import Tuple, Optional
import pickle

try:
    from pyhdf.SD import SD, SDC
    HDF4_AVAILABLE = True
except ImportError:
    HDF4_AVAILABLE = False
    print("Warning: pyhdf not available. Install with: pip install pyhdf")

try:
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not available. Install with: pip install rasterio")

try:
    from pyproj import Proj, Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    print("Warning: pyproj not available. Install with: pip install pyproj")

warnings.filterwarnings('ignore')


# MODIS Sinusoidal projection parameters
MODIS_SIN_PROJ = ("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 "
                  "+b=6371007.181 +units=m +no_defs")


class GLASSGPPProcessor:
    """Process GLASS GPP HDF tiles into global grid."""

    def __init__(self, glass_dir: str, year: int = 2010):
        """
        Initialize GLASS GPP processor.

        Args:
            glass_dir: Directory containing GLASS GPP_YEARLY tiles
            year: Year to process (default: 2010)
        """
        self.glass_dir = Path(glass_dir)
        self.year = year
        self.tile_dir = self.glass_dir / f"{year}_001"

        # MODIS tile parameters
        self.tile_size = 2400  # 2400x2400 pixels at 500m = 1200km
        self.pixel_size = 463.31271653  # meters (MODIS 500m actual)

        # Target output grid
        self.target_resolution = 0.5  # degrees
        self.target_lat = np.arange(-89.75, 90, self.target_resolution)
        self.target_lon = np.arange(-179.75, 180, self.target_resolution)

    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        if not HDF4_AVAILABLE:
            print("ERROR: pyhdf is required but not installed.")
            print("Install with: pip install pyhdf")
            return False
        if not RASTERIO_AVAILABLE:
            print("ERROR: rasterio is required but not installed.")
            print("Install with: pip install rasterio")
            return False
        if not PYPROJ_AVAILABLE:
            print("ERROR: pyproj is required but not installed.")
            print("Install with: pip install pyproj")
            return False
        return True

    def get_tile_files(self) -> list:
        """Get list of GLASS GPP HDF tiles for specified year."""
        if not self.tile_dir.exists():
            raise FileNotFoundError(f"Tile directory not found: {self.tile_dir}")

        hdf_files = list(self.tile_dir.glob("GLASS12E11.*.hdf"))
        print(f"Found {len(hdf_files)} GLASS GPP tiles for {self.year}")
        return sorted(hdf_files)

    def parse_tile_name(self, filename: str) -> Tuple[int, int]:
        """
        Extract MODIS tile coordinates from filename.

        Args:
            filename: HDF filename (e.g., GLASS12E11.V60.A2010001.h13v09.2022100.hdf)

        Returns:
            Tuple of (h_tile, v_tile)
        """
        parts = filename.split('.')
        for part in parts:
            if part.startswith('h') and 'v' in part:
                h = int(part[1:3])
                v = int(part[4:6])
                return h, v
        raise ValueError(f"Cannot parse tile coordinates from: {filename}")

    def read_gpp_tile(self, hdf_file: Path) -> Optional[np.ndarray]:
        """
        Read GPP data from a single HDF tile.

        Args:
            hdf_file: Path to GLASS HDF file

        Returns:
            GPP data array (2400x2400) or None if error
        """
        try:
            hdf = SD(str(hdf_file), SDC.READ)

            # Get GPP dataset (dataset name may vary - check common names)
            dataset_names = list(hdf.datasets().keys())
            gpp_names = [n for n in dataset_names if 'GPP' in n.upper() or 'Gross' in n]

            if not gpp_names:
                # Try first dataset
                gpp_names = [dataset_names[0]]

            dataset = hdf.select(gpp_names[0])
            gpp_data = dataset.get()

            # Get scaling factor and offset
            attrs = dataset.attributes()
            scale_factor = attrs.get('scale_factor', 1.0)
            add_offset = attrs.get('add_offset', 0.0)
            fill_value = attrs.get('_FillValue', -9999)

            # Apply scaling
            gpp_data = gpp_data.astype(np.float32)
            gpp_data[gpp_data == fill_value] = np.nan
            gpp_data = gpp_data * scale_factor + add_offset

            # Filter invalid values
            gpp_data[gpp_data < 0] = np.nan
            gpp_data[gpp_data > 5000] = np.nan  # Max reasonable GPP

            hdf.end()

            return gpp_data

        except Exception as e:
            print(f"Error reading {hdf_file.name}: {e}")
            return None

    def get_tile_bounds_geographic(self, h: int, v: int) -> Tuple[float, float, float, float]:
        """
        Get geographic bounds for a MODIS tile.

        Args:
            h: Horizontal tile index (0-35)
            v: Vertical tile index (0-17)

        Returns:
            Tuple of (west, south, east, north) in degrees
        """
        # MODIS sinusoidal tile bounds in projection coordinates
        tile_width = self.tile_size * self.pixel_size  # meters

        # Calculate tile bounds in sinusoidal projection
        x_min = (h - 18) * tile_width
        x_max = x_min + tile_width
        y_max = (9 - v) * tile_width
        y_min = y_max - tile_width

        # Transform corners to geographic coordinates
        transformer = Transformer.from_crs(
            MODIS_SIN_PROJ,
            "EPSG:4326",
            always_xy=True
        )

        # Transform four corners
        corners_x = [x_min, x_max, x_max, x_min]
        corners_y = [y_min, y_min, y_max, y_max]
        lons, lats = transformer.transform(corners_x, corners_y)

        west = min(lons)
        east = max(lons)
        south = min(lats)
        north = max(lats)

        return west, south, east, north

    def reproject_tile_to_geographic(self, gpp_data: np.ndarray,
                                     h: int, v: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reproject a single tile from MODIS sinusoidal to geographic coordinates.

        Args:
            gpp_data: GPP data array (2400x2400)
            h: Horizontal tile index
            v: Vertical tile index

        Returns:
            Tuple of (reprojected_data, lons, lats)
        """
        # Get tile bounds
        west, south, east, north = self.get_tile_bounds_geographic(h, v)

        # Calculate tile extent in sinusoidal projection
        tile_width = self.tile_size * self.pixel_size
        x_min = (h - 18) * tile_width
        y_max = (9 - v) * tile_width

        # Source transform (sinusoidal)
        src_transform = from_bounds(
            x_min, y_max - tile_width, x_min + tile_width, y_max,
            self.tile_size, self.tile_size
        )

        # Calculate output dimensions for this tile at target resolution
        out_width = int((east - west) / self.target_resolution) + 1
        out_height = int((north - south) / self.target_resolution) + 1

        # Destination transform (geographic)
        dst_transform = from_bounds(west, south, east, north, out_width, out_height)

        # Prepare output array
        reprojected = np.full((out_height, out_width), np.nan, dtype=np.float32)

        # Reproject using rasterio
        reproject(
            gpp_data,
            reprojected,
            src_transform=src_transform,
            src_crs=CRS.from_string(MODIS_SIN_PROJ),
            dst_transform=dst_transform,
            dst_crs=CRS.from_epsg(4326),
            resampling=Resampling.average
        )

        # Generate coordinate arrays
        lons = np.linspace(west, east, out_width)
        lats = np.linspace(north, south, out_height)

        return reprojected, lons, lats

    def aggregate_to_target_grid(self, tile_data: dict) -> xr.DataArray:
        """
        Aggregate all reprojected tiles onto target 0.5° grid.

        Args:
            tile_data: Dictionary of {tile_key: (gpp_data, lons, lats)}

        Returns:
            Global GPP DataArray at 0.5° resolution
        """
        print("Aggregating tiles to 0.5° global grid...")

        # Initialize output grid
        global_gpp = np.full(
            (len(self.target_lat), len(self.target_lon)),
            np.nan,
            dtype=np.float32
        )
        count_grid = np.zeros_like(global_gpp, dtype=np.int16)

        # Process each tile
        for tile_key, (gpp_data, lons, lats) in tqdm(tile_data.items(), desc="Aggregating"):
            # Find indices in target grid
            lat_indices = np.searchsorted(self.target_lat, lats)
            lon_indices = np.searchsorted(self.target_lon, lons)

            # Add tile data to global grid (average overlapping values)
            for i in range(len(lats)):
                for j in range(len(lons)):
                    if not np.isnan(gpp_data[i, j]):
                        lat_idx = lat_indices[i]
                        lon_idx = lon_indices[j]

                        if 0 <= lat_idx < len(self.target_lat) and 0 <= lon_idx < len(self.target_lon):
                            if np.isnan(global_gpp[lat_idx, lon_idx]):
                                global_gpp[lat_idx, lon_idx] = gpp_data[i, j]
                                count_grid[lat_idx, lon_idx] = 1
                            else:
                                # Average overlapping values
                                global_gpp[lat_idx, lon_idx] += gpp_data[i, j]
                                count_grid[lat_idx, lon_idx] += 1

        # Compute averages for overlapping areas
        mask = count_grid > 1
        global_gpp[mask] = global_gpp[mask] / count_grid[mask]

        # Create DataArray
        gpp_da = xr.DataArray(
            global_gpp,
            dims=['lat', 'lon'],
            coords={'lat': self.target_lat, 'lon': self.target_lon},
            name='gpp_yearly',
            attrs={
                'units': 'gC m-2 year-1',
                'long_name': 'Gross Primary Production (Yearly)',
                'source': 'GLASS GPP_YEARLY V60',
                'resolution': '0.5 degrees',
                'year': str(self.year),
                'processing_date': str(np.datetime64('today'))
            }
        )

        # Print statistics
        valid_data = gpp_da.values[~np.isnan(gpp_da.values)]
        print(f"\nGlobal GPP Statistics:")
        print(f"  Valid pixels: {len(valid_data):,}")
        print(f"  Min: {valid_data.min():.1f} gC m⁻² yr⁻¹")
        print(f"  Max: {valid_data.max():.1f} gC m⁻² yr⁻¹")
        print(f"  Mean: {valid_data.mean():.1f} gC m⁻² yr⁻¹")
        print(f"  Median: {np.median(valid_data):.1f} gC m⁻² yr⁻¹")

        return gpp_da

    def process_all_tiles(self, output_file: Optional[str] = None) -> xr.DataArray:
        """
        Complete workflow: Read all tiles, reproject, and create global grid.

        Args:
            output_file: Path to save NetCDF output (optional)

        Returns:
            Global GPP DataArray
        """
        print("="*70)
        print(f"GLASS GPP Global Processing - Year {self.year}")
        print("="*70)

        # Check dependencies
        if not self.check_dependencies():
            raise RuntimeError("Required dependencies not installed")

        # Get tile files
        hdf_files = self.get_tile_files()

        if len(hdf_files) == 0:
            raise FileNotFoundError(f"No HDF tiles found in {self.tile_dir}")

        # Process each tile
        print(f"\nProcessing {len(hdf_files)} tiles...")
        tile_data = {}

        for hdf_file in tqdm(hdf_files, desc="Reading tiles"):
            # Parse tile coordinates
            h, v = self.parse_tile_name(hdf_file.name)

            # Read GPP data
            gpp_data = self.read_gpp_tile(hdf_file)
            if gpp_data is None:
                continue

            # Reproject to geographic
            try:
                reprojected, lons, lats = self.reproject_tile_to_geographic(gpp_data, h, v)
                tile_data[f"h{h:02d}v{v:02d}"] = (reprojected, lons, lats)
            except Exception as e:
                print(f"Error reprojecting tile h{h:02d}v{v:02d}: {e}")
                continue

        print(f"Successfully processed {len(tile_data)}/{len(hdf_files)} tiles")

        # Aggregate to global grid
        global_gpp = self.aggregate_to_target_grid(tile_data)

        # Save output
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            global_gpp.to_netcdf(output_path)
            print(f"\n✓ Global GPP grid saved to: {output_path}")

        print("\n" + "="*70)
        print("Processing complete!")
        print("="*70)

        return global_gpp


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Process GLASS GPP tiles into global 0.5° grid'
    )
    parser.add_argument(
        '--glass-dir',
        type=str,
        default='../../ancillary/glass/GPP_YEARLY',
        help='Directory containing GLASS GPP_YEARLY tiles'
    )
    parser.add_argument(
        '--year',
        type=int,
        default=2010,
        help='Year to process (default: 2010)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../../ancillary/glass/global_gpp_yearly_0.5deg.nc',
        help='Output NetCDF file path'
    )

    args = parser.parse_args()

    try:
        # Initialize processor
        processor = GLASSGPPProcessor(args.glass_dir, args.year)

        # Process all tiles
        global_gpp = processor.process_all_tiles(output_file=args.output)

        # Create quick visualization
        try:
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs

            fig, ax = plt.subplots(
                figsize=(15, 8),
                subplot_kw={'projection': ccrs.PlateCarree()}
            )

            global_gpp.plot(
                ax=ax,
                cmap='YlGn',
                vmin=0,
                vmax=np.nanpercentile(global_gpp.values, 98),
                transform=ccrs.PlateCarree(),
                add_colorbar=True,
                cbar_kwargs={'label': 'GPP (gC m⁻² year⁻¹)', 'shrink': 0.8}
            )

            ax.coastlines(linewidth=0.5)
            ax.set_title(f'Global GLASS GPP - {args.year}', fontsize=14, pad=20)

            output_fig = Path(args.output).with_suffix('.png')
            plt.savefig(output_fig, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Visualization saved to: {output_fig}")
            plt.close()

        except Exception as e:
            print(f"Note: Could not create visualization: {e}")

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
