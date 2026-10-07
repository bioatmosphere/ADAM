"""
Global DEM (Digital Elevation Model) Data Downloader

This module downloads global elevation data from GMTED2010 (Global Multi-resolution 
Terrain Elevation Data 2010) provided by USGS Earth Resources Observation and Science 
(EROS) Center. The data provides global elevation coverage at multiple resolutions.

Features:
- Downloads GMTED2010 global elevation data
- Supports multiple resolution options (7.5 arc-seconds, 15 arc-seconds, 30 arc-seconds)
- Automatic tile management and mosaicking
- Point extraction for specific coordinates
- Integration with TAM pipeline data formats

Data source: USGS GMTED2010
Resolution options:
- 7.5 arc-seconds (~250m at equator)
- 15 arc-seconds (~500m at equator) 
- 30 arc-seconds (~1km at equator)

Author: TAM Development Team
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import logging

try:
    import rasterio
    from rasterio.merge import merge
    from rasterio.mask import mask
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError as e:
    print(f"Missing required packages for raster processing: {e}")
    print("Install with: pip install rasterio geopandas shapely")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dem_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GlobalDEMDownloader:
    """Download and process global DEM data from GMTED2010."""
    
    def __init__(self, base_dir: str = "../ancillary/elevation", resolution: str = "30arc"):
        """
        Initialize the DEM downloader.
        
        Args:
            base_dir: Base directory for storing elevation data
            resolution: Resolution option ('7p5arc', '15arc', '30arc')
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Resolution settings
        self.resolution = resolution
        self.resolution_info = {
            "7p5arc": {"folder": "7p5_arc_second", "suffix": "7p5", "approx_res": "250m"},
            "15arc": {"folder": "15_arc_second", "suffix": "15", "approx_res": "500m"},
            "30arc": {"folder": "30_arc_second", "suffix": "30", "approx_res": "1km"}
        }
        
        if resolution not in self.resolution_info:
            raise ValueError(f"Resolution must be one of: {list(self.resolution_info.keys())}")
        
        self.res_info = self.resolution_info[resolution]
        
        # GMTED2010 base URL - using USGS direct source
        self.base_url = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/topo/downloads/GMTED/Grid_ZipFiles"
        
        # Create resolution-specific directory
        self.data_dir = self.base_dir / self.res_info["folder"]
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DEM downloader for {resolution} (~{self.res_info['approx_res']}) resolution")
        logger.info(f"Data directory: {self.data_dir}")
    
    def get_tile_list(self) -> List[str]:
        """
        Get list of available DEM tiles for the specified resolution.
        
        Returns:
            List of tile filenames available for download
        """
        # GMTED2010 tile naming pattern for different products
        # We'll focus on the mean elevation product (be75, mn75, etc.)
        
        tiles = []
        suffix = self.res_info["suffix"]
        
        # Generate tile names based on GMTED2010 grid system
        # Global coverage tiles for mean elevation product
        if suffix == "30":
            # 30 arc-second tiles (fewer, larger tiles)
            tile_prefixes = [
                "10n000e", "10n030e", "10n060e", "10n090e", "10n120e", "10n150e",
                "10s000e", "10s030e", "10s060e", "10s090e", "10s120e", "10s150e",
                "30n000e", "30n030e", "30n060e", "30n090e", "30n120e", "30n150e",
                "50n000e", "50n030e", "50n060e", "50n090e", "50n120e", "50n150e",
                "70n000e", "70n030e", "70n060e", "70n090e", "70n120e", "70n150e"
            ]
        else:
            # 7.5 and 15 arc-second tiles (more numerous, smaller tiles)
            tile_prefixes = []
            for lat in range(-60, 90, 30):  # -60 to 80 degrees latitude
                for lon in range(-180, 180, 30):  # -180 to 150 degrees longitude
                    lat_str = f"{abs(lat):02d}{'n' if lat >= 0 else 's'}"
                    lon_str = f"{abs(lon):03d}{'e' if lon >= 0 else 'w'}"
                    tile_prefixes.append(f"{lat_str}{lon_str}")
        
        # Create full tile names
        for prefix in tile_prefixes:
            tile_name = f"{prefix}_20101117_gmted_mea{suffix}.tif"
            tiles.append(tile_name)
        
        logger.info(f"Generated {len(tiles)} tile names for {suffix} arc-second resolution")
        return tiles
    
    def download_tile(self, tile_name: str, max_retries: int = 3) -> bool:
        """
        Download a specific DEM tile.
        
        Args:
            tile_name: Name of the tile to download
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if download successful, False otherwise
        """
        tile_path = self.data_dir / tile_name
        
        # Skip if already downloaded
        if tile_path.exists():
            logger.info(f"Tile already exists: {tile_name}")
            return True
        
        # Construct download URL
        url = f"{self.base_url}/{self.res_info['folder']}/{tile_name}"
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {tile_name} (attempt {attempt + 1}/{max_retries})")
                
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()
                
                # Get file size for progress bar
                total_size = int(response.headers.get('content-length', 0))
                
                with open(tile_path, 'wb') as f:
                    with tqdm(
                        desc=tile_name,
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                # Verify download
                if tile_path.exists() and tile_path.stat().st_size > 0:
                    logger.info(f"Successfully downloaded: {tile_name}")
                    return True
                else:
                    logger.warning(f"Download verification failed: {tile_name}")
                    tile_path.unlink(missing_ok=True)
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Download attempt {attempt + 1} failed for {tile_name}: {e}")
                tile_path.unlink(missing_ok=True)
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
            
            except Exception as e:
                logger.error(f"Unexpected error downloading {tile_name}: {e}")
                tile_path.unlink(missing_ok=True)
                break
        
        logger.error(f"Failed to download {tile_name} after {max_retries} attempts")
        return False
    
    def download_global_dem(self, max_concurrent: int = 3) -> None:
        """
        Download all available global DEM tiles.
        
        Args:
            max_concurrent: Maximum number of concurrent downloads
        """
        tiles = self.get_tile_list()
        
        logger.info(f"Starting download of {len(tiles)} DEM tiles")
        logger.info(f"Resolution: {self.resolution} (~{self.res_info['approx_res']})")
        logger.info(f"Target directory: {self.data_dir}")
        
        successful_downloads = 0
        failed_downloads = []
        
        for i, tile_name in enumerate(tiles, 1):
            logger.info(f"Processing tile {i}/{len(tiles)}: {tile_name}")
            
            if self.download_tile(tile_name):
                successful_downloads += 1
            else:
                failed_downloads.append(tile_name)
            
            # Brief pause between downloads to be respectful to server
            time.sleep(0.5)
        
        logger.info(f"Download summary:")
        logger.info(f"  Successful: {successful_downloads}")
        logger.info(f"  Failed: {len(failed_downloads)}")
        
        if failed_downloads:
            logger.warning(f"Failed downloads: {failed_downloads}")
            
            # Save failed downloads list
            failed_file = self.data_dir / "failed_downloads.txt"
            with open(failed_file, 'w') as f:
                for tile in failed_downloads:
                    f.write(f"{tile}\n")
            logger.info(f"Failed downloads list saved to: {failed_file}")
    
    def extract_point_elevations(self, coordinates_df: pd.DataFrame, 
                                lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
        """
        Extract elevation values for specific coordinates.
        
        Args:
            coordinates_df: DataFrame containing lat/lon coordinates
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            
        Returns:
            DataFrame with added elevation column
        """
        logger.info(f"Extracting elevation for {len(coordinates_df)} points")
        
        # Get available DEM files
        dem_files = list(self.data_dir.glob("*.tif"))
        
        if not dem_files:
            raise FileNotFoundError(f"No DEM files found in {self.data_dir}")
        
        logger.info(f"Found {len(dem_files)} DEM files")
        
        # Create GeoDataFrame from coordinates
        geometry = [Point(lon, lat) for lon, lat in 
                   zip(coordinates_df[lon_col], coordinates_df[lat_col])]
        gdf = gpd.GeoDataFrame(coordinates_df.copy(), geometry=geometry, crs="EPSG:4326")
        
        # Initialize elevation column
        gdf['elevation'] = np.nan
        
        # Process each DEM file
        for dem_file in tqdm(dem_files, desc="Processing DEM tiles"):
            try:
                with rasterio.open(dem_file) as src:
                    # Get bounds of current DEM tile
                    bounds = src.bounds
                    
                    # Find points within this tile's bounds
                    mask_within = (
                        (gdf.geometry.x >= bounds.left) & 
                        (gdf.geometry.x <= bounds.right) &
                        (gdf.geometry.y >= bounds.bottom) & 
                        (gdf.geometry.y <= bounds.top) &
                        gdf['elevation'].isna()  # Only process points without elevation
                    )
                    
                    points_in_tile = gdf[mask_within]
                    
                    if len(points_in_tile) > 0:
                        # Extract elevation values
                        coords = [(point.x, point.y) for point in points_in_tile.geometry]
                        elevations = [x[0] for x in src.sample(coords)]
                        
                        # Update elevation values
                        gdf.loc[mask_within, 'elevation'] = elevations
                        
                        logger.info(f"Extracted {len(points_in_tile)} elevations from {dem_file.name}")
            
            except Exception as e:
                logger.warning(f"Error processing {dem_file.name}: {e}")
                continue
        
        # Convert back to regular DataFrame
        result_df = pd.DataFrame(gdf.drop('geometry', axis=1))
        
        # Check for missing elevations
        missing_count = result_df['elevation'].isna().sum()
        if missing_count > 0:
            logger.warning(f"{missing_count} points could not be assigned elevation values")
        
        logger.info(f"Elevation extraction completed")
        logger.info(f"Elevation range: {result_df['elevation'].min():.1f} to {result_df['elevation'].max():.1f} meters")
        
        return result_df
    
    def create_global_mosaic(self, output_path: Optional[str] = None) -> str:
        """
        Create a global mosaic from downloaded DEM tiles.
        
        Args:
            output_path: Output path for mosaic file
            
        Returns:
            Path to created mosaic file
        """
        if output_path is None:
            output_path = self.data_dir / f"global_dem_{self.resolution}.tif"
        
        dem_files = list(self.data_dir.glob("*.tif"))
        
        if not dem_files:
            raise FileNotFoundError(f"No DEM files found in {self.data_dir}")
        
        logger.info(f"Creating global mosaic from {len(dem_files)} tiles")
        logger.info(f"Output: {output_path}")
        
        # Open all DEM files
        src_files_to_mosaic = []
        for dem_file in dem_files:
            src = rasterio.open(dem_file)
            src_files_to_mosaic.append(src)
        
        try:
            # Create mosaic
            mosaic, out_trans = merge(src_files_to_mosaic)
            
            # Update metadata
            out_meta = src_files_to_mosaic[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "compress": "lzw"
            })
            
            # Write mosaic
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(mosaic)
            
            logger.info(f"Global mosaic created: {output_path}")
            
        finally:
            # Close all source files
            for src in src_files_to_mosaic:
                src.close()
        
        return str(output_path)


def main():
    """Main function to demonstrate DEM download functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download global DEM data")
    parser.add_argument("--resolution", choices=["7p5arc", "15arc", "30arc"], 
                       default="30arc", help="DEM resolution")
    parser.add_argument("--download", action="store_true", 
                       help="Download DEM tiles")
    parser.add_argument("--extract-points", type=str,
                       help="CSV file with coordinates for elevation extraction")
    parser.add_argument("--mosaic", action="store_true",
                       help="Create global mosaic from downloaded tiles")
    parser.add_argument("--base-dir", default="../ancillary/elevation",
                       help="Base directory for DEM data")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = GlobalDEMDownloader(args.base_dir, args.resolution)
    
    if args.download:
        logger.info("Starting global DEM download...")
        downloader.download_global_dem()
    
    if args.extract_points:
        logger.info(f"Extracting elevations for points in {args.extract_points}")
        
        # Load coordinates
        coords_df = pd.read_csv(args.extract_points)
        
        # Extract elevations
        result_df = downloader.extract_point_elevations(coords_df)
        
        # Save results
        output_file = f"elevation_points_{args.resolution}.csv"
        result_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")
    
    if args.mosaic:
        logger.info("Creating global mosaic...")
        mosaic_path = downloader.create_global_mosaic()
        logger.info(f"Mosaic created: {mosaic_path}")
    
    # If no specific action requested, show status
    if not any([args.download, args.extract_points, args.mosaic]):
        dem_files = list(downloader.data_dir.glob("*.tif"))
        logger.info(f"Current status:")
        logger.info(f"  Resolution: {args.resolution}")
        logger.info(f"  Data directory: {downloader.data_dir}")
        logger.info(f"  Downloaded tiles: {len(dem_files)}")
        
        if dem_files:
            logger.info("Use --extract-points <csv_file> to extract elevations")
            logger.info("Use --mosaic to create global mosaic")
        else:
            logger.info("Use --download to download DEM tiles")


if __name__ == "__main__":
    main()