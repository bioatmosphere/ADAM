"""
SRTM Global DEM Data Downloader

This module downloads global elevation data from SRTM (Shuttle Radar Topography Mission)
provided by NASA/USGS. The data provides near-global elevation coverage at 1 arc-second 
and 3 arc-second resolution.

Features:
- Downloads SRTM 1 arc-second (30m) and 3 arc-second (90m) data
- Uses NASA EarthData API
- Point extraction for specific coordinates
- Integration with TAM pipeline data formats

Data source: NASA SRTMGL1/SRTMGL3 via OpenTopography or direct NASA sources
Resolution options:
- 1 arc-second (~30m at equator) - SRTMGL1
- 3 arc-second (~90m at equator) - SRTMGL3

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
    from shapely.geometry import Point, box
except ImportError as e:
    print(f"Missing required packages for raster processing: {e}")
    print("Install with: pip install rasterio geopandas shapely")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('srtm_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SRTMDownloader:
    """Download and process global SRTM elevation data."""
    
    def __init__(self, base_dir: str = "../ancillary/elevation", resolution: str = "3arc"):
        """
        Initialize the SRTM downloader.
        
        Args:
            base_dir: Base directory for storing elevation data
            resolution: Resolution option ('1arc' for 30m, '3arc' for 90m)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Resolution settings
        self.resolution = resolution
        self.resolution_info = {
            "1arc": {"folder": "SRTM_1arc", "product": "SRTMGL1", "approx_res": "30m"},
            "3arc": {"folder": "SRTM_3arc", "product": "SRTMGL3", "approx_res": "90m"}
        }
        
        if resolution not in self.resolution_info:
            raise ValueError(f"Resolution must be one of: {list(self.resolution_info.keys())}")
        
        self.res_info = self.resolution_info[resolution]
        
        # OpenTopography SRTM API base URL
        self.base_url = "https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster"
        
        # Alternative: Direct NASA EarthData URLs (requires authentication)
        self.nasa_base_url = "https://e4ftl01.cr.usgs.gov/MEASURES"
        
        # Create resolution-specific directory
        self.data_dir = self.base_dir / self.res_info["folder"]
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized SRTM downloader for {resolution} (~{self.res_info['approx_res']}) resolution")
        logger.info(f"Data directory: {self.data_dir}")
    
    def get_srtm_tiles_for_bounds(self, min_lat: float, max_lat: float, 
                                 min_lon: float, max_lon: float) -> List[str]:
        """
        Get SRTM tile names covering the specified geographic bounds.
        
        Args:
            min_lat, max_lat: Latitude bounds
            min_lon, max_lon: Longitude bounds
            
        Returns:
            List of SRTM tile names
        """
        tiles = []
        
        # SRTM tiles are 1-degree squares
        for lat in range(int(np.floor(min_lat)), int(np.ceil(max_lat)) + 1):
            for lon in range(int(np.floor(min_lon)), int(np.ceil(max_lon)) + 1):
                # Skip areas outside SRTM coverage (approximately 60°N to 56°S)
                if lat < -56 or lat > 60:
                    continue
                
                # Format tile name: N/S followed by latitude, E/W followed by longitude
                lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
                lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
                tile_name = f"{lat_str}{lon_str}"
                tiles.append(tile_name)
        
        return tiles
    
    def get_global_srtm_tiles(self) -> List[str]:
        """
        Get all available SRTM tiles for global coverage.
        
        Returns:
            List of all SRTM tile names
        """
        # SRTM covers approximately 60°N to 56°S
        return self.get_srtm_tiles_for_bounds(-56, 60, -180, 180)
    
    def download_srtm_tile(self, tile_name: str, source: str = "opentopo", 
                          max_retries: int = 3) -> bool:
        """
        Download a specific SRTM tile.
        
        Args:
            tile_name: SRTM tile name (e.g., 'N37W122')
            source: Data source ('opentopo' or 'usgs')
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if download successful, False otherwise
        """
        if source == "opentopo":
            return self._download_from_opentopo(tile_name, max_retries)
        elif source == "usgs":
            return self._download_from_usgs(tile_name, max_retries)
        else:
            raise ValueError("Source must be 'opentopo' or 'usgs'")
    
    def _download_from_opentopo(self, tile_name: str, max_retries: int) -> bool:
        """Download from OpenTopography."""
        file_name = f"{tile_name}.{self.res_info['product']}.hgt.zip"
        tile_path = self.data_dir / file_name
        
        # Skip if already downloaded
        if tile_path.exists():
            logger.info(f"Tile already exists: {file_name}")
            return True
        
        # OpenTopography URL structure
        url = f"{self.base_url}/{self.res_info['product']}/{file_name}"
        
        return self._download_file(url, tile_path, max_retries)
    
    def _download_from_usgs(self, tile_name: str, max_retries: int) -> bool:
        """Download from USGS EarthData (requires authentication)."""
        # This would require NASA EarthData login credentials
        # For now, implement as placeholder
        logger.warning("USGS EarthData download not yet implemented - requires authentication")
        return False
    
    def _download_file(self, url: str, file_path: Path, max_retries: int) -> bool:
        """Generic file download with retries."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {file_path.name} (attempt {attempt + 1}/{max_retries})")
                
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()
                
                # Get file size for progress bar
                total_size = int(response.headers.get('content-length', 0))
                
                with open(file_path, 'wb') as f:
                    with tqdm(
                        desc=file_path.name,
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
                if file_path.exists() and file_path.stat().st_size > 0:
                    logger.info(f"Successfully downloaded: {file_path.name}")
                    return True
                else:
                    logger.warning(f"Download verification failed: {file_path.name}")
                    file_path.unlink(missing_ok=True)
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Download attempt {attempt + 1} failed for {file_path.name}: {e}")
                file_path.unlink(missing_ok=True)
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
            
            except Exception as e:
                logger.error(f"Unexpected error downloading {file_path.name}: {e}")
                file_path.unlink(missing_ok=True)
                break
        
        logger.error(f"Failed to download {file_path.name} after {max_retries} attempts")
        return False
    
    def extract_and_convert_hgt(self, zip_path: Path) -> Optional[Path]:
        """
        Extract HGT file from zip and convert to GeoTIFF.
        
        Args:
            zip_path: Path to downloaded zip file
            
        Returns:
            Path to converted GeoTIFF file, or None if failed
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract HGT file
                hgt_files = [f for f in zip_ref.namelist() if f.endswith('.hgt')]
                
                if not hgt_files:
                    logger.error(f"No HGT file found in {zip_path}")
                    return None
                
                hgt_file = hgt_files[0]
                hgt_path = self.data_dir / hgt_file
                
                with zip_ref.open(hgt_file) as source, open(hgt_path, 'wb') as target:
                    target.write(source.read())
                
                # Convert HGT to GeoTIFF
                tiff_path = hgt_path.with_suffix('.tif')
                self._convert_hgt_to_geotiff(hgt_path, tiff_path)
                
                # Clean up HGT file
                hgt_path.unlink()
                
                return tiff_path
                
        except Exception as e:
            logger.error(f"Error extracting {zip_path}: {e}")
            return None
    
    def _convert_hgt_to_geotiff(self, hgt_path: Path, tiff_path: Path) -> None:
        """Convert HGT binary file to GeoTIFF with proper georeferencing."""
        try:
            # Parse coordinates from filename (e.g., N37W122.hgt)
            filename = hgt_path.stem
            
            # Extract latitude
            if filename[0] == 'N':
                lat = int(filename[1:3])
            else:  # 'S'
                lat = -int(filename[1:3])
            
            # Extract longitude  
            if filename[3] == 'E':
                lon = int(filename[4:7])
            else:  # 'W'
                lon = -int(filename[4:7])
            
            # Determine array size based on resolution
            if self.resolution == "1arc":
                size = 3601  # 1 arc-second
            else:  # 3arc
                size = 1201  # 3 arc-second
            
            # Read HGT data
            with open(hgt_path, 'rb') as f:
                elevation = np.frombuffer(f.read(), dtype='>i2').reshape((size, size))
            
            # Handle no-data values
            elevation = elevation.astype(np.float32)
            elevation[elevation == -32768] = np.nan
            
            # Set up georeferencing
            pixel_size = 1.0 / (size - 1)  # degrees per pixel
            
            # SRTM data is stored from top-left, south-north
            transform = rasterio.transform.from_bounds(
                lon, lat, lon + 1, lat + 1, size, size
            )
            
            # Write GeoTIFF
            with rasterio.open(
                tiff_path, 'w',
                driver='GTiff',
                height=size, width=size,
                count=1, dtype=elevation.dtype,
                crs='EPSG:4326',
                transform=transform,
                compress='lzw',
                nodata=np.nan
            ) as dst:
                dst.write(elevation, 1)
            
            logger.info(f"Converted {hgt_path.name} to {tiff_path.name}")
            
        except Exception as e:
            logger.error(f"Error converting {hgt_path} to GeoTIFF: {e}")
            raise
    
    def download_tiles_for_points(self, coordinates_df: pd.DataFrame,
                                 lat_col: str = "lat", lon_col: str = "lon") -> None:
        """
        Download only the SRTM tiles needed for specific coordinate points.
        
        Args:
            coordinates_df: DataFrame containing lat/lon coordinates
            lat_col: Name of latitude column
            lon_col: Name of longitude column
        """
        # Get bounding box of all points
        min_lat = coordinates_df[lat_col].min()
        max_lat = coordinates_df[lat_col].max()
        min_lon = coordinates_df[lon_col].min()
        max_lon = coordinates_df[lon_col].max()
        
        logger.info(f"Coordinate bounds: {min_lat:.2f}°S to {max_lat:.2f}°N, "
                   f"{min_lon:.2f}°W to {max_lon:.2f}°E")
        
        # Get required tiles
        required_tiles = self.get_srtm_tiles_for_bounds(min_lat, max_lat, min_lon, max_lon)
        
        logger.info(f"Need {len(required_tiles)} SRTM tiles for {len(coordinates_df)} points")
        
        successful_downloads = 0
        failed_downloads = []
        
        for i, tile_name in enumerate(required_tiles, 1):
            logger.info(f"Processing tile {i}/{len(required_tiles)}: {tile_name}")
            
            # Download tile
            if self.download_srtm_tile(tile_name, source="opentopo"):
                # Extract and convert to GeoTIFF
                zip_file = self.data_dir / f"{tile_name}.{self.res_info['product']}.hgt.zip"
                
                if zip_file.exists():
                    tiff_path = self.extract_and_convert_hgt(zip_file)
                    if tiff_path:
                        successful_downloads += 1
                        # Clean up zip file
                        zip_file.unlink()
                    else:
                        failed_downloads.append(tile_name)
                else:
                    failed_downloads.append(tile_name)
            else:
                failed_downloads.append(tile_name)
            
            # Brief pause between downloads
            time.sleep(0.5)
        
        logger.info(f"Download summary:")
        logger.info(f"  Successful: {successful_downloads}")
        logger.info(f"  Failed: {len(failed_downloads)}")
        
        if failed_downloads:
            logger.warning(f"Failed downloads: {failed_downloads}")
    
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
        
        # Get available SRTM files
        srtm_files = list(self.data_dir.glob("*.tif"))
        
        if not srtm_files:
            raise FileNotFoundError(f"No SRTM files found in {self.data_dir}")
        
        logger.info(f"Found {len(srtm_files)} SRTM files")
        
        # Create GeoDataFrame from coordinates
        geometry = [Point(lon, lat) for lon, lat in 
                   zip(coordinates_df[lon_col], coordinates_df[lat_col])]
        gdf = gpd.GeoDataFrame(coordinates_df.copy(), geometry=geometry, crs="EPSG:4326")
        
        # Initialize elevation column
        gdf['elevation'] = np.nan
        
        # Process each SRTM file
        for srtm_file in tqdm(srtm_files, desc="Processing SRTM tiles"):
            try:
                with rasterio.open(srtm_file) as src:
                    # Get bounds of current SRTM tile
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
                        
                        logger.info(f"Extracted {len(points_in_tile)} elevations from {srtm_file.name}")
            
            except Exception as e:
                logger.warning(f"Error processing {srtm_file.name}: {e}")
                continue
        
        # Convert back to regular DataFrame
        result_df = pd.DataFrame(gdf.drop('geometry', axis=1))
        
        # Check for missing elevations
        missing_count = result_df['elevation'].isna().sum()
        if missing_count > 0:
            logger.warning(f"{missing_count} points could not be assigned elevation values")
        
        valid_elevations = result_df['elevation'].dropna()
        if len(valid_elevations) > 0:
            logger.info(f"Elevation extraction completed")
            logger.info(f"Elevation range: {valid_elevations.min():.1f} to {valid_elevations.max():.1f} meters")
        
        return result_df


def main():
    """Main function to demonstrate SRTM download functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download SRTM elevation data")
    parser.add_argument("--resolution", choices=["1arc", "3arc"], 
                       default="3arc", help="SRTM resolution")
    parser.add_argument("--points-file", type=str,
                       help="CSV file with coordinates for elevation extraction")
    parser.add_argument("--download-global", action="store_true",
                       help="Download all global SRTM tiles (warning: very large!)")
    parser.add_argument("--base-dir", default="../ancillary/elevation",
                       help="Base directory for SRTM data")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = SRTMDownloader(args.base_dir, args.resolution)
    
    if args.points_file:
        logger.info(f"Processing coordinates from {args.points_file}")
        
        # Load coordinates
        coords_df = pd.read_csv(args.points_file)
        
        # Download required tiles
        downloader.download_tiles_for_points(coords_df)
        
        # Extract elevations
        result_df = downloader.extract_point_elevations(coords_df)
        
        # Save results
        output_file = f"elevation_points_{args.resolution}_srtm.csv"
        result_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")
    
    elif args.download_global:
        logger.warning("Global download will download thousands of tiles - this may take hours!")
        response = input("Continue? (y/N): ")
        if response.lower() == 'y':
            tiles = downloader.get_global_srtm_tiles()
            logger.info(f"Starting download of {len(tiles)} SRTM tiles")
            # Implementation would go here
        else:
            logger.info("Global download cancelled")
    
    else:
        # Show status
        srtm_files = list(downloader.data_dir.glob("*.tif"))
        logger.info(f"Current status:")
        logger.info(f"  Resolution: {args.resolution}")
        logger.info(f"  Data directory: {downloader.data_dir}")
        logger.info(f"  Downloaded tiles: {len(srtm_files)}")
        
        logger.info("Usage examples:")
        logger.info("  --points-file coordinates.csv  # Download tiles for specific points")
        logger.info("  --download-global              # Download all global SRTM tiles")


if __name__ == "__main__":
    main()