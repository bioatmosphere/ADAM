"""
Optimized soil moisture data processor for ADAM project.

This script provides comprehensive functionality for downloading, processing, and visualizing
soil moisture data from Google Drive and NetCDF files.

Features:
- Concurrent downloads and processing
- Robust error handling and logging
- Memory-efficient data processing
- Comprehensive validation
- Multiple output formats and visualizations

Dependencies:
    uv add gdown xarray matplotlib cartopy tqdm netcdf4 h5netcdf
"""

import os
import sys
import argparse
import zipfile
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Configure matplotlib for better performance
plt.rcParams['figure.max_open_warning'] = 0
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    gdown = None


class SoilMoistureProcessor:
    """
    Optimized soil moisture data processor with comprehensive functionality.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the soil moisture processor.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.logger = self._setup_logging()
        
        # File configurations for Google Drive downloads
        self.files_config = {
            "ec_ors.nc": "1-0Ze2XfDWyOqgxpetQqf5HeVxDxX_y7_",
            "olc_ors.nc": "1-UZ7FEHbqoAXHq6Wa3b5zme8vPeX02kf"
        }
        
        # Soil moisture variable names to try
        self.soil_moisture_vars = ['sm', 'soil_moisture', 'swvl1', 'moisture', 'sm_surface']
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('SoilMoistureProcessor')
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _validate_path(self, path: Union[str, Path], must_exist: bool = True) -> Path:
        """
        Validate and convert path to Path object.
        
        Args:
            path: Input path
            must_exist: Whether path must exist
            
        Returns:
            Validated Path object
            
        Raises:
            FileNotFoundError: If path must exist but doesn't
            ValueError: If path is invalid
        """
        if not path:
            raise ValueError("Path cannot be empty")
            
        path_obj = Path(path)
        
        if must_exist and not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
            
        return path_obj
    
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        Validate DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def download_drive_folder(self, folder_url: str, output_dir: Union[str, Path]) -> bool:
        """
        Download entire Google Drive folder with error handling.
        
        Args:
            folder_url: Google Drive folder URL
            output_dir: Local directory to save files
            
        Returns:
            True if successful, False otherwise
        """
        if not GDOWN_AVAILABLE:
            self.logger.error("gdown library not available. Install with: uv add gdown")
            return False
            
        try:
            output_path = self._validate_path(output_dir, must_exist=False)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Downloading folder from: {folder_url}")
            self.logger.info(f"Output directory: {output_path}")
            
            result = gdown.download_folder(
                folder_url,
                output=str(output_path),
                quiet=not self.verbose,
                use_cookies=False
            )
            
            if result:
                downloaded_files = list(output_path.rglob('*'))
                self.logger.info(f"✓ Downloaded {len(downloaded_files)} files successfully")
                return True
            else:
                self.logger.error("✗ Folder download failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error downloading folder: {e}")
            return False
    
    def download_drive_file(self, file_id: str, output_path: Union[str, Path]) -> bool:
        """
        Download single file from Google Drive with validation.
        
        Args:
            file_id: Google Drive file ID
            output_path: Local path to save file
            
        Returns:
            True if successful, False otherwise
        """
        if not GDOWN_AVAILABLE:
            self.logger.error("gdown library not available. Install with: uv add gdown")
            return False
            
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file already exists
            if output_file.exists():
                self.logger.info(f"File already exists: {output_file}")
                return True
            
            self.logger.info(f"Downloading file: {output_file.name}")
            
            file_url = f"https://drive.google.com/uc?id={file_id}"
            result = gdown.download(file_url, str(output_file), quiet=not self.verbose)
            
            if result and output_file.exists():
                file_size = output_file.stat().st_size
                self.logger.info(f"✓ Download completed: {output_file.name} ({file_size:,} bytes)")
                return True
            else:
                self.logger.error(f"✗ Download failed: {output_file.name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error downloading file: {e}")
            return False
    
    def download_files_concurrent(self, output_dir: Union[str, Path], max_workers: int = 3) -> Dict[str, bool]:
        """
        Download multiple files concurrently for better performance.
        
        Args:
            output_dir: Directory to save files
            max_workers: Maximum number of concurrent downloads
            
        Returns:
            Dictionary mapping filenames to success status
        """
        output_path = self._validate_path(output_dir, must_exist=False)
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.download_drive_file, file_id, output_path / filename): filename
                for filename, file_id in self.files_config.items()
            }
            
            for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Downloading files"):
                filename = future_to_file[future]
                try:
                    results[filename] = future.result()
                except Exception as e:
                    self.logger.error(f"Error downloading {filename}: {e}")
                    results[filename] = False
        
        success_count = sum(results.values())
        self.logger.info(f"✓ Successfully downloaded {success_count}/{len(self.files_config)} files")
        return results
    
    def extract_compressed_files(self, zip_dir: Union[str, Path], extract_dir: Union[str, Path]) -> None:
        """
        Extract compressed files with improved error handling.
        
        Args:
            zip_dir: Directory containing zip files
            extract_dir: Directory to extract files to
        """
        zip_path = self._validate_path(zip_dir)
        extract_path = self._validate_path(extract_dir, must_exist=False)
        extract_path.mkdir(parents=True, exist_ok=True)
        
        zip_files = list(zip_path.glob("*.zip"))
        if not zip_files:
            self.logger.warning(f"No zip files found in {zip_path}")
            return
        
        self.logger.info(f"Extracting {len(zip_files)} zip files")
        
        for zip_file in tqdm(zip_files, desc="Extracting files"):
            try:
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    subfolder = extract_path / zip_file.stem
                    subfolder.mkdir(exist_ok=True)
                    
                    # Extract with progress tracking
                    for member in zf.namelist():
                        zf.extract(member, subfolder)
                    
                self.logger.info(f"Extracted {zip_file.name} to {subfolder}")
                
            except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
                self.logger.error(f"Error with zip file {zip_file}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error extracting {zip_file}: {e}")
    
    def _detect_soil_moisture_variable(self, dataset: xr.Dataset) -> Tuple[str, xr.DataArray]:
        """
        Detect and process soil moisture variable from dataset.
        
        Args:
            dataset: xarray Dataset
            
        Returns:
            Tuple of (variable_name, processed_data_array)
            
        Raises:
            ValueError: If no soil moisture variable found
        """
        # Try to find soil moisture variable
        found_var = None
        for var_name in self.soil_moisture_vars:
            if var_name in dataset:
                found_var = var_name
                break
        
        if not found_var:
            available_vars = list(dataset.data_vars.keys())
            raise ValueError(f"No soil moisture variable found. Available variables: {available_vars}")
        
        data_array = dataset[found_var]
        self.logger.info(f"Using variable: {found_var}")
        self.logger.info(f"Dimensions: {data_array.dims}")
        self.logger.info(f"Shape: {data_array.shape}")
        
        # Process based on dimensions
        if len(data_array.dims) == 4:  # time, depth, lat, lon
            processed_data = data_array[:, 0, :, :].mean(dim='time')  # Top layer, annual mean
            self.logger.info("Processing 4D data: time, depth, lat, lon -> annual mean at surface")
        elif len(data_array.dims) == 3:  # time, lat, lon
            processed_data = data_array.mean(dim='time')
            self.logger.info("Processing 3D data: time, lat, lon -> annual mean")
        else:
            processed_data = data_array
            self.logger.warning(f"Unexpected dimensions: {data_array.dims}")
        
        return found_var, processed_data
    
    def extract_soil_moisture_points(self, nc_file_path: Union[str, Path], 
                                   lat_lon_file: Union[str, Path], 
                                   output_file: Optional[Union[str, Path]] = None,
                                   chunk_size: int = 100) -> pd.DataFrame:
        """
        Extract soil moisture values at specific coordinates with optimized performance.
        
        Args:
            nc_file_path: Path to NetCDF file
            lat_lon_file: Path to CSV file with lat/lon coordinates
            output_file: Optional output CSV file path
            chunk_size: Process coordinates in chunks for better performance
            
        Returns:
            DataFrame with lat, lon, and soil moisture values
        """
        # Validate inputs
        nc_path = self._validate_path(nc_file_path)
        coords_path = self._validate_path(lat_lon_file)
        
        # Load coordinates
        coords_df = pd.read_csv(coords_path)
        self._validate_dataframe(coords_df, ['lat', 'lon'])
        
        self.logger.info(f"Loading soil moisture data from: {nc_path}")
        self.logger.info(f"Extracting for {len(coords_df)} coordinate points")
        
        # Load dataset with chunking for better performance
        try:
            with xr.open_dataset(nc_path, chunks={'time': 12}) as ds:
                var_name, annual_mean = self._detect_soil_moisture_variable(ds)
                
                # Extract values in chunks for better memory management
                results = []
                total_chunks = (len(coords_df) + chunk_size - 1) // chunk_size
                
                for i in tqdm(range(0, len(coords_df), chunk_size), 
                             total=total_chunks, desc="Extracting points"):
                    chunk_coords = coords_df.iloc[i:i+chunk_size]
                    
                    for _, row in chunk_coords.iterrows():
                        lat, lon = row['lat'], row['lon']
                        
                        try:
                            # Use nearest neighbor interpolation
                            point_data = annual_mean.sel(lat=lat, lon=lon, method='nearest')
                            soil_moisture_value = float(point_data.values)
                            
                            # Validate extracted value
                            if np.isnan(soil_moisture_value) or not np.isfinite(soil_moisture_value):
                                soil_moisture_value = np.nan
                            
                            results.append({
                                'lat': lat,
                                'lon': lon,
                                'soil_moisture': soil_moisture_value
                            })
                            
                        except Exception as e:
                            self.logger.warning(f"Error processing point ({lat}, {lon}): {e}")
                            results.append({
                                'lat': lat,
                                'lon': lon,
                                'soil_moisture': np.nan
                            })
                
                # Create results DataFrame
                results_df = pd.DataFrame(results)
                
                # Save results if output file specified
                if output_file:
                    output_path = Path(output_file)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    results_df.to_csv(output_path, index=False)
                    self.logger.info(f"Results saved to: {output_path}")
                
                # Print summary statistics
                self._print_extraction_summary(results_df)
                
                return results_df
                
        except Exception as e:
            self.logger.error(f"Error extracting soil moisture data: {e}")
            raise
    
    def _print_extraction_summary(self, results_df: pd.DataFrame) -> None:
        """Print extraction summary statistics."""
        valid_data = results_df['soil_moisture'].dropna()
        
        self.logger.info(f"\\nExtraction Summary:")
        self.logger.info(f"Total points: {len(results_df)}")
        self.logger.info(f"Valid data points: {len(valid_data)}")
        self.logger.info(f"Missing data points: {len(results_df) - len(valid_data)}")
        self.logger.info(f"Coverage: {len(valid_data)/len(results_df)*100:.1f}%")
        
        if len(valid_data) > 0:
            self.logger.info(f"Soil moisture range: {valid_data.min():.3f} to {valid_data.max():.3f}")
            self.logger.info(f"Mean: {valid_data.mean():.3f}, Median: {valid_data.median():.3f}")
    
    def visualize_global_soil_moisture(self, file_path: Union[str, Path], 
                                     output_file: Optional[Union[str, Path]] = None) -> None:
        """
        Create enhanced global soil moisture visualization.
        
        Args:
            file_path: Path to NetCDF file
            output_file: Optional output path for the map
        """
        nc_path = self._validate_path(file_path)
        
        try:
            with xr.open_dataset(nc_path) as ds:
                var_name, annual_mean = self._detect_soil_moisture_variable(ds)
                
                # Create enhanced figure
                fig, ax = plt.subplots(figsize=(16, 10), 
                                     subplot_kw={'projection': ccrs.PlateCarree()})
                
                # Define custom colormap for soil moisture
                colors = ['#8B4513', '#CD853F', '#DEB887', '#F0E68C', '#ADFF2F', '#00FF00', '#0000FF']
                n_bins = 20
                cmap = mcolors.LinearSegmentedColormap.from_list('soil_moisture', colors, N=n_bins)
                
                # Plot with improved styling
                im = annual_mean.plot(ax=ax,
                                    cmap=cmap,
                                    transform=ccrs.PlateCarree(),
                                    add_colorbar=False,
                                    vmin=0.0, vmax=0.5)
                
                # Add enhanced map features
                ax.coastlines(resolution='50m', color='black', linewidth=0.8)
                ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.7)
                ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
                ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
                
                # Add gridlines
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
                cbar.set_label('Soil Moisture (volumetric water content)', fontsize=12)
                
                # Set title
                file_name = nc_path.stem
                plt.title(f'Global Soil Moisture Distribution - {file_name}', 
                         fontsize=16, pad=20, weight='bold')
                
                # Save figure
                if output_file is None:
                    output_file = nc_path.parent / f"{file_name}_global_soil_moisture.png"
                
                plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                self.logger.info(f"Global soil moisture map saved to: {output_file}")
                
                plt.close(fig)  # Close to free memory
                
        except Exception as e:
            self.logger.error(f"Error creating global visualization: {e}")
            raise
    
    def map_soil_moisture_points(self, points_file: Union[str, Path], 
                               output_file: Optional[Union[str, Path]] = None) -> None:
        """
        Create optimized point visualization with enhanced styling.
        
        Args:
            points_file: Path to CSV file with soil moisture point data
            output_file: Optional output path for the map
        """
        points_path = self._validate_path(points_file)
        
        try:
            # Load and process data
            df = pd.read_csv(points_path)
            self._validate_dataframe(df, ['lat', 'lon', 'soil_moisture'])
            
            # Clean data
            unique_df = df.drop_duplicates(subset=['lat', 'lon'])
            valid_df = unique_df.dropna(subset=['soil_moisture'])
            
            self.logger.info(f"Data summary:")
            self.logger.info(f"  Total points: {len(df)}")
            self.logger.info(f"  Unique coordinates: {len(unique_df)}")
            self.logger.info(f"  Valid measurements: {len(valid_df)}")
            
            if len(valid_df) == 0:
                self.logger.warning("No valid data points found for mapping")
                return
            
            # Create enhanced figure
            fig, ax = plt.subplots(figsize=(16, 10), 
                                 subplot_kw={'projection': ccrs.PlateCarree()})
            
            # Add map features
            ax.coastlines(resolution='50m', color='black', linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.6)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.4)
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.4)
            ax.set_global()
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            
            # Create scatter plot with enhanced styling
            scatter = ax.scatter(valid_df['lon'], valid_df['lat'],
                               c=valid_df['soil_moisture'],
                               cmap='RdYlBu_r',
                               s=80,
                               alpha=0.8,
                               edgecolors='black',
                               linewidths=0.5,
                               transform=ccrs.PlateCarree())
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                              pad=0.05, shrink=0.8)
            cbar.set_label('Soil Moisture (volumetric water content)', fontsize=12)
            
            # Enhanced title
            stats_text = f"Range: {valid_df['soil_moisture'].min():.3f}-{valid_df['soil_moisture'].max():.3f}"
            plt.title(f'Global Soil Moisture Data Points\\n{len(valid_df)} locations | {stats_text}', 
                     fontsize=16, pad=20, weight='bold')
            
            # Save figure
            if output_file is None:
                output_file = points_path.parent / 'soil_moisture_points_map_enhanced.png'
            
            plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            self.logger.info(f"Points map saved to: {output_file}")
            
            plt.close(fig)  # Close to free memory
            
            # Print detailed statistics
            self._print_detailed_statistics(valid_df)
            
        except Exception as e:
            self.logger.error(f"Error creating points map: {e}")
            raise
    
    def _print_detailed_statistics(self, df: pd.DataFrame) -> None:
        """Print detailed statistics for soil moisture data."""
        soil_moisture = df['soil_moisture']
        
        self.logger.info(f"\\nDetailed Statistics:")
        self.logger.info(f"  Count: {len(soil_moisture)}")
        self.logger.info(f"  Mean: {soil_moisture.mean():.4f}")
        self.logger.info(f"  Median: {soil_moisture.median():.4f}")
        self.logger.info(f"  Std Dev: {soil_moisture.std():.4f}")
        self.logger.info(f"  Min: {soil_moisture.min():.4f}")
        self.logger.info(f"  Max: {soil_moisture.max():.4f}")
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95]
        self.logger.info(f"  Percentiles:")
        for p in percentiles:
            value = np.percentile(soil_moisture, p)
            self.logger.info(f"    {p}th: {value:.4f}")


def main():
    """Main function with enhanced command line interface."""
    parser = argparse.ArgumentParser(
        description='Optimized soil moisture data processor for ADAM project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Download all files
  python soil_moisture.py --mode individual --verbose
  
  # Extract soil moisture for coordinates
  python soil_moisture.py --mode points --file-path data.nc --lat-lon-file coords.csv
  
  # Create enhanced visualizations
  python soil_moisture.py --mode map --points-file extracted_points.csv --verbose
        '''
    )
    
    parser.add_argument('--mode', 
                       choices=['folder', 'individual', 'extract', 'visualize', 'points', 'map'], 
                       default='individual',
                       help='Operation mode')
    
    parser.add_argument('--folder-url', type=str,
                       default='https://drive.google.com/drive/folders/1bm57jo6yUHGJ0P-sfPwA4NM5VCzSLoUr',
                       help='Google Drive folder URL')
    
    parser.add_argument('--output-dir', '-o', type=str,
                       default='../../ancillary/soilmoisture',
                       help='Output directory for downloaded files')
    
    parser.add_argument('--extract-dir', type=str,
                       default='../../ancillary/soilmoisture/extracted',
                       help='Directory to extract compressed files')
    
    parser.add_argument('--file-path', type=str,
                       help='Path to NetCDF file for processing')
    
    parser.add_argument('--lat-lon-file', type=str,
                       default='../../productivity/earth/lat_lon.csv',
                       help='CSV file with lat/lon coordinates')
    
    parser.add_argument('--output-points', type=str,
                       default='../../ancillary/soilmoisture/soil_moisture_points.csv',
                       help='Output CSV file for point extraction')
    
    parser.add_argument('--points-file', type=str,
                       help='CSV file with extracted soil moisture points')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    parser.add_argument('--max-workers', type=int, default=3,
                       help='Maximum number of concurrent downloads')
    
    parser.add_argument('--chunk-size', type=int, default=100,
                       help='Chunk size for processing coordinates')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = SoilMoistureProcessor(verbose=args.verbose)
    
    print("=== Optimized Soil Moisture Data Processor ===")
    print(f"Mode: {args.mode}")
    print(f"Verbose: {args.verbose}")
    print()
    
    try:
        if args.mode == 'folder':
            success = processor.download_drive_folder(args.folder_url, args.output_dir)
            if success:
                print(f"\\n✓ Folder downloaded to: {args.output_dir}")
            else:
                print("✗ Folder download failed")
        
        elif args.mode == 'individual':
            results = processor.download_files_concurrent(args.output_dir, args.max_workers)
            success_count = sum(results.values())
            print(f"\\n✓ Downloaded {success_count}/{len(results)} files")
        
        elif args.mode == 'extract':
            processor.extract_compressed_files(args.output_dir, args.extract_dir)
            print(f"✓ Files extracted to: {args.extract_dir}")
        
        elif args.mode == 'visualize':
            if not args.file_path:
                # Find NetCDF files automatically
                nc_files = list(Path(args.extract_dir).glob("*.nc"))
                if not nc_files:
                    print("Error: No NetCDF files found. Specify --file-path")
                    return
                file_path = nc_files[0]
            else:
                file_path = args.file_path
            
            processor.visualize_global_soil_moisture(file_path)
            print("✓ Global visualization completed")
        
        elif args.mode == 'points':
            if not args.file_path:
                print("Error: --file-path is required for point extraction")
                return
            
            result_df = processor.extract_soil_moisture_points(
                args.file_path, 
                args.lat_lon_file, 
                args.output_points,
                args.chunk_size
            )
            print(f"✓ Extracted data for {len(result_df)} coordinates")
        
        elif args.mode == 'map':
            points_file = args.points_file or args.output_points
            processor.map_soil_moisture_points(points_file)
            print("✓ Points map visualization completed")
        
        print("\\n=== Processing Complete ===")
        
    except Exception as e:
        print(f"\\n✗ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()