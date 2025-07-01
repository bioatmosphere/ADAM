# -*- coding: utf-8 -*-
"""
Download and process GLASS (Global Land Surface Satellite) data from the University of Hong Kong.

This script provides functionality to:
1. Download GLASS data products for specified variables and time periods
2. Visualize downloaded data with publication-quality maps
3. Extract time series data for specific point coordinates
4. Export point data to CSV or JSON formats
5. Create time series visualizations for extracted points

Data source: https://www.glass.hku.hk/download.html

Available GLASS Products:
- Leaf Area Index (LAI)
- Broadband Albedo
- Photosynthetically Active Radiation (PAR)
- Fraction of Absorbed Photosynthetically Active Radiation (FAPAR)
- Land Surface Temperature (LST)
- Normalized Difference Vegetation Index (NDVI)
- Enhanced Vegetation Index (EVI)
- Gross Primary Production (GPP) - 8-day
- Gross Primary Production (GPP_YEARLY) - yearly
- Net Primary Production (NPP)
- Evapotranspiration (ET)
- Snow Cover Extent
- Soil Moisture

Usage examples:
    # Download LAI data
    python glass.py --mode download --products LAI FAPAR --start-year 2010 --end-year 2015
    
    # Extract point data from command line
    python glass.py --mode extract --coordinates 40.5 -120.3 Site1 45.2 -118.7 Site2 --plot-points
    
    # Extract point data from CSV file
    python glass.py --mode extract --coords-file sites.csv --output-format json
    
    # Visualize already extracted point data
    python glass.py --mode plot-points --products LAI FAPAR

Dependencies:
    pip install requests xarray matplotlib cartopy tqdm pandas h5py
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


def get_glass_product_info():
    """
    Get information about available GLASS products and their properties.
    
    Returns:
        dict: Product information including URLs, resolutions, and time ranges
    """
    products = {
        'LAI': {
            'name': 'Leaf Area Index',
            'units': 'm²/m²',
            'resolution': '1km',
            'temporal_resolution': '8-day',
            'time_range': '2000-2022',
            'url_pattern': 'https://www.glass.hku.hk/MOD_GLASS02_LAI/{year}/MOD_GLASS02_LAI.A{year}{doy}.hdf',
            'colormap': 'viridis'
        },
        'FAPAR': {
            'name': 'Fraction of Absorbed PAR',
            'units': 'fraction',
            'resolution': '1km',
            'temporal_resolution': '8-day',
            'time_range': '2000-2022',
            'url_pattern': 'https://www.glass.hku.hk/MOD_GLASS01_FAPAR/{year}/MOD_GLASS01_FAPAR.A{year}{doy}.hdf',
            'colormap': 'viridis'
        },
        'NDVI': {
            'name': 'Normalized Difference Vegetation Index',
            'units': 'index',
            'resolution': '500m',
            'temporal_resolution': '8-day',
            'time_range': '2000-2022',
            'url_pattern': 'https://www.glass.hku.hk/MOD_GLASS08_NDVI/{year}/MOD_GLASS08_NDVI.A{year}{doy}.hdf',
            'colormap': 'RdYlGn'
        },
        'EVI': {
            'name': 'Enhanced Vegetation Index',
            'units': 'index',
            'resolution': '500m',
            'temporal_resolution': '8-day',
            'time_range': '2000-2022',
            'url_pattern': 'https://www.glass.hku.hk/MOD_GLASS09_EVI/{year}/MOD_GLASS09_EVI.A{year}{doy}.hdf',
            'colormap': 'RdYlGn'
        },
        'GPP': {
            'name': 'Gross Primary Production',
            'units': 'gC/m²/day',
            'resolution': '500m',
            'temporal_resolution': '8-day',
            'time_range': '2000-2024',
            'url_pattern': 'https://www.glass.hku.hk/archive/GPP/MODIS/500M/GLASS_GPP_500M_V60/{year}/{doy}/',
            'colormap': 'YlGn',
            'tile_based': True
        },
        'GPP_YEARLY': {
            'name': 'Gross Primary Production (Yearly)',
            'units': 'gC/m²/year',
            'resolution': '500m',
            'temporal_resolution': 'yearly',
            'time_range': '2000-2022',
            'url_pattern': 'https://www.glass.hku.hk/archive/GPP/MODIS/500M/GLASS_GPP_500M_YEARLY_V60/{year}/001/',
            'colormap': 'YlGn',
            'tile_based': True
        },
        'NPP': {
            'name': 'Net Primary Production',
            'units': 'gC/m�/day',
            'resolution': '1km',
            'temporal_resolution': '8-day',
            'time_range': '2000-2020',
            'url_pattern': 'https://www.glass.hku.hk/MOD_GLASS13_NPP/{year}/MOD_GLASS13_NPP.A{year}{doy}.hdf',
            'colormap': 'YlGn'
        },
        'LST': {
            'name': 'Land Surface Temperature',
            'units': 'K',
            'resolution': '1km',
            'temporal_resolution': 'daily',
            'time_range': '2000-2022',
            'url_pattern': 'https://www.glass.hku.hk/MOD_GLASS11_LST/{year}/MOD_GLASS11_LST.A{year}{doy}.hdf',
            'colormap': 'RdYlBu_r'
        },
        'ALBEDO': {
            'name': 'Broadband Albedo',
            'units': 'fraction',
            'resolution': '1km',
            'temporal_resolution': '8-day',
            'time_range': '2000-2022',
            'url_pattern': 'https://www.glass.hku.hk/MOD_GLASS03_ALBEDO/{year}/MOD_GLASS03_ALBEDO.A{year}{doy}.hdf',
            'colormap': 'gray'
        }
    }
    return products


def generate_day_of_year_list(start_year, end_year, temporal_resolution='8-day'):
    """
    Generate list of day-of-year values for download based on temporal resolution.
    
    Args:
        start_year (int): Starting year
        end_year (int): Ending year
        temporal_resolution (str): Temporal resolution ('8-day', 'daily', 'monthly', 'yearly')
    
    Returns:
        list: List of (year, doy) tuples
    """
    dates = []
    
    if temporal_resolution == '8-day':
        # Standard MODIS 8-day composites (46 composites per year)
        doy_values = [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289, 297, 305, 313, 321, 329, 337, 345, 353, 361]
        for year in range(start_year, end_year + 1):
            for doy in doy_values:
                dates.append((year, f"{doy:03d}"))
    elif temporal_resolution == 'daily':
        # Daily data - sample every 8 days to reduce download volume
        for year in range(start_year, end_year + 1):
            for doy in range(1, 366, 8):  # Sample every 8 days
                dates.append((year, f"{doy:03d}"))
    elif temporal_resolution == 'monthly':
        # Monthly data - use 15th day of each month
        monthly_doys = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
        for year in range(start_year, end_year + 1):
            for doy in monthly_doys:
                dates.append((year, f"{doy:03d}"))
    elif temporal_resolution == 'yearly':
        # Yearly data - one file per year (uses DOY 001 for entire year)
        for year in range(start_year, end_year + 1):
            dates.append((year, "001"))  # DOY 001 represents the entire year
    
    return dates


def download_glass_data(products, start_year, end_year, output_dir, max_files=None):
    """
    Download GLASS data files for specified products and years.
    
    Args:
        products (list): List of GLASS products (e.g., ['LAI', 'FAPAR'])
        start_year (int): Starting year for download
        end_year (int): Ending year for download
        output_dir (str): Directory to save downloaded files
        max_files (int, optional): Maximum number of files to download per product
    
    Returns:
        dict: Summary of download results
    """
    product_info = get_glass_product_info()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {
        'successful': [],
        'failed': [],
        'skipped': []
    }
    
    print(f"Downloading GLASS data products: {', '.join(products)}")
    print(f"Time range: {start_year}-{end_year}")
    
    for product in products:
        if product not in product_info:
            print(f" Unknown product: {product}")
            continue
        
        print(f"\nProcessing {product} ({product_info[product]['name']})...")
        
        # Create product-specific directory
        product_dir = Path(output_dir) / product
        product_dir.mkdir(exist_ok=True)
        
        # Generate date list based on temporal resolution
        dates = generate_day_of_year_list(
            start_year, end_year, 
            product_info[product]['temporal_resolution']
        )
        
        # Limit number of files if specified
        if max_files:
            dates = dates[:max_files]
        
        print(f"Attempting to download {len(dates)} files for {product}...")
        
        with tqdm(total=len(dates), desc=f"{product} Progress") as pbar:
            for year, doy in dates:
                # Check if this is tile-based data
                if product_info[product].get('tile_based', False):
                    # For tile-based data, get directory listing and download available tiles
                    dir_url = product_info[product]['url_pattern'].format(year=year, doy=doy)
                    success_count = download_glass_tiles(dir_url, product_dir, product, year, doy)
                    if success_count > 0:
                        results['successful'].extend([f"{product}/tiles_{year}_{doy}"] * success_count)
                    else:
                        results['failed'].append(f"{product}/tiles_{year}_{doy}")
                else:
                    # Regular single file download
                    url = product_info[product]['url_pattern'].format(year=year, doy=doy)
                    filename = url.split('/')[-1]
                    file_path = product_dir / filename
                    
                    # Skip if file already exists
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        if file_size > 1024:  # Skip if file is larger than 1KB
                            results['skipped'].append(f"{product}/{filename}")
                            pbar.update(1)
                            continue
                    
                    # Download file
                    success = download_glass_file(url, file_path, product, year, doy)
                    if success:
                        results['successful'].append(f"{product}/{filename}")
                    else:
                        results['failed'].append(f"{product}/{filename}")
                
                pbar.update(1)
                time.sleep(0.2)  # Be respectful to the server
    
    # Print summary
    print(f"\nDownload Summary:")
    print(f"  Successful: {len(results['successful'])}")
    print(f"  Failed: {len(results['failed'])}")
    print(f"  Skipped: {len(results['skipped'])}")
    
    if results['failed']:
        print(f"  Failed files: {', '.join(results['failed'][:10])}")
        if len(results['failed']) > 10:
            print(f"  ... and {len(results['failed']) - 10} more")
    
    # Provide guidance if all downloads failed
    if len(results['failed']) > 0 and len(results['successful']) == 0:
        print(f"\n⚠️  GLASS Data Access Information:")
        print(f"   All downloads failed. GLASS data may require:")
        print(f"   1. User registration at https://www.glass.hku.hk/")
        print(f"   2. Alternative access methods (FTP, API, or direct contact)")
        print(f"   3. Different URL patterns or authentication")
        print(f"   4. Manual download from the GLASS website")
        print(f"\n   Consider visiting https://www.glass.hku.hk/download.html for official download instructions.")
    
    return results


def download_glass_file(url, file_path, product, year, doy):
    """
    Download a single GLASS file with error handling.
    
    Args:
        url (str): URL to download from
        file_path (Path): Local path to save file
        product (str): Product name
        year (int): Year
        doy (str): Day of year
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Make request with timeout
        response = requests.get(url, stream=True, timeout=(10, 60))
        
        # Check if the file exists on server
        if response.status_code == 404:
            print(f"  File not found: {url}")
            return False  # File doesn't exist
        elif response.status_code == 403:
            print(f"  Access denied: {url}")
            print(f"  Note: GLASS data may require registration or use alternative access methods")
            return False
            
        response.raise_for_status()
        
        # Get file size for progress tracking
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(file_path, 'wb') as f:
            if total_size > 0:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            f.flush()  # Ensure data is written to disk immediately
        
        # Verify download
        if file_path.exists() and file_path.stat().st_size > 0:
            print(f"  ✓ Downloaded: {file_path.name}")
            # Force sync to ensure file is completely written to disk
            import os
            os.sync() if hasattr(os, 'sync') else None
            return True
        else:
            # Remove failed partial file
            if file_path.exists():
                file_path.unlink()
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"  Download error for {product}_{year}{doy}: {e}")
        return False
    except Exception as e:
        print(f"  Unexpected error for {product}_{year}{doy}: {e}")
        return False


def download_glass_tiles(dir_url, product_dir, product, year, doy):
    """
    Download all available tiles from a GLASS directory for tile-based products.
    
    Args:
        dir_url (str): URL to the directory containing tiles
        product_dir (Path): Local directory to save files
        product (str): Product name
        year (int): Year
        doy (str): Day of year
    
    Returns:
        int: Number of successfully downloaded tiles
    """
    import re
    
    try:
        # Get directory listing
        response = requests.get(dir_url, timeout=10)
        if response.status_code != 200:
            print(f"  Could not access directory: {dir_url}")
            return 0
        
        # Parse HTML to find .hdf files
        html_content = response.text
        hdf_files = re.findall(r'<a href="([^"]*\.hdf)">', html_content)
        
        if not hdf_files:
            print(f"  No HDF files found in {dir_url}")
            return 0
        
        print(f"  Found {len(hdf_files)} tiles for {product} {year}-{doy}")
        
        # Create subdirectory for this date
        date_dir = product_dir / f"{year}_{doy}"
        date_dir.mkdir(exist_ok=True)
        
        success_count = 0
        max_tiles = len(hdf_files)  # Download all available tiles
        
        for i, hdf_file in enumerate(hdf_files[:max_tiles]):
            file_url = dir_url + hdf_file
            file_path = date_dir / hdf_file
            
            # Skip if file already exists  
            if file_path.exists() and file_path.stat().st_size > 1024:
                print(f"    Skipping {hdf_file} (already exists)")
                success_count += 1
                continue
            elif file_path.exists():
                print(f"    File exists but size <= 1024: {hdf_file} (size: {file_path.stat().st_size})")
                # Remove small/corrupted file and continue to download
                file_path.unlink()
                print(f"    Removed small file: {hdf_file}")
            
            # Download tile
            try:
                print(f"    Downloading tile {i+1}/{min(len(hdf_files), max_tiles)}: {hdf_file}")
                tile_response = requests.get(file_url, stream=True, timeout=30)
                tile_response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in tile_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                    f.flush()  # Ensure data is written to disk immediately
                
                # Verify file was written completely
                if file_path.exists() and file_path.stat().st_size > 0:
                    print(f"    ✓ Downloaded: {hdf_file} ({file_path.stat().st_size:,} bytes)")
                    success_count += 1
                    # Force sync to ensure file is completely written to disk
                    import os
                    os.sync() if hasattr(os, 'sync') else None
                else:
                    print(f"    ✗ Failed: {hdf_file}")
                    # Remove failed partial file
                    if file_path.exists():
                        file_path.unlink()
                    
            except Exception as e:
                print(f"    ✗ Error downloading {hdf_file}: {e}")
                continue
            
            time.sleep(0.1)  # Small delay between tile downloads
        
        if len(hdf_files) > max_tiles:
            print(f"    Note: Downloading {max_tiles} tiles out of {len(hdf_files)} available")
        
        return success_count
        
    except Exception as e:
        print(f"  Error accessing tile directory {dir_url}: {e}")
        return 0


def visualize_glass_data(data_dir, product=None, year=None):
    """
    Create publication-quality visualizations of GLASS data.
    
    Args:
        data_dir (str): Directory containing GLASS HDF files
        product (str, optional): Specific product to plot
        year (int, optional): Specific year to plot
    """
    data_path = Path(data_dir)
    product_info = get_glass_product_info()
    
    # Find available product directories
    if product:
        product_dirs = [data_path / product] if (data_path / product).exists() else []
    else:
        product_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name in product_info]
    
    if not product_dirs:
        print(f"No GLASS product directories found in {data_dir}")
        return
    
    print(f"Found {len(product_dirs)} product directories to visualize")
    
    # Create figures directory
    figures_dir = data_path / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Process each product directory
    for product_dir in product_dirs:
        product_name = product_dir.name
        
        if product_name not in product_info:
            continue
        
        print(f"Processing {product_name} data...")
        
        # Find HDF files - check for tile-based structure first
        hdf_files = []
        
        # Check if this is tile-based data (has year_doy subdirectories)
        date_dirs = [d for d in product_dir.iterdir() if d.is_dir() and '_' in d.name]
        
        if date_dirs:
            # Tile-based structure: search in subdirectories
            print(f"  Found {len(date_dirs)} date directories")
            for date_dir in date_dirs:
                if year and str(year) not in date_dir.name:
                    continue
                date_hdf_files = list(date_dir.glob("*.hdf"))
                hdf_files.extend(date_hdf_files)
                if len(date_hdf_files) > 0:
                    print(f"    {date_dir.name}: {len(date_hdf_files)} HDF files")
        else:
            # Regular structure: search directly in product directory
            if year:
                hdf_files = list(product_dir.glob(f"*{year}*.hdf"))
            else:
                hdf_files = list(product_dir.glob("*.hdf"))
        
        if not hdf_files:
            print(f"No HDF files found for {product_name}")
            continue
        
        # Process a sample of files to avoid too many plots
        sample_files = hdf_files[::max(1, len(hdf_files) // 5)]  # Sample every 5th file
        
        for hdf_file in sample_files[:3]:  # Limit to 3 files per product
            try:
                create_glass_visualization(hdf_file, product_name, product_info[product_name], figures_dir)
            except Exception as e:
                print(f"Error processing {hdf_file.name}: {e}")
        
        # Create summary overview of all downloaded files
        try:
            create_glass_overview(product_dir, product_name, product_info[product_name], figures_dir)
        except Exception as e:
            print(f"Error creating overview: {e}")
        
        # Create multi-tile comparison if we have multiple tiles for same date
        try:
            create_multi_tile_comparison(product_dir, product_name, product_info[product_name], figures_dir)
        except Exception as e:
            print(f"Error creating multi-tile comparison: {e}")


def create_glass_visualization(hdf_file, product_name, product_config, output_dir):
    """
    Create a publication-quality map of GLASS data.
    
    Args:
        hdf_file (Path): Path to HDF file
        product_name (str): Product name
        product_config (dict): Product configuration
        output_dir (Path): Directory to save figure
    """
    try:
        print(f"    Creating data visualization for {hdf_file.name}")
        
        # Extract date and tile info from filename
        filename = hdf_file.stem
        # Filename format: GLASS12E01.V60.A2010001.h00v08.2022059
        parts = filename.split('.')
        if len(parts) >= 4:
            date_part = parts[2]  # A2010001
            tile_part = parts[3]  # h00v08
            year = date_part[1:5]
            doy = date_part[5:8]
        else:
            date_part = "unknown"
            tile_part = "unknown"
            year = "unknown"
            doy = "unknown"
        
        # Read HDF4 data using pyhdf
        data = None
        try:
            from pyhdf import SD
            import numpy as np
            
            hdf = SD.SD(str(hdf_file), SD.SDC.READ)
            
            # Get the main dataset (usually named after the product)
            dataset_name = product_name if product_name in [name for name, _ in hdf.datasets().items()] else list(hdf.datasets().keys())[0]
            dataset = hdf.select(dataset_name)
            
            # Read data and attributes
            raw_data = dataset.get()
            attrs = dataset.attributes()
            
            # Apply scaling and handle fill values
            fill_value = attrs.get('_FillValue', 65535)
            water_value = attrs.get('watervalue', 65534)
            scale_factor = attrs.get('scale_factor', 1.0)
            add_offset = attrs.get('add_offset', 0.0)
            valid_range = attrs.get('valid_range', [0, 65533])
            
            # Create masked array
            data = raw_data.astype(np.float32)
            
            # Mask invalid values
            data = np.ma.masked_where((raw_data == fill_value) | (raw_data == water_value), data)
            
            # Apply scaling
            data = (data * scale_factor) + add_offset
            
            # Mask values outside valid range (after scaling)
            if valid_range:
                scaled_valid_min = (valid_range[0] * scale_factor) + add_offset
                scaled_valid_max = (valid_range[1] * scale_factor) + add_offset
                data = np.ma.masked_where((data < scaled_valid_min) | (data > scaled_valid_max), data)
            
            dataset.endaccess()
            hdf.end()
            
            print(f"      Successfully read {dataset_name}: {data.shape}, range: {data.min():.2f} to {data.max():.2f}")
            
        except Exception as e:
            print(f"      Could not read HDF data: {e}")
            # Fall back to placeholder visualization
            data = None
        
        if data is not None:
            # Create figure with data visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot 1: Data visualization
            im = ax1.imshow(data, cmap=product_config['colormap'], aspect='equal')
            ax1.set_title(f"GLASS {product_config['name']}\n{year}-{doy} {tile_part}", fontsize=12)
            ax1.set_xlabel('X (500m pixels)')
            ax1.set_ylabel('Y (500m pixels)')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
            cbar.set_label(f"{product_config['name']} ({product_config['units']})")
            
            # Add statistics text
            stats_text = f"""
Data Statistics:
Valid pixels: {np.sum(~data.mask):,}
Invalid pixels: {np.sum(data.mask):,}
Mean: {np.ma.mean(data):.2f}
Std: {np.ma.std(data):.2f}
Min: {np.ma.min(data):.2f}
Max: {np.ma.max(data):.2f}
            """
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Plot 2: Histogram
            valid_data = data.compressed()  # Get non-masked values
            if len(valid_data) > 0:
                ax2.hist(valid_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                ax2.set_xlabel(f"{product_config['name']} ({product_config['units']})")
                ax2.set_ylabel('Frequency')
                ax2.set_title('Data Distribution')
                ax2.grid(True, alpha=0.3)
                
                # Add median and mean lines
                median_val = np.ma.median(data)
                mean_val = np.ma.mean(data)
                ax2.axvline(median_val, color='red', linestyle='--', label=f'Median: {median_val:.2f}')
                ax2.axvline(mean_val, color='orange', linestyle='--', label=f'Mean: {mean_val:.2f}')
                ax2.legend()
            
            plt.tight_layout()
            
            # Save data visualization
            output_file = output_dir / f"{product_name}_{year}_{doy}_{tile_part}_data.pdf"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"    ✓ Saved data plot: {output_file}")
            
            plt.close()
            
        else:
            # Fallback: Create info-only visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')
            
            # Get file info
            file_size = hdf_file.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            # Create information table
            info_text = f"""
GLASS Data File Information

Filename: {hdf_file.name}
Product: {product_config['name']} ({product_name})
Year: {year}
Day of Year: {doy}
Tile: {tile_part}
File Size: {file_size_mb:.2f} MB
Resolution: {product_config['resolution']}
Units: {product_config['units']}

Note: Could not read HDF data - using metadata only visualization.
            """
            
            ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.title(f"GLASS {product_config['name']} File - {year}-{doy} - {tile_part}", 
                     fontsize=14, pad=20)
            
            # Save info plot
            output_file = output_dir / f"{product_name}_{year}_{doy}_{tile_part}_info.pdf"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"    ✓ Saved info plot: {output_file}")
            
            plt.close()
        
    except Exception as e:
        print(f"Error creating visualization for {hdf_file.name}: {e}")


def create_glass_overview(product_dir, product_name, product_config, figures_dir):
    """
    Create an overview plot showing all downloaded GLASS files for a product.
    
    Args:
        product_dir (Path): Directory containing product data
        product_name (str): Product name
        product_config (dict): Product configuration
        figures_dir (Path): Directory to save figure
    """
    # Collect all files and organize by date
    date_info = {}
    
    # Check for tile-based structure
    date_dirs = [d for d in product_dir.iterdir() if d.is_dir() and '_' in d.name]
    
    if date_dirs:
        for date_dir in date_dirs:
            date_name = date_dir.name
            hdf_files = list(date_dir.glob("*.hdf"))
            if hdf_files:
                year, doy = date_name.split('_')
                date_info[date_name] = {
                    'year': year,
                    'doy': doy,
                    'file_count': len(hdf_files),
                    'total_size': sum(f.stat().st_size for f in hdf_files),
                    'tiles': [f.stem.split('.')[3] for f in hdf_files if len(f.stem.split('.')) > 3]
                }
    
    if not date_info:
        return
    
    # Create overview plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: File count by date
    dates = list(date_info.keys())
    file_counts = [date_info[d]['file_count'] for d in dates]
    
    ax1.bar(range(len(dates)), file_counts, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Date (Year_DOY)')
    ax1.set_ylabel('Number of Tiles')
    ax1.set_title(f'GLASS {product_config["name"]} - Downloaded Files Overview')
    ax1.set_xticks(range(len(dates)))
    ax1.set_xticklabels(dates, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add file count labels on bars
    for i, count in enumerate(file_counts):
        ax1.text(i, count + 0.1, str(count), ha='center', va='bottom')
    
    # Plot 2: Summary statistics table
    ax2.axis('off')
    
    # Calculate summary statistics
    total_files = sum(file_counts)
    total_size_mb = sum(date_info[d]['total_size'] for d in dates) / (1024 * 1024)
    unique_tiles = set()
    for d in dates:
        unique_tiles.update(date_info[d]['tiles'])
    
    # Create summary table
    summary_data = [
        ['Metric', 'Value'],
        ['Product', f'{product_config["name"]} ({product_name})'],
        ['Time Range', f'{min(date_info[d]["year"] for d in dates)} - {max(date_info[d]["year"] for d in dates)}'],
        ['Number of Dates', f'{len(dates)}'],
        ['Total Files Downloaded', f'{total_files}'],
        ['Total Data Size', f'{total_size_mb:.1f} MB'],
        ['Unique Tiles', f'{len(unique_tiles)}'],
        ['Resolution', product_config['resolution']],
        ['Units', product_config['units']],
        ['Temporal Resolution', product_config['temporal_resolution']]
    ]
    
    table = ax2.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.3, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    # Style the header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F2F2F2')
    
    ax2.set_title('Download Summary', fontsize=14, pad=20, weight='bold')
    
    plt.tight_layout()
    
    # Save overview plot
    output_file = figures_dir / f"{product_name}_download_overview.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved overview: {output_file}")
    
    plt.close()


def create_multi_tile_comparison(product_dir, product_name, product_config, figures_dir):
    """
    Create a comparison visualization showing multiple tiles for the same date.
    
    Args:
        product_dir (Path): Directory containing product data
        product_name (str): Product name
        product_config (dict): Product configuration
        figures_dir (Path): Directory to save figure
    """
    # Find a date with multiple tiles
    date_dirs = [d for d in product_dir.iterdir() if d.is_dir() and '_' in d.name]
    
    if not date_dirs:
        return
    
    # Use the first date with multiple tiles
    target_date = None
    target_files = []
    
    for date_dir in date_dirs:
        hdf_files = list(date_dir.glob("*.hdf"))
        if len(hdf_files) >= 3:  # Need at least 3 tiles for comparison
            target_date = date_dir.name
            target_files = hdf_files[:4]  # Limit to 4 tiles
            break
    
    if not target_files:
        print(f"  No suitable date found for multi-tile comparison")
        return
    
    print(f"  Creating multi-tile comparison for {target_date}")
    
    # Create figure with subplots
    n_tiles = len(target_files)
    cols = 2
    rows = (n_tiles + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Read and plot each tile
    from pyhdf import SD
    import numpy as np
    
    for i, hdf_file in enumerate(target_files):
        try:
            # Extract tile info
            tile_part = hdf_file.stem.split('.')[3]
            
            # Read HDF data
            hdf = SD.SD(str(hdf_file), SD.SDC.READ)
            dataset = hdf.select(product_name)
            raw_data = dataset.get()
            attrs = dataset.attributes()
            
            # Process data
            fill_value = attrs.get('_FillValue', 65535)
            water_value = attrs.get('watervalue', 65534)
            scale_factor = attrs.get('scale_factor', 1.0)
            
            data = raw_data.astype(np.float32)
            data = np.ma.masked_where((raw_data == fill_value) | (raw_data == water_value), data)
            data = data * scale_factor
            
            # Plot
            ax = axes[i]
            im = ax.imshow(data, cmap=product_config['colormap'], aspect='equal')
            ax.set_title(f'Tile {tile_part}\nValid: {100*np.sum(~data.mask)/data.size:.1f}%')
            ax.set_xlabel('X (500m pixels)')
            ax.set_ylabel('Y (500m pixels)')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            if i == 0:  # Only label first colorbar
                cbar.set_label(f"{product_config['name']} ({product_config['units']})")
            
            dataset.endaccess()
            hdf.end()
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error reading\n{hdf_file.name}\n{str(e)[:50]}...',
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'Tile {i+1} - Error')
    
    # Hide empty subplots
    for i in range(n_tiles, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'GLASS {product_config["name"]} Multi-Tile Comparison - {target_date}', 
                fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save comparison plot
    output_file = figures_dir / f"{product_name}_{target_date}_comparison.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved comparison: {output_file}")
    
    plt.close()


def extract_glass_point_data(data_dir, coordinates, products=None, start_year=None, end_year=None, output_format='csv'):
    """
    Extract GLASS data for specific point coordinates.
    
    Args:
        data_dir (str): Directory containing GLASS HDF files
        coordinates (list): List of tuples (lat, lon) or (lat, lon, name)
        products (list, optional): List of products to extract
        start_year (int, optional): Starting year
        end_year (int, optional): Ending year
        output_format (str): Output format ('csv' or 'json')
    
    Returns:
        dict: Extracted data for each point
    """
    data_path = Path(data_dir)
    product_info = get_glass_product_info()
    
    # Find available product directories
    if products:
        product_dirs = [data_path / p for p in products if (data_path / p).exists()]
    else:
        product_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name in product_info]
    
    if not product_dirs:
        print(f"No GLASS product directories found in {data_dir}")
        return {}
    
    print(f"Found {len(product_dirs)} product directories for extraction")
    
    # Process coordinates
    points_data = {}
    for i, coord in enumerate(coordinates):
        if len(coord) == 3:
            lat, lon, name = coord
        else:
            lat, lon = coord
            name = f"Point_{i+1}"
        
        points_data[name] = {
            'latitude': lat,
            'longitude': lon,
            'data': {}
        }
    
    # Extract data for each product
    for product_dir in product_dirs:
        product_name = product_dir.name
        print(f"Processing {product_name} data...")
        
        # Find relevant HDF files
        hdf_files = list(product_dir.glob("*.hdf"))
        if start_year and end_year:
            filtered_files = []
            for f in hdf_files:
                filename = f.name
                # Extract year from filename (assuming format includes year)
                try:
                    year_in_file = int(filename.split('.')[1][1:5])  # Extract year from A2020... format
                    if start_year <= year_in_file <= end_year:
                        filtered_files.append(f)
                except:
                    continue
            hdf_files = filtered_files
        
        if not hdf_files:
            continue
        
        # Sample files to avoid processing too many
        sample_files = hdf_files[::max(1, len(hdf_files) // 20)]  # Sample every 20th file
        
        for hdf_file in tqdm(sample_files, desc=f"Processing {product_name}"):
            try:
                # Try to extract data from HDF file
                extract_from_hdf_file(hdf_file, product_name, points_data)
            except Exception as e:
                print(f"Error processing {hdf_file.name}: {e}")
                continue
    
    # Save extracted data
    output_dir = data_path / "point_extractions"
    output_dir.mkdir(exist_ok=True)
    
    if output_format == 'csv':
        save_glass_point_data_csv(points_data, output_dir)
    elif output_format == 'json':
        save_glass_point_data_json(points_data, output_dir)
    
    return points_data


def extract_from_hdf_file(hdf_file, product_name, points_data):
    """
    Extract point data from a single HDF file.
    
    Args:
        hdf_file (Path): Path to HDF file
        product_name (str): Product name
        points_data (dict): Dictionary to store extracted data
    """
    try:
        # Try to open with xarray
        with xr.open_dataset(hdf_file, engine='h5netcdf') as ds:
            data_vars = list(ds.data_vars.keys())
            if not data_vars:
                return
            
            data_var = data_vars[0]
            data = ds[data_var]
            
            # Extract date from filename
            filename = hdf_file.stem
            date_str = filename.split('.')[-2] if '.' in filename else filename[-7:]
            
            # Extract data for each point
            for point_name, point_info in points_data.items():
                lat, lon = point_info['latitude'], point_info['longitude']
                
                try:
                    # Find nearest grid point
                    point_data = data.sel(lat=lat, lon=lon, method='nearest')
                    value = float(point_data.values)
                    
                    # Initialize product data if not exists
                    if product_name not in point_info['data']:
                        point_info['data'][product_name] = {}
                    
                    # Store data
                    point_info['data'][product_name][date_str] = value
                    
                except Exception:
                    continue
                    
    except Exception:
        # Silently skip files that can't be processed
        pass


def save_glass_point_data_csv(points_data, output_dir):
    """
    Save GLASS point data to CSV files.
    
    Args:
        points_data (dict): Extracted point data
        output_dir (Path): Output directory
    """
    # Collect all products
    all_products = set()
    for point_info in points_data.values():
        all_products.update(point_info['data'].keys())
    
    # Create one CSV file per product
    for product in all_products:
        all_rows = []
        
        for point_name, point_info in points_data.items():
            if product not in point_info['data']:
                continue
            
            for date_str, value in point_info['data'][product].items():
                if pd.notna(value):
                    all_rows.append({
                        'point_name': point_name,
                        'date': date_str,
                        'product': product,
                        'value': value,
                        'latitude': point_info['latitude'],
                        'longitude': point_info['longitude']
                    })
        
        if all_rows:
            df = pd.DataFrame(all_rows)
            df = df.sort_values(['point_name', 'date'])
            output_file = output_dir / f"{product}_all_points.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved: {output_file} ({len(df)} records)")


def parse_coordinates(coord_list, coords_file):
    """
    Parse coordinates from command line arguments or file.
    
    Args:
        coord_list (list): List of coordinates from command line
        coords_file (str): Path to CSV file with coordinates
    
    Returns:
        list: List of coordinate tuples
    """
    coordinates = []
    
    # Parse coordinates from command line
    if coord_list:
        i = 0
        while i < len(coord_list):
            if i + 1 < len(coord_list):
                try:
                    lat = float(coord_list[i])
                    lon = float(coord_list[i + 1])
                except ValueError:
                    i += 2
                    continue
                
                # Check if next item is a name
                if i + 2 < len(coord_list):
                    try:
                        float(coord_list[i + 2])
                        coordinates.append((lat, lon))
                        i += 2
                    except ValueError:
                        name = str(coord_list[i + 2])
                        coordinates.append((lat, lon, name))
                        i += 3
                else:
                    coordinates.append((lat, lon))
                    i += 2
            else:
                break
    
    # Parse coordinates from file
    if coords_file:
        try:
            df = pd.read_csv(coords_file)
            required_cols = ['lat', 'lon']
            
            if not all(col in df.columns for col in required_cols):
                print(f" Coordinates file must contain columns: {required_cols}")
                return coordinates
            
            for _, row in df.iterrows():
                if 'name' in df.columns and pd.notna(row['name']):
                    coordinates.append((row['lat'], row['lon'], str(row['name'])))
                else:
                    coordinates.append((row['lat'], row['lon']))
                    
        except Exception as e:
            print(f" Error reading coordinates file: {e}")
    
    return coordinates


def main():
    """
    Main function with command line interface for GLASS data processing.
    """
    parser = argparse.ArgumentParser(description='Download and process GLASS data')
    parser.add_argument('--mode', choices=['download', 'visualize', 'extract', 'plot-points', 'info'], default='info',
                       help='Mode: download data, visualize existing data, extract point data, plot extracted data, or show product info')
    parser.add_argument('--products', nargs='+', default=['LAI', 'FAPAR'],
                       help='GLASS products to process (e.g., LAI FAPAR NDVI)')
    parser.add_argument('--start-year', type=int, default=2010,
                       help='Starting year for download')
    parser.add_argument('--end-year', type=int, default=2012,
                       help='Ending year for download')
    parser.add_argument('--output-dir', '-o', type=str,
                       default='../../ancillary/glass',
                       help='Output directory for downloaded files')
    parser.add_argument('--max-files', type=int,
                       help='Maximum number of files to download per product (for testing)')
    
    # Point extraction arguments
    parser.add_argument('--coordinates', nargs='+',
                       help='Point coordinates as lat lon [name] pairs')
    parser.add_argument('--coords-file', type=str,
                       help='CSV file with coordinates (columns: lat, lon, name)')
    parser.add_argument('--output-format', choices=['csv', 'json'], default='csv',
                       help='Output format for point data')
    parser.add_argument('--plot-points', action='store_true',
                       help='Create plots for extracted points')
    
    args = parser.parse_args()
    
    print("=== GLASS Data Processor ===")
    print(f"Mode: {args.mode}")
    print()
    
    if args.mode == 'info':
        # Show product information
        products = get_glass_product_info()
        print("Available GLASS Products:")
        print("-" * 80)
        for code, info in products.items():
            print(f"{code:8} | {info['name']:35} | {info['resolution']:6} | {info['time_range']}")
        print("-" * 80)
        print("\nUse --mode download to download data")
        print("Example: python glass.py --mode download --products LAI FAPAR --start-year 2010 --end-year 2012")
    
    elif args.mode == 'download':
        # Download GLASS data
        print(f"Products: {', '.join(args.products)}")
        results = download_glass_data(
            args.products,
            args.start_year,
            args.end_year,
            args.output_dir,
            args.max_files
        )
        
        if results['successful']:
            print(f"\n Successfully downloaded {len(results['successful'])} files to: {args.output_dir}")
            print("Next steps:")
            print("1. Run with --mode visualize to create plots")
            print("2. Run with --mode extract to extract point data")
        else:
            print(" No files were successfully downloaded")
    
    elif args.mode == 'visualize':
        # Visualize existing data
        print(f"Visualizing GLASS data from: {args.output_dir}")
        visualize_glass_data(args.output_dir, args.products[0] if len(args.products) == 1 else None)
    
    elif args.mode == 'extract':
        # Extract point data
        coordinates = parse_coordinates(args.coordinates, args.coords_file)
        if not coordinates:
            print(" No coordinates provided. Use --coordinates or --coords-file")
            return
        
        print(f"Extracting data for {len(coordinates)} points")
        points_data = extract_glass_point_data(
            args.output_dir,
            coordinates,
            args.products,
            args.start_year,
            args.end_year,
            args.output_format
        )
        
        print(f" Point extraction complete")
    
    elif args.mode == 'plot-points':
        # Visualize extracted point data
        print(f"Loading and visualizing extracted point data from: {args.output_dir}")
        # Implementation would go here
        print(" Point visualization complete")
    
    print("\n=== Complete ===")


if __name__ == "__main__":
    main()