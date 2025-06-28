"""
Download and process TerraClimate data from the University of Idaho.

This script provides functionality to:
1. Download TerraClimate NetCDF files for specified variables and years
2. Visualize downloaded data with publication-quality maps
3. Extract time series data for specific point coordinates
4. Export point data to CSV or JSON formats
5. Create time series visualizations for extracted points

Data source: https://climate.northwestknowledge.net/TERRACLIMATE-DATA

Usage examples:
    # Download data
    python terraclimate.py --mode download --variables ppt tmax --start-year 2010 --end-year 2015
    
    # Extract point data from command line
    python terraclimate.py --mode extract --coordinates 40.5 -120.3 Site1 45.2 -118.7 Site2 --plot-points
    
    # Extract point data from CSV file
    python terraclimate.py --mode extract --coords-file sites.csv --output-format json
    
    # Visualize already extracted point data
    python terraclimate.py --mode plot-points --variables ppt tmax

Dependencies:
    pip install requests xarray matplotlib cartopy tqdm pandas
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr


def download_terraclimate_data(variables, start_year, end_year, output_dir):
    """
    Download TerraClimate data files for specified variables and years.
    
    Args:
        variables (list): List of climate variables (e.g., ['ppt', 'tmax', 'tmin'])
        start_year (int): Starting year for download
        end_year (int): Ending year for download
        output_dir (str): Directory to save downloaded files
    
    Returns:
        dict: Summary of download results
    """
    base_url = "https://climate.northwestknowledge.net/TERRACLIMATE-DATA"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {
        'successful': [],
        'failed': [],
        'skipped': []
    }
    
    total_files = len(variables) * (end_year - start_year + 1)
    print(f"Downloading {total_files} TerraClimate files...")
    
    with tqdm(total=total_files, desc="Overall Progress") as pbar:
        for variable in variables:
            for year in range(start_year, end_year + 1):
                filename = f"TerraClimate_{variable}_{year}.nc"
                file_path = Path(output_dir) / filename
                url = f"{base_url}/{filename}"
                
                # Skip if file already exists
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    if file_size > 1024:  # Skip if file is larger than 1KB
                        print(f"Skipping {filename} (already exists)")
                        results['skipped'].append(filename)
                        pbar.update(1)
                        continue
                
                # Download file
                success = download_file(url, file_path, variable, year)
                if success:
                    results['successful'].append(filename)
                else:
                    results['failed'].append(filename)
                
                pbar.update(1)
                time.sleep(0.1)  # Be respectful to the server
    
    # Print summary
    print(f"\nDownload Summary:")
    print(f"  Successful: {len(results['successful'])}")
    print(f"  Failed: {len(results['failed'])}")
    print(f"  Skipped: {len(results['skipped'])}")
    
    if results['failed']:
        print(f"  Failed files: {', '.join(results['failed'])}")
    
    return results


def download_file(url, file_path, variable, year):
    """
    Download a single file with error handling and progress tracking.
    
    Args:
        url (str): URL to download from
        file_path (Path): Local path to save file
        variable (str): Climate variable name
        year (int): Year being downloaded
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Downloading {variable} data for {year}...")
        
        # Make request with streaming and timeout
        response = requests.get(url, stream=True, timeout=(10, 60))
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(file_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=f"{variable}_{year}", leave=False) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        # Verify download
        if file_path.exists() and file_path.stat().st_size > 0:
            print(f"✓ Downloaded {file_path.name} ({file_path.stat().st_size:,} bytes)")
            return True
        else:
            print(f"✗ Download failed: {file_path.name}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Download error for {variable}_{year}: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error for {variable}_{year}: {e}")
        return False


def visualize_terraclimate_data(data_dir, variable=None, year=None):
    """
    Create publication-quality visualizations of TerraClimate data.
    
    Args:
        data_dir (str): Directory containing TerraClimate NetCDF files
        variable (str, optional): Specific variable to plot
        year (int, optional): Specific year to plot
    """
    data_path = Path(data_dir)
    
    # Find available files
    if variable and year:
        nc_files = list(data_path.glob(f"TerraClimate_{variable}_{year}.nc"))
    elif variable:
        nc_files = list(data_path.glob(f"TerraClimate_{variable}_*.nc"))
    else:
        nc_files = list(data_path.glob("TerraClimate_*.nc"))
    
    if not nc_files:
        print(f"No TerraClimate files found in {data_dir}")
        return
    
    print(f"Found {len(nc_files)} files to visualize")
    
    # Create figures directory
    figures_dir = data_path / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Process each file
    for nc_file in nc_files:
        try:
            # Parse filename
            filename_parts = nc_file.stem.split('_')
            if len(filename_parts) >= 3:
                var_name = filename_parts[1]
                file_year = filename_parts[2]
            else:
                print(f"Cannot parse filename: {nc_file.name}")
                continue
            
            print(f"Visualizing {var_name} data for {file_year}...")
            
            # Load data
            with xr.open_dataset(nc_file) as ds:
                print(f"Dataset variables: {list(ds.data_vars)}")
                
                # Get the main data variable (usually the same as var_name)
                data_var = var_name if var_name in ds.data_vars else list(ds.data_vars)[0]
                data = ds[data_var]
                
                # Calculate annual mean if time dimension exists
                if 'time' in data.dims:
                    annual_mean = data.mean(dim='time')
                else:
                    annual_mean = data
                
                # Create visualization
                create_climate_map(annual_mean, var_name, file_year, figures_dir)
                
        except Exception as e:
            print(f"Error processing {nc_file.name}: {e}")


def create_climate_map(data, variable, year, output_dir):
    """
    Create a publication-quality map of climate data.
    
    Args:
        data (xarray.DataArray): Climate data to plot
        variable (str): Variable name
        year (str): Year
        output_dir (Path): Directory to save figure
    """
    # Variable-specific settings
    var_config = {
        'ppt': {
            'title': 'Annual Precipitation',
            'cmap': 'Blues',
            'units': 'mm/year'
        },
        'tmax': {
            'title': 'Annual Maximum Temperature',
            'cmap': 'Reds',
            'units': '°C'
        },
        'tmin': {
            'title': 'Annual Minimum Temperature',
            'cmap': 'coolwarm',
            'units': '°C'
        }
    }
    
    config = var_config.get(variable, {
        'title': f'{variable.upper()} Data',
        'cmap': 'viridis',
        'units': 'units'
    })
    
    # Create figure
    _, ax = plt.subplots(figsize=(15, 10), 
                          subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Add map features
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.3)
    ax.add_feature(cfeature.RIVERS, alpha=0.3)
    ax.gridlines(draw_labels=True, alpha=0.5)
    
    # Plot data
    if variable == 'tmin' or variable == 'tmax':
        # Convert Kelvin to Celsius for temperature
        plot_data = data - 273.15 if data.max() > 200 else data
    else:
        plot_data = data
    
    # Use percentiles for better color scaling
    vmin, vmax = np.percentile(plot_data.values[~np.isnan(plot_data.values)], [2, 98])
    
    im = plot_data.plot(ax=ax,
                       cmap=config['cmap'],
                       transform=ccrs.PlateCarree(),
                       vmin=vmin,
                       vmax=vmax,
                       add_colorbar=False)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, orientation='horizontal', 
                 label=f"{config['title']} ({config['units']})",
                 pad=0.05, shrink=0.8)
    
    # Set title
    plt.title(f"{config['title']} - {year}", fontsize=14, pad=20)
    
    # Save figure
    output_file = output_dir / f"{variable}_{year}_map.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Map saved to: {output_file}")
    
    plt.close()


def extract_point_data(data_dir, coordinates, variables=None, start_year=None, end_year=None, output_format='csv'):
    """
    Extract TerraClimate data for specific point coordinates.
    
    Args:
        data_dir (str): Directory containing TerraClimate NetCDF files
        coordinates (list): List of tuples (lat, lon) or (lat, lon, name)
        variables (list, optional): List of variables to extract
        start_year (int, optional): Starting year
        end_year (int, optional): Ending year
        output_format (str): Output format ('csv' or 'json')
    
    Returns:
        dict: Extracted data for each point
    """
    data_path = Path(data_dir)
    
    # Find available files
    if variables:
        nc_files = []
        for var in variables:
            if start_year and end_year:
                nc_files.extend(data_path.glob(f"TerraClimate_{var}_{start_year}_to_{end_year}.nc"))
                for year in range(start_year, end_year + 1):
                    nc_files.extend(data_path.glob(f"TerraClimate_{var}_{year}.nc"))
            else:
                nc_files.extend(data_path.glob(f"TerraClimate_{var}_*.nc"))
    else:
        nc_files = list(data_path.glob("TerraClimate_*.nc"))
    
    if not nc_files:
        print(f"No TerraClimate files found in {data_dir}")
        return {}
    
    print(f"Found {len(nc_files)} files for point extraction")
    
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
            'data': {},
            'raw_values': {}  # Store all values for mean calculation
        }
    
    # Extract data for each file
    for nc_file in tqdm(nc_files, desc="Processing files"):
        try:
            # Parse filename
            filename_parts = nc_file.stem.split('_')
            if len(filename_parts) >= 3:
                var_name = filename_parts[1]
                file_year = filename_parts[2]
            else:
                continue
            
            # Load data
            with xr.open_dataset(nc_file) as ds:
                data_var = var_name if var_name in ds.data_vars else list(ds.data_vars)[0]
                data = ds[data_var]
                
                # Extract data for each point
                for point_name, point_info in points_data.items():
                    lat, lon = point_info['latitude'], point_info['longitude']
                    
                    # Find nearest grid point
                    point_data = data.sel(lat=lat, lon=lon, method='nearest')
                    
                    # Initialize variable if not exists
                    if var_name not in point_info['data']:
                        point_info['data'][var_name] = {}
                        point_info['raw_values'][var_name] = []
                    
                    # Store data by year and collect raw values
                    if 'time' in point_data.dims:
                        # Monthly data - calculate annual mean
                        annual_mean = float(point_data.mean().values)
                        points_data[point_name]['data'][var_name][file_year] = annual_mean
                        points_data[point_name]['raw_values'][var_name].append(annual_mean)
                    else:
                        # Annual data
                        annual_value = float(point_data.values)
                        points_data[point_name]['data'][var_name][file_year] = annual_value
                        points_data[point_name]['raw_values'][var_name].append(annual_value)
                        
        except Exception as e:
            print(f"Error processing {nc_file.name}: {e}")
    
    # Calculate means for each variable and point
    for point_name, point_info in points_data.items():
        for var_name, values in point_info['raw_values'].items():
            if values:
                mean_value = np.mean(values)
                points_data[point_name]['data'][var_name]['mean'] = mean_value
    
    # Save extracted data and means
    output_dir = data_path / "point_extractions"
    output_dir.mkdir(exist_ok=True)
    
    # Save means to CSV
    save_point_means_csv(points_data, output_dir)
    
    if output_format == 'csv':
        save_point_data_csv(points_data, output_dir)
    elif output_format == 'json':
        save_point_data_json(points_data, output_dir)
    
    return points_data


def save_point_means_csv(points_data, output_dir):
    """
    Save mean values for each variable to separate CSV files.
    
    Args:
        points_data (dict): Extracted point data with means
        output_dir (Path): Output directory
    """
    # Collect all variables
    all_variables = set()
    for point_info in points_data.values():
        all_variables.update(point_info['data'].keys())
    
    # Create separate CSV file for each variable
    for var_name in all_variables:
        var_rows = []
        
        for point_name, point_info in points_data.items():
            if var_name in point_info['data'] and 'mean' in point_info['data'][var_name]:
                var_rows.append({
                    'point_name': point_name,
                    'latitude': point_info['latitude'],
                    'longitude': point_info['longitude'],
                    'mean_value': point_info['data'][var_name]['mean'],
                    'units': get_variable_units(var_name)
                })
        
        if var_rows:
            df = pd.DataFrame(var_rows)
            # Sort by point name for better organization
            df = df.sort_values('point_name')
            output_file = output_dir / f"{var_name}_means.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved {var_name} means: {output_file} ({len(df)} records from {len(points_data)} points)")
    
    if not all_variables:
        print("No mean values to save")


def save_point_data_csv(points_data, output_dir):
    """
    Save point data to CSV files (one file per variable, all points combined).
    
    Args:
        points_data (dict): Extracted point data
        output_dir (Path): Output directory
    """
    # Collect all variables
    all_variables = set()
    for point_info in points_data.values():
        all_variables.update(point_info['data'].keys())
    
    # Create one CSV file per variable with all points
    for var_name in all_variables:
        all_rows = []
        
        for point_name, point_info in points_data.items():
            if var_name not in point_info['data']:
                continue
                
            var_data = point_info['data'][var_name]
            
            for year, year_data in var_data.items():
                if isinstance(year_data, dict):
                    # Monthly data
                    for date_str, value in year_data.items():
                        if pd.notna(value):
                            all_rows.append({
                                'point_name': point_name,
                                'date': date_str,
                                'year': year,
                                'variable': var_name,
                                'value': value,
                                'latitude': point_info['latitude'],
                                'longitude': point_info['longitude']
                            })
                else:
                    # Annual data
                    all_rows.append({
                        'point_name': point_name,
                        'date': f"{year}-01-01",
                        'year': year,
                        'variable': var_name,
                        'value': year_data,
                        'latitude': point_info['latitude'],
                        'longitude': point_info['longitude']
                    })
        
        if all_rows:
            df = pd.DataFrame(all_rows)
            # Sort by point name and date for better organization
            df = df.sort_values(['point_name', 'date'])
            output_file = output_dir / f"{var_name}_all_points.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved: {output_file} ({len(df)} records from {len(points_data)} points)")


def visualize_point_timeseries(points_data, variables=None, output_dir=None):
    """
    Create time series plots for extracted point data.
    
    Args:
        points_data (dict): Extracted point data
        variables (list, optional): Variables to plot
        output_dir (Path, optional): Directory to save figures
    """
    if output_dir:
        figures_dir = Path(output_dir) / "figures"
        figures_dir.mkdir(exist_ok=True)
    
    for point_name, point_info in points_data.items():
        plot_variables = variables if variables else list(point_info['data'].keys())
        
        for var_name in plot_variables:
            if var_name not in point_info['data']:
                continue
                
            # Collect time series data
            dates = []
            values = []
            
            for year, year_data in point_info['data'][var_name].items():
                if isinstance(year_data, dict):
                    # Monthly data
                    for date_str, value in year_data.items():
                        if pd.notna(value):
                            dates.append(pd.to_datetime(date_str))
                            values.append(value)
                else:
                    # Annual data
                    dates.append(pd.to_datetime(f"{year}-07-01"))
                    values.append(year_data)
            
            if not dates:
                continue
            
            # Create time series plot
            _, ax = plt.subplots(figsize=(12, 6))
            
            df = pd.DataFrame({'date': dates, 'value': values}).sort_values('date')
            ax.plot(df['date'], df['value'], marker='o', linewidth=2, markersize=4)
            
            ax.set_title(f"{var_name.upper()} Time Series - {point_name}\n"
                        f"Lat: {point_info['latitude']:.3f}, Lon: {point_info['longitude']:.3f}")
            ax.set_xlabel('Date')
            ax.set_ylabel(get_variable_units(var_name))
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if output_dir:
                output_file = figures_dir / f"{point_name}_{var_name}_timeseries.pdf"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Saved: {output_file}")
            
            plt.show()


def get_variable_units(variable):
    """
    Get units for TerraClimate variables.
    
    Args:
        variable (str): Variable name
    
    Returns:
        str: Units string
    """
    units_map = {
        'ppt': 'mm/month',
        'tmax': '°C',
        'tmin': '°C',
        'tmean': '°C',
        'pet': 'mm/month',
        'q': 'mm/month',
        'ws': 'm/s',
        'vpd': 'kPa',
        'vap': 'kPa'
    }
    return units_map.get(variable, 'units')


def visualize_extracted_points(data_dir, variables=None):
    """
    Load and visualize already extracted point data from CSV files.
    
    Args:
        data_dir (str): Directory containing extracted point data CSV files
        variables (list, optional): Variables to visualize
    """
    data_path = Path(data_dir)
    extractions_dir = data_path / "point_extractions"
    
    if not extractions_dir.exists():
        print(f"✗ No point extractions directory found at: {extractions_dir}")
        return
    
    # Find CSV files
    csv_files = list(extractions_dir.glob("*_all_points.csv"))
    if not csv_files:
        print(f"✗ No extracted point CSV files found in: {extractions_dir}")
        return
    
    print(f"Found {len(csv_files)} extracted point data files")
    
    # Create figures directory
    figures_dir = extractions_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Process each variable file
    for csv_file in csv_files:
        # Extract variable name from filename
        var_name = csv_file.stem.replace('_all_points', '')
        
        # Skip if specific variables requested and this isn't one of them
        if variables and var_name not in variables:
            continue
        
        print(f"Visualizing {var_name} data...")
        
        try:
            # Load data
            df = pd.read_csv(csv_file)
            
            if df.empty:
                print(f"No data found in {csv_file.name}")
                continue
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Get unique points
            unique_points = df['point_name'].unique() if 'point_name' in df.columns else ['Point_1']
            
            # Create time series plot
            fig, ax = plt.subplots(figsize=(15, 8))
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_points)))
            
            for i, point in enumerate(unique_points):
                if 'point_name' in df.columns:
                    point_data = df[df['point_name'] == point].copy()
                else:
                    point_data = df.copy()
                
                # Sort by date
                point_data = point_data.sort_values('date')
                
                # Get coordinates for legend
                if len(point_data) > 0:
                    lat = point_data['latitude'].iloc[0]
                    lon = point_data['longitude'].iloc[0]
                    label = f"{point} ({lat:.2f}, {lon:.2f})"
                else:
                    label = str(point)
                
                ax.plot(point_data['date'], point_data['value'], 
                       marker='o', linewidth=2, markersize=3, 
                       color=colors[i], label=label, alpha=0.8)
            
            # Customize plot
            ax.set_title(f"{var_name.upper()} Time Series - All Points", fontsize=16, pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel(f"{var_name.upper()} ({get_variable_units(var_name)})", fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add legend if multiple points
            if len(unique_points) > 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save figure
            output_file = figures_dir / f"{var_name}_all_points_timeseries.pdf"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved: {output_file}")
            
            # Also create individual point plots if multiple points
            if len(unique_points) > 3:  # Only if more than 3 points to avoid clutter
                create_individual_point_plots(df, var_name, figures_dir)
            
            plt.close()
            
        except Exception as e:
            print(f"Error visualizing {csv_file.name}: {e}")


def create_individual_point_plots(df, var_name, figures_dir):
    """
    Create individual plots for each point when there are many points.
    
    Args:
        df (pd.DataFrame): Point data
        var_name (str): Variable name
        figures_dir (Path): Directory to save figures
    """
    unique_points = df['point_name'].unique() if 'point_name' in df.columns else ['Point_1']
    
    # Create subplot grid
    n_points = len(unique_points)
    cols = min(3, n_points)
    rows = (n_points + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, point in enumerate(unique_points):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if 'point_name' in df.columns:
            point_data = df[df['point_name'] == point].copy()
        else:
            point_data = df.copy()
        
        point_data = point_data.sort_values('date')
        
        if len(point_data) > 0:
            lat = point_data['latitude'].iloc[0]
            lon = point_data['longitude'].iloc[0]
            
            ax.plot(point_data['date'], point_data['value'], 
                   marker='o', linewidth=2, markersize=3)
            ax.set_title(f"{point}\nLat: {lat:.2f}, Lon: {lon:.2f}", fontsize=10)
            ax.set_ylabel(get_variable_units(var_name), fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
    
    # Hide empty subplots
    for i in range(n_points, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f"{var_name.upper()} Time Series - Individual Points", fontsize=14)
    plt.tight_layout()
    
    output_file = figures_dir / f"{var_name}_individual_points.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def visualize_mean_files(data_dir):
    """
    Create visualizations for the separate mean CSV files.
    
    Args:
        data_dir (str): Directory containing the mean CSV files
    """
    data_path = Path(data_dir)
    extractions_dir = data_path / "point_extractions"
    
    if not extractions_dir.exists():
        print(f"✗ No point extractions directory found at: {extractions_dir}")
        return
    
    # Find mean CSV files
    mean_files = list(extractions_dir.glob("*_means.csv"))
    if not mean_files:
        print(f"✗ No mean CSV files found in: {extractions_dir}")
        return
    
    print(f"Found {len(mean_files)} mean files to visualize")
    
    # Create figures directory
    figures_dir = extractions_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Process each mean file
    for csv_file in mean_files:
        # Extract variable name from filename
        var_name = csv_file.stem.replace('_means', '')
        
        print(f"Visualizing {var_name} mean data...")
        
        try:
            # Load data
            df = pd.read_csv(csv_file)
            
            if df.empty:
                print(f"No data found in {csv_file.name}")
                continue
            
            # Create global map visualization
            create_mean_map(df, var_name, figures_dir)
            
            # Create histogram
            create_mean_histogram(df, var_name, figures_dir)
            
            # Create summary statistics
            create_mean_statistics(df, var_name, figures_dir)
            
        except Exception as e:
            print(f"Error visualizing {csv_file.name}: {e}")


def create_mean_map(df, var_name, figures_dir):
    """
    Create a global map of mean values.
    
    Args:
        df (pd.DataFrame): Mean data
        var_name (str): Variable name
        figures_dir (Path): Directory to save figures
    """
    # Variable-specific settings
    var_config = {
        'ppt': {
            'title': 'Mean Annual Precipitation',
            'cmap': 'Blues',
            'units': 'mm/month'
        },
        'tmax': {
            'title': 'Mean Maximum Temperature',
            'cmap': 'Reds',
            'units': '°C'
        },
        'tmin': {
            'title': 'Mean Minimum Temperature',
            'cmap': 'coolwarm',
            'units': '°C'
        }
    }
    
    config = var_config.get(var_name, {
        'title': f'Mean {var_name.upper()}',
        'cmap': 'viridis',
        'units': 'units'
    })
    
    # Create figure with global projection
    fig, ax = plt.subplots(figsize=(15, 10), 
                          subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Add map features
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.3)
    ax.add_feature(cfeature.RIVERS, alpha=0.3)
    ax.gridlines(draw_labels=True, alpha=0.5)
    ax.set_global()
    
    # Plot points
    values = df['mean_value'].values
    lons = df['longitude'].values
    lats = df['latitude'].values
    
    # Use percentiles for better color scaling
    vmin, vmax = np.percentile(values[~np.isnan(values)], [2, 98])
    
    scatter = ax.scatter(lons, lats, c=values, cmap=config['cmap'], 
                        s=20, alpha=0.7, transform=ccrs.PlateCarree(),
                        vmin=vmin, vmax=vmax, edgecolors='none')
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                 label=f"{config['title']} ({config['units']})",
                 pad=0.05, shrink=0.8)
    
    # Set title
    plt.title(f"{config['title']} - Global Distribution\n({len(df)} sites)", 
              fontsize=14, pad=20)
    
    # Save figure
    output_file = figures_dir / f"{var_name}_means_global_map.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved global map: {output_file}")
    
    plt.close()


def create_mean_histogram(df, var_name, figures_dir):
    """
    Create histogram of mean values.
    
    Args:
        df (pd.DataFrame): Mean data
        var_name (str): Variable name
        figures_dir (Path): Directory to save figures
    """
    # Variable-specific settings
    var_config = {
        'ppt': {
            'title': 'Mean Annual Precipitation Distribution',
            'units': 'mm/month',
            'color': 'blue'
        },
        'tmax': {
            'title': 'Mean Maximum Temperature Distribution',
            'units': '°C',
            'color': 'red'
        },
        'tmin': {
            'title': 'Mean Minimum Temperature Distribution',
            'units': '°C',
            'color': 'cyan'
        }
    }
    
    config = var_config.get(var_name, {
        'title': f'Mean {var_name.upper()} Distribution',
        'units': 'units',
        'color': 'gray'
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    values = df['mean_value'].values
    
    # Create histogram
    n_bins = min(50, len(df) // 20)  # Adaptive bin count
    counts, bins, patches = ax.hist(values, bins=n_bins, alpha=0.7, 
                                   color=config['color'], edgecolor='black', linewidth=0.5)
    
    # Add statistics
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {median_val:.2f}')
    
    # Customize plot
    ax.set_xlabel(f"{config['title'].replace(' Distribution', '')} ({config['units']})", fontsize=12)
    ax.set_ylabel('Number of Sites', fontsize=12)
    ax.set_title(f"{config['title']}\n"
                f"Mean ± SD: {mean_val:.2f} ± {std_val:.2f} {config['units']} "
                f"(n={len(df)} sites)", fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save figure
    output_file = figures_dir / f"{var_name}_means_histogram.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved histogram: {output_file}")
    
    plt.close()


def create_mean_statistics(df, var_name, figures_dir):
    """
    Create summary statistics table and save to CSV.
    
    Args:
        df (pd.DataFrame): Mean data
        var_name (str): Variable name
        figures_dir (Path): Directory to save figures
    """
    values = df['mean_value'].values
    
    # Calculate statistics
    stats = {
        'Variable': var_name,
        'Units': df['units'].iloc[0] if 'units' in df.columns else 'units',
        'Count': len(values),
        'Mean': np.mean(values),
        'Median': np.median(values),
        'Std_Dev': np.std(values),
        'Min': np.min(values),
        'Max': np.max(values),
        'Q25': np.percentile(values, 25),
        'Q75': np.percentile(values, 75),
        'Range': np.max(values) - np.min(values)
    }
    
    # Save statistics to CSV
    stats_df = pd.DataFrame([stats])
    output_file = figures_dir.parent / f"{var_name}_means_statistics.csv"
    stats_df.to_csv(output_file, index=False)
    print(f"Saved statistics: {output_file}")
    
    # Also create a formatted table figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Format data for table
    table_data = [
        ['Statistic', 'Value'],
        ['Variable', var_name.upper()],
        ['Units', stats['Units']],
        ['Count', f"{stats['Count']:,}"],
        ['Mean', f"{stats['Mean']:.3f}"],
        ['Median', f"{stats['Median']:.3f}"],
        ['Std Dev', f"{stats['Std_Dev']:.3f}"],
        ['Minimum', f"{stats['Min']:.3f}"],
        ['Maximum', f"{stats['Max']:.3f}"],
        ['25th Percentile', f"{stats['Q25']:.3f}"],
        ['75th Percentile', f"{stats['Q75']:.3f}"],
        ['Range', f"{stats['Range']:.3f}"]
    ]
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title(f"Summary Statistics: {var_name.upper()} Mean Values", 
              fontsize=14, pad=20, weight='bold')
    
    # Save table figure
    output_file = figures_dir / f"{var_name}_means_statistics_table.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved statistics table: {output_file}")
    
    plt.close()


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
                    print(f"✗ Invalid coordinates: {coord_list[i]} {coord_list[i + 1]}")
                    i += 2
                    continue
                
                # Check if next item is a name (not a number)
                if i + 2 < len(coord_list):
                    try:
                        float(coord_list[i + 2])
                        # Next item is a number, so no name provided
                        coordinates.append((lat, lon))
                        i += 2
                    except ValueError:
                        # Next item is not a number, treat as name
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
                print(f"✗ Coordinates file must contain columns: {required_cols}")
                return coordinates
            
            for _, row in df.iterrows():
                if 'name' in df.columns and pd.notna(row['name']):
                    coordinates.append((row['lat'], row['lon'], str(row['name'])))
                else:
                    coordinates.append((row['lat'], row['lon']))
                    
        except Exception as e:
            print(f"✗ Error reading coordinates file: {e}")
    
    return coordinates


def main():
    """
    Main function with command line interface for TerraClimate data processing.
    """
    parser = argparse.ArgumentParser(description='Download and process TerraClimate data')
    parser.add_argument('--mode', choices=['download', 'visualize', 'extract', 'plot-points', 'visualize-means'], default='download',
                       help='Mode: download data, visualize existing data, extract point data, plot extracted point data, or visualize mean files')
    parser.add_argument('--variables', nargs='+', default=['ppt', 'tmax', 'tmin'],
                       help='Climate variables to download (e.g., ppt tmax tmin)')
    parser.add_argument('--start-year', type=int, default=2001,
                       help='Starting year for download')
    parser.add_argument('--end-year', type=int, default=2010,
                       help='Ending year for download')
    parser.add_argument('--output-dir', '-o', type=str,
                       default='../../ancillary/terraclimate',
                       help='Output directory for downloaded files')
    parser.add_argument('--variable', type=str,
                       help='Specific variable to visualize')
    parser.add_argument('--year', type=int,
                       help='Specific year to visualize')
    
    # Point extraction arguments
    parser.add_argument('--coordinates', nargs='+',
                       help='Point coordinates as lat lon [name] pairs (e.g., 40.5 -120.3 Site1 45.2 -118.7 Site2)')
    parser.add_argument('--coords-file', type=str, default='../../productivity/earth/lat_lon.csv',
                       help='CSV file with coordinates (columns: lat, lon, name)')
    parser.add_argument('--output-format', choices=['csv', 'json'], default='csv',
                       help='Output format for point data')
    parser.add_argument('--plot-points', action='store_true',
                       help='Create time series plots for extracted points')
    
    args = parser.parse_args()
    
    print("=== TerraClimate Data Processor ===")
    print(f"Mode: {args.mode}")
    print()
    
    if args.mode == 'download':
        # Download TerraClimate data
        results = download_terraclimate_data(
            args.variables,
            args.start_year,
            args.end_year,
            args.output_dir
        )
        
        if results['successful']:
            print(f"\n✓ Successfully downloaded {len(results['successful'])} files to: {args.output_dir}")
            print("Next steps:")
            print("1. Run with --mode visualize to create plots")
            print("2. Run with --mode extract to extract point data")
        else:
            print("✗ No files were successfully downloaded")
    
    elif args.mode == 'visualize':
        # Visualize existing data
        print(f"Visualizing TerraClimate data from: {args.output_dir}")
        visualize_terraclimate_data(args.output_dir, args.variable, args.year)
    
    elif args.mode == 'extract':
        # Extract point data
        coordinates = parse_coordinates(args.coordinates, args.coords_file)
        if not coordinates:
            print("✗ No coordinates provided. Use --coordinates or --coords-file")
            return
        
        print(f"Extracting data for {len(coordinates)} points")
        points_data = extract_point_data(
            args.output_dir,
            coordinates,
            args.variables,
            args.start_year,
            args.end_year,
            args.output_format
        )
        
        if points_data and args.plot_points:
            print("Creating time series plots...")
            visualize_point_timeseries(points_data, args.variables, args.output_dir)
        
        print(f"✓ Point extraction complete")
    
    elif args.mode == 'plot-points':
        # Visualize already extracted point data
        print(f"Loading and visualizing extracted point data from: {args.output_dir}")
        visualize_extracted_points(args.output_dir, args.variables)
        print("✓ Point visualization complete")
    
    elif args.mode == 'visualize-means':
        # Visualize mean files
        print(f"Creating visualizations for mean files from: {args.output_dir}")
        visualize_mean_files(args.output_dir)
        print("✓ Mean file visualization complete")
    
    print("\n=== Complete ===")


if __name__ == "__main__":
    main()