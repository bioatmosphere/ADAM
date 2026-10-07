#!/usr/bin/env python3
"""
Create global maps of GLASS GPP data for each date separately.

This script reads all tiles for each 8-day period and creates comprehensive
global maps showing GPP values across the entire world.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from datetime import datetime, timedelta
from pyhdf import SD
import warnings
import math

# Import pyproj for proper projection handling
try:
    from pyproj import Proj, transform, CRS
    PYPROJ_AVAILABLE = True
except ImportError:
    print("pyproj not available - install with: uv add pyproj")
    PYPROJ_AVAILABLE = False

warnings.filterwarnings('ignore')

def parse_tile_metadata(hdf_file_path):
    """
    Parse tile metadata from HDF file to get exact coordinates.
    
    Args:
        hdf_file_path (str): Path to HDF file
    
    Returns:
        dict: Metadata including exact coordinates
    """
    try:
        hdf = SD.SD(str(hdf_file_path), SD.SDC.READ)
        attrs = hdf.attributes()
        
        # Parse StructMetadata to get exact coordinates
        struct_meta = attrs.get('StructMetadata.0', '')
        
        # Extract UpperLeftPointMtrs and LowerRightMtrs
        import re
        ul_match = re.search(r'UpperLeftPointMtrs=\(([^,]+),([^)]+)\)', struct_meta)
        lr_match = re.search(r'LowerRightMtrs=\(([^,]+),([^)]+)\)', struct_meta)
        
        if ul_match and lr_match:
            ul_x, ul_y = float(ul_match.group(1)), float(ul_match.group(2))
            lr_x, lr_y = float(lr_match.group(1)), float(lr_match.group(2))
            
            hdf.end()
            return {
                'upper_left': (ul_x, ul_y),
                'lower_right': (lr_x, lr_y),
                'projection': 'sinusoidal',
                'sphere_radius': 6371007.181
            }
        
        hdf.end()
        return None
        
    except Exception as e:
        print(f"Could not parse metadata from {hdf_file_path}: {e}")
        return None

def get_modis_projection():
    """Get the exact MODIS sinusoidal projection."""
    if PYPROJ_AVAILABLE:
        # MODIS sinusoidal projection with exact parameters
        modis_proj = Proj(proj='sinu', 
                         R=6371007.181,  # MODIS sphere radius
                         x_0=0,          # False easting
                         y_0=0,          # False northing
                         lon_0=0)        # Central meridian
        return modis_proj
    else:
        return None

def modis_to_geographic_pyproj(x, y):
    """Convert MODIS sinusoidal to geographic using pyproj."""
    if PYPROJ_AVAILABLE:
        from pyproj import Transformer
        
        # Create transformer from MODIS sinusoidal to WGS84
        # Use the CRS class for more reliable projection definition
        from pyproj import CRS
        
        modis_crs = CRS.from_proj4("+proj=sinu +R=6371007.181 +x_0=0 +y_0=0 +lon_0=0")
        wgs84_crs = CRS.from_epsg(4326)
        
        transformer = Transformer.from_crs(modis_crs, wgs84_crs, always_xy=True)
        
        lon, lat = transformer.transform(x, y)
        return lon, lat
    else:
        return None, None

def get_modis_tile_bounds_from_metadata(h_tile, v_tile, sample_file=None):
    """
    Calculate tile bounds using actual tile metadata if available.
    
    Args:
        h_tile (int): Horizontal tile number
        v_tile (int): Vertical tile number  
        sample_file (str): Path to sample HDF file for this tile
    
    Returns:
        tuple: (lon_min, lon_max, lat_min, lat_max) in degrees
    """
    if sample_file and PYPROJ_AVAILABLE:
        metadata = parse_tile_metadata(sample_file)
        if metadata:
            ul_x, ul_y = metadata['upper_left']
            lr_x, lr_y = metadata['lower_right']
            
            # Convert corners using pyproj for accurate transformation
            corners = [(ul_x, ul_y), (lr_x, ul_y), (lr_x, lr_y), (ul_x, lr_y)]
            geo_corners = []
            
            for x, y in corners:
                lon, lat = modis_to_geographic_pyproj(x, y)
                if lon is not None and lat is not None:
                    # Handle longitude wrapping
                    if lon > 180:
                        lon -= 360
                    elif lon < -180:
                        lon += 360
                    geo_corners.append((lon, lat))
            
            if len(geo_corners) == 4:
                lons = [c[0] for c in geo_corners]
                lats = [c[1] for c in geo_corners]
                return min(lons), max(lons), min(lats), max(lats)
    
    # Fallback to calculated bounds
    return get_modis_tile_bounds_calculated(h_tile, v_tile)

def get_modis_tile_bounds_calculated(h_tile, v_tile):
    """
    Calculate tile bounds using standard MODIS grid parameters.
    
    This uses the official MODIS sinusoidal projection specification.
    """
    # MODIS sinusoidal projection parameters
    R = 6371007.181  # Earth radius (sphere)
    tile_size = 1111950.519667  # Tile size in meters
    
    # MODIS global grid parameters
    # The grid covers -180° to +180° longitude and approximately -65.4° to +83.6° latitude
    grid_width = 36  # tiles
    grid_height = 18  # tiles
    
    # Calculate the upper-left corner of the global grid in projection coordinates
    total_width = grid_width * tile_size  # Total width in meters
    grid_origin_x = -total_width / 2.0    # Center at 0° longitude
    
    # For latitude, the grid is designed to cover the land areas
    # Using approximate values based on MODIS specification
    grid_origin_y = R * math.radians(90 - 10.0)  # Start at ~80°N
    
    # Calculate this tile's bounds in projection coordinates
    x_min = grid_origin_x + (h_tile * tile_size)
    x_max = grid_origin_x + ((h_tile + 1) * tile_size)
    y_max = grid_origin_y - (v_tile * tile_size)
    y_min = grid_origin_y - ((v_tile + 1) * tile_size)
    
    # Convert to geographic coordinates
    lat_min = math.degrees(y_min / R)
    lat_max = math.degrees(y_max / R)
    
    # Calculate longitude at center latitude
    lat_center = (lat_min + lat_max) / 2.0
    lat_center_rad = math.radians(lat_center)
    
    if abs(lat_center) < 89.9:
        cos_lat = math.cos(lat_center_rad)
        lon_min = math.degrees(x_min / (R * cos_lat))
        lon_max = math.degrees(x_max / (R * cos_lat))
    else:
        lon_min = -180.0
        lon_max = 180.0
    
    # Clamp to valid ranges
    lon_min = max(-180.0, min(180.0, lon_min))
    lon_max = max(-180.0, min(180.0, lon_max))
    lat_min = max(-90.0, min(90.0, lat_min))
    lat_max = max(-90.0, min(90.0, lat_max))
    
    return lon_min, lon_max, lat_min, lat_max

def get_modis_tile_bounds(h_tile, v_tile):
    """
    Calculate accurate geographic bounds for a MODIS tile.
    """
    return get_modis_tile_bounds_calculated(h_tile, v_tile)

def get_tile_bounds(h_tile, v_tile):
    """
    Wrapper function for MODIS tile bounds calculation.
    """
    return get_modis_tile_bounds(h_tile, v_tile)

def sinusoidal_to_geographic(x, y, R=6371007.181):
    """
    Convert sinusoidal projection coordinates to geographic coordinates.
    
    Args:
        x (float): X coordinate in sinusoidal projection (meters)
        y (float): Y coordinate in sinusoidal projection (meters)
        R (float): Earth radius in meters
    
    Returns:
        tuple: (longitude, latitude) in degrees
    """
    # Latitude is straightforward
    lat = math.degrees(y / R)
    
    # Longitude depends on latitude
    lat_rad = math.radians(lat)
    if abs(lat) < 89.9:
        lon = math.degrees(x / (R * math.cos(lat_rad)))
    else:
        lon = 0.0  # Longitude undefined at poles
    
    return lon, lat

def geographic_to_sinusoidal(lon, lat, R=6371007.181):
    """
    Convert geographic coordinates to sinusoidal projection coordinates.
    
    Args:
        lon (float): Longitude in degrees
        lat (float): Latitude in degrees
        R (float): Earth radius in meters
    
    Returns:
        tuple: (x, y) in sinusoidal projection (meters)
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    x = R * lon_rad * math.cos(lat_rad)
    y = R * lat_rad
    
    return x, y

def create_coordinate_grid(h_tile, v_tile, data_shape):
    """
    Create coordinate grids for a MODIS tile.
    
    Args:
        h_tile (int): Horizontal tile number
        v_tile (int): Vertical tile number
        data_shape (tuple): Shape of the data array (rows, cols)
    
    Returns:
        tuple: (lon_grid, lat_grid) - 2D arrays of coordinates
    """
    rows, cols = data_shape
    
    # MODIS parameters
    R = 6371007.181
    tile_size = 1111950.519667
    pixel_size = tile_size / 2400  # 2400 pixels per tile
    grid_origin_x = -20015109.354000
    grid_origin_y = 10007554.677000
    
    # Calculate tile corner in projection coordinates
    tile_x_min = grid_origin_x + (h_tile * tile_size)
    tile_y_max = grid_origin_y - (v_tile * tile_size)
    
    # Create pixel coordinate arrays
    x_coords = np.linspace(tile_x_min, tile_x_min + tile_size, cols)
    y_coords = np.linspace(tile_y_max, tile_y_max - tile_size, rows)
    
    # Create coordinate grids
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    
    # Convert to geographic coordinates
    lon_grid = np.zeros_like(x_grid)
    lat_grid = np.zeros_like(y_grid)
    
    for i in range(rows):
        for j in range(cols):
            lon_grid[i, j], lat_grid[i, j] = sinusoidal_to_geographic(x_grid[i, j], y_grid[i, j], R)
    
    return lon_grid, lat_grid

def test_tile_bounds():
    """
    Test function to verify tile bounds calculation with known examples.
    """
    # Test some known tiles
    test_cases = [
        (18, 9, "Center tile (should be near 0°, 0°)"),
        (0, 0, "Northwest corner"),
        (35, 17, "Southeast corner"),
        (10, 5, "North America"),
        (25, 8, "Europe/Africa"),
        (12, 4, "Eastern US"),
        (8, 5, "Western US")
    ]
    
    print("Testing MODIS tile bounds calculation:")
    print("=" * 70)
    for h, v, description in test_cases:
        lon_min, lon_max, lat_min, lat_max = get_tile_bounds(h, v)
        lon_center = (lon_min + lon_max) / 2.0
        lat_center = (lat_min + lat_max) / 2.0
        
        # Calculate tile width and height in degrees
        lon_width = lon_max - lon_min
        lat_height = lat_max - lat_min
        
        print(f"h{h:02d}v{v:02d} ({description}):")
        print(f"  Bounds: {lon_min:7.2f}° to {lon_max:7.2f}°E, {lat_min:6.2f}° to {lat_max:6.2f}°N")
        print(f"  Center: {lon_center:7.2f}°E, {lat_center:6.2f}°N")
        print(f"  Size:   {lon_width:7.2f}° × {lat_height:6.2f}°")
        print()

def read_hdf_data(hdf_file_path):
    """
    Read data from a GLASS HDF file.
    
    Args:
        hdf_file_path (Path): Path to HDF file
    
    Returns:
        numpy.ndarray or None: Data array or None if failed
    """
    try:
        hdf = SD.SD(str(hdf_file_path), SD.SDC.READ)
        datasets = hdf.datasets()
        
        # Find the main data variable (usually GPP, not QC)
        main_dataset = None
        for dataset_name in datasets.keys():
            if 'QC' not in dataset_name.upper() and 'GPP' in dataset_name.upper():
                main_dataset = dataset_name
                break
        
        if not main_dataset:
            # Fallback: use first non-QC dataset
            for dataset_name in datasets.keys():
                if 'QC' not in dataset_name.upper():
                    main_dataset = dataset_name
                    break
        
        if main_dataset:
            dataset = hdf.select(main_dataset)
            data = dataset.get()
            
            # Apply scaling if available
            attrs = dataset.attributes()
            scale_factor = attrs.get('scale_factor', 1.0)
            add_offset = attrs.get('add_offset', 0.0)
            
            # Convert to float for calculations
            data = data.astype(np.float32)
            data = (data * scale_factor) + add_offset
            
            # Mask invalid values (common fill values are negative or very large)
            valid_mask = (data >= 0) & (data <= 50)  # Reasonable range for GPP (g C/m²/day)
            data_masked = np.ma.masked_where(~valid_mask, data)
            
            dataset.endaccess()
            hdf.end()
            
            return data_masked
        
        hdf.end()
        return None
        
    except Exception as e:
        print(f"    Warning: Could not read {hdf_file_path.name}: {e}")
        return None

def create_global_map(data_dir, product_name, date_name, output_dir):
    """
    Create a global map for a specific date showing all available tiles.
    
    Args:
        data_dir (Path): Directory containing the date subdirectory
        product_name (str): Product name (e.g., 'GPP')
        date_name (str): Date string (e.g., '2010_001')
        output_dir (Path): Directory to save the output map
    
    Returns:
        bool: True if successful, False otherwise
    """
    date_dir = data_dir / date_name
    
    if not date_dir.exists():
        print(f"    Warning: Date directory {date_dir} not found")
        return False
    
    # Get all HDF files for this date
    hdf_files = list(date_dir.glob("*.hdf"))
    
    if not hdf_files:
        print(f"    Warning: No HDF files found in {date_dir}")
        return False
    
    year_str, doy = date_name.split('_')
    
    # Convert DOY to actual date
    try:
        date_obj = datetime.strptime(f"{year_str}{doy.zfill(3)}", "%Y%j")
        date_str = date_obj.strftime("%Y-%m-%d")
    except:
        date_str = f"{year_str}-{doy}"
    
    print(f"    Processing {len(hdf_files)} tiles for {date_str}...")
    
    # Create figure with PlateCarree projection for final display
    fig = plt.figure(figsize=(24, 14))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.7, color='gray')
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.2)
    ax.set_global()
    
    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Define the MODIS Sinusoidal projection for data transformation
    modis_sinusoidal = ccrs.Sinusoidal(central_longitude=0.0)
    
    # Create a grid for assembling tiles more efficiently
    tile_grid = {}
    valid_data_range = []
    
    # First pass: collect all data for global scaling
    for hdf_file in hdf_files:
        try:
            # Extract tile info from filename
            filename_parts = hdf_file.stem.split('.')
            if len(filename_parts) < 4:
                continue
            
            tile_id = filename_parts[3]  # e.g., h18v02
            h_tile = int(tile_id[1:3])
            v_tile = int(tile_id[4:6])
            
            # Get exact tile bounds from HDF metadata
            lon_min, lon_max, lat_min, lat_max = get_modis_tile_bounds_from_metadata(h_tile, v_tile, hdf_file)
            
            # Read data
            data = read_hdf_data(hdf_file)
            
            if data is not None and not data.mask.all():
                # Store tile info for plotting
                tile_grid[(h_tile, v_tile)] = {
                    'data': data,
                    'bounds': (lon_min, lon_max, lat_min, lat_max),
                    'file': hdf_file
                }
                
                # Collect valid data for global scaling
                valid_pixels = data.compressed()
                if len(valid_pixels) > 0:
                    valid_data_range.extend(valid_pixels)
                
        except Exception as e:
            print(f"    Warning: Error processing {hdf_file.name}: {e}")
            continue
    
    if not tile_grid:
        print(f"    Warning: No valid data found for {date_name}")
        return False
    
    # Calculate global statistics for consistent color scaling
    if len(valid_data_range) == 0:
        print(f"    Warning: No valid pixels found for {date_name}")
        return False
    
    valid_data_array = np.array(valid_data_range)
    vmin = np.percentile(valid_data_array, 2)
    vmax = np.percentile(valid_data_array, 98)
    
    print(f"      Data range: {vmin:.2f} to {vmax:.2f} g C/m²/day")
    print(f"      Valid tiles: {len(tile_grid)}")
    
    # Plot all tiles with consistent scaling and proper projection handling
    images = []
    plotted_tiles = 0
    
    for (h_tile, v_tile), tile_info in tile_grid.items():
        data = tile_info['data']
        lon_min, lon_max, lat_min, lat_max = tile_info['bounds']
        
        # Skip tiles with invalid bounds
        if lon_min == lon_max or lat_min == lat_max:
            continue
            
        # Create a properly sampled version of the data for display
        # MODIS tiles are 2400x2400 pixels, downsample for faster rendering
        sample_factor = 8  # More aggressive downsampling for speed
        data_sampled = data[::sample_factor, ::sample_factor]
        
        # Ensure data is not all masked
        if data_sampled.mask.all():
            continue
        
        try:
            # Create coordinate grids for this tile in MODIS projection
            rows, cols = data_sampled.shape
            
            # Get actual MODIS coordinates from metadata if available
            metadata = parse_tile_metadata(tile_info['file'])
            if metadata and PYPROJ_AVAILABLE:
                ul_x, ul_y = metadata['upper_left']
                lr_x, lr_y = metadata['lower_right']
                
                # Create coordinate arrays for the downsampled data
                x_coords = np.linspace(ul_x, lr_x, cols)
                y_coords = np.linspace(ul_y, lr_y, rows)
                x_grid, y_grid = np.meshgrid(x_coords, y_coords)
                
                # Transform all coordinates to geographic
                from pyproj import Transformer, CRS
                
                modis_crs = CRS.from_proj4("+proj=sinu +R=6371007.181 +x_0=0 +y_0=0 +lon_0=0")
                wgs84_crs = CRS.from_epsg(4326)
                
                transformer = Transformer.from_crs(modis_crs, wgs84_crs, always_xy=True)
                
                lon_grid, lat_grid = transformer.transform(x_grid, y_grid)
                
                # Use pcolormesh for proper coordinate transformation
                im = ax.pcolormesh(lon_grid, lat_grid, data_sampled,
                                 transform=ccrs.PlateCarree(),
                                 cmap='YlGn',
                                 alpha=0.8,
                                 vmin=vmin, vmax=vmax,
                                 shading='auto')
                images.append(im)
                plotted_tiles += 1
            else:
                # Fallback to extent-based plotting
                im = ax.imshow(data_sampled, 
                              extent=[lon_min, lon_max, lat_min, lat_max],
                              transform=ccrs.PlateCarree(),
                              cmap='YlGn',
                              alpha=0.8,
                              vmin=vmin, vmax=vmax,
                              interpolation='bilinear',
                              origin='upper')
                images.append(im)
                plotted_tiles += 1
            
        except Exception as e:
            print(f"      Warning: Could not plot tile h{h_tile:02d}v{v_tile:02d}: {e}")
            continue
    
    print(f"      Successfully plotted: {plotted_tiles} tiles")
    
    # Add colorbar using the last image
    if images:
        cbar = plt.colorbar(images[-1], ax=ax, shrink=0.6, pad=0.02, aspect=30)
        cbar.set_label('Gross Primary Production (g C/m²/day)', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
    
    # Add title with comprehensive information
    title = f"GLASS {product_name} Global Coverage - {date_str}\n"
    title += f"Day of Year: {doy} | Tiles: {len(tile_grid)} | "
    title += f"Data Range: {vmin:.1f} - {vmax:.1f} g C/m²/day"
    
    plt.title(title, fontsize=16, pad=20)
    
    # Add statistics text box
    stats_text = f"""Global Statistics:
Valid tiles: {len(tile_grid)}
Mean GPP: {np.mean(valid_data_array):.2f}
Median GPP: {np.median(valid_data_array):.2f}
Std Dev: {np.std(valid_data_array):.2f}
Valid pixels: {len(valid_data_array):,}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the map
    output_file = output_dir / f"{product_name}_global_{date_name}.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.1)
    
    plt.close()
    
    print(f"      ✓ Saved: {output_file}")
    return True

def create_global_maps(data_dir, product='GPP', year=None, max_dates=None):
    """
    Create global maps for all dates in the dataset.
    
    Args:
        data_dir (str): Path to data directory containing product subdirectories
        product (str): Product name (default: 'GPP')
        year (int, optional): Specific year to process
        max_dates (int, optional): Maximum number of dates to process
    """
    data_path = Path(data_dir)
    product_dir = data_path / product
    
    if not product_dir.exists():
        print(f"Error: Product directory {product_dir} not found")
        return
    
    # Create output directory
    output_dir = data_path / "global_maps"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating global maps for {product} data...")
    print(f"Data directory: {product_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find all date directories
    date_dirs = [d for d in product_dir.iterdir() if d.is_dir() and '_' in d.name]
    date_dirs = sorted(date_dirs, key=lambda x: x.name)
    
    # Filter by year if specified
    if year:
        date_dirs = [d for d in date_dirs if d.name.startswith(str(year))]
    
    # Limit number of dates if specified
    if max_dates:
        date_dirs = date_dirs[:max_dates]
    
    if not date_dirs:
        print(f"No date directories found in {product_dir}")
        return
    
    print(f"Found {len(date_dirs)} dates to process")
    
    success_count = 0
    for i, date_dir in enumerate(date_dirs, 1):
        date_name = date_dir.name
        print(f"  [{i:2d}/{len(date_dirs)}] Processing {date_name}...")
        
        if create_global_map(product_dir, product, date_name, output_dir):
            success_count += 1
        else:
            print(f"    ✗ Failed to create map for {date_name}")
    
    print(f"\n✓ Global mapping complete!")
    print(f"  Successfully created: {success_count}/{len(date_dirs)} maps")
    print(f"  Output directory: {output_dir}")
    
    # Create an index file
    create_map_index(output_dir, product, success_count)

def create_map_index(output_dir, product, map_count):
    """Create an HTML index file listing all generated maps."""
    index_file = output_dir / "index.html"
    
    # Get all PNG files
    png_files = sorted(output_dir.glob(f"{product}_global_*.png"))
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>GLASS {product} Global Maps</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .map-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }}
        .map-item {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
        .map-item img {{ max-width: 100%; height: auto; }}
        .map-title {{ font-weight: bold; margin-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>GLASS {product} Global Maps</h1>
        <p>Generated {map_count} global maps showing 8-day GPP composites</p>
    </div>
    
    <div class="map-grid">
"""
    
    for png_file in png_files:
        # Extract date from filename
        filename_parts = png_file.stem.split('_')
        if len(filename_parts) >= 3:
            date_name = '_'.join(filename_parts[2:])
            year_str, doy = date_name.split('_')
            
            try:
                date_obj = datetime.strptime(f"{year_str}{doy.zfill(3)}", "%Y%j")
                display_date = date_obj.strftime("%Y-%m-%d")
            except:
                display_date = date_name
            
            html_content += f"""
        <div class="map-item">
            <div class="map-title">{display_date} (DOY {doy})</div>
            <img src="{png_file.name}" alt="{product} map for {display_date}">
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    with open(index_file, 'w') as f:
        f.write(html_content)
    
    print(f"  ✓ Created index file: {index_file}")

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Create global maps of GLASS GPP data')
    parser.add_argument('--data-dir', '-d',
                       help='Directory containing GLASS data')
    parser.add_argument('--product', '-p', default='GPP',
                       help='Product name (default: GPP)')
    parser.add_argument('--year', '-y', type=int,
                       help='Specific year to process (optional)')
    parser.add_argument('--max-dates', '-m', type=int,
                       help='Maximum number of dates to process (for testing)')
    parser.add_argument('--test-bounds', action='store_true',
                       help='Test tile bounds calculation and exit')
    
    args = parser.parse_args()
    
    if args.test_bounds:
        test_tile_bounds()
        return
    
    if not args.data_dir:
        print("Error: --data-dir is required when not testing bounds")
        return
    
    print("=== GLASS Global Mapper ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Product: {args.product}")
    if args.year:
        print(f"Year filter: {args.year}")
    if args.max_dates:
        print(f"Max dates: {args.max_dates}")
    print()
    
    create_global_maps(args.data_dir, args.product, args.year, args.max_dates)

if __name__ == "__main__":
    main()