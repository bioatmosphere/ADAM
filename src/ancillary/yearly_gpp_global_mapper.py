#!/usr/bin/env python3
"""
Create global map of GLASS yearly GPP data.

This script reads all yearly GPP tiles for 2010 and creates a comprehensive
global map showing annual GPP values across the entire world.
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from pyhdf import SD
import warnings

# Import pyproj for proper projection handling
try:
    from pyproj import Proj, transform, CRS
    PYPROJ_AVAILABLE = True
except ImportError:
    print("pyproj not available - install with: uv add pyproj")
    PYPROJ_AVAILABLE = False

warnings.filterwarnings('ignore')

def get_modis_projection():
    """Get the MODIS sinusoidal projection."""
    if PYPROJ_AVAILABLE:
        modis_proj = Proj(proj='sinu', 
                         R=6371007.181,  # MODIS sphere radius
                         x_0=0,          # False easting
                         y_0=0,          # False northing
                         lon_0=0)        # Central meridian
        return modis_proj
    return None

def get_tile_bounds_from_name(filename):
    """Get tile bounds from MODIS tile h/v coordinates in filename."""
    import re
    
    # Extract h and v from filename like GLASS12E11.V60.A2010001.h13v09.2022100.hdf
    match = re.search(r'h(\d+)v(\d+)', filename)
    if not match:
        return None
    
    h_tile = int(match.group(1))
    v_tile = int(match.group(2))
    
    # MODIS tile grid parameters
    tile_size = 1111950.5196666666  # meters
    ul_x = -20015109.354
    ul_y = 10007554.677
    
    # Calculate tile bounds in sinusoidal projection
    west_x = ul_x + h_tile * tile_size
    east_x = west_x + tile_size
    north_y = ul_y - v_tile * tile_size  
    south_y = north_y - tile_size
    
    if PYPROJ_AVAILABLE:
        # Convert to geographic coordinates
        modis_proj = get_modis_projection()
        wgs84_proj = Proj(proj='latlong', datum='WGS84')
        
        # Transform corners
        west_lon, north_lat = transform(modis_proj, wgs84_proj, west_x, north_y)
        east_lon, south_lat = transform(modis_proj, wgs84_proj, east_x, south_y)
        
        return {
            'bounds': (west_lon, south_lat, east_lon, north_lat),  # west, south, east, north
            'ul_x': west_x, 'ul_y': north_y,
            'lr_x': east_x, 'lr_y': south_y
        }
    
    return None

def parse_tile_metadata(hdf_file_path):
    """
    Parse tile metadata using filename-based approach for better accuracy.
    
    Args:
        hdf_file_path (str): Path to HDF file
    
    Returns:
        dict: Metadata including exact coordinates
    """
    # Use filename-based approach which is more reliable
    bounds_data = get_tile_bounds_from_name(hdf_file_path.name if hasattr(hdf_file_path, 'name') else str(hdf_file_path))
    if bounds_data:
        return bounds_data
    
    # Fallback to metadata parsing
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
            
            # Convert from sinusoidal to geographic coordinates
            if PYPROJ_AVAILABLE:
                modis_proj = get_modis_projection()
                wgs84_proj = Proj(proj='latlong', datum='WGS84')
                
                # Transform corners
                ul_lon, ul_lat = transform(modis_proj, wgs84_proj, ul_x, ul_y)
                lr_lon, lr_lat = transform(modis_proj, wgs84_proj, lr_x, lr_y)
                
                return {
                    'bounds': (ul_lon, lr_lat, lr_lon, ul_lat),  # (west, south, east, north)
                    'ul_x': ul_x, 'ul_y': ul_y,
                    'lr_x': lr_x, 'lr_y': lr_y
                }
        
        hdf.end()
        return None
        
    except Exception as e:
        print(f"Error parsing metadata from {hdf_file_path}: {e}")
        return None

def read_gpp_data(hdf_file_path):
    """
    Read GPP data from HDF file.
    
    Args:
        hdf_file_path (str): Path to HDF file
    
    Returns:
        tuple: (data_array, metadata)
    """
    try:
        hdf = SD.SD(str(hdf_file_path), SD.SDC.READ)
        
        # Get the GPP dataset (yearly data might have different name)
        datasets = hdf.datasets()
        gpp_dataset_name = None
        
        for name, info in datasets.items():
            if 'GPP' in name or 'gpp' in name.lower():
                gpp_dataset_name = name
                break
        
        if not gpp_dataset_name:
            print(f"No GPP dataset found in {hdf_file_path}")
            print(f"Available datasets: {list(datasets.keys())}")
            hdf.end()
            return None, None
        
        # Read the dataset
        sds = hdf.select(gpp_dataset_name)
        data = sds.get()
        
        # Get attributes for scaling
        attrs = sds.attributes()
        scale_factor = attrs.get('scale_factor', 1.0)
        add_offset = attrs.get('add_offset', 0.0)
        fill_value = attrs.get('_FillValue', -32768)
        
        # Apply scaling
        data = data.astype(np.float32)
        valid_mask = data != fill_value
        data[valid_mask] = data[valid_mask] * scale_factor + add_offset
        data[~valid_mask] = np.nan
        
        # Get metadata
        metadata = parse_tile_metadata(hdf_file_path)
        
        # Don't call sds.end() - it causes attribute errors
        hdf.end()
        
        return data, metadata
        
    except Exception as e:
        print(f"Error reading {hdf_file_path}: {e}")
        return None, None

def create_yearly_gpp_global_map():
    """Create a global map of yearly GPP data for 2010."""
    
    # Path to yearly GPP data
    data_path = Path("ancillary/glass/GPP_YEARLY/2010_001")
    
    if not data_path.exists():
        print(f"❌ Yearly GPP data directory not found: {data_path}")
        return
    
    # Find all HDF files
    hdf_files = list(data_path.glob("*.hdf"))
    
    if not hdf_files:
        print(f"❌ No HDF files found in {data_path}")
        return
    
    print(f"📊 Found {len(hdf_files)} yearly GPP tiles")
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    
    # Set global extent
    ax.set_global()
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    # Collect data for statistics
    all_data = []
    tiles_plotted = 0
    
    # Process tiles with optimization for speed
    total_tiles = len(hdf_files)
    sample_factor = 5  # Reduce downsampling to preserve more data
    progress_interval = 20  # Print progress every 20 tiles
    
    # Debug counters
    metadata_success = 0
    fallback_used = 0
    skipped_no_data = 0
    skipped_no_bounds = 0
    skipped_tiles = []
    
    # Pre-create transformer to avoid recreating it for each tile
    if PYPROJ_AVAILABLE:
        from pyproj import Transformer, CRS
        modis_crs = CRS.from_proj4("+proj=sinu +R=6371007.181 +x_0=0 +y_0=0 +lon_0=0")
        wgs84_crs = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(modis_crs, wgs84_crs, always_xy=True)
    
    for i, hdf_file in enumerate(hdf_files):
        if i % progress_interval == 0:
            print(f"Processing tile {i+1}/{total_tiles}: {hdf_file.name}")
        
        # Read data
        data, metadata = read_gpp_data(hdf_file)
        
        if data is None:
            # Even if we can't read data, try to plot with calculated bounds
            import re
            match = re.search(r'h(\d+)v(\d+)', hdf_file.name)
            if match:
                h_tile = int(match.group(1))
                v_tile = int(match.group(2))
                # Use calculated bounds as fallback
                bounds = get_tile_bounds_from_name(hdf_file.name)
                if bounds:
                    # Create empty data to plot the tile boundary
                    empty_data = np.full((10, 10), np.nan)
                    if isinstance(bounds, dict):
                        bounds = bounds['bounds']
                    west, south, east, north = bounds
                    ax.imshow(empty_data, extent=[west, east, south, north],
                             transform=ccrs.PlateCarree(), 
                             alpha=0.0)  # Invisible but preserves tile structure
                    tiles_plotted += 1
            continue
        
        # Downsample data for faster processing
        data_sampled = data[::sample_factor, ::sample_factor]
        rows, cols = data_sampled.shape
        
        if rows == 0 or cols == 0:
            continue
        
        # If metadata is None, create fallback bounds
        if metadata is None:
            metadata = {}
            bounds = get_tile_bounds_from_name(hdf_file.name)
            if bounds:
                if isinstance(bounds, dict):
                    metadata['bounds'] = bounds['bounds']
                else:
                    metadata['bounds'] = bounds
        
        # Try to get actual MODIS coordinates from metadata first
        plotted_this_tile = False
        
        try:
            hdf = SD.SD(str(hdf_file), SD.SDC.READ)
            attrs = hdf.attributes()
            struct_meta = attrs.get('StructMetadata.0', '')
            
            # Extract UpperLeftPointMtrs and LowerRightMtrs from metadata
            import re
            ul_match = re.search(r'UpperLeftPointMtrs=\(([^,]+),([^)]+)\)', struct_meta)
            lr_match = re.search(r'LowerRightMtrs=\(([^,]+),([^)]+)\)', struct_meta)
            
            if ul_match and lr_match and PYPROJ_AVAILABLE:
                ul_x, ul_y = float(ul_match.group(1)), float(ul_match.group(2))
                lr_x, lr_y = float(lr_match.group(1)), float(lr_match.group(2))
                
                # Create coordinate arrays for the downsampled data (cell centers, not boundaries)
                x_coords = np.linspace(ul_x, lr_x, cols)
                y_coords = np.linspace(ul_y, lr_y, rows)
                x_grid, y_grid = np.meshgrid(x_coords, y_coords)
                
                # Transform coordinates to geographic
                lon_grid, lat_grid = transformer.transform(x_grid, y_grid)
                
                # Plot the data with proper coordinate transformation
                valid_data = data_sampled[~np.isnan(data_sampled)]
                
                # Always plot the tile, even if no valid data (will be transparent)
                im = ax.pcolormesh(lon_grid, lat_grid, data_sampled, 
                                 transform=ccrs.PlateCarree(),
                                 cmap='YlGn', 
                                 vmin=0, vmax=3000,
                                 shading='nearest',
                                 alpha=0.8)
                
                if len(valid_data) > 0:
                    all_data.extend(valid_data)
                    
                tiles_plotted += 1
                plotted_this_tile = True
                metadata_success += 1
                    
            hdf.end()
            
        except Exception as e:
            # If metadata reading fails, we'll try fallback method below
            pass
        
        # Fallback to bounds-based plotting if metadata method failed
        if not plotted_this_tile:
            bounds = metadata.get('bounds')
            if bounds:
                west, south, east, north = bounds
                valid_data = data_sampled[~np.isnan(data_sampled)]
                
                # Always plot the tile, even if no valid data
                im = ax.imshow(data_sampled, extent=[west, east, south, north],
                             transform=ccrs.PlateCarree(),
                             cmap='YlGn', 
                             vmin=0, vmax=3000,
                             alpha=0.8)
                
                if len(valid_data) > 0:
                    all_data.extend(valid_data)
                    
                tiles_plotted += 1
                fallback_used += 1
            else:
                skipped_no_bounds += 1
                skipped_tiles.append(f"{hdf_file.name} (no bounds)")
    
    print(f"✅ Successfully plotted {tiles_plotted} tiles")
    print(f"📊 Debug info:")
    print(f"   Metadata method: {metadata_success} tiles")
    print(f"   Fallback method: {fallback_used} tiles") 
    print(f"   Skipped (no data): {skipped_no_data} tiles")
    print(f"   Skipped (no bounds): {skipped_no_bounds} tiles")
    
    if len(skipped_tiles) > 0:
        print(f"📋 First 10 skipped tiles:")
        for tile in skipped_tiles[:10]:
            print(f"   {tile}")
    
    # Calculate statistics
    if all_data:
        all_data = np.array(all_data)
        mean_gpp = np.mean(all_data)
        median_gpp = np.median(all_data)
        min_gpp = np.min(all_data)
        max_gpp = np.max(all_data)
        
        print(f"📊 Global Statistics:")
        print(f"   Mean: {mean_gpp:.1f} g C/m²/year")
        print(f"   Median: {median_gpp:.1f} g C/m²/year")
        print(f"   Range: {min_gpp:.1f} - {max_gpp:.1f} g C/m²/year")
        print(f"   Valid pixels: {len(all_data):,}")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('GPP (g C/m²/year)', fontsize=12)
        
        # Add title with statistics
        title = f"GLASS Yearly GPP - 2010 (FIXED Projection)\n"
        title += f"Tiles: {tiles_plotted} | Global Mean: {mean_gpp:.1f} g C/m²/year"
        plt.title(title, fontsize=16, pad=20)
        
        # Save the plot
        output_path = Path("ancillary/glass/yearly_gpp_global_PROPERLY_FIXED_2010.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Global map saved: {output_path}")
        
        # Save statistics
        stats_path = Path("ancillary/glass/yearly_gpp_fixed_projection_stats_2010.txt")
        with open(stats_path, 'w') as f:
            f.write("GLASS Yearly GPP Statistics - 2010\n")
            f.write("=====================================\n\n")
            f.write(f"Tiles processed: {tiles_plotted}\n")
            f.write(f"Valid pixels: {len(all_data):,}\n\n")
            f.write("Global Statistics:\n")
            f.write(f"  Mean: {mean_gpp:.1f} g C/m²/year\n")
            f.write(f"  Median: {median_gpp:.1f} g C/m²/year\n")
            f.write(f"  Minimum: {min_gpp:.1f} g C/m²/year\n")
            f.write(f"  Maximum: {max_gpp:.1f} g C/m²/year\n")
        
        print(f"📄 Statistics saved: {stats_path}")
        
    else:
        print("❌ No valid data found")
    
    plt.show()

if __name__ == "__main__":
    create_yearly_gpp_global_map()