#!/usr/bin/env python3
"""
Efficient yearly GLASS GPP global mapper with data sampling.
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from pyhdf import SD
import warnings

# Import pyproj
try:
    from pyproj import Proj, transform
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

warnings.filterwarnings('ignore')

def get_modis_projection():
    """Get the MODIS sinusoidal projection."""
    if PYPROJ_AVAILABLE:
        return Proj(proj='sinu', R=6371007.181, x_0=0, y_0=0, lon_0=0)
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
        
        return (west_lon, south_lat, east_lon, north_lat)  # west, south, east, north
    
    return None

def parse_tile_metadata(hdf_file_path):
    """Parse tile metadata using filename-based approach for better accuracy."""
    # Use filename-based approach which is more reliable
    bounds = get_tile_bounds_from_name(hdf_file_path.name)
    if bounds:
        return bounds
    
    # Fallback to metadata parsing
    try:
        hdf = SD.SD(str(hdf_file_path), SD.SDC.READ)
        attrs = hdf.attributes()
        struct_meta = attrs.get('StructMetadata.0', '')
        
        import re
        ul_match = re.search(r'UpperLeftPointMtrs=\(([^,]+),([^)]+)\)', struct_meta)
        lr_match = re.search(r'LowerRightMtrs=\(([^,]+),([^)]+)\)', struct_meta)
        
        if ul_match and lr_match and PYPROJ_AVAILABLE:
            ul_x, ul_y = float(ul_match.group(1)), float(ul_match.group(2))
            lr_x, lr_y = float(lr_match.group(1)), float(lr_match.group(2))
            
            modis_proj = get_modis_projection()
            wgs84_proj = Proj(proj='latlong', datum='WGS84')
            
            ul_lon, ul_lat = transform(modis_proj, wgs84_proj, ul_x, ul_y)
            lr_lon, lr_lat = transform(modis_proj, wgs84_proj, lr_x, lr_y)
            
            hdf.end()
            return (ul_lon, lr_lat, lr_lon, ul_lat)  # west, south, east, north
        
        hdf.end()
        return None
    except:
        return None

def read_gpp_data_sampled(hdf_file_path, sample_factor=10):
    """Read and sample GPP data from HDF file."""
    try:
        hdf = SD.SD(str(hdf_file_path), SD.SDC.READ)
        sds = hdf.select('GPP')
        data = sds.get()
        
        # Sample the data to reduce processing time
        data = data[::sample_factor, ::sample_factor]
        
        # Get attributes
        attrs = sds.attributes()
        scale_factor = attrs.get('scale_factor', 0.01)
        add_offset = attrs.get('add_offset', 0.0)
        fill_value = attrs.get('_FillValue', 4294967295)
        
        # Apply scaling
        data = data.astype(np.float32)
        valid_mask = data != fill_value
        data[valid_mask] = data[valid_mask] * scale_factor + add_offset
        data[~valid_mask] = np.nan
        
        # Get metadata using filename-based approach
        bounds = parse_tile_metadata(hdf_file_path)
        
        hdf.end()
        return data, bounds
        
    except Exception as e:
        return None, None

def create_efficient_yearly_gpp_map():
    """Create an efficient global map of yearly GPP data."""
    
    data_path = Path("ancillary/glass/GPP_YEARLY/2010_001")
    hdf_files = list(data_path.glob("*.hdf"))
    
    print(f"📊 Processing {len(hdf_files)} yearly GPP tiles (sampled)...")
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.6)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.4)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.4)
    ax.set_global()
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    # Process every 3rd tile for efficiency
    all_data = []
    tiles_plotted = 0
    sample_tiles = hdf_files[::3]  # Process every 3rd tile
    
    print(f"📊 Sampling {len(sample_tiles)} tiles for faster processing...")
    
    for i, hdf_file in enumerate(sample_tiles):
        if i % 20 == 0:
            print(f"  Processing sample {i+1}/{len(sample_tiles)}...")
        
        data, bounds = read_gpp_data_sampled(hdf_file, sample_factor=20)
        
        if data is None or bounds is None:
            continue
        
        west, south, east, north = bounds
        
        # Create coordinate arrays
        lons = np.linspace(west, east, data.shape[1] + 1)
        lats = np.linspace(north, south, data.shape[0] + 1)
        
        # Plot data
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 100:  # Only plot if enough valid data
            im = ax.pcolormesh(lons, lats, data, 
                             transform=ccrs.PlateCarree(),
                             cmap='YlGn', 
                             vmin=0, vmax=3000,
                             shading='flat',
                             alpha=0.8)
            all_data.extend(valid_data)
            tiles_plotted += 1
    
    print(f"✅ Successfully plotted {tiles_plotted} tiles")
    
    # Calculate statistics
    if all_data:
        all_data = np.array(all_data)
        mean_gpp = np.mean(all_data)
        median_gpp = np.median(all_data)
        min_gpp = np.min(all_data)
        max_gpp = np.max(all_data)
        
        print(f"📊 Global Statistics (sampled):")
        print(f"   Mean: {mean_gpp:.1f} g C/m²/year")
        print(f"   Median: {median_gpp:.1f} g C/m²/year")
        print(f"   Range: {min_gpp:.1f} - {max_gpp:.1f} g C/m²/year")
        print(f"   Valid pixels: {len(all_data):,}")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('GPP (g C/m²/year)', fontsize=12)
        
        # Add title
        title = f"GLASS Yearly GPP - 2010 (Corrected Projection)\n"
        title += f"Sample Tiles: {tiles_plotted} | Global Mean: {mean_gpp:.1f} g C/m²/year"
        plt.title(title, fontsize=16, pad=20)
        
        # Save plot
        output_path = Path("ancillary/glass/yearly_gpp_global_corrected_2010.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_path}")
        
        # Save statistics
        stats_path = Path("ancillary/glass/yearly_gpp_corrected_stats_2010.txt")
        with open(stats_path, 'w') as f:
            f.write("GLASS Yearly GPP Statistics (Sampled) - 2010\n")
            f.write("===============================================\n\n")
            f.write(f"Sample tiles processed: {tiles_plotted} (every 3rd tile)\n")
            f.write(f"Spatial sampling: Every 20th pixel\n")
            f.write(f"Valid pixels: {len(all_data):,}\n\n")
            f.write("Global Statistics:\n")
            f.write(f"  Mean: {mean_gpp:.1f} g C/m²/year\n")
            f.write(f"  Median: {median_gpp:.1f} g C/m²/year\n")
            f.write(f"  Minimum: {min_gpp:.1f} g C/m²/year\n")
            f.write(f"  Maximum: {max_gpp:.1f} g C/m²/year\n")
        
        print(f"📄 Saved: {stats_path}")
        
        plt.show()
        return True
        
    else:
        print("❌ No valid data found")
        return False

if __name__ == "__main__":
    create_efficient_yearly_gpp_map()