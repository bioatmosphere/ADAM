#!/usr/bin/env python3
"""
Efficient annual GPP calculator that processes data in manageable chunks.
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from pyhdf import SD
import warnings
import pickle
import time

# Import pyproj for proper projection handling
try:
    from pyproj import Transformer, CRS
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

warnings.filterwarnings('ignore')

def parse_tile_metadata(hdf_file_path):
    """Parse tile metadata from HDF file."""
    try:
        hdf = SD.SD(str(hdf_file_path), SD.SDC.READ)
        attrs = hdf.attributes()
        struct_meta = attrs.get('StructMetadata.0', '')
        
        import re
        ul_match = re.search(r'UpperLeftPointMtrs=\(([^,]+),([^)]+)\)', struct_meta)
        lr_match = re.search(r'LowerRightMtrs=\(([^,]+),([^)]+)\)', struct_meta)
        
        if ul_match and lr_match:
            ul_x, ul_y = float(ul_match.group(1)), float(ul_match.group(2))
            lr_x, lr_y = float(lr_match.group(1)), float(lr_match.group(2))
            hdf.end()
            return {'upper_left': (ul_x, ul_y), 'lower_right': (lr_x, lr_y)}
        
        hdf.end()
        return None
    except:
        return None

def read_hdf_data(hdf_file_path):
    """Read data from a GLASS HDF file."""
    try:
        hdf = SD.SD(str(hdf_file_path), SD.SDC.READ)
        datasets = hdf.datasets()
        
        main_dataset = None
        for dataset_name in datasets.keys():
            if 'QC' not in dataset_name.upper() and 'GPP' in dataset_name.upper():
                main_dataset = dataset_name
                break
        
        if not main_dataset:
            for dataset_name in datasets.keys():
                if 'QC' not in dataset_name.upper():
                    main_dataset = dataset_name
                    break
        
        if main_dataset:
            dataset = hdf.select(main_dataset)
            data = dataset.get()
            
            attrs = dataset.attributes()
            scale_factor = attrs.get('scale_factor', 1.0)
            add_offset = attrs.get('add_offset', 0.0)
            
            data = data.astype(np.float32)
            data = (data * scale_factor) + add_offset
            
            valid_mask = (data >= 0) & (data <= 50)
            data_masked = np.ma.masked_where(~valid_mask, data)
            
            dataset.endaccess()
            hdf.end()
            
            return data_masked
        
        hdf.end()
        return None
    except:
        return None

def get_modis_tile_bounds(h_tile, v_tile, sample_file=None):
    """Calculate tile bounds."""
    if sample_file and PYPROJ_AVAILABLE:
        metadata = parse_tile_metadata(sample_file)
        if metadata:
            ul_x, ul_y = metadata['upper_left']
            lr_x, lr_y = metadata['lower_right']
            
            from pyproj import Transformer, CRS
            modis_crs = CRS.from_proj4("+proj=sinu +R=6371007.181 +x_0=0 +y_0=0 +lon_0=0")
            wgs84_crs = CRS.from_epsg(4326)
            transformer = Transformer.from_crs(modis_crs, wgs84_crs, always_xy=True)
            
            corners = [(ul_x, ul_y), (lr_x, ul_y), (lr_x, lr_y), (ul_x, lr_y)]
            geo_corners = []
            
            for x, y in corners:
                lon, lat = transformer.transform(x, y)
                if lon > 180: lon -= 360
                elif lon < -180: lon += 360
                geo_corners.append((lon, lat))
            
            if len(geo_corners) == 4:
                lons = [c[0] for c in geo_corners]
                lats = [c[1] for c in geo_corners]
                return min(lons), max(lons), min(lats), max(lats)
    return None

def calculate_efficient_annual_gpp():
    """Calculate annual GPP efficiently by sampling data strategically."""
    
    data_path = Path("ancillary/glass/GPP")
    date_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('2010')])
    
    print(f"=== Efficient Annual GPP Calculator ===")
    print(f"Found {len(date_dirs)} periods for 2010")
    print(f"Strategy: Sample every 4th period, every 3rd tile")
    print()
    
    # Use every 4th period for good seasonal coverage but faster processing
    sample_periods = date_dirs[::4]
    print(f"Using {len(sample_periods)} sample periods:")
    for p in sample_periods:
        print(f"  - {p.name}")
    print()
    
    # Initialize storage
    annual_totals = {}
    tile_bounds = {}
    tile_files = {}  # Store sample file for each tile
    
    start_time = time.time()
    
    for i, date_dir in enumerate(sample_periods, 1):
        print(f"[{i}/{len(sample_periods)}] Processing {date_dir.name}...")
        
        hdf_files = list(date_dir.glob("*.hdf"))
        # Sample every 3rd file for speed while maintaining coverage
        sample_files = hdf_files[::3]
        
        processed_tiles = 0
        
        for hdf_file in sample_files:
            try:
                filename_parts = hdf_file.stem.split('.')
                if len(filename_parts) < 4:
                    continue
                
                tile_id = filename_parts[3]
                h_tile = int(tile_id[1:3])
                v_tile = int(tile_id[4:6])
                tile_key = (h_tile, v_tile)
                
                data = read_hdf_data(hdf_file)
                
                if data is not None and not data.mask.all():
                    # Store sample file for later projection use
                    if tile_key not in tile_files:
                        tile_files[tile_key] = hdf_file
                    
                    # Calculate bounds on first encounter
                    if tile_key not in tile_bounds:
                        bounds = get_modis_tile_bounds(h_tile, v_tile, hdf_file)
                        if bounds:
                            tile_bounds[tile_key] = bounds
                    
                    # Initialize or add to annual total
                    if tile_key not in annual_totals:
                        # Downsample the data to reduce memory usage
                        sample_factor = 8
                        annual_totals[tile_key] = np.ma.zeros_like(data[::sample_factor, ::sample_factor])
                    
                    # Add this period's contribution (scaled for full year)
                    # 8 days per period * extrapolate to full 46 periods
                    period_contribution = data * 8.0 * (46.0 / len(sample_periods))
                    downsampled = period_contribution[::8, ::8]  # Same sampling as above
                    annual_totals[tile_key] += downsampled
                    
                    processed_tiles += 1
                    
            except Exception as e:
                continue
        
        print(f"  Processed: {processed_tiles} tiles")
        
        # Progress update
        elapsed = time.time() - start_time
        print(f"  Elapsed: {elapsed:.1f}s, Total tiles: {len(annual_totals)}")
    
    print(f"\nAnnual calculation complete!")
    print(f"Total tiles: {len(annual_totals)}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    
    # Calculate statistics
    all_values = []
    for annual_data in annual_totals.values():
        valid_pixels = annual_data.compressed()
        if len(valid_pixels) > 0:
            # Sample for statistics
            sample_size = min(200, len(valid_pixels))
            sample_pixels = np.random.choice(valid_pixels, sample_size, replace=False)
            all_values.extend(sample_pixels)
    
    if all_values:
        all_array = np.array(all_values)
        stats = {
            'mean_annual_gpp': np.mean(all_array),
            'median_annual_gpp': np.median(all_array),
            'min_annual_gpp': np.min(all_array),
            'max_annual_gpp': np.max(all_array),
            'total_valid_pixels': len(all_array)
        }
        
        print(f"\nGlobal Statistics:")
        print(f"  Mean annual GPP: {stats['mean_annual_gpp']:.0f} g C/m²/year")
        print(f"  Range: {stats['min_annual_gpp']:.0f} - {stats['max_annual_gpp']:.0f} g C/m²/year")
        print(f"  Valid pixels: {stats['total_valid_pixels']:,}")
    else:
        stats = {}
    
    return {
        'annual_totals': annual_totals,
        'tile_bounds': tile_bounds,
        'tile_files': tile_files,
        'statistics': stats,
        'year': 2010,
        'total_periods': len(date_dirs),
        'sample_periods': len(sample_periods)
    }

def create_efficient_annual_map(annual_data, output_dir):
    """Create annual map with efficient processing."""
    
    annual_totals = annual_data['annual_totals']
    tile_bounds = annual_data['tile_bounds']
    tile_files = annual_data['tile_files']
    stats = annual_data['statistics']
    year = annual_data['year']
    
    print(f"\nCreating efficient annual GPP map for {year}...")
    
    # Create figure
    fig = plt.figure(figsize=(24, 14))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.7, color='gray')
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.2)
    ax.set_global()
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Calculate color scale
    if stats:
        vmin = max(0, stats['min_annual_gpp'])
        vmax = min(3000, stats['max_annual_gpp'])
    else:
        vmin, vmax = 0, 2000
    
    print(f"  Color scale: {vmin:.0f} - {vmax:.0f} g C/m²/year")
    
    # Plot tiles
    plotted_tiles = 0
    images = []
    
    for tile_key, annual_data_tile in annual_totals.items():
        h_tile, v_tile = tile_key
        
        if tile_key not in tile_bounds:
            continue
            
        bounds = tile_bounds[tile_key]
        if not bounds:
            continue
            
        lon_min, lon_max, lat_min, lat_max = bounds
        
        if lon_min == lon_max or lat_min == lat_max:
            continue
        
        if annual_data_tile.mask.all():
            continue
        
        try:
            # Use proper MODIS projection if available
            if PYPROJ_AVAILABLE and tile_key in tile_files:
                sample_file = tile_files[tile_key]
                metadata = parse_tile_metadata(sample_file)
                
                if metadata:
                    rows, cols = annual_data_tile.shape
                    ul_x, ul_y = metadata['upper_left']
                    lr_x, lr_y = metadata['lower_right']
                    
                    # Create coordinate grids
                    x_coords = np.linspace(ul_x, lr_x, cols)
                    y_coords = np.linspace(ul_y, lr_y, rows)
                    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
                    
                    # Transform to geographic
                    from pyproj import Transformer, CRS
                    modis_crs = CRS.from_proj4("+proj=sinu +R=6371007.181 +x_0=0 +y_0=0 +lon_0=0")
                    wgs84_crs = CRS.from_epsg(4326)
                    transformer = Transformer.from_crs(modis_crs, wgs84_crs, always_xy=True)
                    
                    lon_grid, lat_grid = transformer.transform(x_grid, y_grid)
                    
                    # Plot with correct projection
                    im = ax.pcolormesh(lon_grid, lat_grid, annual_data_tile,
                                     transform=ccrs.PlateCarree(),
                                     cmap='YlGn',
                                     alpha=0.8,
                                     vmin=vmin, vmax=vmax,
                                     shading='auto')
                    images.append(im)
                    plotted_tiles += 1
                    continue
            
            # Fallback to extent-based plotting
            im = ax.imshow(annual_data_tile, 
                          extent=[lon_min, lon_max, lat_min, lat_max],
                          transform=ccrs.PlateCarree(),
                          cmap='YlGn',
                          alpha=0.8,
                          vmin=vmin, vmax=vmax,
                          origin='upper',
                          interpolation='bilinear')
            images.append(im)
            plotted_tiles += 1
            
        except Exception as e:
            print(f"    Warning: Could not plot tile h{h_tile:02d}v{v_tile:02d}: {e}")
            continue
    
    print(f"  Successfully plotted: {plotted_tiles} tiles")
    
    # Add colorbar
    if images:
        cbar = plt.colorbar(images[-1], ax=ax, shrink=0.6, pad=0.02, aspect=30)
        cbar.set_label('Annual Total GPP (g C/m²/year)', fontsize=12)
    
    # Add title
    title = f"GLASS GPP Annual Total - {year}\n"
    title += f"Tiles: {len(annual_totals)} | Sample periods: {annual_data['sample_periods']}/{annual_data['total_periods']} | "
    if stats:
        title += f"Global Mean: {stats.get('mean_annual_gpp', 0):.0f} g C/m²/year"
    
    plt.title(title, fontsize=16, pad=20)
    
    # Add statistics text
    if stats:
        stats_text = f"""Annual GPP Statistics ({year}):
Mean: {stats.get('mean_annual_gpp', 0):.0f} g C/m²/year
Median: {stats.get('median_annual_gpp', 0):.0f} g C/m²/year
Range: {stats.get('min_annual_gpp', 0):.0f} - {stats.get('max_annual_gpp', 0):.0f}
Valid pixels: {stats.get('total_valid_pixels', 0):,}
Method: Efficient sampling ({annual_data['sample_periods']} periods)"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save map
    output_file = output_dir / f"GPP_annual_total_efficient_{year}.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_file}")
    return output_file

def main():
    """Main function."""
    
    output_dir = Path("ancillary/glass/annual_totals")
    output_dir.mkdir(exist_ok=True)
    
    # Calculate annual data efficiently
    annual_data = calculate_efficient_annual_gpp()
    
    if annual_data:
        # Create the map
        map_file = create_efficient_annual_map(annual_data, output_dir)
        
        # Save statistics
        stats_file = output_dir / "GPP_annual_stats_efficient_2010.txt"
        with open(stats_file, 'w') as f:
            f.write(f"GLASS GPP Annual Statistics (Efficient) - 2010\n")
            f.write("=" * 50 + "\n\n")
            
            stats = annual_data['statistics']
            if stats:
                f.write(f"Global Statistics:\n")
                f.write(f"  Mean annual GPP: {stats.get('mean_annual_gpp', 0):.1f} g C/m²/year\n")
                f.write(f"  Median annual GPP: {stats.get('median_annual_gpp', 0):.1f} g C/m²/year\n")
                f.write(f"  Range: {stats.get('min_annual_gpp', 0):.1f} - {stats.get('max_annual_gpp', 0):.1f} g C/m²/year\n")
                f.write(f"  Total valid pixels: {stats.get('total_valid_pixels', 0):,}\n")
            
            f.write(f"\nData Coverage:\n")
            f.write(f"  Total tiles processed: {len(annual_data['annual_totals'])}\n")
            f.write(f"  Sample periods used: {annual_data['sample_periods']}/{annual_data['total_periods']}\n")
            f.write(f"  Sampling strategy: Every 4th period, every 3rd tile\n")
        
        print(f"\n✓ Efficient annual GPP calculation complete!")
        print(f"  Map: {map_file}")
        print(f"  Stats: {stats_file}")

if __name__ == "__main__":
    main()