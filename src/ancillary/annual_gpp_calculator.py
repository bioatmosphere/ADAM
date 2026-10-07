#!/usr/bin/env python3
"""
Calculate annual total GPP for 2010 from GLASS 8-day composite data.

This script:
1. Reads all 8-day GPP composites for 2010
2. Accumulates them to calculate annual total GPP
3. Creates a global map of annual GPP totals
4. Provides summary statistics
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from datetime import datetime
from pyhdf import SD
import warnings

# Import pyproj for proper projection handling
try:
    from pyproj import Transformer, CRS
    PYPROJ_AVAILABLE = True
except ImportError:
    print("pyproj not available - install with: uv add pyproj")
    PYPROJ_AVAILABLE = False

warnings.filterwarnings('ignore')

def parse_tile_metadata(hdf_file_path):
    """Parse tile metadata from HDF file to get exact coordinates."""
    try:
        hdf = SD.SD(str(hdf_file_path), SD.SDC.READ)
        attrs = hdf.attributes()
        
        # Parse StructMetadata to get exact coordinates
        struct_meta = attrs.get('StructMetadata.0', '')
        
        # Extract UpperLeftPointMtrs and LowerRightMtrs
        import re
        ul_match = re.search(r'UpperLeftPointMtrs=\\(([^,]+),([^)]+)\\)', struct_meta)
        lr_match = re.search(r'LowerRightMtrs=\\(([^,]+),([^)]+)\\)', struct_meta)
        
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
        return None

def modis_to_geographic_pyproj(x, y):
    """Convert MODIS sinusoidal to geographic using pyproj."""
    if PYPROJ_AVAILABLE:
        from pyproj import Transformer, CRS
        
        modis_crs = CRS.from_proj4("+proj=sinu +R=6371007.181 +x_0=0 +y_0=0 +lon_0=0")
        wgs84_crs = CRS.from_epsg(4326)
        
        transformer = Transformer.from_crs(modis_crs, wgs84_crs, always_xy=True)
        
        lon, lat = transformer.transform(x, y)
        return lon, lat
    else:
        return None, None

def get_modis_tile_bounds_from_metadata(h_tile, v_tile, sample_file=None):
    """Calculate tile bounds using actual tile metadata if available."""
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
    
    # Fallback to calculated bounds (simplified version)
    return None

def read_hdf_data(hdf_file_path):
    """Read data from a GLASS HDF file."""
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

def calculate_annual_gpp(data_dir, product='GPP', year=2010):
    """
    Calculate annual total GPP by summing all 8-day composites.
    
    Args:
        data_dir (str): Path to data directory containing product subdirectories
        product (str): Product name (default: 'GPP')
        year (int): Year to process (default: 2010)
    
    Returns:
        dict: Contains annual totals, tile info, and statistics
    """
    data_path = Path(data_dir)
    product_dir = data_path / product
    
    if not product_dir.exists():
        print(f"Error: Product directory {product_dir} not found")
        return None
    
    print(f"Calculating annual total GPP for {year}...")
    print(f"Data directory: {product_dir}")
    
    # Find all date directories for the specified year
    date_dirs = [d for d in product_dir.iterdir() if d.is_dir() and d.name.startswith(str(year))]
    date_dirs = sorted(date_dirs, key=lambda x: x.name)
    
    if not date_dirs:
        print(f"No date directories found for {year} in {product_dir}")
        return None
    
    print(f"Found {len(date_dirs)} 8-day periods to process")
    
    # Initialize storage for annual totals
    annual_totals = {}  # {(h_tile, v_tile): annual_sum_array}
    tile_bounds = {}    # {(h_tile, v_tile): (lon_min, lon_max, lat_min, lat_max)}
    valid_periods = {}  # {(h_tile, v_tile): count_of_valid_periods}
    
    # Process each 8-day period
    for i, date_dir in enumerate(date_dirs, 1):
        date_name = date_dir.name
        print(f"  [{i:2d}/{len(date_dirs)}] Processing {date_name}...")
        
        # Get all HDF files for this date
        hdf_files = list(date_dir.glob("*.hdf"))
        
        if not hdf_files:
            print(f"    Warning: No HDF files found in {date_dir}")
            continue
        
        period_tiles = 0
        
        # Process each tile for this period
        for hdf_file in hdf_files:
            try:
                # Extract tile info from filename
                filename_parts = hdf_file.stem.split('.')
                if len(filename_parts) < 4:
                    continue
                
                tile_id = filename_parts[3]  # e.g., h18v02
                h_tile = int(tile_id[1:3])
                v_tile = int(tile_id[4:6])
                tile_key = (h_tile, v_tile)
                
                # Read data
                data = read_hdf_data(hdf_file)
                
                if data is not None and not data.mask.all():
                    # Store tile bounds on first encounter
                    if tile_key not in tile_bounds:
                        bounds = get_modis_tile_bounds_from_metadata(h_tile, v_tile, hdf_file)
                        if bounds:
                            tile_bounds[tile_key] = bounds
                    
                    # Initialize annual total array for this tile if needed
                    if tile_key not in annual_totals:
                        annual_totals[tile_key] = np.ma.zeros_like(data)
                        valid_periods[tile_key] = 0
                    
                    # Add this period's data to annual total
                    # GPP is in g C/m²/day, so multiply by 8 to get 8-day total
                    period_total = data * 8.0
                    annual_totals[tile_key] += period_total
                    valid_periods[tile_key] += 1
                    period_tiles += 1
                
            except Exception as e:
                print(f"    Warning: Error processing {hdf_file.name}: {e}")
                continue
        
        print(f"    Processed: {period_tiles} tiles")
    
    if not annual_totals:
        print(f"No valid data found for {year}")
        return None
    
    print(f"\nAnnual GPP calculation complete!")
    print(f"  Total tiles with data: {len(annual_totals)}")
    
    # Calculate statistics
    all_annual_values = []
    for tile_key, annual_data in annual_totals.items():
        valid_pixels = annual_data.compressed()
        if len(valid_pixels) > 0:
            all_annual_values.extend(valid_pixels)
    
    if all_annual_values:
        all_annual_array = np.array(all_annual_values)
        stats = {
            'mean_annual_gpp': np.mean(all_annual_array),
            'median_annual_gpp': np.median(all_annual_array),
            'min_annual_gpp': np.min(all_annual_array),
            'max_annual_gpp': np.max(all_annual_array),
            'std_annual_gpp': np.std(all_annual_array),
            'total_valid_pixels': len(all_annual_array)
        }
        
        print(f"  Global statistics:")
        print(f"    Mean annual GPP: {stats['mean_annual_gpp']:.1f} g C/m²/year")
        print(f"    Median annual GPP: {stats['median_annual_gpp']:.1f} g C/m²/year")
        print(f"    Range: {stats['min_annual_gpp']:.1f} - {stats['max_annual_gpp']:.1f} g C/m²/year")
        print(f"    Valid pixels: {stats['total_valid_pixels']:,}")
    else:
        stats = {}
    
    return {
        'annual_totals': annual_totals,
        'tile_bounds': tile_bounds,
        'valid_periods': valid_periods,
        'statistics': stats,
        'year': year,
        'total_periods': len(date_dirs)
    }

def create_annual_gpp_map(annual_data, output_dir):
    """Create a global map showing annual total GPP."""
    
    if not annual_data:
        print("No annual data to map")
        return
    
    annual_totals = annual_data['annual_totals']
    tile_bounds = annual_data['tile_bounds']
    stats = annual_data['statistics']
    year = annual_data['year']
    
    print(f"\nCreating annual GPP map for {year}...")
    
    # Create figure with PlateCarree projection
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
    
    # Calculate global color scale from actual data
    all_values = []
    for annual_total in annual_totals.values():
        if not annual_total.mask.all():
            valid_pixels = annual_total.compressed()
            if len(valid_pixels) > 0:
                # Sample pixels for color scale calculation
                sample_size = min(1000, len(valid_pixels))
                sample_pixels = np.random.choice(valid_pixels, sample_size, replace=False)
                all_values.extend(sample_pixels)
    
    if all_values and len(all_values) > 100:
        all_array = np.array(all_values)
        vmin = max(0, np.percentile(all_array, 2))
        vmax = min(4000, np.percentile(all_array, 98))
        print(f"  Data-based color scale: {vmin:.0f} - {vmax:.0f} g C/m²/year")
    else:
        # Fallback to stats-based scale
        if stats:
            vmin = max(0, stats.get('min_annual_gpp', 0))
            vmax = min(4000, stats.get('max_annual_gpp', 3000))
        else:
            vmin, vmax = 0, 3000
        print(f"  Fallback color scale: {vmin:.0f} - {vmax:.0f} g C/m²/year")
    
    # Plot all tiles
    images = []
    plotted_tiles = 0
    
    for tile_key, annual_total in annual_totals.items():
        h_tile, v_tile = tile_key
        
        if tile_key not in tile_bounds:
            continue
            
        bounds = tile_bounds[tile_key]
        if not bounds:
            continue
            
        lon_min, lon_max, lat_min, lat_max = bounds
        
        # Skip tiles with invalid bounds
        if lon_min == lon_max or lat_min == lat_max:
            continue
            
        # Downsample for display (annual totals are large arrays)
        sample_factor = 15
        annual_sampled = annual_total[::sample_factor, ::sample_factor]
        
        # Ensure data is not all masked
        if annual_sampled.mask.all():
            continue
        
        try:
            # Use proper coordinate transformation with pcolormesh for accurate projection
            if PYPROJ_AVAILABLE and bounds:
                # Find a sample file for this tile to get metadata
                data_path = Path("ancillary/glass/GPP")
                sample_files = []
                
                # Search through date directories for this tile
                for date_dir in data_path.iterdir():
                    if date_dir.is_dir() and date_dir.name.startswith('2010'):
                        tile_files = list(date_dir.glob(f"*h{h_tile:02d}v{v_tile:02d}*.hdf"))
                        if tile_files:
                            sample_files.extend(tile_files[:1])  # Just take first one found
                            break
                
                if sample_files:
                    metadata = parse_tile_metadata(sample_files[0])
                    
                    if metadata:
                        rows, cols = annual_sampled.shape
                        ul_x, ul_y = metadata['upper_left']
                        lr_x, lr_y = metadata['lower_right']
                        
                        # Create coordinate arrays for the downsampled data
                        x_coords = np.linspace(ul_x, lr_x, cols)
                        y_coords = np.linspace(ul_y, lr_y, rows)
                        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
                        
                        # Transform coordinates to geographic using pyproj
                        from pyproj import Transformer, CRS
                        
                        modis_crs = CRS.from_proj4("+proj=sinu +R=6371007.181 +x_0=0 +y_0=0 +lon_0=0")
                        wgs84_crs = CRS.from_epsg(4326)
                        
                        transformer = Transformer.from_crs(modis_crs, wgs84_crs, always_xy=True)
                        
                        lon_grid, lat_grid = transformer.transform(x_grid, y_grid)
                        
                        # Use pcolormesh for proper coordinate transformation
                        im = ax.pcolormesh(lon_grid, lat_grid, annual_sampled,
                                         transform=ccrs.PlateCarree(),
                                         cmap='YlGn',
                                         alpha=0.8,
                                         vmin=vmin, vmax=vmax,
                                         shading='auto')
                        images.append(im)
                        plotted_tiles += 1
                        continue
            
            # Fallback to extent-based plotting if pyproj fails
            im = ax.imshow(annual_sampled, 
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
    
    # Add colorbar
    if images:
        cbar = plt.colorbar(images[-1], ax=ax, shrink=0.6, pad=0.02, aspect=30)
        cbar.set_label('Annual Total GPP (g C/m²/year)', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
    
    # Add title
    title = f"GLASS GPP Annual Total - {year}\\n"
    title += f"Tiles: {len(annual_totals)} | Periods: {annual_data['total_periods']} | "
    if stats:
        title += f"Global Mean: {stats.get('mean_annual_gpp', 0):.0f} g C/m²/year"
    
    plt.title(title, fontsize=16, pad=20)
    
    # Add statistics text box
    if stats:
        stats_text = f"""Annual GPP Statistics ({year}):
Mean: {stats.get('mean_annual_gpp', 0):.0f} g C/m²/year
Median: {stats.get('median_annual_gpp', 0):.0f} g C/m²/year
Range: {stats.get('min_annual_gpp', 0):.0f} - {stats.get('max_annual_gpp', 0):.0f}
Valid pixels: {stats.get('total_valid_pixels', 0):,}
8-day periods: {annual_data['total_periods']}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the map
    output_file = output_dir / f"GPP_annual_total_{year}.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.1)
    
    plt.close()
    
    print(f"      ✓ Saved: {output_file}")
    return output_file

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Calculate annual total GPP from GLASS data')
    parser.add_argument('--data-dir', '-d', default='ancillary/glass',
                       help='Directory containing GLASS data')
    parser.add_argument('--product', '-p', default='GPP',
                       help='Product name (default: GPP)')
    parser.add_argument('--year', '-y', type=int, default=2010,
                       help='Year to process (default: 2010)')
    parser.add_argument('--output-dir', '-o', 
                       help='Output directory (default: same as data dir)')
    
    args = parser.parse_args()
    
    if not args.output_dir:
        args.output_dir = f"{args.data_dir}/annual_totals"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=== GLASS Annual GPP Calculator ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Product: {args.product}")
    print(f"Year: {args.year}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Calculate annual totals
    annual_data = calculate_annual_gpp(args.data_dir, args.product, args.year)
    
    if annual_data:
        # Create annual map
        create_annual_gpp_map(annual_data, output_dir)
        
        # Save statistics to file
        stats_file = output_dir / f"GPP_annual_stats_{args.year}.txt"
        with open(stats_file, 'w') as f:
            f.write(f"GLASS GPP Annual Statistics - {args.year}\\n")
            f.write("=" * 50 + "\\n\\n")
            
            stats = annual_data['statistics']
            if stats:
                f.write(f"Global Statistics:\\n")
                f.write(f"  Mean annual GPP: {stats.get('mean_annual_gpp', 0):.1f} g C/m²/year\\n")
                f.write(f"  Median annual GPP: {stats.get('median_annual_gpp', 0):.1f} g C/m²/year\\n")
                f.write(f"  Minimum annual GPP: {stats.get('min_annual_gpp', 0):.1f} g C/m²/year\\n")
                f.write(f"  Maximum annual GPP: {stats.get('max_annual_gpp', 0):.1f} g C/m²/year\\n")
                f.write(f"  Standard deviation: {stats.get('std_annual_gpp', 0):.1f} g C/m²/year\\n")
                f.write(f"  Total valid pixels: {stats.get('total_valid_pixels', 0):,}\\n")
            
            f.write(f"\\nData Coverage:\\n")
            f.write(f"  Total tiles processed: {len(annual_data['annual_totals'])}\\n")
            f.write(f"  Total 8-day periods: {annual_data['total_periods']}\\n")
            
            # Tile-specific statistics
            f.write(f"\\nTile Coverage Summary:\\n")
            for tile_key, periods in annual_data['valid_periods'].items():
                h_tile, v_tile = tile_key
                f.write(f"  h{h_tile:02d}v{v_tile:02d}: {periods}/{annual_data['total_periods']} periods\\n")
        
        print(f"✓ Statistics saved: {stats_file}")
        print(f"✓ Annual GPP calculation complete!")
    else:
        print("Failed to calculate annual GPP")

if __name__ == "__main__":
    main()