"""
Extract soil data from existing GeoTIFF rasters for specific coordinates.

This script extracts soil values from global rasters at specific coordinates,
bypassing the SoilGrids REST API.
"""

import pandas as pd
import rasterio
from rasterio.warp import transform
from pyproj import Transformer
from pathlib import Path
import numpy as np
from osgeo import gdal

# Enable GDAL network access for VRT files
gdal.SetConfigOption('GDAL_HTTP_UNSAFESSL', 'YES')
gdal.SetConfigOption('GDAL_HTTP_TIMEOUT', '300')
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif,.vrt')
gdal.SetConfigOption('VSI_CACHE', 'YES')
gdal.SetConfigOption('VSI_CACHE_SIZE', '100000000')


def extract_values_from_raster(raster_path, coords_df):
    """
    Extract raster values at specific coordinates with CRS transformation.

    Args:
        raster_path: Path to GeoTIFF raster file
        coords_df: DataFrame with 'lat' and 'lon' columns (WGS84)

    Returns:
        list: Extracted values for each coordinate
    """
    values = []

    with rasterio.open(raster_path) as src:
        # Check if raster CRS matches WGS84
        if src.crs and src.crs != 'EPSG:4326':
            # Transform coordinates from WGS84 to raster CRS
            lons = coords_df['lon'].values
            lats = coords_df['lat'].values

            # Transform using rasterio's transform function
            xs, ys = transform('EPSG:4326', src.crs, lons, lats)
            coords = list(zip(xs, ys))
        else:
            # No transformation needed
            coords = [(row['lon'], row['lat']) for _, row in coords_df.iterrows()]

        # Sample all coordinates at once
        try:
            samples = list(src.sample(coords))

            for sample in samples:
                value = sample[0]  # Get first band value

                # Handle nodata values
                if value == src.nodata or (isinstance(value, (int, float)) and (np.isnan(value) or value == -32768 or value < -9999)):
                    values.append(None)
                else:
                    values.append(float(value))

        except Exception as e:
            print(f"  Error during sampling: {e}")
            import traceback
            traceback.print_exc()
            # Return None for all on error
            values = [None] * len(coords_df)

    return values


def main():
    print("="*80)
    print("EXTRACTING SOIL DATA FROM RASTERS")
    print("="*80)

    # Load coordinates
    coords_file = Path("productivity/earth/missing_soil_coords.csv")
    print(f"\nLoading coordinates: {coords_file}")
    coords_df = pd.read_csv(coords_file)
    print(f"  ✓ {len(coords_df):,} coordinates loaded")

    # Define VRT files to extract from (access remotely via HTTP)
    base_url = "/vsicurl/https://files.isric.org/soilgrids/latest/data"
    rasters = {
        'soil_carbon_stock': f'{base_url}/soc/soc_0-5cm_mean.vrt',
        'clay_content': f'{base_url}/clay/clay_0-5cm_mean.vrt',
        'silt_content': f'{base_url}/silt/silt_0-5cm_mean.vrt',
        'sand_content': f'{base_url}/sand/sand_0-5cm_mean.vrt',
        'nitrogen_content': f'{base_url}/nitrogen/nitrogen_0-5cm_mean.vrt',
        'cation_exchange_capacity': f'{base_url}/cec/cec_0-5cm_mean.vrt',
        'ph_in_water': f'{base_url}/phh2o/phh2o_0-5cm_mean.vrt',
        'bulk_density': f'{base_url}/bdod/bdod_0-5cm_mean.vrt',
        'coarse_fragments': f'{base_url}/cfvo/cfvo_0-5cm_mean.vrt',
    }

    # Extract from each raster
    results_df = coords_df[['lat', 'lon']].copy()

    for prop_name, raster_path in rasters.items():
        # Don't check .exists() for /vsicurl/ paths
        if isinstance(raster_path, Path) and not str(raster_path).startswith('/vsicurl/'):
            if not raster_path.exists():
                print(f"\n✗ Skipping {prop_name}: {raster_path} not found")
                continue

        raster_name = raster_path.split('/')[-1] if isinstance(raster_path, str) else raster_path.name
        print(f"\n  Extracting {prop_name} from {raster_name}...")
        values = extract_values_from_raster(raster_path, coords_df)
        results_df[prop_name] = values

        n_valid = sum(1 for v in values if v is not None)
        print(f"    ✓ Extracted {n_valid:,}/{len(values):,} valid values ({n_valid/len(values)*100:.1f}%)")

    # Save results
    output_file = Path("ancillary/soilgrids/extracted_soil_data.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")
    print(f"  Total records: {len(results_df):,}")
    print(f"  Columns: {', '.join(results_df.columns)}")

    # Summary statistics
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    for col in results_df.columns:
        if col not in ['lat', 'lon']:
            valid = results_df[col].notna().sum()
            print(f"  {col:30s}: {valid:,}/{len(results_df):,} ({valid/len(results_df)*100:.1f}%)")

    return results_df


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
