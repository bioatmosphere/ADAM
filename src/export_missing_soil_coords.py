"""
Export coordinates for samples missing soil/elevation data

Author: TAM Development Team
"""

import pandas as pd
from pathlib import Path


def main():
    print("="*80)
    print("EXPORTING COORDINATES FOR MISSING SOIL/ELEVATION DATA")
    print("="*80)

    # Load merged dataset
    data_path = Path("productivity/earth/aggregated_data.csv")
    print(f"\nLoading: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  ✓ {len(df):,} samples")

    # Identify samples missing soil/elevation data
    soil_vars = ['soil_carbon_stock', 'clay_content', 'silt_content', 'sand_content',
                 'nitrogen_content', 'cation_exchange_capacity', 'ph_in_water',
                 'bulk_density', 'coarse_fragments', 'soil_moisture', 'elevation']

    missing_soil = df[soil_vars].isna().any(axis=1)
    missing_df = df[missing_soil].copy()

    n_missing = len(missing_df)
    print(f"\nSamples missing soil/elevation data: {n_missing:,}")
    print(f"  Percentage: {n_missing/len(df)*100:.1f}%")

    # Check which variables are missing
    print(f"\nMissing by variable:")
    for var in soil_vars:
        n_var_missing = missing_df[var].isna().sum()
        print(f"  {var:30s}: {n_var_missing:,} ({n_var_missing/n_missing*100:.1f}%)")

    # Prepare coordinates for export
    coords_df = missing_df[['lat', 'lon']].copy()
    coords_df['name'] = [f"sample_{i}" for i in range(len(coords_df))]

    # Verify coordinates are valid
    valid_coords = (
        (coords_df['lat'] >= -90) & (coords_df['lat'] <= 90) &
        (coords_df['lon'] >= -180) & (coords_df['lon'] <= 180)
    )

    n_invalid = (~valid_coords).sum()
    if n_invalid > 0:
        print(f"\n⚠ Warning: {n_invalid} samples have invalid coordinates!")
        coords_df = coords_df[valid_coords].copy()

    print(f"\nValid coordinates to export: {len(coords_df):,}")

    # Export to CSV (format compatible with soilgrids extraction)
    output_path = Path("productivity/earth/missing_soil_coords.csv")
    coords_df.to_csv(output_path, index=False)
    print(f"\n✓ Coordinates exported to: {output_path}")

    # Print geographic range
    print(f"\nGeographic range:")
    print(f"  Latitude:  {coords_df['lat'].min():.2f} to {coords_df['lat'].max():.2f}")
    print(f"  Longitude: {coords_df['lon'].min():.2f} to {coords_df['lon'].max():.2f}")

    # Print first 10
    print(f"\nFirst 10 coordinates:")
    print(coords_df.head(10).to_string(index=False))

    print("\n" + "="*80)
    print("✓ EXPORT COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Extract soil data from SoilGrids API")
    print("  2. Extract elevation data")
    print("  3. Merge back into aggregated dataset")

    return coords_df


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
