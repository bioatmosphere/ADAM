"""
Export coordinates for samples missing TerraClimate data

This script identifies samples in the merged dataset that are missing
climate variables and exports their coordinates for TerraClimate extraction.

Author: TAM Development Team
"""

import pandas as pd
from pathlib import Path


def export_missing_climate_coordinates():
    """Export coordinates of samples missing climate data."""
    print("="*80)
    print("EXPORTING COORDINATES FOR MISSING CLIMATE DATA")
    print("="*80)

    # Load merged dataset
    data_path = Path("productivity/earth/aggregated_data.csv")
    print(f"\nLoading merged dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  ✓ Loaded {len(df)} samples")

    # Identify samples missing climate data
    climate_vars = ['aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd']
    missing_climate = df[climate_vars].isna().any(axis=1)

    missing_df = df[missing_climate].copy()
    n_missing = len(missing_df)

    print(f"\nSamples missing climate data: {n_missing}")
    print(f"  Percentage: {n_missing/len(df)*100:.1f}%")

    # Check which climate variables are missing
    print(f"\nMissing climate variables breakdown:")
    for var in climate_vars:
        n_var_missing = missing_df[var].isna().sum()
        print(f"  {var}: {n_var_missing} ({n_var_missing/n_missing*100:.1f}%)")

    # Prepare coordinates for export
    coords_df = missing_df[['lat', 'lon']].copy()

    # Add optional name column (use index as name)
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

    print(f"\nValid coordinates to export: {len(coords_df)}")

    # Export to CSV
    output_path = Path("productivity/earth/missing_climate_coords.csv")
    coords_df.to_csv(output_path, index=False)
    print(f"\n✓ Coordinates exported to: {output_path}")

    # Print geographic distribution
    print(f"\nGeographic range:")
    print(f"  Latitude:  {coords_df['lat'].min():.2f} to {coords_df['lat'].max():.2f}")
    print(f"  Longitude: {coords_df['lon'].min():.2f} to {coords_df['lon'].max():.2f}")

    # Print sample coordinates
    print(f"\nFirst 10 coordinates:")
    print(coords_df.head(10).to_string(index=False))

    return coords_df


def main():
    """Main execution."""
    try:
        coords_df = export_missing_climate_coordinates()

        print("\n" + "="*80)
        print("✓ EXPORT COMPLETE")
        print("="*80)
        print("\nNext step:")
        print("  Run TerraClimate extraction:")
        print("    python src/ancillary/terraclimate.py --mode extract \\")
        print("      --coords-file productivity/earth/missing_climate_coords.csv \\")
        print("      --variables aet pet ppt tmax tmin vpd \\")
        print("      --start-year 2001 --end-year 2020")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
