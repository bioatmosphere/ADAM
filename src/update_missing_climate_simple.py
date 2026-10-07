"""
Update Missing Climate Data - Simple Approach

Fills in ONLY missing climate values in aggregated dataset,
preserving all existing data.

Author: TAM Development Team
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():
    print("="*80)
    print("UPDATING MISSING CLIMATE VALUES")
    print("="*80)

    # Load aggregated dataset
    print("\nLoading aggregated dataset...")
    data_path = Path("productivity/earth/aggregated_data.csv")
    df = pd.read_csv(data_path)
    print(f"  ✓ {len(df):,} samples loaded")

    # Check initial climate completeness
    climate_vars = ['aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd']
    initial_complete = df[climate_vars].notna().all(axis=1).sum()
    print(f"  Samples with all 6 climate variables: {initial_complete:,}")

    # Load newly extracted climate data
    print("\nLoading newly extracted climate means...")
    extract_dir = Path("ancillary/terraclimate/point_extractions")

    # Start with first variable to get coordinates
    first_file = extract_dir / "aet_means.csv"
    climate_df = pd.read_csv(first_file)
    climate_df = climate_df.rename(columns={'latitude': 'lat', 'longitude': 'lon', 'mean_value': 'aet'})
    print(f"  ✓ aet: {len(climate_df):,} points")

    # Load other variables
    for var in ['pet', 'ppt', 'tmax', 'tmin', 'vpd']:
        var_file = extract_dir / f"{var}_means.csv"
        var_df = pd.read_csv(var_file)
        var_df = var_df.rename(columns={'mean_value': var})

        # Merge on point_name
        climate_df = climate_df.merge(
            var_df[['point_name', var]],
            on='point_name',
            how='left'
        )
        print(f"  ✓ {var}: {len(var_df):,} points")

    print(f"\n  Extracted climate data: {len(climate_df):,} points with {len(climate_vars)} variables")

    # Update missing values
    print("\nUpdating missing climate values...")
    updates = {var: 0 for var in climate_vars}

    for idx, row in df.iterrows():
        # Check if any climate variable is missing
        if not df.loc[idx, climate_vars].isna().any():
            continue  # Skip rows with complete climate data

        lat, lon = row['lat'], row['lon']

        # Find nearest point in climate data (within 0.01 degrees)
        dist = np.sqrt(
            (climate_df['lat'] - lat)**2 +
            (climate_df['lon'] - lon)**2
        )

        min_dist_idx = dist.idxmin()
        min_dist = dist[min_dist_idx]

        if min_dist <= 0.01:  # Match within ~1km
            # Update only missing values
            for var in climate_vars:
                if pd.isna(df.loc[idx, var]) and pd.notna(climate_df.loc[min_dist_idx, var]):
                    df.loc[idx, var] = climate_df.loc[min_dist_idx, var]
                    updates[var] += 1

    # Print update summary
    print("\n  Updates by variable:")
    for var, count in updates.items():
        print(f"    {var}: {count:,} values updated")

    # Verify improvements
    final_complete = df[climate_vars].notna().all(axis=1).sum()
    print(f"\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    print(f"\nSamples with all 6 climate variables:")
    print(f"  Before: {initial_complete:,}")
    print(f"  After:  {final_complete:,}")
    print(f"  Gain: +{final_complete - initial_complete:,}")

    # Check complete cases for modeling
    essential_features = [
        'BNPP_fraction',
        'aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd',
        'soil_carbon_stock', 'clay_content', 'silt_content', 'sand_content',
        'nitrogen_content', 'ph_in_water', 'bulk_density', 'coarse_fragments',
        'elevation', 'soil_moisture'
    ]

    existing_features = [f for f in essential_features if f in df.columns]
    complete_cases_before = 3677  # From previous analysis
    complete_cases_after = df[existing_features].notna().all(axis=1).sum()

    print(f"\nComplete cases (all {len(existing_features)} features):")
    print(f"  Before: {complete_cases_before:,}")
    print(f"  After:  {complete_cases_after:,}")
    print(f"  Gain: +{complete_cases_after - complete_cases_before:,} ({(complete_cases_after/complete_cases_before - 1)*100:.1f}% increase)")

    # Save updated dataset
    print("\n" + "="*80)
    print("SAVING")
    print("="*80)

    df.to_csv(data_path, index=False)
    print(f"\n✓ Updated: {data_path}")
    print(f"  Total samples: {len(df):,}")
    print(f"  Complete cases: {complete_cases_after:,}")

    print("\n" + "="*80)
    print("✓ UPDATE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
