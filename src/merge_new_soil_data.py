"""
Merge newly extracted soil data into aggregated dataset.

Updates only missing soil values in the aggregated dataset,
preserving all existing data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():
    print("="*80)
    print("MERGING NEW SOIL DATA INTO AGGREGATED DATASET")
    print("="*80)

    # Load aggregated dataset
    print("\nLoading aggregated dataset...")
    data_path = Path("productivity/earth/aggregated_data.csv")
    df = pd.read_csv(data_path)
    print(f"  ✓ {len(df):,} samples loaded")

    # Check initial soil completeness
    soil_vars = ['soil_carbon_stock', 'clay_content', 'silt_content', 'sand_content',
                 'nitrogen_content', 'cation_exchange_capacity', 'ph_in_water',
                 'bulk_density', 'coarse_fragments']

    initial_complete = df[soil_vars].notna().all(axis=1).sum()
    print(f"  Samples with all 9 soil variables: {initial_complete:,}")

    # Load newly extracted soil data
    print("\nLoading newly extracted soil data...")
    soil_data_path = Path("ancillary/soilgrids/extracted_soil_data.csv")
    soil_df = pd.read_csv(soil_data_path)
    print(f"  ✓ {len(soil_df):,} points extracted")

    # Show extraction success rate
    for var in soil_vars:
        n_valid = soil_df[var].notna().sum()
        print(f"    {var:30s}: {n_valid:,}/{len(soil_df):,} ({n_valid/len(soil_df)*100:.1f}%)")

    # Update missing values
    print("\nUpdating missing soil values...")
    updates = {var: 0 for var in soil_vars}

    for idx, row in df.iterrows():
        # Check if any soil variable is missing
        if not df.loc[idx, soil_vars].isna().any():
            continue  # Skip rows with complete soil data

        lat, lon = row['lat'], row['lon']

        # Find nearest point in extracted data (within 0.01 degrees)
        dist = np.sqrt(
            (soil_df['lat'] - lat)**2 +
            (soil_df['lon'] - lon)**2
        )

        min_dist_idx = dist.idxmin()
        min_dist = dist[min_dist_idx]

        if min_dist <= 0.01:  # Match within ~1km
            # Update only missing values
            for var in soil_vars:
                if pd.isna(df.loc[idx, var]) and pd.notna(soil_df.loc[min_dist_idx, var]):
                    df.loc[idx, var] = soil_df.loc[min_dist_idx, var]
                    updates[var] += 1

    # Print update summary
    print("\n  Updates by variable:")
    for var, count in updates.items():
        print(f"    {var:30s}: {count:,} values updated")

    # Verify improvements
    final_complete = df[soil_vars].notna().all(axis=1).sum()
    print(f"\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    print(f"\nSamples with all 9 soil variables:")
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
    complete_cases_after = df[existing_features].notna().all(axis=1).sum()

    print(f"\nComplete cases (all {len(existing_features)} features):")
    print(f"  Before: 3,677")
    print(f"  After:  {complete_cases_after:,}")
    print(f"  Gain: +{complete_cases_after - 3677:,}")

    # Save updated dataset
    print("\n" + "="*80)
    print("SAVING")
    print("="*80)

    df.to_csv(data_path, index=False)
    print(f"\n✓ Updated: {data_path}")
    print(f"  Total samples: {len(df):,}")
    print(f"  Complete cases: {complete_cases_after:,}")

    print("\n" + "="*80)
    print("✓ MERGE COMPLETE")
    print("="*80)

    return df


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
