"""
Merge elevation and soil moisture data into aggregated dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():
    print("="*80)
    print("MERGING ELEVATION AND SOIL MOISTURE DATA")
    print("="*80)

    # Load aggregated dataset
    print("\nLoading aggregated dataset...")
    data_path = Path("productivity/earth/aggregated_data.csv")
    df = pd.read_csv(data_path)
    print(f"  ✓ {len(df):,} samples loaded")

    # Check initial completeness
    initial_elevation = df['elevation'].notna().sum()
    initial_sm = df['soil_moisture'].notna().sum()
    print(f"  Initial elevation coverage: {initial_elevation:,}/{len(df):,} ({initial_elevation/len(df)*100:.1f}%)")
    print(f"  Initial soil moisture coverage: {initial_sm:,}/{len(df):,} ({initial_sm/len(df)*100:.1f}%)")

    # Load extracted elevation data
    print("\nLoading elevation data...")
    elev_path = Path("ancillary/elevation/extracted_elevation.csv")
    elev_df = pd.read_csv(elev_path)
    print(f"  ✓ {len(elev_df):,} elevations extracted")
    print(f"  Valid elevations: {elev_df['elevation'].notna().sum():,}")

    # Load extracted soil moisture (use EC file, or merge both)
    print("\nLoading soil moisture data...")
    sm_ec_path = Path("ancillary/soilmoisture/extracted_sm_ec.csv")
    sm_olc_path = Path("ancillary/soilmoisture/extracted_sm_olc.csv")

    sm_ec_df = pd.read_csv(sm_ec_path)
    sm_olc_df = pd.read_csv(sm_olc_path)

    print(f"  ✓ EC soil moisture: {sm_ec_df['soil_moisture'].notna().sum():,}/{len(sm_ec_df):,} valid")
    print(f"  ✓ OLC soil moisture: {sm_olc_df['soil_moisture'].notna().sum():,}/{len(sm_olc_df):,} valid")

    # Merge EC and OLC (use average where both available, fallback to either)
    sm_df = sm_ec_df[['lat', 'lon']].copy()
    sm_df['soil_moisture'] = sm_ec_df['soil_moisture'].fillna(sm_olc_df['soil_moisture'])
    # Where both have data, average them
    both_valid = sm_ec_df['soil_moisture'].notna() & sm_olc_df['soil_moisture'].notna()
    sm_df.loc[both_valid, 'soil_moisture'] = (
        sm_ec_df.loc[both_valid, 'soil_moisture'] +
        sm_olc_df.loc[both_valid, 'soil_moisture']
    ) / 2

    print(f"  Combined soil moisture: {sm_df['soil_moisture'].notna().sum():,} valid")

    # Update elevation
    print("\nUpdating elevation values...")
    elev_updates = 0
    for idx, row in df.iterrows():
        if not pd.isna(df.loc[idx, 'elevation']):
            continue  # Skip if elevation already exists

        lat, lon = row['lat'], row['lon']
        dist = np.sqrt((elev_df['lat'] - lat)**2 + (elev_df['lon'] - lon)**2)
        min_dist_idx = dist.idxmin()

        if dist[min_dist_idx] <= 0.01 and pd.notna(elev_df.loc[min_dist_idx, 'elevation']):
            df.loc[idx, 'elevation'] = elev_df.loc[min_dist_idx, 'elevation']
            elev_updates += 1

    print(f"  ✓ Updated {elev_updates:,} elevation values")

    # Update soil moisture
    print("\nUpdating soil moisture values...")
    sm_updates = 0
    for idx, row in df.iterrows():
        if not pd.isna(df.loc[idx, 'soil_moisture']):
            continue  # Skip if soil moisture already exists

        lat, lon = row['lat'], row['lon']
        dist = np.sqrt((sm_df['lat'] - lat)**2 + (sm_df['lon'] - lon)**2)
        min_dist_idx = dist.idxmin()

        if dist[min_dist_idx] <= 0.01 and pd.notna(sm_df.loc[min_dist_idx, 'soil_moisture']):
            df.loc[idx, 'soil_moisture'] = sm_df.loc[min_dist_idx, 'soil_moisture']
            sm_updates += 1

    print(f"  ✓ Updated {sm_updates:,} soil moisture values")

    # Verify improvements
    final_elevation = df['elevation'].notna().sum()
    final_sm = df['soil_moisture'].notna().sum()

    print(f"\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    print(f"\nElevation coverage:")
    print(f"  Before: {initial_elevation:,} ({initial_elevation/len(df)*100:.1f}%)")
    print(f"  After:  {final_elevation:,} ({final_elevation/len(df)*100:.1f}%)")
    print(f"  Gain: +{final_elevation - initial_elevation:,}")

    print(f"\nSoil moisture coverage:")
    print(f"  Before: {initial_sm:,} ({initial_sm/len(df)*100:.1f}%)")
    print(f"  After:  {final_sm:,} ({final_sm/len(df)*100:.1f}%)")
    print(f"  Gain: +{final_sm - initial_sm:,}")

    # Check complete cases
    essential_features = [
        'BNPP_fraction',
        'aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd',
        'soil_carbon_stock', 'clay_content', 'silt_content', 'sand_content',
        'nitrogen_content', 'ph_in_water', 'bulk_density', 'coarse_fragments',
        'elevation', 'soil_moisture'
    ]

    existing_features = [f for f in essential_features if f in df.columns]
    complete_cases = df[existing_features].notna().all(axis=1).sum()

    print(f"\nComplete cases (all {len(existing_features)} features):")
    print(f"  Before: 4,457")
    print(f"  After:  {complete_cases:,}")
    print(f"  Gain: +{complete_cases - 4457:,}")
    print(f"  Progress to target (~5,110): {complete_cases/5110*100:.1f}%")

    # Save updated dataset
    print("\n" + "="*80)
    print("SAVING")
    print("="*80)

    df.to_csv(data_path, index=False)
    print(f"\n✓ Updated: {data_path}")
    print(f"  Total samples: {len(df):,}")
    print(f"  Complete cases: {complete_cases:,}")

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
