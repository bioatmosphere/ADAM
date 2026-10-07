"""
Expanded ForC Data Processing - Extract Maximum BNPP/ANPP/TNPP Data

This expands beyond the current 529 samples by:
1. Including fine + coarse root components
2. Using all ANPP levels (0, 1, 2, foliage, woody)
3. Calculating TNPP from ANPP + BNPP where available
4. Including more forest types and sites

Current: 529 samples
Target: 1,000-1,500 samples

Author: TAM Development Team
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_forc_measurements():
    """Load the main ForC measurements table."""
    print("Loading ForC measurements...")
    forc_path = Path("../../productivity/forc/data/ForC_measurements.csv")

    if not forc_path.exists():
        raise FileNotFoundError(f"ForC measurements not found at {forc_path}")

    df = pd.read_csv(forc_path)
    print(f"  Loaded {len(df)} total measurements")
    print(f"  Unique sites: {df['sites.sitename'].nunique()}")
    print(f"  Unique variables: {df['variable.name'].nunique()}")

    return df


def load_forc_sites():
    """Load ForC sites table with coordinates and metadata."""
    print("\nLoading ForC sites...")
    sites_path = Path("../../productivity/forc/data/ForC_sites.csv")

    if not sites_path.exists():
        raise FileNotFoundError(f"ForC sites not found at {sites_path}")

    sites = pd.read_csv(sites_path, encoding='latin-1')  # Handle special characters
    print(f"  Loaded {len(sites)} sites")

    # Filter to sites with valid coordinates
    sites_valid = sites[
        (sites['lat'].notna()) &
        (sites['lon'].notna()) &
        (sites['lat'] >= -90) & (sites['lat'] <= 90) &
        (sites['lon'] >= -180) & (sites['lon'] <= 180)
    ].copy()

    print(f"  Sites with valid coordinates: {len(sites_valid)}")

    return sites_valid


def extract_bnpp_measurements(df):
    """
    Extract all BNPP-related measurements.

    Includes:
    - BNPP_root_C (total root production)
    - BNPP_root_fine_C (fine root production)
    - BNPP_root_coarse_C (coarse root production)
    """
    print("\nExtracting BNPP measurements...")

    bnpp_variables = [
        'BNPP_root_C',
        'BNPP_root_fine_C',
        'BNPP_root_coarse_C'
    ]

    bnpp_df = df[df['variable.name'].isin(bnpp_variables)].copy()
    print(f"  Found {len(bnpp_df)} BNPP measurements")

    for var in bnpp_variables:
        count = (bnpp_df['variable.name'] == var).sum()
        print(f"    {var}: {count}")

    # Rename for consistency
    bnpp_df = bnpp_df.rename(columns={'mean': 'BNPP'})

    return bnpp_df


def extract_anpp_measurements(df):
    """
    Extract all ANPP-related measurements.

    Includes:
    - ANPP_0_C (foliage only)
    - ANPP_1_C (foliage + woody)
    - ANPP_2_C (comprehensive ANPP)
    - ANPP_foliage_C
    - ANPP_woody_C
    """
    print("\nExtracting ANPP measurements...")

    anpp_variables = [
        'ANPP_0_C',
        'ANPP_1_C',
        'ANPP_2_C',
        'ANPP_foliage_C',
        'ANPP_woody_C',
        'ANPP_woody_stem_C',
        'ANPP_woody_branch_C'
    ]

    anpp_df = df[df['variable.name'].isin(anpp_variables)].copy()
    print(f"  Found {len(anpp_df)} ANPP measurements")

    for var in anpp_variables:
        count = (anpp_df['variable.name'] == var).sum()
        if count > 0:
            print(f"    {var}: {count}")

    # Rename for consistency
    anpp_df = anpp_df.rename(columns={'mean': 'ANPP'})

    return anpp_df


def extract_tnpp_measurements(df):
    """
    Extract TNPP measurements and components.
    """
    print("\nExtracting TNPP measurements...")

    tnpp_variables = [
        'NPP_1_C',  # Most common comprehensive NPP
        'NPP_2_C',
        'NPP_3_C'
    ]

    tnpp_df = df[df['variable.name'].isin(tnpp_variables)].copy()
    print(f"  Found {len(tnpp_df)} TNPP measurements")

    for var in tnpp_variables:
        count = (tnpp_df['variable.name'] == var).sum()
        if count > 0:
            print(f"    {var}: {count}")

    # Rename for consistency
    tnpp_df = tnpp_df.rename(columns={'mean': 'TNPP'})

    return tnpp_df


def merge_productivity_data(bnpp_df, anpp_df, tnpp_df, sites_df):
    """
    Merge BNPP, ANPP, TNPP measurements by site and year.
    Calculate BNPP_fraction where possible.
    """
    print("\n" + "="*60)
    print("Merging productivity measurements...")
    print("="*60)

    # Key columns for merging
    merge_keys = ['sites.sitename', 'start.date', 'end.date']

    # Start with BNPP
    merged = bnpp_df[merge_keys + ['BNPP']].copy()
    merged['has_bnpp'] = True

    # Add ANPP
    anpp_subset = anpp_df[merge_keys + ['ANPP']].copy()
    merged = merged.merge(anpp_subset, on=merge_keys, how='outer', suffixes=('', '_anpp'))
    merged['has_anpp'] = merged['ANPP'].notna()

    # Add TNPP
    tnpp_subset = tnpp_df[merge_keys + ['TNPP']].copy()
    merged = merged.merge(tnpp_subset, on=merge_keys, how='outer', suffixes=('', '_tnpp'))
    merged['has_tnpp'] = merged['TNPP'].notna()

    print(f"  Total merged records: {len(merged)}")
    print(f"  Records with BNPP: {merged['has_bnpp'].sum()}")
    print(f"  Records with ANPP: {merged['has_anpp'].sum()}")
    print(f"  Records with TNPP: {merged['has_tnpp'].sum()}")

    # Calculate TNPP from components where missing
    has_both = merged['BNPP'].notna() & merged['ANPP'].notna()
    missing_tnpp = merged['TNPP'].isna()
    can_calculate = has_both & missing_tnpp

    print(f"\n  Calculating TNPP from ANPP + BNPP: {can_calculate.sum()} records")
    merged.loc[can_calculate, 'TNPP'] = merged.loc[can_calculate, 'BNPP'] + merged.loc[can_calculate, 'ANPP']
    merged.loc[can_calculate, 'TNPP_calculated'] = True

    # Calculate BNPP_fraction
    valid_for_fraction = merged['BNPP'].notna() & merged['TNPP'].notna() & (merged['TNPP'] > 0)
    merged.loc[valid_for_fraction, 'BNPP_fraction'] = merged.loc[valid_for_fraction, 'BNPP'] / merged.loc[valid_for_fraction, 'TNPP']

    print(f"  Records with BNPP_fraction: {merged['BNPP_fraction'].notna().sum()}")

    # Merge with site data
    site_columns = ['sites.sitename', 'lat', 'lon', 'masl']
    # Add optional columns if they exist
    optional_cols = ['biogeog', 'FAO.ecozone', 'Koeppen', 'continent', 'country']
    for col in optional_cols:
        if col in sites_df.columns:
            site_columns.append(col)

    merged = merged.merge(
        sites_df[site_columns],
        left_on='sites.sitename',
        right_on='sites.sitename',
        how='left'
    )

    # Add metadata
    merged['Data_Source'] = 'ForC_expanded'
    merged['Units'] = 'Mg C ha⁻¹ yr⁻¹'

    return merged


def apply_quality_filters(df):
    """
    Apply quality filters to remove unrealistic values.
    """
    print("\n" + "="*60)
    print("Applying quality filters...")
    print("="*60)

    initial_count = len(df)

    # Filter 1: Valid coordinates
    df = df[(df['lat'].notna()) & (df['lon'].notna())].copy()
    print(f"  After coordinate filter: {len(df)} ({len(df)/initial_count*100:.1f}%)")

    # Filter 2: Valid BNPP values (remove negatives and extreme outliers)
    valid_bnpp = df['BNPP'].notna()
    df = df[~valid_bnpp | ((df['BNPP'] > 0) & (df['BNPP'] < 50))].copy()  # < 50 Mg C/ha/yr
    print(f"  After BNPP range filter: {len(df)} ({len(df)/initial_count*100:.1f}%)")

    # Filter 3: Valid BNPP_fraction (0.05 to 0.95)
    has_fraction = df['BNPP_fraction'].notna()
    valid_fraction = (df['BNPP_fraction'] >= 0.05) & (df['BNPP_fraction'] <= 0.95)
    df = df[~has_fraction | valid_fraction].copy()
    print(f"  After BNPP_fraction filter: {len(df)} ({len(df)/initial_count*100:.1f}%)")

    # Filter 4: Remove duplicates (same site, date, values)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['sites.sitename', 'start.date', 'BNPP', 'ANPP'], keep='first')
    print(f"  After deduplication: {len(df)} (removed {before_dedup - len(df)} duplicates)")

    return df


def save_expanded_data(df, output_path="../../productivity/forc/ForC_BNPP_ANPP_TNPP_expanded.csv"):
    """Save the expanded dataset."""
    print("\n" + "="*60)
    print("Saving expanded ForC dataset...")
    print("="*60)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}")

    print(f"\nDataset summary:")
    print(f"  Total records: {len(df)}")
    print(f"  Records with BNPP: {df['BNPP'].notna().sum()}")
    print(f"  Records with ANPP: {df['ANPP'].notna().sum()}")
    print(f"  Records with TNPP: {df['TNPP'].notna().sum()}")
    print(f"  Records with BNPP_fraction: {df['BNPP_fraction'].notna().sum()}")

    print(f"\nBNPP_fraction statistics:")
    if df['BNPP_fraction'].notna().any():
        print(f"  Mean: {df['BNPP_fraction'].mean():.4f}")
        print(f"  Median: {df['BNPP_fraction'].median():.4f}")
        print(f"  Min: {df['BNPP_fraction'].min():.4f}")
        print(f"  Max: {df['BNPP_fraction'].max():.4f}")

    print(f"\nEcozone distribution:")
    if 'FAO.ecozone' in df.columns:
        print(df['FAO.ecozone'].value_counts().head(10))
    elif 'biogeog' in df.columns:
        print(df['biogeog'].value_counts().head(10))


def compare_with_original():
    """Compare expanded dataset with original."""
    print("\n" + "="*60)
    print("COMPARISON: Expanded vs Original")
    print("="*60)

    try:
        original = pd.read_csv("../../productivity/forc/ForC_BNPP_ANPP_TNPP_processed.csv")
        expanded = pd.read_csv("../../productivity/forc/ForC_BNPP_ANPP_TNPP_expanded.csv")

        print(f"Original dataset:")
        print(f"  Total records: {len(original)}")
        print(f"  With BNPP_fraction: {original['BNPP_fraction'].notna().sum()}")

        print(f"\nExpanded dataset:")
        print(f"  Total records: {len(expanded)}")
        print(f"  With BNPP_fraction: {expanded['BNPP_fraction'].notna().sum()}")

        print(f"\nGain:")
        print(f"  Additional records: +{len(expanded) - len(original)}")
        print(f"  Additional with BNPP_fraction: +{expanded['BNPP_fraction'].notna().sum() - original['BNPP_fraction'].notna().sum()}")

        gain_pct = ((len(expanded) - len(original)) / len(original)) * 100
        print(f"  Percentage increase: {gain_pct:.1f}%")

    except FileNotFoundError as e:
        print(f"Could not compare: {e}")


def main():
    """Main execution function."""
    print("="*60)
    print("EXPANDED ForC DATA PROCESSING")
    print("Extracting Maximum BNPP/ANPP/TNPP Data")
    print("="*60)

    try:
        # 1. Load data
        measurements = load_forc_measurements()
        sites = load_forc_sites()

        # 2. Extract productivity components
        bnpp_df = extract_bnpp_measurements(measurements)
        anpp_df = extract_anpp_measurements(measurements)
        tnpp_df = extract_tnpp_measurements(measurements)

        # 3. Merge
        merged = merge_productivity_data(bnpp_df, anpp_df, tnpp_df, sites)

        # 4. Apply filters
        cleaned = apply_quality_filters(merged)

        # 5. Save
        save_expanded_data(cleaned)

        # 6. Compare
        compare_with_original()

        print("\n" + "="*60)
        print("✓ EXPANDED FORC PROCESSING COMPLETE!")
        print("="*60)
        print("\nNext step: Merge with grassland data using data_aggregation.py")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
