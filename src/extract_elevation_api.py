"""
Extract elevation data using Open-Elevation API.
"""

import pandas as pd
import requests
import time
from pathlib import Path
import json


def extract_elevation_batch(coords_list, batch_size=100):
    """
    Extract elevation for coordinates using Open-Elevation API.

    API allows up to 100 locations per request.
    """

    all_elevations = []
    base_url = "https://api.open-elevation.com/api/v1/lookup"

    n_batches = (len(coords_list) + batch_size - 1) // batch_size

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(coords_list))
        batch = coords_list[start_idx:end_idx]

        # Format locations for API
        locations = "|".join([f"{lat},{lon}" for lat, lon in batch])

        try:
            response = requests.get(f"{base_url}?locations={locations}", timeout=60)

            if response.status_code == 200:
                data = response.json()
                elevations = [result['elevation'] for result in data['results']]
                all_elevations.extend(elevations)
                print(f"  ✓ Batch {i+1}/{n_batches}: {len(elevations)} elevations")
            else:
                print(f"  ✗ Batch {i+1}/{n_batches}: HTTP {response.status_code}")
                # Fill with None for failed batch
                all_elevations.extend([None] * len(batch))

            # Rate limiting: wait 1 second between requests
            if i < n_batches - 1:
                time.sleep(1)

        except Exception as e:
            print(f"  ✗ Batch {i+1}/{n_batches}: {e}")
            all_elevations.extend([None] * len(batch))

    return all_elevations


def main():
    print("="*80)
    print("EXTRACTING ELEVATION DATA VIA OPEN-ELEVATION API")
    print("="*80)

    # Load coordinates
    coords_file = Path("productivity/earth/missing_elev_sm_coords.csv")
    print(f"\nLoading coordinates: {coords_file}")
    coords_df = pd.read_csv(coords_file)
    print(f"  ✓ {len(coords_df):,} coordinates loaded")

    # Prepare coordinate list
    coords_list = list(zip(coords_df['lat'], coords_df['lon']))

    # Extract elevation in batches
    print(f"\nExtracting elevation data in batches of 100...")
    print(f"  Total batches: {(len(coords_list) + 99) // 100}")
    print(f"  Estimated time: ~{(len(coords_list) + 99) // 100} minutes")

    elevations = extract_elevation_batch(coords_list, batch_size=100)

    # Create results dataframe
    results_df = coords_df.copy()
    results_df['elevation'] = elevations

    # Save results
    output_file = Path("ancillary/elevation/extracted_elevation.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)

    n_valid = results_df['elevation'].notna().sum()
    print(f"\n✓ Saved to: {output_file}")
    print(f"  Valid elevations: {n_valid:,}/{len(results_df):,} ({n_valid/len(results_df)*100:.1f}%)")

    # Statistics
    if n_valid > 0:
        elev_data = results_df['elevation'].dropna()
        print(f"\nElevation statistics:")
        print(f"  Min:    {elev_data.min():.1f}m")
        print(f"  Max:    {elev_data.max():.1f}m")
        print(f"  Mean:   {elev_data.mean():.1f}m")
        print(f"  Median: {elev_data.median():.1f}m")

    print("\n" + "="*80)
    print("✓ ELEVATION EXTRACTION COMPLETE")
    print("="*80)

    return results_df


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
