"""Download SoilGrids VRT files for all properties."""

import requests
from pathlib import Path

# SoilGrids property codes (file structure names)
PROPERTIES = {
    'soc': 'soil_carbon_stock',      # Soil Organic Carbon
    'clay': 'clay_content',
    'silt': 'silt_content',
    'sand': 'sand_content',
    'nitrogen': 'nitrogen_content',
    'cec': 'cation_exchange_capacity',
    'phh2o': 'ph_in_water',
    'bdod': 'bulk_density',
    'cfvo': 'coarse_fragments'
}

BASE_URL = "https://files.isric.org/soilgrids/latest/data"
DEPTH = "0-5cm"  # Surface layer (most representative)
OUTPUT_DIR = Path("ancillary/soilgrids/vrts")

def download_vrt(prop_code, prop_name):
    """Download VRT file for a property."""
    url = f"{BASE_URL}/{prop_code}/{prop_code}_{DEPTH}_mean.vrt"
    output_path = OUTPUT_DIR / f"{prop_code}_{DEPTH}_mean.vrt"

    print(f"Downloading {prop_name} ({prop_code})...")
    print(f"  URL: {url}")

    try:
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            output_path.write_bytes(response.content)
            size_mb = len(response.content) / 1024 / 1024
            print(f"  ✓ Downloaded: {output_path} ({size_mb:.2f} MB)")
            return True
        else:
            print(f"  ✗ HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    print("="*80)
    print("DOWNLOADING SOILGRIDS VRT FILES")
    print("="*80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download each property
    success_count = 0
    for prop_code, prop_name in PROPERTIES.items():
        if download_vrt(prop_code, prop_name):
            success_count += 1
        print()

    print("="*80)
    print(f"✓ Successfully downloaded {success_count}/{len(PROPERTIES)} VRT files")
    print("="*80)

    # List downloaded files
    vrt_files = list(OUTPUT_DIR.glob("*.vrt"))
    print(f"\nDownloaded files in {OUTPUT_DIR}:")
    for vrt_file in sorted(vrt_files):
        size_mb = vrt_file.stat().st_size / 1024 / 1024
        print(f"  {vrt_file.name:40s} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
