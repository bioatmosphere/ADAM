#!/usr/bin/env python3
"""
Export global soil data at 0.5° resolution using Google Earth Engine.

Uses OpenLandMap soil datasets which are derived from SoilGrids and available in GEE.

Prerequisites:
1. Install earthengine-api: pip install earthengine-api
2. Authenticate: earthengine authenticate
3. Initialize: ee.Initialize()

Author: TAM Development Team
"""

import ee
from pathlib import Path
import time


def initialize_gee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize()
        print("Google Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"Failed to initialize GEE: {e}")
        print("\nTo authenticate, run:")
        print("  earthengine authenticate")
        return False


# OpenLandMap soil asset mapping
# Bands: b0=0cm, b10=10cm, b30=30cm, b60=60cm, b100=100cm, b200=200cm
OPENLANDMAP_ASSETS = {
    'clay_content': {
        'asset': 'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02',
        'band': 'b0',  # 0cm depth
        'scale': 1.0,  # Already in %
    },
    'sand_content': {
        'asset': 'OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02',
        'band': 'b0',
        'scale': 1.0,
    },
    'soil_carbon_stock': {
        'asset': 'OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02',
        'band': 'b0',
        'scale': 0.1,  # g/kg to %
    },
    'ph_in_water': {
        'asset': 'OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02',
        'band': 'b0',
        'scale': 0.1,  # pH*10 to pH
    },
    'bulk_density': {
        'asset': 'OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02',
        'band': 'b0',
        'scale': 0.01,  # kg/m3 / 100 to g/cm3
    },
    'water_content': {
        'asset': 'OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01',
        'band': 'b0',
        'scale': 1.0,
    },
}


def export_soil_property(prop_name: str, config: dict,
                         output_dir: str = "soil_0p5deg",
                         scale: float = 55660) -> ee.batch.Task:
    """
    Export a soil property at coarse resolution.

    Args:
        prop_name: Property name for output file
        config: Asset configuration dict
        output_dir: Google Drive folder for export
        scale: Output resolution in meters (55660m ≈ 0.5°)

    Returns:
        Export task
    """
    asset_id = config['asset']
    band = config['band']

    print(f"  Loading {asset_id} band {band}...")

    try:
        image = ee.Image(asset_id).select(band)

        # Apply scale factor if needed
        if config['scale'] != 1.0:
            image = image.multiply(config['scale'])

    except Exception as e:
        print(f"    Error loading asset: {e}")
        return None

    # Define global bounds
    global_bounds = ee.Geometry.Rectangle([-180, -60, 180, 85], 'EPSG:4326', False)

    # Create export task
    task_name = f"{prop_name}_0p5deg"

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=task_name,
        folder=output_dir,
        fileNamePrefix=task_name,
        region=global_bounds,
        scale=scale,
        crs='EPSG:4326',
        maxPixels=1e9,
        fileFormat='GeoTIFF'
    )

    task.start()
    print(f"    Started export: {task_name}")

    return task


def export_all_soil_properties(output_dir: str = "soil_0p5deg"):
    """Export all available soil properties."""

    if not initialize_gee():
        return

    print(f"\nExporting {len(OPENLANDMAP_ASSETS)} soil properties at 0.5° resolution")
    print(f"Output folder (Google Drive): {output_dir}")
    print("=" * 60)

    tasks = []
    for prop_name, config in OPENLANDMAP_ASSETS.items():
        print(f"\n{prop_name}:")
        task = export_soil_property(prop_name, config, output_dir)
        if task:
            tasks.append((prop_name, task))

    print("\n" + "=" * 60)
    print(f"Started {len(tasks)} export tasks")
    print("\nTo monitor progress:")
    print("  - Go to https://code.earthengine.google.com/tasks")
    print("  - Or run: earthengine task list")
    print("\nOnce complete, download from Google Drive and place in:")
    print("  ancillary/soilgrids/global/")

    return tasks


def check_task_status(tasks: list):
    """Check status of export tasks."""
    print("\nTask Status:")
    print("-" * 40)

    for name, task in tasks:
        status = task.status()
        state = status.get('state', 'UNKNOWN')
        print(f"  {name}: {state}")

        if state == 'FAILED':
            error = status.get('error_message', 'Unknown error')
            print(f"    Error: {error}")


def main():
    """Main function to export soil data."""
    print("=" * 60)
    print("SOIL DATA EXPORT VIA GOOGLE EARTH ENGINE")
    print("Using OpenLandMap datasets (derived from SoilGrids)")
    print("=" * 60)

    tasks = export_all_soil_properties()

    if tasks:
        print("\n" + "=" * 60)
        print("EXPORT TASKS STARTED")
        print("=" * 60)
        print("\nThis can take 10-30 minutes per property.")
        print("You can close this script and monitor at:")
        print("  https://code.earthengine.google.com/tasks")

        # Check status
        print("\nChecking status in 30 seconds...")
        time.sleep(30)
        check_task_status(tasks)


if __name__ == "__main__":
    main()
