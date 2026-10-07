"""
Quick test script to verify GLASS processing setup.

This script checks:
1. Required dependencies are installed
2. GLASS HDF tiles are available
3. Basic HDF file reading works
4. Sample tile can be processed

Usage:
    python test_glass_setup.py
"""

import sys
from pathlib import Path


def test_dependencies():
    """Test if required packages are installed."""
    print("="*60)
    print("TESTING DEPENDENCIES")
    print("="*60)

    results = {}

    # Test pyhdf
    try:
        from pyhdf.SD import SD, SDC
        print("✓ pyhdf installed")
        results['pyhdf'] = True
    except ImportError:
        print("✗ pyhdf NOT installed")
        print("  Install with: conda install -c conda-forge pyhdf")
        results['pyhdf'] = False

    # Test rasterio
    try:
        import rasterio
        print(f"✓ rasterio installed (version {rasterio.__version__})")
        results['rasterio'] = True
    except ImportError:
        print("✗ rasterio NOT installed")
        print("  Install with: pip install rasterio")
        results['rasterio'] = False

    # Test pyproj
    try:
        import pyproj
        print(f"✓ pyproj installed (version {pyproj.__version__})")
        results['pyproj'] = True
    except ImportError:
        print("✗ pyproj NOT installed")
        print("  Install with: pip install pyproj")
        results['pyproj'] = False

    # Test xarray
    try:
        import xarray as xr
        print(f"✓ xarray installed (version {xr.__version__})")
        results['xarray'] = True
    except ImportError:
        print("✗ xarray NOT installed")
        print("  Install with: pip install xarray")
        results['xarray'] = False

    # Test numpy
    try:
        import numpy as np
        print(f"✓ numpy installed (version {np.__version__})")
        results['numpy'] = True
    except ImportError:
        print("✗ numpy NOT installed")
        results['numpy'] = False

    print()
    all_installed = all(results.values())
    if all_installed:
        print("✓ All dependencies installed!")
        return True
    else:
        print("✗ Some dependencies missing. Please install them.")
        return False


def test_data_availability():
    """Test if GLASS HDF tiles are available."""
    print("="*60)
    print("TESTING DATA AVAILABILITY")
    print("="*60)

    glass_dir = Path("../../ancillary/glass/GPP_YEARLY/2010_001")

    if not glass_dir.exists():
        print(f"✗ GLASS directory not found: {glass_dir}")
        print("  Please download GLASS GPP_YEARLY data first.")
        return False

    hdf_files = list(glass_dir.glob("GLASS12E11.*.hdf"))

    if len(hdf_files) == 0:
        print(f"✗ No HDF files found in {glass_dir}")
        return False

    print(f"✓ Found {len(hdf_files)} GLASS GPP HDF tiles")
    print(f"  Directory: {glass_dir}")
    print(f"  Example files:")
    for f in hdf_files[:3]:
        print(f"    - {f.name}")

    return True


def test_hdf_reading():
    """Test reading a sample HDF file."""
    print("\n" + "="*60)
    print("TESTING HDF FILE READING")
    print("="*60)

    try:
        from pyhdf.SD import SD, SDC
    except ImportError:
        print("✗ Cannot import pyhdf - skipping HDF test")
        return False

    import numpy as np

    glass_dir = Path("../../ancillary/glass/GPP_YEARLY/2010_001")
    hdf_files = list(glass_dir.glob("GLASS12E11.*.hdf"))

    if len(hdf_files) == 0:
        print("✗ No HDF files available for testing")
        return False

    # Test first file
    test_file = hdf_files[0]
    print(f"Testing file: {test_file.name}")

    try:
        hdf = SD(str(test_file), SDC.READ)

        # Get datasets
        datasets = hdf.datasets()
        print(f"✓ File opened successfully")
        print(f"  Datasets found: {list(datasets.keys())}")

        # Try to read first dataset
        dataset_name = list(datasets.keys())[0]
        dataset = hdf.select(dataset_name)
        data = dataset.get()

        print(f"✓ Dataset '{dataset_name}' read successfully")
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Value range: {data.min()} to {data.max()}")

        # Get attributes
        attrs = dataset.attributes()
        if 'scale_factor' in attrs:
            print(f"  Scale factor: {attrs['scale_factor']}")
        if 'add_offset' in attrs:
            print(f"  Add offset: {attrs['add_offset']}")

        hdf.end()

        print("✓ HDF reading test PASSED")
        return True

    except Exception as e:
        print(f"✗ Error reading HDF file: {e}")
        return False


def test_sample_processing():
    """Test processing a small sample tile."""
    print("\n" + "="*60)
    print("TESTING SAMPLE TILE PROCESSING")
    print("="*60)

    try:
        from pyhdf.SD import SD, SDC
        import numpy as np
        from pyproj import Transformer
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("  Skipping processing test")
        return False

    glass_dir = Path("../../ancillary/glass/GPP_YEARLY/2010_001")
    hdf_files = list(glass_dir.glob("GLASS12E11.*.hdf"))

    if len(hdf_files) == 0:
        print("✗ No HDF files available")
        return False

    test_file = hdf_files[0]
    print(f"Processing: {test_file.name}")

    try:
        # Read HDF file
        hdf = SD(str(test_file), SDC.READ)
        dataset_name = list(hdf.datasets().keys())[0]
        dataset = hdf.select(dataset_name)
        data = dataset.get()

        # Apply scaling
        attrs = dataset.attributes()
        scale_factor = attrs.get('scale_factor', 1.0)
        add_offset = attrs.get('add_offset', 0.0)

        data = data.astype(np.float32)
        data = data * scale_factor + add_offset

        hdf.end()

        print(f"✓ Data processed successfully")
        print(f"  Scaled data range: {np.nanmin(data):.2f} to {np.nanmax(data):.2f}")
        print(f"  Valid pixels: {np.sum(~np.isnan(data)):,} / {data.size:,}")

        # Test coordinate transformation
        MODIS_SIN_PROJ = ("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 "
                         "+b=6371007.181 +units=m +no_defs")

        transformer = Transformer.from_crs(
            MODIS_SIN_PROJ,
            "EPSG:4326",
            always_xy=True
        )

        # Test transform one point
        x_test = 0
        y_test = 0
        lon, lat = transformer.transform(x_test, y_test)

        print(f"✓ Coordinate transformation working")
        print(f"  Test point (0, 0) → ({lon:.2f}°, {lat:.2f}°)")

        print("\n✓ Sample processing test PASSED")
        return True

    except Exception as e:
        print(f"✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("GLASS PROCESSING SETUP TEST")
    print("="*60)
    print()

    results = {}

    # Test 1: Dependencies
    results['dependencies'] = test_dependencies()

    # Test 2: Data availability
    results['data'] = test_data_availability()

    # Test 3: HDF reading (only if dependencies OK)
    if results['dependencies']:
        results['hdf_reading'] = test_hdf_reading()
    else:
        results['hdf_reading'] = False

    # Test 4: Sample processing (only if everything else OK)
    if results['dependencies'] and results['data']:
        results['processing'] = test_sample_processing()
    else:
        results['processing'] = False

    # Final summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("You can now run the GLASS processor:")
        print("  python glass_global_processor.py --year 2010")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please fix the issues above before processing GLASS data.")
    print("="*60)

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
