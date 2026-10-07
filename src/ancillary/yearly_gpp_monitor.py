#!/usr/bin/env python3
"""
Monitor yearly GPP download progress and create a sample visualization.
"""

import time
from pathlib import Path
import subprocess

def monitor_yearly_gpp_download():
    """Monitor the download progress of yearly GPP data."""
    
    gpp_yearly_dir = Path("ancillary/glass/GPP_YEARLY/2010_001")
    
    print("=== GLASS Yearly GPP Download Monitor ===")
    print()
    
    if not gpp_yearly_dir.exists():
        print("❌ Yearly GPP download not started yet")
        print("Run: python glass.py --mode download --products GPP_YEARLY --start-year 2010 --end-year 2010")
        return
    
    expected_total = 288  # Total MODIS tiles for global coverage
    
    while True:
        # Count downloaded files
        hdf_files = list(gpp_yearly_dir.glob("*.hdf"))
        downloaded = len(hdf_files)
        progress = (downloaded / expected_total) * 100
        
        # Get total size
        try:
            result = subprocess.run(['du', '-sh', str(gpp_yearly_dir.parent)], 
                                  capture_output=True, text=True)
            total_size = result.stdout.split()[0] if result.returncode == 0 else "unknown"
        except:
            total_size = "unknown"
        
        print(f"\r📊 Progress: {downloaded}/{expected_total} tiles ({progress:.1f}%) | Size: {total_size}", end="")
        
        if downloaded >= expected_total:
            print(f"\n\n✅ Download complete! {downloaded} tiles downloaded")
            break
        elif downloaded > 0:
            # Show sample files
            if downloaded <= 5:
                print(f"\n📁 Recent files:")
                for i, hdf_file in enumerate(hdf_files[-3:]):
                    print(f"   {i+1}. {hdf_file.name}")
        
        time.sleep(10)  # Check every 10 seconds

def create_yearly_vs_8day_comparison():
    """Create a comparison between yearly and 8-day GPP data."""
    
    yearly_dir = Path("ancillary/glass/GPP_YEARLY/2010_001")
    daily_dir = Path("ancillary/glass/GPP/2010_185")  # Mid-year 8-day data
    
    if not yearly_dir.exists():
        print("❌ Yearly GPP data not available")
        return
    
    if not daily_dir.exists():
        print("❌ 8-day GPP data not available")
        return
    
    # Find a common tile
    yearly_files = list(yearly_dir.glob("*.hdf"))
    daily_files = list(daily_dir.glob("*.hdf"))
    
    yearly_tiles = {f.name.split('.')[3] for f in yearly_files}
    daily_tiles = {f.name.split('.')[3] for f in daily_files}
    
    common_tiles = yearly_tiles & daily_tiles
    
    if common_tiles:
        sample_tile = list(common_tiles)[0]
        print(f"\n📊 Data Comparison for tile {sample_tile}:")
        print(f"   Yearly GPP: {len([f for f in yearly_files if sample_tile in f.name])} file(s)")
        print(f"   8-day GPP:  {len([f for f in daily_files if sample_tile in f.name])} file(s)")
        print()
        print("📈 Analysis potential:")
        print("   - Compare annual totals vs. summed 8-day periods")
        print("   - Validate yearly aggregation methodology") 
        print("   - Analyze seasonal patterns vs. annual means")
        print("   - Create time series analysis")
        
        # Show file details
        yearly_file = [f for f in yearly_files if sample_tile in f.name][0]
        daily_file = [f for f in daily_files if sample_tile in f.name][0]
        
        yearly_size = yearly_file.stat().st_size / 1024 / 1024
        daily_size = daily_file.stat().st_size / 1024 / 1024
        
        print(f"\n📁 File sizes:")
        print(f"   Yearly: {yearly_size:.1f} MB ({yearly_file.name})")
        print(f"   8-day:  {daily_size:.1f} MB ({daily_file.name})")
    else:
        print("❌ No common tiles found between yearly and 8-day data")

def main():
    """Main function."""
    
    yearly_dir = Path("ancillary/glass/GPP_YEARLY")
    
    if yearly_dir.exists():
        print("✅ Yearly GPP download directory exists")
        
        # Show progress
        monitor_yearly_gpp_download()
        
        # Create comparison if data is available
        create_yearly_vs_8day_comparison()
        
        print("\n🎯 Next steps:")
        print("1. Wait for download to complete (288 tiles total)")
        print("2. Use: python glass.py --mode visualize --products GPP_YEARLY") 
        print("3. Create annual maps using the yearly data")
        print("4. Compare with calculated annual totals from 8-day data")
        
    else:
        print("❌ Yearly GPP download not started")
        print("\n🚀 To start download:")
        print("python glass.py --mode download --products GPP_YEARLY --start-year 2010 --end-year 2010")

if __name__ == "__main__":
    main()