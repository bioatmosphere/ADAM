#!/usr/bin/env python3
"""
Quick test script to check GLASS URL accessibility for 2010_025
"""

import requests
from pathlib import Path

def test_glass_url():
    """Test if GLASS URLs are accessible for various 2010 dates"""
    
    # Test multiple dates
    dates_to_test = ['2010_001', '2010_009', '2010_017', '2010_025', '2010_033', '2010_041']
    
    for date in dates_to_test:
        year, doy = date.split('_')
        base_url = f"https://www.glass.hku.hk/archive/GPP/MODIS/500M/GLASS_GPP_500M_V60/{year}/{doy}/"
        
        print(f"\nTesting URL: {base_url}")
        
        try:
            # Try to access the directory listing
            response = requests.get(base_url, timeout=10)
        
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                print("✓ URL is accessible")
                
                # Look for .hdf files in the response
                if '.hdf' in response.text:
                    print("✓ HDF files found in directory listing")
                    # Count how many .hdf files are mentioned
                    hdf_count = response.text.count('.hdf')
                    print(f"  Found {hdf_count} .hdf file references")
                else:
                    print("✗ No HDF files found in directory listing")
                    
            else:
                print(f"✗ URL not accessible (status: {response.status_code})")
                
        except requests.exceptions.Timeout:
            print("✗ Request timed out")
        except requests.exceptions.ConnectionError:
            print("✗ Connection error")
        except Exception as e:
            print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_glass_url()