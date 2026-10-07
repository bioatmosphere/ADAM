#!/usr/bin/env python3
"""
Calculate annual GPP totals for different ecosystem types using selected tiles.
"""

import numpy as np
from pathlib import Path
from pyhdf import SD
import warnings

warnings.filterwarnings('ignore')

def read_hdf_data(hdf_file_path):
    """Read data from a GLASS HDF file."""
    try:
        hdf = SD.SD(str(hdf_file_path), SD.SDC.READ)
        datasets = hdf.datasets()
        
        # Find the main GPP dataset
        main_dataset = None
        for dataset_name in datasets.keys():
            if 'QC' not in dataset_name.upper() and 'GPP' in dataset_name.upper():
                main_dataset = dataset_name
                break
        
        if not main_dataset:
            for dataset_name in datasets.keys():
                if 'QC' not in dataset_name.upper():
                    main_dataset = dataset_name
                    break
        
        if main_dataset:
            dataset = hdf.select(main_dataset)
            data = dataset.get()
            
            # Apply scaling
            attrs = dataset.attributes()
            scale_factor = attrs.get('scale_factor', 1.0)
            add_offset = attrs.get('add_offset', 0.0)
            
            data = data.astype(np.float32)
            data = (data * scale_factor) + add_offset
            
            # Mask invalid values
            valid_mask = (data >= 0) & (data <= 50)
            data_masked = np.ma.masked_where(~valid_mask, data)
            
            dataset.endaccess()
            hdf.end()
            
            return data_masked
        
        hdf.end()
        return None
        
    except Exception as e:
        return None

def analyze_ecosystem_annual_gpp():
    """Analyze annual GPP for different ecosystem types using representative tiles."""
    
    data_path = Path("ancillary/glass/GPP")
    
    # Define representative tiles for different ecosystem types
    # Based on typical MODIS tile locations for major biomes
    ecosystem_tiles = {
        'Amazon_Tropical_Forest': ['h11v08', 'h12v08', 'h11v09'],  # Amazon basin
        'Congo_Tropical_Forest': ['h19v07', 'h20v07', 'h19v08'],   # Congo basin
        'Boreal_Forest_Canada': ['h11v03', 'h12v03', 'h13v03'],   # Canadian boreal
        'Boreal_Forest_Siberia': ['h21v03', 'h22v03', 'h23v03'],  # Siberian boreal
        'US_Temperate_Forest': ['h12v04', 'h13v04'],               # Eastern US forests
        'European_Temperate': ['h18v03', 'h19v03'],               # European forests
        'US_Croplands': ['h10v04', 'h11v04'],                     # Midwest croplands
        'Mediterranean': ['h17v04', 'h18v04'],                     # Mediterranean
        'Australian_Savanna': ['h29v11', 'h30v11'],               # Northern Australia
        'Sahara_Desert': ['h17v07', 'h18v07'],                    # Sahara
        'Arctic_Tundra': ['h11v02', 'h12v02'],                    # Arctic tundra
        'Southeast_Asia_Forest': ['h27v08', 'h28v08']             # SE Asian forests
    }
    
    # Find all 2010 date directories
    date_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('2010')]
    date_dirs = sorted(date_dirs)
    
    print(f"Ecosystem Annual GPP Analysis for 2010")
    print(f"======================================")
    print(f"Total 8-day periods: {len(date_dirs)}")
    print(f"Ecosystem types: {len(ecosystem_tiles)}")
    print()
    
    ecosystem_results = {}
    
    for ecosystem, tiles in ecosystem_tiles.items():
        print(f"Analyzing {ecosystem}...")
        
        annual_totals = []
        valid_periods = 0
        
        # Process each 8-day period
        for date_dir in date_dirs:
            period_values = []
            
            # Look for files matching our target tiles
            for tile_id in tiles:
                matching_files = list(date_dir.glob(f"*{tile_id}*.hdf"))
                
                for hdf_file in matching_files:
                    data = read_hdf_data(hdf_file)
                    
                    if data is not None and not data.mask.all():
                        # Convert daily to 8-day total and sample for speed
                        period_total = data * 8.0
                        valid_pixels = period_total.compressed()
                        
                        if len(valid_pixels) > 0:
                            # Sample 10% of pixels for speed
                            sample_size = max(100, len(valid_pixels) // 10)
                            if len(valid_pixels) > sample_size:
                                sampled_pixels = np.random.choice(valid_pixels, sample_size, replace=False)
                            else:
                                sampled_pixels = valid_pixels
                            
                            period_values.extend(sampled_pixels)
            
            if period_values:
                period_mean = np.mean(period_values)
                annual_totals.append(period_mean)
                valid_periods += 1
        
        if annual_totals:
            # Calculate annual statistics
            annual_array = np.array(annual_totals)
            total_annual_gpp = np.sum(annual_array)
            mean_8day_gpp = np.mean(annual_array)
            
            ecosystem_results[ecosystem] = {
                'annual_total': total_annual_gpp,
                'mean_8day_gpp': mean_8day_gpp,
                'valid_periods': valid_periods,
                'seasonal_pattern': annual_totals
            }
            
            print(f"  Annual total GPP: {total_annual_gpp:.0f} g C/m²/year")
            print(f"  Mean 8-day GPP: {mean_8day_gpp:.1f} g C/m²")
            print(f"  Valid periods: {valid_periods}/{len(date_dirs)}")
            
            # Seasonal analysis
            if len(annual_totals) >= 4:
                quarters = [
                    np.mean(annual_totals[0:12]),    # Q1 (Jan-Mar)
                    np.mean(annual_totals[12:24]),   # Q2 (Apr-Jun)
                    np.mean(annual_totals[24:36]),   # Q3 (Jul-Sep)
                    np.mean(annual_totals[36:46])    # Q4 (Oct-Dec)
                ]
                peak_quarter = np.argmax(quarters)
                quarter_names = ['Winter', 'Spring', 'Summer', 'Fall']
                print(f"  Peak season: {quarter_names[peak_quarter]} ({quarters[peak_quarter]:.1f} g C/m²)")
        else:
            print(f"  No valid data found")
        
        print()
    
    # Summary comparison
    print("Ecosystem Comparison Summary:")
    print("=" * 60)
    print(f"{'Ecosystem':<25} {'Annual GPP':<15} {'Peak Season':<12}")
    print("-" * 60)
    
    sorted_ecosystems = sorted(ecosystem_results.items(), 
                              key=lambda x: x[1]['annual_total'], reverse=True)
    
    for ecosystem, results in sorted_ecosystems:
        quarterly_data = results['seasonal_pattern']
        if len(quarterly_data) >= 4:
            quarters = [
                np.mean(quarterly_data[0:12]),
                np.mean(quarterly_data[12:24]),
                np.mean(quarterly_data[24:36]),
                np.mean(quarterly_data[36:46])
            ]
            peak_quarter = np.argmax(quarters)
            quarter_names = ['Winter', 'Spring', 'Summer', 'Fall']
            peak_season = quarter_names[peak_quarter]
        else:
            peak_season = 'Unknown'
        
        annual_gpp = results['annual_total']
        print(f"{ecosystem:<25} {annual_gpp:>8.0f} g C/m²/yr {peak_season:<12}")
    
    print()
    
    # Global productivity classification
    print("Productivity Classification:")
    print("-" * 30)
    for ecosystem, results in sorted_ecosystems:
        annual_gpp = results['annual_total']
        if annual_gpp > 2000:
            category = "Very High"
        elif annual_gpp > 1500:
            category = "High"
        elif annual_gpp > 1000:
            category = "Moderate"
        elif annual_gpp > 500:
            category = "Low"
        else:
            category = "Very Low"
        
        print(f"{ecosystem:<25} {category}")
    
    # Save detailed results
    results_file = Path("ancillary/glass/annual_totals/ecosystem_annual_gpp_2010.txt")
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("GLASS GPP Ecosystem Analysis - 2010\\n")
        f.write("=" * 40 + "\\n\\n")
        
        for ecosystem, results in sorted_ecosystems:
            f.write(f"{ecosystem}:\\n")
            f.write(f"  Annual total GPP: {results['annual_total']:.0f} g C/m²/year\\n")
            f.write(f"  Mean 8-day GPP: {results['mean_8day_gpp']:.1f} g C/m²\\n")
            f.write(f"  Valid periods: {results['valid_periods']}/{len(date_dirs)}\\n")
            f.write(f"  Target tiles: {', '.join(ecosystem_tiles[ecosystem])}\\n")
            f.write("\\n")
    
    print(f"✓ Detailed results saved: {results_file}")
    
    return ecosystem_results

if __name__ == "__main__":
    analyze_ecosystem_annual_gpp()