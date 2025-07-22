""" 
Aggregate processed data from different sources into a single dataset for machine learing.

This script fetches data from various sources, processes them, and aggregate them into a single DataFrame
    for machine learning (deep learning) next.

What it does:
    - Reads processed grassland and forest BNPP data from CSV files.
    - Performs unit conversions to standardize BNPP measurements:
        * Forest data: Mg C ha⁻¹ yr⁻¹ → g C m⁻² yr⁻¹ (×100 conversion factor)
        * Grassland data: g m⁻² yr⁻¹ (assuming carbon equivalent)
    - Renames columns to ensure consistency across datasets.
    - Combines the data into a single DataFrame.
    - Plots the spatial distribution of data points on a map.

**NOTE**: It extracts the latitude and longitude of data points to be used for extracting
    data from the ancillary data sources like weather, soil, and other environmental data.
    
The data sources include:
    - grassland BNPP data: processed from Dryad global grassland database (grassland_bnpp_data.csv)
    - forest BNPP data: processed from ForC database (ForC_BNPP_root_C_processed.csv)
    - other potential sources in the future.
    - ancillary data from various sources:
        * TerraClimate: climate variables (aet, pet, ppt, tmax, tmin, vpd)
        * GLASS: Gross Primary Production (GPP) satellite data from yearly aggregations
        * SoilGrids: soil properties (carbon stock, texture, nutrients, pH, bulk density)
        * Soil Moisture: EC ORS dataset (volumetric water content)
        * Elevation: SRTM-based elevation data (meters above sea level)

Check out https://github.com/NVIDIA-Omniverse-blueprints/earth2-weather-analytics 
    for inspirations for structure and data sources.

MCP: https://claude.ai/share/1d62b6fc-271b-422f-8c0d-f9be2efdd8ed

"""

import pandas as pd
import os
from pathlib import Path
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Import scipy for spatial distance calculations and outlier detection
try:
    from scipy.spatial.distance import cdist
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not available. Spatial merging will use simplified approach.")
    SCIPY_AVAILABLE = False

def process_productivity_data(df):
    """Process grassland/forest productivity data with standard column renaming and unit conversion."""
    # Make column names consistent
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)
    
    # Handle BNPP column - different names in grassland vs forest data
    if 'mean' in df.columns:
        df.rename(columns={'mean': 'BNPP'}, inplace=True)
    elif 'BNPP' in df.columns:
        # Grassland data already has BNPP column
        pass
    
    # Handle biome/ecosystem type columns
    if 'dominant.veg' in df.columns:
        df.rename(columns={'dominant.veg': 'biome'}, inplace=True)
    elif 'dominant.life.form' in df.columns:
        df.rename(columns={'dominant.life.form': 'biome'}, inplace=True)
    elif 'Grassland type' in df.columns:
        df.rename(columns={'Grassland type': 'biome'}, inplace=True)
    
    # Determine data source and handle unit conversions
    data_source = None
    if 'Data_Source' in df.columns:
        if 'Grassland' in str(df['Data_Source'].iloc[0]):
            data_source = 'grassland'
        elif 'ForC' in str(df['Data_Source'].iloc[0]):
            data_source = 'forest'
    elif 'Variable_Name' in df.columns and 'BNPP_root_C' in str(df['Variable_Name'].iloc[0]):
        data_source = 'forest'
    elif 'Grassland type' in df.columns:
        data_source = 'grassland'
    
    # Unit conversions to standardize BNPP values
    if 'BNPP' in df.columns and data_source:
        if data_source == 'forest':
            # Convert forest data from Mg C ha⁻¹ yr⁻¹ to g C m⁻² yr⁻¹
            # 1 Mg C ha⁻¹ yr⁻¹ = 1000 kg C / 10000 m² / yr = 0.1 kg C m⁻² yr⁻¹ = 100 g C m⁻² yr⁻¹
            df['BNPP'] = df['BNPP'] * 100  # Convert to g C m⁻² yr⁻¹
            df['BNPP_units'] = 'g C m⁻² yr⁻¹'
            print(f"Converted forest BNPP from Mg C ha⁻¹ yr⁻¹ to g C m⁻² yr⁻¹")
        elif data_source == 'grassland':
            # Grassland data is in g/m²/year, assuming this is already carbon equivalent
            # If not carbon-specific, we may need additional conversion factors
            df['BNPP_units'] = 'g m⁻² yr⁻¹'
            print(f"Grassland BNPP units: g m⁻² yr⁻¹ (assuming carbon equivalent)")
    
    # Add data source identifier
    if data_source:
        df['data_source'] = data_source
    elif 'data_source' not in df.columns:
        df['data_source'] = 'productivity'
    
    return df

def process_terraclimate_data(df, file_path):
    """Process TerraClimate mean data with appropriate column naming."""
    # Ensure consistent lat/lon column names
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
    
    # Extract variable name from file path and rename mean_value column
    file_name = file_path.stem
    if '_means' in file_name:
        var_name = file_name.replace('_means', '')
        if 'mean_value' in df.columns:
            df.rename(columns={'mean_value': var_name}, inplace=True)
    
    # Add data source identifier
    df['data_source'] = 'terraclimate'
    
    return df

def process_glass_data(df, file_path):
    """Process GLASS GPP data with appropriate column naming."""
    # Ensure consistent lat/lon column names
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
    
    # Extract variable name from file path and rename value column
    file_name = file_path.stem
    if 'GPP_YEARLY' in file_name:
        # Rename 'value' column to 'gpp_yearly' for clarity
        if 'value' in df.columns:
            df.rename(columns={'value': 'gpp_yearly'}, inplace=True)
    
    # Add data source identifier
    df['data_source'] = 'glass'
    
    # Convert units from gC m-2 year-1 to more standard units if needed
    # Keep original units for now, can be converted later if needed
    
    return df

def process_soilgrids_data(df, file_path):
    """Process SoilGrids soil data with appropriate column naming and unit conversions."""
    # Ensure consistent lat/lon column names
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
    
    # Handle unit conversions for specific soil properties
    file_name = file_path.stem
    
    # pH data needs to be divided by 10 (stored as pH*10)
    if 'ph_data_points' in file_name and 'ph_in_water' in df.columns:
        df['ph_in_water'] = df['ph_in_water'] / 10.0
        print(f"Converted pH values from pH*10 to actual pH")
    
    # Bulk density is in cg/cm³ (centigrams per cubic centimeter)
    if 'bulk_density_data_points' in file_name and 'bulk_density' in df.columns:
        # Convert from cg/cm³ to g/cm³ by dividing by 100
        df['bulk_density'] = df['bulk_density'] / 100.0
        print(f"Converted bulk density from cg/cm³ to g/cm³")
    
    # Nitrogen content is in cg/kg (centigrams per kilogram)
    if 'nitrogen_data_points' in file_name and 'nitrogen_content' in df.columns:
        # Convert from cg/kg to g/kg by dividing by 100
        df['nitrogen_content'] = df['nitrogen_content'] / 100.0
        print(f"Converted nitrogen content from cg/kg to g/kg")
    
    # Coarse fragments are in promille (‰) - no conversion needed
    if 'coarse_fragments_data_points' in file_name and 'coarse_fragments' in df.columns:
        print(f"Coarse fragments in promille (‰) - no conversion needed")
    
    # Cation exchange capacity is in mmol(c)/kg - no conversion needed
    if 'cec_data_points' in file_name and 'cation_exchange_capacity' in df.columns:
        print(f"CEC in mmol(c)/kg - no conversion needed")
    
    # Soil carbon stock is in tonnes C/ha - no conversion needed for now
    if 'soil_carbon_points' in file_name and 'soil_carbon_stock' in df.columns:
        print(f"Soil carbon stock in tonnes C/ha - no conversion needed")
    
    # Clay, silt, sand content are in g/kg - could convert to percentage
    for texture in ['clay_content', 'silt_content', 'sand_content']:
        if texture in df.columns:
            # Convert from g/kg to percentage by dividing by 10
            df[texture] = df[texture] / 10.0
            print(f"Converted {texture} from g/kg to percentage")
    
    # Add data source identifier
    df['data_source'] = 'soilgrids'
    
    return df

def process_soil_moisture_data(df, file_path):
    """Process soil moisture data with appropriate column naming."""
    # Ensure consistent lat/lon column names
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
    
    # Soil moisture is already in volumetric water content (fraction)
    # Values typically range from 0.0 to 1.0, but more commonly 0.1 to 0.5
    if 'soil_moisture' in df.columns:
        print(f"Soil moisture in volumetric water content (fraction) - no conversion needed")
    
    # Add data source identifier
    df['data_source'] = 'soil_moisture'
    
    return df

def process_elevation_data(df, file_path):
    """Process elevation data with appropriate column naming."""
    # Ensure consistent lat/lon column names
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
    
    # Elevation is in meters above sea level
    if 'elevation' in df.columns:
        print(f"Elevation in meters above sea level - no conversion needed")
    
    # Add data source identifier
    df['data_source'] = 'elevation'
    
    return df

def detect_statistical_outliers(df, column='BNPP', method='iqr', multiplier=1.5):
    """
    Detect statistical outliers in BNPP data using various methods.
    
    Args:
        df: DataFrame with BNPP data
        column: Column name to check for outliers (default: 'BNPP')
        method: Method for outlier detection ('iqr', 'z_score', 'modified_z_score')
        multiplier: Multiplier for outlier threshold (default: 1.5 for IQR, 3 for z-score)
    
    Returns:
        pandas.Series: Boolean mask indicating outliers (True = outlier)
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found in DataFrame")
        return pd.Series([False] * len(df))
    
    values = df[column].dropna()
    if len(values) == 0:
        print(f"Warning: No valid values found in column '{column}'")
        return pd.Series([False] * len(df))
    
    outliers = pd.Series([False] * len(df), index=df.index)
    
    if method == 'iqr':
        # Interquartile Range (IQR) method
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        print(f"IQR outlier detection: bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
        
    elif method == 'z_score':
        # Z-score method
        if multiplier == 1.5:  # Default was for IQR, adjust for z-score
            multiplier = 3.0
        
        mean_val = values.mean()
        std_val = values.std()
        z_scores = np.abs((df[column] - mean_val) / std_val)
        outliers = z_scores > multiplier
        print(f"Z-score outlier detection: threshold {multiplier}, mean={mean_val:.2f}, std={std_val:.2f}")
        
    elif method == 'modified_z_score':
        # Modified Z-score using median absolute deviation (MAD)
        if multiplier == 1.5:  # Default was for IQR, adjust for modified z-score
            multiplier = 3.5
        
        median_val = values.median()
        mad = np.median(np.abs(values - median_val))
        modified_z_scores = 0.6745 * (df[column] - median_val) / mad
        outliers = np.abs(modified_z_scores) > multiplier
        print(f"Modified Z-score outlier detection: threshold {multiplier}, median={median_val:.2f}, MAD={mad:.2f}")
    
    else:
        print(f"Warning: Unknown outlier detection method '{method}'. Using IQR method.")
        return detect_statistical_outliers(df, column, method='iqr', multiplier=multiplier)
    
    return outliers

def detect_domain_outliers(df, column='BNPP', data_source_col='data_source'):
    """
    Detect domain-specific outliers based on biological/ecological knowledge.
    
    Args:
        df: DataFrame with BNPP data
        column: Column name to check for outliers (default: 'BNPP')
        data_source_col: Column indicating data source (forest vs grassland)
    
    Returns:
        pandas.Series: Boolean mask indicating outliers (True = outlier)
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found in DataFrame")
        return pd.Series([False] * len(df))
    
    outliers = pd.Series([False] * len(df), index=df.index)
    
    # Domain-specific thresholds based on ecological literature
    # Units: g C m⁻² yr⁻¹
    domain_thresholds = {
        'forest': {
            'min': 0,      # Minimum: 0 g C m⁻² yr⁻¹
            'max': 2000,   # Maximum: ~2000 g C m⁻² yr⁻¹ (very productive forests)
            'typical_max': 1500  # Typical maximum for most forests
        },
        'grassland': {
            'min': 0,      # Minimum: 0 g C m⁻² yr⁻¹
            'max': 1000,   # Maximum: ~1000 g C m⁻² yr⁻¹ (very productive grasslands)
            'typical_max': 800   # Typical maximum for most grasslands
        },
        'general': {
            'min': 0,      # Minimum: 0 g C m⁻² yr⁻¹
            'max': 2500,   # Conservative maximum across all ecosystems
            'typical_max': 2000
        }
    }
    
    # Check for negative values (always outliers)
    negative_outliers = df[column] < 0
    outliers = outliers | negative_outliers
    
    # Check for extremely high values
    if data_source_col in df.columns:
        # Apply ecosystem-specific thresholds
        for ecosystem in ['forest', 'grassland']:
            mask = df[data_source_col] == ecosystem
            if mask.any():
                thresholds = domain_thresholds[ecosystem]
                ecosystem_outliers = (df[column] > thresholds['max']) & mask
                outliers = outliers | ecosystem_outliers
                
                print(f"{ecosystem.capitalize()} outliers: {ecosystem_outliers.sum()} points > {thresholds['max']} g C m⁻² yr⁻¹")
    else:
        # Apply general thresholds
        thresholds = domain_thresholds['general']
        high_outliers = df[column] > thresholds['max']
        outliers = outliers | high_outliers
        print(f"General outliers: {high_outliers.sum()} points > {thresholds['max']} g C m⁻² yr⁻¹")
    
    if negative_outliers.any():
        print(f"Negative value outliers: {negative_outliers.sum()} points < 0 g C m⁻² yr⁻¹")
    
    return outliers

def detect_geographic_outliers(df, column='BNPP', lat_col='lat', lon_col='lon'):
    """
    Detect geographic outliers by checking for unusual values in specific regions.
    
    Args:
        df: DataFrame with BNPP data
        column: Column name to check for outliers (default: 'BNPP')
        lat_col: Latitude column name
        lon_col: Longitude column name
    
    Returns:
        pandas.Series: Boolean mask indicating outliers (True = outlier)
    """
    if not all(col in df.columns for col in [column, lat_col, lon_col]):
        print(f"Warning: Required columns not found in DataFrame")
        return pd.Series([False] * len(df))
    
    outliers = pd.Series([False] * len(df), index=df.index)
    
    # Define geographic regions with expected BNPP ranges
    regions = {
        'arctic': {'lat_min': 60, 'lat_max': 90, 'max_bnpp': 200},
        'tropical': {'lat_min': -23.5, 'lat_max': 23.5, 'max_bnpp': 2000},
        'temperate': {'lat_min': 23.5, 'lat_max': 60, 'max_bnpp': 1500},
        'southern_temperate': {'lat_min': -60, 'lat_max': -23.5, 'max_bnpp': 1500},
        'antarctic': {'lat_min': -90, 'lat_max': -60, 'max_bnpp': 100}
    }
    
    for region_name, bounds in regions.items():
        # Identify points in this region
        in_region = (df[lat_col] >= bounds['lat_min']) & (df[lat_col] <= bounds['lat_max'])
        
        if in_region.any():
            # Check for values exceeding regional maximum
            region_outliers = (df[column] > bounds['max_bnpp']) & in_region
            outliers = outliers | region_outliers
            
            if region_outliers.any():
                print(f"{region_name.capitalize()} region outliers: {region_outliers.sum()} points > {bounds['max_bnpp']} g C m⁻² yr⁻¹")
    
    return outliers

def visualize_outliers(df, column='BNPP', outlier_methods=['iqr', 'domain'], output_dir='../productivity/earth'):
    """
    Create visualizations to show detected outliers.
    
    Args:
        df: DataFrame with BNPP data
        column: Column name to visualize (default: 'BNPP')
        outlier_methods: List of outlier detection methods to apply
        output_dir: Directory to save visualizations
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found in DataFrame")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect outliers using different methods
    outlier_masks = {}
    for method in outlier_methods:
        if method == 'iqr':
            outlier_masks[method] = detect_statistical_outliers(df, column, method='iqr')
        elif method == 'z_score':
            outlier_masks[method] = detect_statistical_outliers(df, column, method='z_score')
        elif method == 'modified_z_score':
            outlier_masks[method] = detect_statistical_outliers(df, column, method='modified_z_score')
        elif method == 'domain':
            outlier_masks[method] = detect_domain_outliers(df, column)
        elif method == 'geographic':
            outlier_masks[method] = detect_geographic_outliers(df, column)
    
    # Create combined outlier mask
    combined_outliers = pd.Series([False] * len(df), index=df.index)
    for mask in outlier_masks.values():
        combined_outliers = combined_outliers | mask
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Histogram with outliers highlighted
    ax1 = axes[0, 0]
    values = df[column].dropna()
    ax1.hist(values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Highlight outliers
    for method, mask in outlier_masks.items():
        outlier_values = df.loc[mask, column].dropna()
        if len(outlier_values) > 0:
            ax1.hist(outlier_values, bins=50, alpha=0.8, 
                    label=f'{method} outliers ({len(outlier_values)})', 
                    histtype='step', linewidth=2)
    
    ax1.set_xlabel(f'{column} (g C m⁻² yr⁻¹)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('BNPP Distribution with Outliers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot with outliers
    ax2 = axes[0, 1]
    bp = ax2.boxplot(values, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax2.set_ylabel(f'{column} (g C m⁻² yr⁻¹)')
    ax2.set_title('BNPP Box Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter plot: BNPP vs Latitude
    ax3 = axes[1, 0]
    if 'lat' in df.columns:
        # Plot normal points
        normal_mask = ~combined_outliers
        ax3.scatter(df.loc[normal_mask, 'lat'], df.loc[normal_mask, column], 
                   alpha=0.6, s=30, color='blue', label='Normal')
        
        # Plot outliers
        if combined_outliers.any():
            ax3.scatter(df.loc[combined_outliers, 'lat'], df.loc[combined_outliers, column], 
                       alpha=0.8, s=50, color='red', marker='x', label='Outliers')
        
        ax3.set_xlabel('Latitude')
        ax3.set_ylabel(f'{column} (g C m⁻² yr⁻¹)')
        ax3.set_title('BNPP vs Latitude')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Geographic distribution of outliers
    ax4 = axes[1, 1]
    if 'lat' in df.columns and 'lon' in df.columns:
        # Plot normal points
        normal_mask = ~combined_outliers
        ax4.scatter(df.loc[normal_mask, 'lon'], df.loc[normal_mask, 'lat'], 
                   alpha=0.6, s=30, color='blue', label='Normal')
        
        # Plot outliers
        if combined_outliers.any():
            ax4.scatter(df.loc[combined_outliers, 'lon'], df.loc[combined_outliers, 'lat'], 
                       alpha=0.8, s=50, color='red', marker='x', label='Outliers')
        
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        ax4.set_title('Geographic Distribution of Outliers')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    output_path = os.path.join(output_dir, f'{column}_outlier_analysis.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Outlier analysis saved to: {output_path}")
    
    # Print summary statistics
    print(f"\nOutlier Detection Summary for {column}:")
    print(f"Total data points: {len(df)}")
    print(f"Valid {column} values: {len(values)}")
    
    for method, mask in outlier_masks.items():
        outlier_count = mask.sum()
        outlier_percentage = (outlier_count / len(df)) * 100
        print(f"{method.upper()} outliers: {outlier_count} ({outlier_percentage:.1f}%)")
    
    combined_count = combined_outliers.sum()
    combined_percentage = (combined_count / len(df)) * 100
    print(f"Combined outliers: {combined_count} ({combined_percentage:.1f}%)")
    
    plt.show()

def remove_outliers(df, column='BNPP', methods=['iqr', 'domain'], remove_method='union'):
    """
    Remove outliers from the dataset based on specified methods.
    
    Args:
        df: DataFrame with BNPP data
        column: Column name to check for outliers (default: 'BNPP')
        methods: List of outlier detection methods to apply
        remove_method: How to combine multiple methods ('union' or 'intersection')
    
    Returns:
        tuple: (cleaned_df, outlier_df, outlier_summary)
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found in DataFrame")
        return df, pd.DataFrame(), {}
    
    # Detect outliers using different methods
    outlier_masks = {}
    for method in methods:
        if method == 'iqr':
            outlier_masks[method] = detect_statistical_outliers(df, column, method='iqr')
        elif method == 'z_score':
            outlier_masks[method] = detect_statistical_outliers(df, column, method='z_score')
        elif method == 'modified_z_score':
            outlier_masks[method] = detect_statistical_outliers(df, column, method='modified_z_score')
        elif method == 'domain':
            outlier_masks[method] = detect_domain_outliers(df, column)
        elif method == 'geographic':
            outlier_masks[method] = detect_geographic_outliers(df, column)
    
    # Combine outlier masks
    if remove_method == 'union':
        # Remove points that are outliers in any method
        combined_outliers = pd.Series([False] * len(df), index=df.index)
        for mask in outlier_masks.values():
            combined_outliers = combined_outliers | mask
    elif remove_method == 'intersection':
        # Remove points that are outliers in all methods
        combined_outliers = pd.Series([True] * len(df), index=df.index)
        for mask in outlier_masks.values():
            combined_outliers = combined_outliers & mask
    else:
        print(f"Warning: Unknown remove_method '{remove_method}'. Using 'union'.")
        combined_outliers = pd.Series([False] * len(df), index=df.index)
        for mask in outlier_masks.values():
            combined_outliers = combined_outliers | mask
    
    # Split data into clean and outlier datasets
    cleaned_df = df[~combined_outliers].copy()
    outlier_df = df[combined_outliers].copy()
    
    # Create summary
    outlier_summary = {
        'original_count': len(df),
        'outlier_count': combined_outliers.sum(),
        'cleaned_count': len(cleaned_df),
        'outlier_percentage': (combined_outliers.sum() / len(df)) * 100,
        'methods_used': methods,
        'remove_method': remove_method
    }
    
    # Add method-specific counts
    for method, mask in outlier_masks.items():
        outlier_summary[f'{method}_outliers'] = mask.sum()
    
    print(f"\nOutlier Removal Summary:")
    print(f"Original dataset: {outlier_summary['original_count']} points")
    print(f"Outliers removed: {outlier_summary['outlier_count']} points ({outlier_summary['outlier_percentage']:.1f}%)")
    print(f"Cleaned dataset: {outlier_summary['cleaned_count']} points")
    print(f"Methods used: {', '.join(methods)}")
    print(f"Combination method: {remove_method}")
    
    return cleaned_df, outlier_df, outlier_summary

def merge_datasets_by_location(productivity_df, climate_df, tolerance=0.1):
    """
    Merge productivity and climate datasets based on lat/lon coordinates.
    
    Args:
        productivity_df: DataFrame with productivity data (must have 'lat', 'lon' columns)
        climate_df: DataFrame with climate data (must have 'lat', 'lon' columns)
        tolerance: Distance tolerance in degrees for matching coordinates
    
    Returns:
        Merged DataFrame with climate variables added to productivity data
    """
    # Extract coordinates
    prod_coords = productivity_df[['lat', 'lon']].values
    clim_coords = climate_df[['lat', 'lon']].values
    
    # Calculate distances between all productivity and climate points
    if SCIPY_AVAILABLE:
        distances = cdist(prod_coords, clim_coords)
    else:
        # Fallback: simple Euclidean distance calculation
        distances = np.sqrt(
            ((prod_coords[:, np.newaxis, 0] - clim_coords[np.newaxis, :, 0]) ** 2) +
            ((prod_coords[:, np.newaxis, 1] - clim_coords[np.newaxis, :, 1]) ** 2)
        )
    
    # Find nearest climate point for each productivity point
    nearest_indices = np.argmin(distances, axis=1)
    nearest_distances = np.min(distances, axis=1)
    
    # Create a copy of productivity data to avoid modifying original
    merged_df = productivity_df.copy()
    
    # Add climate variables for points within tolerance
    climate_vars = [col for col in climate_df.columns if col not in ['lat', 'lon', 'point_name', 'units', 'data_source', 'date', 'product']]
    
    for var in climate_vars:
        merged_df[var] = np.nan
    
    # Merge climate data where distance is within tolerance
    valid_matches = nearest_distances <= tolerance
    
    for i, (is_valid, clim_idx) in enumerate(zip(valid_matches, nearest_indices)):
        if is_valid:
            for var in climate_vars:
                if var in climate_df.columns:
                    merged_df.iloc[i, merged_df.columns.get_loc(var)] = climate_df.iloc[clim_idx][var]
    
    print(f"Merged {valid_matches.sum()} out of {len(productivity_df)} productivity points with climate data")
    print(f"Climate variables added: {climate_vars}")
    
    return merged_df

def aggregate_data(data_files=[]):
    """Aggregate data from different sources into a single DataFrame.
    
    Handles both productivity data (grassland/forest) and ancillary climate data (TerraClimate).
    """
    
    # Define base directories
    productivity_dir = Path('../productivity').resolve()
    base_dir = Path('..').resolve()
    
    # Initialize dictionaries to hold DataFrames by type
    productivity_dataframes = []
    climate_dataframes = []
    
    # Read each file and append the DataFrame to the appropriate list
    for file in data_files:
        try:
            # Determine file path based on file location
            if file.startswith('../ancillary/'):
                # TerraClimate ancillary data files
                file_path = base_dir / file.replace('../', '')
            else:
                # Productivity data files (grassland/forest)
                file_path = productivity_dir / file
                # Verify productivity directory exists
                if not productivity_dir.exists():
                    raise FileNotFoundError(f"Productivity data directory not found: {productivity_dir}")
            
            if not file_path.is_file():
                print(f"Warning: Data file not found: {file_path}, skipping...")
                continue
                
            print(f"Processing: {file_path}")
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} rows from {file}")

            # Process based on file type
            if 'terraclimate' in str(file_path):
                # TerraClimate data processing
                df = process_terraclimate_data(df, file_path)
                climate_dataframes.append(df)
            elif 'glass' in str(file_path) or 'GPP_YEARLY' in str(file_path):
                # GLASS GPP data processing
                df = process_glass_data(df, file_path)
                climate_dataframes.append(df)
            elif 'soilgrids' in str(file_path):
                # SoilGrids soil data processing
                df = process_soilgrids_data(df, file_path)
                climate_dataframes.append(df)
            elif 'soil_moisture' in str(file_path):
                # Soil moisture data processing
                df = process_soil_moisture_data(df, file_path)
                climate_dataframes.append(df)
            elif 'elevation' in str(file_path):
                # Elevation data processing
                df = process_elevation_data(df, file_path)
                climate_dataframes.append(df)
            else:
                # Productivity data processing (grassland/forest)
                df = process_productivity_data(df)
                productivity_dataframes.append(df)

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    # Combine and merge the datasets
    integrated_df = None
    
    # First, concatenate productivity data
    if productivity_dataframes:
        productivity_df = pd.concat(productivity_dataframes, ignore_index=True)
        productivity_df['data_source'] = 'productivity'
        print(f"Combined productivity data: {len(productivity_df)} rows")
        integrated_df = productivity_df
    
    # Then, process climate data and merge with productivity data if possible  
    if climate_dataframes and integrated_df is not None:
        print("Merging climate variables with productivity data...")
        
        # Process each climate/environmental variable separately to avoid complex merges
        for df in climate_dataframes:
            # Get the variable name from the dataframe columns
            var_cols = [col for col in df.columns if col not in ['lat', 'lon', 'point_name', 'units', 'data_source', 'date', 'product']]
            if var_cols:
                var_name = var_cols[0]
                print(f"  Processing {var_name}...")
                
                # Simple approach: find exact coordinate matches
                matched_count = 0
                for idx, row in integrated_df.iterrows():
                    prod_lat, prod_lon = row['lat'], row['lon']
                    
                    # Find matching environmental data point
                    env_match = df[
                        (abs(df['lat'] - prod_lat) < 0.01) & 
                        (abs(df['lon'] - prod_lon) < 0.01)
                    ]
                    
                    if not env_match.empty:
                        integrated_df.at[idx, var_name] = env_match.iloc[0][var_name]
                        matched_count += 1
                    else:
                        integrated_df.at[idx, var_name] = np.nan
                
                print(f"    Matched {matched_count}/{len(integrated_df)} points for {var_name}")
    
    elif climate_dataframes and integrated_df is None:
        print("Only climate data available, no productivity data to merge with")
        integrated_df = pd.concat(climate_dataframes, ignore_index=True)
    
    if integrated_df is not None and len(integrated_df) > 0:
        # Filter to keep only essential columns: lat, lon, biome, productivity, and climate data
        essential_columns = ['lat', 'lon']
        
        # Add biome column (ecosystem type)
        biome_cols = [col for col in integrated_df.columns if col in ['biome', 'ecosystem', 'vegetation_type', 'land_cover']]
        if biome_cols:
            essential_columns.extend(biome_cols)
            print(f"Found biome column: {biome_cols}")
        else:
            print("Warning: No biome column found")
        
        # Add productivity column (BNPP or similar)
        productivity_cols = [col for col in integrated_df.columns if col in ['BNPP', 'mean', 'productivity']]
        if productivity_cols:
            essential_columns.extend(productivity_cols)
            print(f"Found productivity column: {productivity_cols}")
            # Also include units column if available
            if 'BNPP_units' in integrated_df.columns:
                essential_columns.append('BNPP_units')
        else:
            print("Warning: No productivity column found. Looking for numeric columns...")
            # Fallback: look for numeric columns that might represent productivity
            numeric_cols = integrated_df.select_dtypes(include=[np.number]).columns
            potential_prod_cols = [col for col in numeric_cols if col not in [
                'lat', 'lon', 'aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd', 'gpp_yearly', 
                'Entry_ID', 'Altitude', 'Sampling year', 'BNPP_SE', 'MAT', 'MAP', 'stand.age', 'masl',
                'soil_carbon_stock', 'clay_content', 'silt_content', 'sand_content', 
                'nitrogen_content', 'cation_exchange_capacity', 'ph_in_water', 
                'bulk_density', 'coarse_fragments', 'soil_moisture'
            ]]
            if potential_prod_cols:
                essential_columns.extend(potential_prod_cols[:1])  # Take first one
                print(f"Using {potential_prod_cols[0]} as productivity measure")
        
        # Add climate variables
        climate_cols = [col for col in integrated_df.columns if col in ['aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd', 'gpp_yearly']]
        essential_columns.extend(climate_cols)
        
        # Add soil variables
        soil_cols = [col for col in integrated_df.columns if col in [
            'soil_carbon_stock', 'clay_content', 'silt_content', 'sand_content', 
            'nitrogen_content', 'cation_exchange_capacity', 'ph_in_water', 
            'bulk_density', 'coarse_fragments'
        ]]
        essential_columns.extend(soil_cols)
        
        # Add soil moisture variable
        soil_moisture_cols = [col for col in integrated_df.columns if col in ['soil_moisture']]
        essential_columns.extend(soil_moisture_cols)
        
        # Add elevation variable
        elevation_cols = [col for col in integrated_df.columns if col in ['elevation']]
        essential_columns.extend(elevation_cols)
        
        # Add data source identifier
        if 'data_source' in integrated_df.columns:
            essential_columns.append('data_source')
        
        # Filter the dataframe to keep only essential columns
        available_cols = [col for col in essential_columns if col in integrated_df.columns]
        filtered_df = integrated_df[available_cols].copy()
        
        print(f"Filtered dataset from {integrated_df.shape[1]} to {filtered_df.shape[1]} columns")
        print(f"Essential columns: {available_cols}")
        
        # Write the filtered DataFrame to a CSV file
        output_file = productivity_dir / 'earth' / 'aggregated_data.csv'
        # check if the file exists, if not save it
        if not output_file.exists():
            # Ensure parent directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            filtered_df.to_csv(output_file, index=False)
            print(f"Aggregated data saved to: {output_file}")
        else:
            print(f"File already exists: {output_file}, skipping save.")
        
        # Write only the columns of lat and lon to a separate file
        # this file will be used to extract data from ancillary data sources
        lat_lon_file = productivity_dir / 'earth' / 'lat_lon.csv'
        if not lat_lon_file.exists():
            lat_lon_df = filtered_df[['lat', 'lon']].drop_duplicates()  # Remove duplicates
            lat_lon_df.to_csv(lat_lon_file, index=False)
            print(f"Latitude and longitude data saved to: {lat_lon_file}")
        else:
            print(f"File already exists: {lat_lon_file}, skipping save.")
        
        # Return the filtered DataFrame
        return filtered_df
    else:
        print("No data files were found or processed successfully.")
        return pd.DataFrame()  # Return an empty DataFrame if no files were found


def plot_data_distribution(df):
    """ Plot the spatial distribution of data points

    TODO:
        - Add more map features like lakes, rivers, etc.
        - Customize the legend to show unique PFTs with colors.
        - Add more descriptive titles and labels.

    Parameters:
        df: dataframe
            the dataframe to be plotted.
    """

    # Create a figure with a PlateCarree projection
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Add map features
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.LAKES, alpha=0.5)
    # ax.add_feature(cfeature.LAND, facecolor='lightgray')
    # ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    
    # Set map extent [lon_min, lon_max, lat_min, lat_max]
    ax.set_extent([-180, 180, -90, 90])
    
    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, 
                     linewidth=0.5, 
                     color='gray', 
                     alpha=0.5,
                     linestyle='--')
    gl.top_labels = False  # Remove top labels
    gl.right_labels = False  # Remove right labels
    
    
    # Create a color map for unique PFTs
    unique_cover = df['biome'].unique()
    #colors = plt.cm.Set3(np.linspace(0, 1, len(unique_pfts)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cover)))
    
    # Plot points for each PFT category
    for cover, color in zip(unique_cover, colors):
        #if cover == 'evergreen broadleaf forest':
            mask = df['biome'] == cover
            ax.scatter(df.loc[mask, 'lon'], 
                       df.loc[mask, 'lat'],
                       color=color,
                       s=50,
                       label=cover,
                       transform=ccrs.PlateCarree()
                      )
    
    # Adjust legend position and size
    # ax.legend(title='Type',
    #          bbox_to_anchor=(0.0, 0.1),
    #          loc='lower left',
    #          shadow=True,                # Add shadow
    #          borderpad=1                 # Padding between legend and frame
    #          )
    
    # Set title
    ax.set_title(f'Data ({df.shape[0]}) with BNPP Measurements', fontsize=15)
    
    # Save the figure as a PDF with high resolution
    output_dir = '../productivity/earth'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'aggregated_data_distribution.pdf')
    plt.savefig(output_path, 
                format='pdf',
                dpi=300,              # High resolution
                bbox_inches='tight',  # Prevent cutoff
                facecolor='white',    # White background
                edgecolor='none',     # No edge color
                pad_inches=0.1)       # Add small padding
    print(f"\nMap saved as: {output_path}")
    
    # Display the plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Aggregate data from different sources for machine learning')
    parser.add_argument('--files', '-f', nargs='+', 
                       default=['grassland/grassland_bnpp_data.csv',
                                'forc/ForC_BNPP_root_C_processed.csv',
                                '../ancillary/terraclimate/point_extractions/aet_means.csv',
                                '../ancillary/terraclimate/point_extractions/pet_means.csv',
                                '../ancillary/terraclimate/point_extractions/ppt_means.csv',
                                '../ancillary/terraclimate/point_extractions/tmax_means.csv',
                                '../ancillary/terraclimate/point_extractions/tmin_means.csv',
                                '../ancillary/terraclimate/point_extractions/vpd_means.csv',
                                '../ancillary/glass/point_extractions/GPP_YEARLY_all_points.csv',
                                '../ancillary/soilgrids/soil_carbon_points.csv',
                                '../ancillary/soilgrids/clay_data_points.csv',
                                '../ancillary/soilgrids/silt_data_points.csv',
                                '../ancillary/soilgrids/sand_data_points.csv',
                                '../ancillary/soilgrids/nitrogen_data_points.csv',
                                '../ancillary/soilgrids/cec_data_points.csv',
                                '../ancillary/soilgrids/ph_data_points.csv',
                                '../ancillary/soilgrids/bulk_density_data_points.csv',
                                '../ancillary/soilgrids/coarse_fragments_data_points.csv',
                                '../ancillary/soilmoisture/soil_moisture_points.csv',
                                '../ancillary/elevation_points.csv'],
                       help='List of data files to aggregate (default: grassland, forest, terraclimate, GLASS GPP, SoilGrids, soil moisture, and elevation files)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting the data distribution')
    parser.add_argument('--output-dir', '-o', type=str, default='../productivity',
                       help='Output directory for processed data (default: ../productivity)')
    
    # Outlier detection arguments
    parser.add_argument('--detect-outliers', action='store_true',
                       help='Enable outlier detection and visualization')
    parser.add_argument('--remove-outliers', action='store_true',
                       help='Remove detected outliers from the dataset')
    parser.add_argument('--outlier-methods', nargs='+', 
                       default=['iqr', 'domain'],
                       choices=['iqr', 'z_score', 'modified_z_score', 'domain', 'geographic'],
                       help='Outlier detection methods to use (default: iqr, domain)')
    parser.add_argument('--outlier-combination', type=str, default='union',
                       choices=['union', 'intersection'],
                       help='How to combine multiple outlier detection methods (default: union)')
    parser.add_argument('--bnpp-column', type=str, default='BNPP',
                       help='Column name for BNPP data (default: BNPP)')
    
    args = parser.parse_args()
    
    try:
        # Aggregate data from the specified files
        integrated_data = aggregate_data(args.files)
        
        # Check if the integrated DataFrame is not empty
        if not integrated_data.empty:
            print(f"Integrated DataFrame shape: {integrated_data.shape}")
            print(f"Columns: {list(integrated_data.columns)}")
            print(integrated_data.head())
            
            # Outlier detection and handling
            if args.detect_outliers or args.remove_outliers:
                if args.bnpp_column in integrated_data.columns:
                    print(f"\n{'='*60}")
                    print("OUTLIER DETECTION AND ANALYSIS")
                    print('='*60)
                    
                    # Visualize outliers
                    if args.detect_outliers:
                        print("Creating outlier visualizations...")
                        output_dir = Path(args.output_dir) / 'earth'
                        visualize_outliers(integrated_data, 
                                         column=args.bnpp_column,
                                         outlier_methods=args.outlier_methods,
                                         output_dir=str(output_dir))
                    
                    # Remove outliers if requested
                    if args.remove_outliers:
                        print("Removing outliers from dataset...")
                        cleaned_data, outlier_data, outlier_summary = remove_outliers(
                            integrated_data,
                            column=args.bnpp_column,
                            methods=args.outlier_methods,
                            remove_method=args.outlier_combination
                        )
                        
                        # Save cleaned and outlier datasets
                        output_dir = Path(args.output_dir) / 'earth'
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save cleaned dataset
                        cleaned_file = output_dir / 'aggregated_data_cleaned.csv'
                        cleaned_data.to_csv(cleaned_file, index=False)
                        print(f"Cleaned dataset saved to: {cleaned_file}")
                        
                        # Save outliers dataset
                        outlier_file = output_dir / 'outliers_removed.csv'
                        outlier_data.to_csv(outlier_file, index=False)
                        print(f"Outliers dataset saved to: {outlier_file}")
                        
                        # Save outlier summary
                        summary_file = output_dir / 'outlier_summary.txt'
                        with open(summary_file, 'w') as f:
                            f.write("BNPP Outlier Detection Summary\n")
                            f.write("=" * 40 + "\n\n")
                            for key, value in outlier_summary.items():
                                f.write(f"{key}: {value}\n")
                        print(f"Outlier summary saved to: {summary_file}")
                        
                        # Use cleaned data for further processing
                        integrated_data = cleaned_data
                        print(f"Proceeding with cleaned dataset: {len(integrated_data)} points")
                    
                else:
                    print(f"Warning: BNPP column '{args.bnpp_column}' not found in integrated data.")
                    print(f"Available columns: {list(integrated_data.columns)}")
            
            # Plot the integrated data if needed
            if not args.no_plot:
                plot_data_distribution(integrated_data)
        else:
            print("No data to display.")
            
    except Exception as e:
        print(f"An error occurred during data integration: {str(e)}")


if __name__ == "__main__":
    main()
