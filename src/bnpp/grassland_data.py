"""
Grassland BNPP Data Processing Script

This script downloads and processes global grassland Below-ground Net Primary 
Productivity (BNPP) data from the Dryad repository. It handles data cleaning,
coordinate range processing, and creates comprehensive visualizations.

Data Source: https://datadryad.org/dataset/doi:10.5061/dryad.7sqv9s4vv
Reference: Sun et al. (2022) - Above‐ and belowground net‐primary productivity: 
A field‐based global database of grasslands

Author: ADAM Project
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import requests
import os
from pathlib import Path
from tqdm import tqdm
from zipfile import ZipFile
from glob import glob
from chardet import detect

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_URL = "https://datadryad.org/dataset/doi:10.5061/dryad.7sqv9s4vv"
DOWNLOAD_URL = "http://datadryad.org/api/v2/datasets/doi%253A10.5061%252Fdryad.7sqv9s4vv/download"
OUTPUT_DIR = "../../productivity/grassland"
SOURCE_FILE = f"{OUTPUT_DIR}/source_data.zip"
DATA_DIR = f"{OUTPUT_DIR}/Data_S1"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def download_file(url, output_path):
    """
    Download a file from URL with progress bar and error handling.
    
    Args:
        url (str): URL to download from
        output_path (str): Path where to save the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as file, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
                
        print(f"Download completed: {output_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

def extract_data_files(zip_pattern, extract_path):
    """
    Extract data files from zip archives.
    
    Args:
        zip_pattern (str): Glob pattern to find zip files
        extract_path (str): Path where to extract files
    """
    for zipfile in tqdm(glob(zip_pattern), desc="Unzipping"):
        with ZipFile(zipfile) as fzip:
            fzip.extractall(path=extract_path)

def detect_file_encoding(file_path):
    """
    Detect the encoding of a text file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Detected encoding
    """
    with open(file_path, "rb") as f:
        result = detect(f.read())
        encoding = result['encoding']
        print(f"Detected encoding: {encoding}")
        return encoding

def parse_coordinate_range(coord_str):
    """
    Parse coordinate ranges and return average value.
    Handles formats like: "44.13-45.12", "-110~-110.17", "30.18"
    
    Args:
        coord_str: Coordinate string (may contain ranges)
        
    Returns:
        float: Average coordinate value or NaN if parsing fails
    """
    if pd.isna(coord_str):
        return np.nan
    
    coord_str = str(coord_str).strip()
    
    # Handle positive ranges like "44.13-45.12"
    if '-' in coord_str and not coord_str.startswith('-'):
        parts = coord_str.split('-')
        if len(parts) == 2:
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except ValueError:
                return np.nan
    
    # Handle ranges with ~ separator like "-110~-110.17"
    elif '~' in coord_str:
        parts = coord_str.split('~')
        if len(parts) == 2:
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except ValueError:
                return np.nan
    
    # Handle negative ranges like "-38.02--37.98"
    elif '-' in coord_str and coord_str.count('-') > 1:
        if '--' in coord_str:
            parts = coord_str.split('--')
            if len(parts) == 2:
                try:
                    return (float('-' + parts[0].replace('-', '')) + float('-' + parts[1])) / 2
                except ValueError:
                    return np.nan
    
    # If no range, try to convert to float
    try:
        return float(coord_str)
    except ValueError:
        return np.nan

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================
def process_coordinates(df):
    """
    Process latitude and longitude coordinates, handling ranges.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with processed coordinates
    """
    print("Processing coordinate ranges...")
    
    # Preserve original coordinates
    df['Latitude_orig'] = df['Latitude'].copy()
    df['Longitude_orig'] = df['Longitude'].copy()
    
    # Process coordinate ranges
    df['Latitude'] = df['Latitude'].apply(parse_coordinate_range)
    df['Longitude'] = df['Longitude'].apply(parse_coordinate_range)
    
    # Report processing results
    lat_ranges = df[df['Latitude_orig'].astype(str).str.contains('-|~', na=False)]['Latitude_orig'].nunique()
    lon_ranges = df[df['Longitude_orig'].astype(str).str.contains('-|~', na=False)]['Longitude_orig'].nunique()
    print(f"Processed {lat_ranges} unique latitude ranges")
    print(f"Processed {lon_ranges} unique longitude ranges")
    
    # Show examples
    show_coordinate_examples(df)
    
    return df

def show_coordinate_examples(df):
    """Show examples of coordinate range processing."""
    print("\\nExamples of coordinate range processing:")
    
    lat_range_mask = df['Latitude_orig'].astype(str).str.contains('-|~', na=False)
    lon_range_mask = df['Longitude_orig'].astype(str).str.contains('-|~', na=False)
    
    actual_ranges = df[(lat_range_mask | lon_range_mask) & 
                       ((df['Latitude_orig'].astype(str).str.contains(r'\\d+-\\d+|\\d+~\\d+', na=False)) |
                        (df['Longitude_orig'].astype(str).str.contains(r'\\d+-\\d+|\\d+~\\d+', na=False)))]
    
    if len(actual_ranges) > 0:
        range_examples = actual_ranges[['Location', 'Latitude_orig', 'Latitude', 'Longitude_orig', 'Longitude']].head(3)
        for idx, row in range_examples.iterrows():
            print(f"{row['Location']}: Lat {row['Latitude_orig']} → {row['Latitude']:.4f}, Lon {row['Longitude_orig']} → {row['Longitude']:.4f}")
    else:
        print("No coordinate ranges found to process")

def clean_numeric_columns(df):
    """
    Clean and convert numeric columns to proper data types.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with cleaned numeric columns
    """
    numeric_columns = ['BNPP', 'BNPP_SE', 'ANPP', 'ANPP_SE', 'TNPP', 'MAT', 'MAP']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def extract_bnpp_data(df):
    """
    Extract and process BNPP data with metadata.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Processed BNPP dataset
    """
    # Select relevant columns
    bnpp_columns = ['Entry_ID', 'Site_ID', 'Location', 'Country', 'Continent', 
                   'Latitude', 'Longitude', 'Altitude', 'Sampling year', 
                   'Grassland type', 'Dominant Species', 'BNPP', 'BNPP_SE', 
                   'MAT', 'MAP', 'Sources']
    
    bnpp_data = df[bnpp_columns].copy()
    
    # Remove rows with missing BNPP data
    bnpp_data = bnpp_data.dropna(subset=['BNPP'])
    
    # Add metadata columns
    bnpp_data['Data_Source'] = 'Grassland_Global_Database'
    bnpp_data['Measurement_Type'] = 'BNPP'
    bnpp_data['Units'] = 'g/m²/year'
    
    return bnpp_data

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def analyze_bnpp_data(df, bnpp_data):
    """
    Perform comprehensive analysis of BNPP data.
    
    Args:
        df (pd.DataFrame): Full dataset
        bnpp_data (pd.DataFrame): BNPP-specific dataset
    """
    print(f"\\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\\nBNPP data availability:")
    print(f"Total rows: {len(df)}")
    print(f"Rows with BNPP data: {df['BNPP'].notna().sum()}")
    print(f"Rows missing BNPP data: {df['BNPP'].isna().sum()}")
    
    if df['BNPP'].notna().any():
        print(f"\\nBNPP statistics (g/m²/year):")
        print(f"Mean: {df['BNPP'].mean():.2f}")
        print(f"Median: {df['BNPP'].median():.2f}")
        print(f"Min: {df['BNPP'].min():.2f}")
        print(f"Max: {df['BNPP'].max():.2f}")
        print(f"Standard deviation: {df['BNPP'].std():.2f}")
    
    print(f"\\nSample of processed BNPP data:")
    print(bnpp_data.head())
    
    print(f"\\nGeographic distribution of BNPP data:")
    print(bnpp_data['Continent'].value_counts())
    print(f"\\nCountry distribution:")
    print(bnpp_data['Country'].value_counts().head(10))

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_global_map(bnpp_data):
    """
    Create global distribution map of BNPP data.
    
    Args:
        bnpp_data (pd.DataFrame): BNPP dataset with coordinates
        
    Returns:
        str: Path to saved plot
    """
    print(f"\\nCreating global distribution plot...")
    
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    ax.add_feature(cfeature.LAND, color='lightgray')
    
    # Create scatter plot
    scatter = ax.scatter(bnpp_data['Longitude'], bnpp_data['Latitude'], 
                        c=bnpp_data['BNPP'], cmap='viridis', 
                        s=50, alpha=0.7, edgecolors='black', linewidth=0.5,
                        transform=ccrs.PlateCarree())
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8)
    cbar.set_label('BNPP (g/m²/year)', fontsize=12)
    
    # Set title with point count
    n_points = len(bnpp_data)
    ax.set_title(f'Global Distribution of Grassland Below-ground Net Primary Productivity (BNPP)\\n'
                 f'({n_points} measurement locations)', 
                 fontsize=14, fontweight='bold')
    ax.set_global()
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Add statistics text box
    stats_text = f'''Data Points: {n_points}
Mean BNPP: {bnpp_data['BNPP'].mean():.1f} g/m²/year
Range: {bnpp_data['BNPP'].min():.0f} - {bnpp_data['BNPP'].max():.0f} g/m²/year
Continents: {bnpp_data['Continent'].nunique()}
Countries: {bnpp_data['Country'].nunique()}'''
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    plot_file = f"{OUTPUT_DIR}/grassland_bnpp_global_distribution.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Global distribution plot saved to: {plot_file}")
    
    return plot_file

def create_histograms(bnpp_data):
    """
    Create histograms of BNPP values.
    
    Args:
        bnpp_data (pd.DataFrame): BNPP dataset
        
    Returns:
        str: Path to saved plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Normal histogram
    ax1.hist(bnpp_data['BNPP'], bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('BNPP (g/m²/year)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of BNPP Values')
    ax1.grid(True, alpha=0.3)
    
    # Log-scale histogram
    ax2.hist(bnpp_data['BNPP'], bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('BNPP (g/m²/year)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of BNPP Values (Log Scale)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    hist_file = f"{OUTPUT_DIR}/grassland_bnpp_histogram.png"
    plt.savefig(hist_file, dpi=300, bbox_inches='tight')
    print(f"BNPP histogram saved to: {hist_file}")
    
    return hist_file

def create_continental_comparison(bnpp_data):
    """
    Create continental comparison box plot.
    
    Args:
        bnpp_data (pd.DataFrame): BNPP dataset
        
    Returns:
        str: Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    continents = bnpp_data['Continent'].unique()
    continent_data = [bnpp_data[bnpp_data['Continent'] == cont]['BNPP'].values for cont in continents]
    
    bp = ax.boxplot(continent_data, tick_labels=continents, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    for patch, color in zip(bp['boxes'], colors[:len(continents)]):
        patch.set_facecolor(color)
    
    ax.set_ylabel('BNPP (g/m²/year)')
    ax.set_title('BNPP Distribution by Continent')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    cont_file = f"{OUTPUT_DIR}/grassland_bnpp_by_continent.png"
    plt.savefig(cont_file, dpi=300, bbox_inches='tight')
    print(f"Continental comparison plot saved to: {cont_file}")
    
    return cont_file

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution function."""
    print("=" * 60)
    print("GRASSLAND BNPP DATA PROCESSING")
    print("=" * 60)
    
    # Step 1: Download data
    print("\\n1. DOWNLOADING DATA")
    print("-" * 30)
    success = download_file(DOWNLOAD_URL, SOURCE_FILE)
    if not success:
        print("Failed to download data. Exiting.")
        return
    
    # Step 2: Extract data files
    print("\\n2. EXTRACTING DATA FILES")
    print("-" * 30)
    extract_data_files(f"{OUTPUT_DIR}/*.zip", OUTPUT_DIR)
    
    # Step 3: Load and process data
    print("\\n3. LOADING AND PROCESSING DATA")
    print("-" * 30)
    csv_files = glob(f"{DATA_DIR}/*.csv")
    print(f"CSV files found: {csv_files}")
    
    if not csv_files:
        print("No CSV files found. Exiting.")
        return
    
    # Detect encoding and load data
    encoding = detect_file_encoding(csv_files[0])
    df = pd.read_csv(csv_files[0], encoding=encoding)
    
    # Process coordinates and clean data
    df = process_coordinates(df)
    df = clean_numeric_columns(df)
    
    # Step 4: Extract BNPP data
    print("\\n4. EXTRACTING BNPP DATA")
    print("-" * 30)
    bnpp_data = extract_bnpp_data(df)
    
    # Save processed data
    output_file = f"{OUTPUT_DIR}/grassland_bnpp_data.csv"
    bnpp_data.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Processed BNPP data saved to: {output_file}")
    
    # Step 5: Analyze data
    print("\\n5. DATA ANALYSIS")
    print("-" * 30)
    analyze_bnpp_data(df, bnpp_data)
    
    # Step 6: Create visualizations
    print("\\n6. CREATING VISUALIZATIONS")
    print("-" * 30)
    create_global_map(bnpp_data)
    create_histograms(bnpp_data)
    create_continental_comparison(bnpp_data)
    
    print("\\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total BNPP measurements processed: {len(bnpp_data)}")
    print(f"Geographic coverage: {bnpp_data['Continent'].nunique()} continents, {bnpp_data['Country'].nunique()} countries")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()