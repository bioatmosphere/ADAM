"""
ForC Database BNPP and ANPP Data Processing Script

This script downloads and processes the ForC (Forest Carbon) database from GitHub,
extracting BNPP_root_C (Belowground Net Primary Production) and ANPP_2_C
(Aboveground Net Primary Production) data for analysis. It calculates TNPP (Total NPP)
and BNPP fraction for sites with both measurements.

Data Source: https://github.com/forc-db/ForC
Database: Global forest carbon and flux measurements
Reference: Anderson-Teixeira et al. (2018) ForC: a global database of forest carbon and flux data

Author: ADAM Project
"""

# =============================================================================
# IMPORTS
# =============================================================================
import requests
import zipfile
import io
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
from matplotlib.colors import LogNorm

# =============================================================================
# CONFIGURATION
# =============================================================================
# GitHub repository settings
USER = "forc-db"
REPO = "ForC"
COMMIT_HASH = "407c520e6350917bca42e6bf7d5031dbcc551362"
FOLDER_PATH_IN_REPO = "data"

# Local paths
OUTPUT_DIR = "../../productivity/forc"
DATA_DIR = f"{OUTPUT_DIR}/data"
PROCESSED_FILE = f"{OUTPUT_DIR}/ForC_BNPP_ANPP_TNPP_processed.csv"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"

# Data processing settings
TARGET_VARIABLE_BNPP = "BNPP_root_C"
TARGET_VARIABLE_ANPP = "ANPP_2_C"
ENCODING = "latin-1"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def check_data_exists(data_dir):
    """
    Check if ForC data has already been downloaded and extracted.
    
    Args:
        data_dir (str): Path to the data directory
        
    Returns:
        bool: True if data exists and is complete, False otherwise
    """
    # Check if directory exists
    if not os.path.exists(data_dir):
        return False
    
    # Check for essential ForC files
    essential_files = [
        "ForC_measurements.csv",
        "ForC_sites.csv", 
        "ForC_variables.csv",
        "ForC_methodology.csv"
    ]
    
    for file_name in essential_files:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Missing essential file: {file_name}")
            return False
    
    # Validate file integrity by reading headers
    try:
        measurements_file = os.path.join(data_dir, "ForC_measurements.csv")
        sites_file = os.path.join(data_dir, "ForC_sites.csv")
        
        measurements = pd.read_csv(measurements_file, nrows=1, encoding=ENCODING)
        sites = pd.read_csv(sites_file, nrows=1, encoding=ENCODING)
        
        return True
    except Exception as e:
        print(f"Error reading existing files: {e}")
        return False

def download_file_from_github(user, repo, commit_hash, folder_path, destination_dir):
    """
    Download and extract a specific folder from a GitHub repository.
    
    Args:
        user (str): GitHub username
        repo (str): Repository name
        commit_hash (str): Specific commit hash
        folder_path (str): Folder path within the repository
        destination_dir (str): Local destination directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Construct download URL
        zip_url = f"https://github.com/{user}/{repo}/archive/{commit_hash}.zip"
        zip_folder_prefix = f"{repo}-{commit_hash}/{folder_path}/"
        
        print(f"Downloading from: {zip_url}")
        
        # Download zip file
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()
        print("Download successful. Extracting files...")
        
        # Extract files
        zip_content = io.BytesIO(response.content)
        extracted_count = 0
        
        with zipfile.ZipFile(zip_content, 'r') as zip_ref:
            os.makedirs(destination_dir, exist_ok=True)
            
            for member in zip_ref.infolist():
                if member.filename.startswith(zip_folder_prefix) and not member.filename == zip_folder_prefix:
                    # Calculate relative path
                    relative_path = member.filename.split(zip_folder_prefix, 1)[1]
                    target_path = os.path.join(destination_dir, relative_path)
                    
                    if member.is_dir():
                        os.makedirs(target_path, exist_ok=True)
                    else:
                        # Create parent directory and extract file
                        parent_dir = os.path.dirname(target_path)
                        os.makedirs(parent_dir, exist_ok=True)
                        
                        with zip_ref.open(member) as source, open(target_path, "wb") as target:
                            target.write(source.read())
                        extracted_count += 1
        
        print(f"Successfully extracted {extracted_count} files to: {os.path.abspath(destination_dir)}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
        return False
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip archive.")
        return False
    except Exception as e:
        print(f"Unexpected error during download: {e}")
        return False

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================
def load_forc_data(data_dir):
    """
    Load ForC database files with proper encoding handling.
    
    Args:
        data_dir (str): Path to the ForC data directory
        
    Returns:
        tuple: (measurements, sites, variables, methodology) DataFrames
    """
    try:
        print("Loading ForC database files...")
        
        # Define file paths
        files = {
            'measurements': os.path.join(data_dir, "ForC_measurements.csv"),
            'sites': os.path.join(data_dir, "ForC_sites.csv"),
            'variables': os.path.join(data_dir, "ForC_variables.csv"),
            'methodology': os.path.join(data_dir, "ForC_methodology.csv")
        }
        
        # Load data with encoding handling
        measurements = pd.read_csv(files['measurements'], encoding=ENCODING, low_memory=False)
        sites = pd.read_csv(files['sites'], encoding=ENCODING)
        variables = pd.read_csv(files['variables'], encoding=ENCODING)
        methodology = pd.read_csv(files['methodology'], encoding=ENCODING) if os.path.exists(files['methodology']) else None
        
        print(f"✓ Loaded {len(measurements)} measurements, {len(sites)} sites, {len(variables)} variables")
        if methodology is not None:
            print(f"✓ Loaded {len(methodology)} methodology records")
        
        return measurements, sites, variables, methodology
        
    except Exception as e:
        print(f"Error loading ForC data: {e}")
        return None, None, None, None

def extract_productivity_measurements(measurements, variable_name):
    """
    Extract productivity measurements from the ForC measurements data.

    Args:
        measurements (pd.DataFrame): ForC measurements data
        variable_name (str): Target variable name (e.g., 'BNPP_root_C', 'ANPP_2_C')

    Returns:
        pd.DataFrame: Filtered measurements
    """
    print(f"Extracting {variable_name} measurements...")

    variable_measurements = measurements[measurements['variable.name'] == variable_name].copy()
    print(f"Found {len(variable_measurements)} {variable_name} measurements")

    if len(variable_measurements) == 0:
        print(f"Warning: No {variable_name} measurements found in the database.")

    return variable_measurements

def merge_bnpp_anpp_data(bnpp_data, anpp_data):
    """
    Merge BNPP and ANPP data for sites with both measurements.
    Calculate TNPP and BNPP fraction.

    Args:
        bnpp_data (pd.DataFrame): Processed BNPP data
        anpp_data (pd.DataFrame): Processed ANPP data

    Returns:
        pd.DataFrame: Merged dataset with TNPP and BNPP_fraction
    """
    print("Merging BNPP and ANPP data...")

    # Select key columns from ANPP data to merge
    anpp_columns = ['sites.sitename', 'mean']
    anpp_merge = anpp_data[anpp_columns].copy()
    anpp_merge = anpp_merge.rename(columns={'mean': 'ANPP_2_C'})

    # Group by site and average (in case of multiple measurements per site)
    anpp_merge = anpp_merge.groupby('sites.sitename', as_index=False).mean()

    # Rename BNPP mean column before merge
    bnpp_data_copy = bnpp_data.copy()
    bnpp_data_copy = bnpp_data_copy.rename(columns={'mean': 'BNPP_root_C'})

    # Merge BNPP with ANPP
    merged_data = bnpp_data_copy.merge(
        anpp_merge,
        on='sites.sitename',
        how='left'
    )

    # Calculate TNPP = BNPP + ANPP (for sites with both)
    merged_data['TNPP_C'] = merged_data['BNPP_root_C'] + merged_data['ANPP_2_C']

    # Calculate BNPP fraction = BNPP / TNPP
    merged_data['BNPP_fraction'] = merged_data['BNPP_root_C'] / merged_data['TNPP_C']

    # Count sites with both measurements
    both_count = merged_data['TNPP_C'].notna().sum()
    print(f"✓ Successfully merged data for {both_count} sites with both BNPP and ANPP measurements")
    print(f"  - Mean BNPP: {merged_data['BNPP_root_C'].mean():.3f} Mg C ha-1 yr-1")
    print(f"  - Mean ANPP: {merged_data['ANPP_2_C'].mean():.3f} Mg C ha-1 yr-1 (n={both_count})")
    print(f"  - Mean TNPP: {merged_data['TNPP_C'].mean():.3f} Mg C ha-1 yr-1 (n={both_count})")
    print(f"  - Mean BNPP fraction: {merged_data['BNPP_fraction'].mean():.3f} (n={both_count})")

    return merged_data

def merge_site_data(bnpp_measurements, sites, methodology=None):
    """
    Merge BNPP measurements with site and methodology data.
    
    Args:
        bnpp_measurements (pd.DataFrame): BNPP measurements
        sites (pd.DataFrame): Site information
        methodology (pd.DataFrame, optional): Methodology information
        
    Returns:
        pd.DataFrame: Merged dataset
    """
    print("Merging with site and methodology data...")
    
    # Define site columns to include
    site_columns = [
        'sites.sitename', 'lat', 'lon', 'country', 'continent', 'mat', 'map', 'masl', 
        'climate.notes', 'soil.texture', 'soil.classification', 'Koeppen', 'FAO.ecozone'
    ]
    available_site_columns = [col for col in site_columns if col in sites.columns]
    
    # Merge with sites data
    bnpp_data = bnpp_measurements.merge(
        sites[available_site_columns], 
        on='sites.sitename', 
        how='left'
    )
    
    # Merge with methodology data if available
    if methodology is not None:
        methodology_columns = ['method.ID', 'method.category', 'method.notes']
        available_method_columns = [col for col in methodology_columns if col in methodology.columns]
        
        # Convert method.ID to string for consistent merging
        if 'method.ID' in bnpp_data.columns:
            bnpp_data['method.ID'] = bnpp_data['method.ID'].astype(str)
        if 'method.ID' in methodology.columns:
            methodology['method.ID'] = methodology['method.ID'].astype(str)
        
        bnpp_data = bnpp_data.merge(
            methodology[available_method_columns], 
            on='method.ID', 
            how='left'
        )
    
    return bnpp_data

def clean_and_process_data(bnpp_data):
    """
    Clean and process the merged BNPP dataset.
    
    Args:
        bnpp_data (pd.DataFrame): Raw merged BNPP data
        
    Returns:
        pd.DataFrame: Cleaned and processed dataset
    """
    print("Cleaning and processing BNPP data...")
    
    # Convert numeric columns
    numeric_columns = ['mean', 'lat', 'lon', 'mat', 'map', 'masl', 'stand.age', 'sd', 'se', 'n']
    for col in numeric_columns:
        if col in bnpp_data.columns:
            bnpp_data[col] = pd.to_numeric(bnpp_data[col], errors='coerce')
    
    # Remove rows with missing essential data
    essential_columns = ['mean', 'lat', 'lon']
    bnpp_clean = bnpp_data.dropna(subset=essential_columns).copy()
    
    print(f"Clean BNPP dataset: {len(bnpp_clean)} measurements with complete coordinates")
    
    return bnpp_clean

def add_metadata_columns(productivity_data, variables):
    """
    Add standardized metadata columns to the productivity dataset.

    Args:
        productivity_data (pd.DataFrame): Cleaned productivity data with BNPP, ANPP, TNPP
        variables (pd.DataFrame): Variables information

    Returns:
        pd.DataFrame: Dataset with added metadata
    """
    print("Adding standardized metadata...")

    # Define comprehensive metadata columns (including new productivity columns)
    essential_columns = [
        # Identification
        'measurement.ID', 'sites.sitename', 'plot.name', 'citation.ID',

        # Measurement details
        'variable.name', 'BNPP_root_C', 'ANPP_2_C', 'TNPP_C', 'BNPP_fraction',
        'sd', 'se', 'n', 'original.units',
        'date', 'start.date', 'end.date',

        # Site characteristics
        'lat', 'lon', 'country', 'continent', 'masl',

        # Climate
        'mat', 'map',

        # Vegetation
        'stand.age', 'dominant.life.form', 'dominant.veg', 'scientific.name',

        # Methodology
        'method.ID', 'notes', 'area.sampled', 'depth', 'min.dbh',

        # Quality control
        'conflicts', 'flag.suspicious', 'checked.ori.pub'
    ]

    # Optional columns
    optional_columns = [
        'climate.notes', 'soil.texture', 'soil.classification', 'Koeppen', 'FAO.ecozone',
        'method.category', 'method.notes', 'veg.notes',
        'lower95CI', 'upper95CI', 'covariate_1', 'coV_1.value', 'covariate_2', 'coV_2.value'
    ]

    # Select available columns
    available_columns = [col for col in essential_columns + optional_columns if col in productivity_data.columns]
    productivity_final = productivity_data[available_columns].copy()

    # Add standardized metadata
    productivity_final['Data_Source'] = 'ForC_Database'
    productivity_final['Standard_Units'] = 'Mg C ha-1 yr-1'
    productivity_final['Database_Version'] = COMMIT_HASH[:7]

    return productivity_final

# =============================================================================
# ANALYSIS AND REPORTING FUNCTIONS
# =============================================================================
def analyze_bnpp_data(productivity_data):
    """
    Perform comprehensive analysis of productivity data and display results.

    Args:
        productivity_data (pd.DataFrame): Processed productivity dataset with BNPP, ANPP, TNPP
    """
    print(f"\n{'='*60}")
    print("PRODUCTIVITY DATA ANALYSIS (BNPP, ANPP, TNPP)")
    print(f"{'='*60}")

    # Basic statistics
    print(f"\nOverall Statistics:")
    print(f"Total measurements: {len(productivity_data)}")

    if 'BNPP_root_C' in productivity_data.columns:
        print(f"\nBNPP_root_C Statistics:")
        bnpp_data = productivity_data['BNPP_root_C'].dropna()
        print(f"  N: {len(bnpp_data)}")
        print(f"  Mean: {bnpp_data.mean():.3f} Mg C ha-1 yr-1")
        print(f"  Median: {bnpp_data.median():.3f} Mg C ha-1 yr-1")
        print(f"  Min: {bnpp_data.min():.3f} Mg C ha-1 yr-1")
        print(f"  Max: {bnpp_data.max():.3f} Mg C ha-1 yr-1")
        print(f"  Std Dev: {bnpp_data.std():.3f} Mg C ha-1 yr-1")

    if 'ANPP_2_C' in productivity_data.columns:
        print(f"\nANPP_2_C Statistics:")
        anpp_data = productivity_data['ANPP_2_C'].dropna()
        print(f"  N: {len(anpp_data)}")
        print(f"  Mean: {anpp_data.mean():.3f} Mg C ha-1 yr-1")
        print(f"  Median: {anpp_data.median():.3f} Mg C ha-1 yr-1")
        print(f"  Min: {anpp_data.min():.3f} Mg C ha-1 yr-1")
        print(f"  Max: {anpp_data.max():.3f} Mg C ha-1 yr-1")
        print(f"  Std Dev: {anpp_data.std():.3f} Mg C ha-1 yr-1")

    if 'TNPP_C' in productivity_data.columns:
        print(f"\nTNPP_C Statistics (BNPP + ANPP):")
        tnpp_data = productivity_data['TNPP_C'].dropna()
        print(f"  N: {len(tnpp_data)} (sites with both BNPP and ANPP)")
        print(f"  Mean: {tnpp_data.mean():.3f} Mg C ha-1 yr-1")
        print(f"  Median: {tnpp_data.median():.3f} Mg C ha-1 yr-1")
        print(f"  Min: {tnpp_data.min():.3f} Mg C ha-1 yr-1")
        print(f"  Max: {tnpp_data.max():.3f} Mg C ha-1 yr-1")
        print(f"  Std Dev: {tnpp_data.std():.3f} Mg C ha-1 yr-1")

    if 'BNPP_fraction' in productivity_data.columns:
        print(f"\nBNPP Fraction (BNPP/TNPP) Statistics:")
        frac_data = productivity_data['BNPP_fraction'].dropna()
        print(f"  N: {len(frac_data)}")
        print(f"  Mean: {frac_data.mean():.3f}")
        print(f"  Median: {frac_data.median():.3f}")
        print(f"  Min: {frac_data.min():.3f}")
        print(f"  Max: {frac_data.max():.3f}")
        print(f"  Std Dev: {frac_data.std():.3f}")
    
    # Geographic distribution
    if 'continent' in productivity_data.columns:
        print(f"\nGeographic distribution:")
        print(productivity_data['continent'].value_counts())

    if 'country' in productivity_data.columns:
        print(f"\nTop 10 countries:")
        print(productivity_data['country'].value_counts().head(10))
    
    # Metadata completeness
    print(f"\nMetadata completeness:")
    print(f"Total columns: {len(productivity_data.columns)}")

    key_metadata_fields = ['stand.age', 'dominant.life.form', 'mat', 'map', 'method.ID', 'soil.texture']
    for field in key_metadata_fields:
        if field in productivity_data.columns:
            non_null_count = productivity_data[field].notna().sum()
            completeness = (non_null_count / len(productivity_data)) * 100
            print(f"  {field}: {non_null_count}/{len(productivity_data)} ({completeness:.1f}%)")

    # Temporal coverage
    if 'date' in productivity_data.columns:
        date_info = productivity_data['date'].dropna()
        if len(date_info) > 0:
            print(f"\nTemporal coverage:")
            print(f"  Date information: {len(date_info)}/{len(productivity_data)} measurements")
            try:
                years = pd.to_numeric(date_info, errors='coerce')
                valid_years = years.dropna()
                if len(valid_years) > 0:
                    print(f"  Year range: {int(valid_years.min())} - {int(valid_years.max())}")
            except:
                print(f"  Date formats vary")

    # Methodology coverage
    if 'method.ID' in productivity_data.columns:
        method_counts = productivity_data['method.ID'].value_counts()
        print(f"\nMethodology coverage:")
        print(f"  Unique methods: {len(method_counts)}")
        print(f"  Top 5 methods:")
        for method_id, count in method_counts.head().items():
            print(f"    Method {method_id}: {count} measurements")

def display_sample_data(productivity_data):
    """
    Display sample of processed data.

    Args:
        productivity_data (pd.DataFrame): Processed productivity dataset
    """
    print(f"\nSample of processed data:")
    sample_columns = ['sites.sitename', 'country', 'lat', 'lon', 'BNPP_root_C', 'ANPP_2_C',
                      'TNPP_C', 'BNPP_fraction', 'stand.age', 'dominant.life.form']
    available_sample_columns = [col for col in sample_columns if col in productivity_data.columns]
    print(productivity_data[available_sample_columns].head(10))

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_global_bnpp_map(productivity_data, output_dir):
    """
    Create a global map showing BNPP measurement locations and BNPP fraction.

    Args:
        productivity_data (pd.DataFrame): Processed productivity dataset
        output_dir (str): Directory to save the figure
    """
    if 'lat' not in productivity_data.columns or 'lon' not in productivity_data.columns:
        print("Warning: No coordinate data available for mapping.")
        return

    # Create figures directory
    os.makedirs(output_dir, exist_ok=True)

    # Set up the map
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)

    # Plot BNPP measurements colored by BNPP fraction
    if 'BNPP_fraction' in productivity_data.columns:
        # Use color scale for BNPP fraction values
        plot_data = productivity_data.dropna(subset=['BNPP_fraction'])
        scatter = ax.scatter(plot_data['lon'], plot_data['lat'],
                           c=plot_data['BNPP_fraction'],
                           cmap='RdYlGn_r',
                           s=50,
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5,
                           vmin=0, vmax=1,
                           transform=ccrs.PlateCarree())

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('BNPP Fraction (BNPP/TNPP)', fontsize=12)

    elif 'BNPP_root_C' in productivity_data.columns:
        # Fallback to BNPP values
        plot_data = productivity_data.dropna(subset=['BNPP_root_C'])
        scatter = ax.scatter(plot_data['lon'], plot_data['lat'],
                           c=plot_data['BNPP_root_C'],
                           cmap='viridis',
                           s=50,
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5,
                           transform=ccrs.PlateCarree())

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('BNPP (Mg C ha⁻¹ yr⁻¹)', fontsize=12)
    else:
        # Simple point plot if no values available
        ax.scatter(productivity_data['lon'], productivity_data['lat'],
                  c='red', s=20, alpha=0.7,
                  transform=ccrs.PlateCarree())

    # Set global extent
    ax.set_global()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Title and labels
    tnpp_sites = productivity_data['TNPP_C'].notna().sum()
    plt.title(f'Global Distribution of ForC Productivity Measurements\n'
              f'({len(productivity_data)} BNPP sites, {tnpp_sites} with TNPP)',
              fontsize=16, fontweight='bold')

    # Save the figure
    output_file = os.path.join(output_dir, 'ForC_BNPP_ANPP_global_map.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Global map saved to: {output_file}")
    plt.close()

def create_bnpp_distribution_plots(productivity_data, output_dir):
    """
    Create statistical distribution plots for productivity data.

    Args:
        productivity_data (pd.DataFrame): Processed productivity dataset
        output_dir (str): Directory to save the figures
    """
    # Create figures directory
    os.makedirs(output_dir, exist_ok=True)

    # Set up subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ForC Productivity Data Distribution Analysis', fontsize=16, fontweight='bold')

    # 1. Histogram of BNPP values
    ax1 = axes[0, 0]
    if 'BNPP_root_C' in productivity_data.columns:
        bnpp_data = productivity_data['BNPP_root_C'].dropna()
        ax1.hist(bnpp_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('BNPP (Mg C ha⁻¹ yr⁻¹)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'BNPP Distribution (n={len(bnpp_data)})')
        ax1.grid(True, alpha=0.3)

    # 2. Histogram of ANPP values
    ax2 = axes[0, 1]
    if 'ANPP_2_C' in productivity_data.columns:
        anpp_data = productivity_data['ANPP_2_C'].dropna()
        ax2.hist(anpp_data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('ANPP (Mg C ha⁻¹ yr⁻¹)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'ANPP Distribution (n={len(anpp_data)})')
        ax2.grid(True, alpha=0.3)

    # 3. Histogram of TNPP values
    ax3 = axes[0, 2]
    if 'TNPP_C' in productivity_data.columns:
        tnpp_data = productivity_data['TNPP_C'].dropna()
        ax3.hist(tnpp_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('TNPP (Mg C ha⁻¹ yr⁻¹)')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'TNPP Distribution (n={len(tnpp_data)})')
        ax3.grid(True, alpha=0.3)

    # 4. BNPP fraction histogram
    ax4 = axes[1, 0]
    if 'BNPP_fraction' in productivity_data.columns:
        frac_data = productivity_data['BNPP_fraction'].dropna()
        ax4.hist(frac_data, bins=30, alpha=0.7, color='gold', edgecolor='black')
        ax4.set_xlabel('BNPP Fraction (BNPP/TNPP)')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'BNPP Fraction Distribution (n={len(frac_data)})')
        ax4.axvline(frac_data.mean(), color='red', linestyle='--', label=f'Mean: {frac_data.mean():.3f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. BNPP vs ANPP scatter
    ax5 = axes[1, 1]
    if 'BNPP_root_C' in productivity_data.columns and 'ANPP_2_C' in productivity_data.columns:
        plot_data = productivity_data.dropna(subset=['BNPP_root_C', 'ANPP_2_C'])
        ax5.scatter(plot_data['ANPP_2_C'], plot_data['BNPP_root_C'], alpha=0.6, color='purple')
        ax5.set_xlabel('ANPP (Mg C ha⁻¹ yr⁻¹)')
        ax5.set_ylabel('BNPP (Mg C ha⁻¹ yr⁻¹)')
        ax5.set_title(f'BNPP vs ANPP (n={len(plot_data)})')
        # Add 1:1 line
        max_val = max(plot_data['ANPP_2_C'].max(), plot_data['BNPP_root_C'].max())
        ax5.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='1:1 line')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 6. TNPP by continent
    ax6 = axes[1, 2]
    if 'TNPP_C' in productivity_data.columns and 'continent' in productivity_data.columns:
        plot_data = productivity_data.dropna(subset=['TNPP_C', 'continent'])
        if len(plot_data) > 0:
            sns.boxplot(data=plot_data, x='continent', y='TNPP_C', ax=ax6)
            ax6.set_xlabel('Continent')
            ax6.set_ylabel('TNPP (Mg C ha⁻¹ yr⁻¹)')
            ax6.set_title('TNPP by Continent')
            ax6.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save the figure
    output_file = os.path.join(output_dir, 'ForC_productivity_distributions.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Distribution plots saved to: {output_file}")
    plt.close()

def create_data_coverage_plots(bnpp_data, output_dir):
    """
    Create plots showing data coverage and completeness.
    
    Args:
        bnpp_data (pd.DataFrame): Processed BNPP dataset
        output_dir (str): Directory to save the figures
    """
    # Create figures directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ForC BNPP Data Coverage Analysis', fontsize=16, fontweight='bold')
    
    # 1. Data completeness heatmap
    ax1 = axes[0, 0]
    key_columns = ['mean', 'lat', 'lon', 'stand.age', 'mat', 'map', 'dominant.life.form', 'method.ID']
    available_columns = [col for col in key_columns if col in bnpp_data.columns]
    
    if available_columns:
        completeness_data = bnpp_data[available_columns].notna().mean()
        completeness_df = pd.DataFrame({
            'Variable': completeness_data.index,
            'Completeness': completeness_data.values
        })
        
        bars = ax1.bar(range(len(completeness_df)), completeness_df['Completeness'], 
                      color='lightcoral', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Variables')
        ax1.set_ylabel('Data Completeness (%)')
        ax1.set_title('Metadata Completeness')
        ax1.set_xticks(range(len(completeness_df)))
        ax1.set_xticklabels(completeness_df['Variable'], rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, completeness_df['Completeness']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{pct:.1%}', ha='center', va='bottom', fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'No key columns available', 
                transform=ax1.transAxes, ha='center', va='center')
    
    # 2. Geographic coverage by country
    ax2 = axes[0, 1]
    if 'country' in bnpp_data.columns:
        country_counts = bnpp_data['country'].value_counts().head(10)
        if len(country_counts) > 0:
            bars = ax2.bar(range(len(country_counts)), country_counts.values, 
                          color='lightblue', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Countries')
            ax2.set_ylabel('Number of Measurements')
            ax2.set_title('Top 10 Countries by Measurement Count')
            ax2.set_xticks(range(len(country_counts)))
            ax2.set_xticklabels(country_counts.index, rotation=45, ha='right')
            
            # Add count labels on bars
            for bar, count in zip(bars, country_counts.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{count}', ha='center', va='bottom', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No country data available', 
                    transform=ax2.transAxes, ha='center', va='center')
    else:
        ax2.text(0.5, 0.5, 'No country data available', 
                transform=ax2.transAxes, ha='center', va='center')
    
    # 3. Forest type distribution
    ax3 = axes[1, 0]
    if 'dominant.life.form' in bnpp_data.columns:
        forest_types = bnpp_data['dominant.life.form'].value_counts().head(8)
        if len(forest_types) > 0:
            ax3.pie(forest_types.values, labels=forest_types.index, autopct='%1.1f%%',
                   startangle=90, colors=plt.cm.Set3(range(len(forest_types))))
            ax3.set_title('Distribution by Forest Type')
        else:
            ax3.text(0.5, 0.5, 'No forest type data available', 
                    transform=ax3.transAxes, ha='center', va='center')
    else:
        ax3.text(0.5, 0.5, 'No forest type data available', 
                transform=ax3.transAxes, ha='center', va='center')
    
    # 4. Temporal coverage
    ax4 = axes[1, 1]
    if 'date' in bnpp_data.columns:
        date_info = bnpp_data['date'].dropna()
        if len(date_info) > 0:
            try:
                # Try to convert to years
                years = pd.to_numeric(date_info, errors='coerce').dropna()
                if len(years) > 0:
                    ax4.hist(years, bins=20, alpha=0.7, color='gold', edgecolor='black')
                    ax4.set_xlabel('Year')
                    ax4.set_ylabel('Number of Measurements')
                    ax4.set_title('Temporal Distribution of Measurements')
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'Cannot parse date information', 
                            transform=ax4.transAxes, ha='center', va='center')
            except:
                ax4.text(0.5, 0.5, 'Cannot parse date information', 
                        transform=ax4.transAxes, ha='center', va='center')
        else:
            ax4.text(0.5, 0.5, 'No date data available', 
                    transform=ax4.transAxes, ha='center', va='center')
    else:
        ax4.text(0.5, 0.5, 'No date data available', 
                transform=ax4.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(output_dir, 'ForC_BNPP_coverage.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Coverage plots saved to: {output_file}")
    plt.close()

def create_all_visualizations(bnpp_data):
    """
    Create all visualization plots for BNPP data.
    
    Args:
        bnpp_data (pd.DataFrame): Processed BNPP dataset
    """
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Create figures directory
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Create all plots
    create_global_bnpp_map(bnpp_data, FIGURES_DIR)
    create_bnpp_distribution_plots(bnpp_data, FIGURES_DIR)
    create_data_coverage_plots(bnpp_data, FIGURES_DIR)
    
    print(f"\n✓ All visualizations created and saved to: {FIGURES_DIR}")
    print(f"  - Global map: ForC_BNPP_global_map.png")
    print(f"  - Distribution plots: ForC_BNPP_distributions.png")
    print(f"  - Coverage analysis: ForC_BNPP_coverage.png")

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================
def process_bnpp_data(data_dir):
    """
    Main function to process BNPP_root_C and ANPP_2_C data from ForC database.
    Calculates TNPP and BNPP fraction.

    Args:
        data_dir (str): Path to the ForC data directory
    """
    try:
        print("\n" + "="*60)
        print("PROCESSING FORC PRODUCTIVITY DATA (BNPP, ANPP, TNPP)")
        print("="*60)

        # Load ForC data
        measurements, sites, variables, methodology = load_forc_data(data_dir)
        if measurements is None:
            return False

        # Extract BNPP measurements
        bnpp_measurements = extract_productivity_measurements(measurements, TARGET_VARIABLE_BNPP)
        if len(bnpp_measurements) == 0:
            return False

        # Merge BNPP with site and methodology data
        bnpp_data = merge_site_data(bnpp_measurements, sites, methodology)

        # Clean BNPP data
        bnpp_clean = clean_and_process_data(bnpp_data)

        # Extract ANPP measurements
        anpp_measurements = extract_productivity_measurements(measurements, TARGET_VARIABLE_ANPP)

        # Merge ANPP with site data (simpler, just need coordinates and site info)
        anpp_data = merge_site_data(anpp_measurements, sites, methodology)

        # Clean ANPP data
        anpp_clean = clean_and_process_data(anpp_data)

        # Merge BNPP and ANPP data, calculate TNPP and BNPP_fraction
        productivity_merged = merge_bnpp_anpp_data(bnpp_clean, anpp_clean)

        # Add metadata columns
        productivity_final = add_metadata_columns(productivity_merged, variables)

        # Save processed data
        os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
        productivity_final.to_csv(PROCESSED_FILE, index=False)
        print(f"\nProcessed productivity data saved to: {PROCESSED_FILE}")

        # Display analysis and sample data
        analyze_bnpp_data(productivity_final)
        display_sample_data(productivity_final)

        # Create visualizations
        create_all_visualizations(productivity_final)

        # Success summary
        print(f"\n{'='*60}")
        print("FORC PRODUCTIVITY PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Comprehensive productivity data extraction completed")
        print(f"✓ {len(productivity_final)} BNPP measurements")
        print(f"✓ {productivity_final['TNPP_C'].notna().sum()} sites with both BNPP and ANPP (TNPP calculated)")
        print(f"✓ {len(productivity_final.columns)} metadata fields")
        print(f"✓ Data saved to: {os.path.basename(PROCESSED_FILE)}")
        print(f"✓ Visualizations saved to: {FIGURES_DIR}")

        return True

    except Exception as e:
        print(f"Error processing productivity data: {e}")
        import traceback
        traceback.print_exc()
        return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution function."""
    print("="*60)
    print("FORC DATABASE PRODUCTIVITY EXTRACTION (BNPP + ANPP + TNPP)")
    print("="*60)
    print(f"Target variables: {TARGET_VARIABLE_BNPP}, {TARGET_VARIABLE_ANPP}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 60)

    # Check if data already exists
    data_already_exists = check_data_exists(DATA_DIR)

    if data_already_exists:
        print("✓ ForC data already exists and appears complete.")
        print(f"✓ Data location: {os.path.abspath(DATA_DIR)}")
        print("✓ Skipping download, proceeding to productivity processing...")
        print("-" * 60)

        # Process productivity data directly
        success = process_bnpp_data(DATA_DIR)

    else:
        print("ForC data not found or incomplete. Starting download...")
        print(f"Repository: {USER}/{REPO}")
        print(f"Commit: {COMMIT_HASH}")
        print("-" * 60)

        # Download ForC data
        download_success = download_file_from_github(
            USER, REPO, COMMIT_HASH, FOLDER_PATH_IN_REPO, DATA_DIR
        )

        if download_success:
            # Process productivity data after successful download
            success = process_bnpp_data(DATA_DIR)
        else:
            print("Failed to download ForC data. Exiting.")
            success = False

    # Final status
    if success:
        print(f"\n🎉 ForC productivity extraction completed successfully!")
        print(f"📁 Processed data: {PROCESSED_FILE}")
    else:
        print(f"\n❌ ForC productivity extraction failed.")

    return success

if __name__ == "__main__":
    main()