"""
ForC Database - ALL Productivity Data Extraction

Extends ForC_data.py to extract ALL productivity variables corresponding to BNPP measurements.
This creates a comprehensive productivity dataset including:
- BNPP (belowground)
- ANPP (aboveground)
- NPP (net primary production)
- GPP (gross primary production)
- TBCF (total belowground carbon flux)

For each site with BNPP_root_C measurements, this script extracts ALL available
productivity variables, creating a wide-format dataset for comprehensive analysis.

Author: ADAM Project
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Import from original ForC_data.py
from ForC_data import (
    check_data_exists,
    download_file_from_github,
    load_forc_data,
    DATA_DIR,
    OUTPUT_DIR,
    USER,
    REPO,
    COMMIT_HASH,
    FOLDER_PATH_IN_REPO,
    ENCODING
)

# Configuration
PROCESSED_FILE = f"{OUTPUT_DIR}/ForC_all_productivity_data.csv"
FIGURES_DIR = f"{OUTPUT_DIR}/figures_productivity"

# Define productivity variable groups
PRODUCTIVITY_VARIABLES = {
    'BNPP': [
        'BNPP_root_C', 'BNPP_root_OM',
        'BNPP_root_coarse_C', 'BNPP_root_coarse_OM',
        'BNPP_root_fine_C', 'BNPP_root_fine_OM',
        'BNPP_root.turnover_fine_C', 'BNPP_root.turnover_fine_OM'
    ],
    'ANPP': [
        'ANPP_0_C', 'ANPP_1_C', 'ANPP_2_C',
        'ANPP_foliage_C', 'ANPP_woody_C', 'ANPP_woody_stem_C', 'ANPP_woody_branch_C',
        'ANPP_litterfall_0_C', 'ANPP_litterfall_1_C', 'ANPP_litterfall_2_C',
        'ANPP_repro_C', 'ANPP_folivory_C'
    ],
    'NPP': [
        'NPP_0_C', 'NPP_1_C', 'NPP_2_C', 'NPP_3_C', 'NPP_4_C', 'NPP_5_C',
        'NPP_woody_C', 'NPP_understory_C', 'NPP_litter_C'
    ],
    'GPP': ['GPP_C', 'GPP_cum_C'],
    'TBCF': ['TBCF_C']
}

# Flatten list for easy checking
ALL_PRODUCTIVITY_VARS = []
for var_list in PRODUCTIVITY_VARIABLES.values():
    ALL_PRODUCTIVITY_VARS.extend(var_list)


def get_bnpp_sites(measurements):
    """
    Get list of sites that have BNPP_root_C measurements.

    Args:
        measurements (pd.DataFrame): ForC measurements data

    Returns:
        list: Site names with BNPP measurements
    """
    bnpp_sites = measurements[
        measurements['variable.name'] == 'BNPP_root_C'
    ]['sites.sitename'].unique()

    print(f"Found {len(bnpp_sites)} unique sites with BNPP_root_C measurements")
    return bnpp_sites


def extract_all_productivity_for_sites(measurements, sites, bnpp_sites, variables, methodology=None):
    """
    Extract ALL productivity variables for sites with BNPP measurements.

    Args:
        measurements (pd.DataFrame): ForC measurements data
        sites (pd.DataFrame): Site information
        bnpp_sites (list): Sites with BNPP measurements
        variables (pd.DataFrame): Variable definitions
        methodology (pd.DataFrame, optional): Methodology information

    Returns:
        pd.DataFrame: Comprehensive productivity dataset
    """
    print(f"\n{'='*60}")
    print("EXTRACTING ALL PRODUCTIVITY DATA")
    print(f"{'='*60}")

    # Filter measurements for productivity variables at BNPP sites
    productivity_mask = (
        measurements['variable.name'].isin(ALL_PRODUCTIVITY_VARS) &
        measurements['sites.sitename'].isin(bnpp_sites)
    )

    productivity_data = measurements[productivity_mask].copy()
    print(f"\nExtracted {len(productivity_data)} productivity measurements")

    # Count by variable type
    print(f"\nBreakdown by productivity type:")
    for prod_type, var_list in PRODUCTIVITY_VARIABLES.items():
        count = productivity_data[productivity_data['variable.name'].isin(var_list)].shape[0]
        unique_sites = productivity_data[
            productivity_data['variable.name'].isin(var_list)
        ]['sites.sitename'].nunique()
        print(f"  {prod_type}: {count} measurements from {unique_sites} sites")

    # Merge with site data
    site_columns = [
        'sites.sitename', 'lat', 'lon', 'country', 'continent', 'mat', 'map', 'masl',
        'climate.notes', 'soil.texture', 'soil.classification', 'Koeppen', 'FAO.ecozone'
    ]
    available_site_columns = [col for col in site_columns if col in sites.columns]

    productivity_enriched = productivity_data.merge(
        sites[available_site_columns],
        on='sites.sitename',
        how='left'
    )

    # Merge with methodology if available
    if methodology is not None:
        methodology_columns = ['method.ID', 'method.category', 'method.notes']
        available_method_columns = [col for col in methodology_columns if col in methodology.columns]

        if 'method.ID' in productivity_enriched.columns:
            productivity_enriched['method.ID'] = productivity_enriched['method.ID'].astype(str)
        if 'method.ID' in methodology.columns:
            methodology['method.ID'] = methodology['method.ID'].astype(str)

        productivity_enriched = productivity_enriched.merge(
            methodology[available_method_columns],
            on='method.ID',
            how='left'
        )

    return productivity_enriched


def create_wide_format_dataset(productivity_data):
    """
    Pivot productivity data to wide format: one row per site, columns for each variable.

    Args:
        productivity_data (pd.DataFrame): Long-format productivity data

    Returns:
        pd.DataFrame: Wide-format dataset
    """
    print(f"\n{'='*60}")
    print("CREATING WIDE-FORMAT DATASET")
    print(f"{'='*60}")

    # Convert numeric columns
    numeric_columns = ['mean', 'lat', 'lon', 'mat', 'map', 'masl', 'stand.age', 'sd', 'se', 'n']
    for col in numeric_columns:
        if col in productivity_data.columns:
            productivity_data[col] = pd.to_numeric(productivity_data[col], errors='coerce')

    # For wide format, we'll take the mean if multiple measurements exist
    # Key columns to identify unique sites
    id_columns = ['sites.sitename', 'lat', 'lon', 'country', 'continent',
                  'mat', 'map', 'masl', 'dominant.life.form', 'dominant.veg']

    # Available ID columns
    available_id_cols = [col for col in id_columns if col in productivity_data.columns]

    # Pivot: create columns for each variable.name
    pivot_data = productivity_data.pivot_table(
        index=available_id_cols,
        columns='variable.name',
        values='mean',
        aggfunc='mean'  # Average if multiple measurements per site
    ).reset_index()

    print(f"\nWide-format dataset created:")
    print(f"  Sites: {len(pivot_data)}")
    print(f"  Productivity variables: {len([col for col in pivot_data.columns if col in ALL_PRODUCTIVITY_VARS])}")

    # Calculate data completeness
    print(f"\n  Data completeness for key variables:")
    key_vars = ['BNPP_root_C', 'ANPP_1_C', 'ANPP_2_C', 'NPP_1_C', 'NPP_2_C', 'GPP_C']
    for var in key_vars:
        if var in pivot_data.columns:
            completeness = pivot_data[var].notna().sum()
            pct = (completeness / len(pivot_data)) * 100
            print(f"    {var}: {completeness}/{len(pivot_data)} ({pct:.1f}%)")

    return pivot_data


def create_long_format_dataset(productivity_data):
    """
    Create long-format dataset with comprehensive metadata.

    Args:
        productivity_data (pd.DataFrame): Productivity data

    Returns:
        pd.DataFrame: Long-format dataset
    """
    print(f"\n{'='*60}")
    print("CREATING LONG-FORMAT DATASET")
    print(f"{'='*60}")

    # Convert numeric columns
    numeric_columns = ['mean', 'lat', 'lon', 'mat', 'map', 'masl', 'stand.age', 'sd', 'se', 'n']
    for col in numeric_columns:
        if col in productivity_data.columns:
            productivity_data[col] = pd.to_numeric(productivity_data[col], errors='coerce')

    # Remove rows with missing essential data
    essential_columns = ['mean', 'lat', 'lon', 'variable.name']
    long_clean = productivity_data.dropna(subset=essential_columns).copy()

    # Add productivity type categorization
    def categorize_variable(var_name):
        for prod_type, var_list in PRODUCTIVITY_VARIABLES.items():
            if var_name in var_list:
                return prod_type
        return 'Other'

    long_clean['productivity_type'] = long_clean['variable.name'].apply(categorize_variable)

    # Add standardized metadata
    long_clean['Data_Source'] = 'ForC_Database'
    long_clean['Database_Version'] = COMMIT_HASH[:7]
    long_clean['Dataset'] = 'All_Productivity_Variables'

    print(f"\nLong-format dataset created:")
    print(f"  Total measurements: {len(long_clean)}")
    print(f"  Unique sites: {long_clean['sites.sitename'].nunique()}")
    print(f"  Unique variables: {long_clean['variable.name'].nunique()}")

    return long_clean


def analyze_productivity_relationships(wide_data):
    """
    Analyze relationships between different productivity variables.

    Args:
        wide_data (pd.DataFrame): Wide-format productivity data
    """
    print(f"\n{'='*60}")
    print("PRODUCTIVITY RELATIONSHIPS ANALYSIS")
    print(f"{'='*60}")

    # Key relationships to analyze
    relationships = [
        ('BNPP_root_C', 'ANPP_1_C', 'BNPP vs ANPP'),
        ('BNPP_root_C', 'NPP_1_C', 'BNPP vs NPP'),
        ('BNPP_root_C', 'GPP_C', 'BNPP vs GPP'),
        ('ANPP_1_C', 'NPP_1_C', 'ANPP vs NPP'),
        ('NPP_1_C', 'GPP_C', 'NPP vs GPP')
    ]

    print("\nCorrelations between productivity variables:")
    for var1, var2, label in relationships:
        if var1 in wide_data.columns and var2 in wide_data.columns:
            # Get complete cases
            complete_data = wide_data[[var1, var2]].dropna()
            if len(complete_data) >= 10:
                corr = complete_data[var1].corr(complete_data[var2])
                print(f"  {label}: r = {corr:.3f} (n = {len(complete_data)})")
            else:
                print(f"  {label}: Insufficient data (n = {len(complete_data)})")

    # BNPP/NPP ratios
    if 'BNPP_root_C' in wide_data.columns and 'NPP_1_C' in wide_data.columns:
        ratio_data = wide_data[['BNPP_root_C', 'NPP_1_C']].dropna()
        ratio_data['BNPP_NPP_ratio'] = ratio_data['BNPP_root_C'] / ratio_data['NPP_1_C']

        print(f"\nBNPP/NPP Ratio Statistics (n = {len(ratio_data)}):")
        print(f"  Mean: {ratio_data['BNPP_NPP_ratio'].mean():.3f}")
        print(f"  Median: {ratio_data['BNPP_NPP_ratio'].median():.3f}")
        print(f"  Range: {ratio_data['BNPP_NPP_ratio'].min():.3f} - {ratio_data['BNPP_NPP_ratio'].max():.3f}")
        print(f"  Std: {ratio_data['BNPP_NPP_ratio'].std():.3f}")


def create_productivity_correlation_matrix(wide_data, output_dir):
    """
    Create correlation matrix heatmap for productivity variables.

    Args:
        wide_data (pd.DataFrame): Wide-format productivity data
        output_dir (str): Directory to save figure
    """
    os.makedirs(output_dir, exist_ok=True)

    # Select key productivity variables that exist in data
    key_vars = [
        'BNPP_root_C', 'ANPP_0_C', 'ANPP_1_C', 'ANPP_2_C',
        'NPP_1_C', 'NPP_2_C', 'GPP_C', 'TBCF_C'
    ]
    available_vars = [var for var in key_vars if var in wide_data.columns]

    if len(available_vars) < 3:
        print("Not enough variables for correlation matrix")
        return

    # Calculate correlation matrix
    corr_data = wide_data[available_vars].corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)

    plt.title('Correlation Matrix: Forest Productivity Variables\n(ForC Database)',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'productivity_correlation_matrix.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Correlation matrix saved to: {output_file}")
    plt.close()


def create_productivity_scatter_plots(wide_data, output_dir):
    """
    Create scatter plots showing relationships between productivity variables.

    Args:
        wide_data (pd.DataFrame): Wide-format productivity data
        output_dir (str): Directory to save figure
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define key relationships to plot
    plot_configs = [
        ('BNPP_root_C', 'ANPP_1_C', 'BNPP vs ANPP'),
        ('BNPP_root_C', 'NPP_1_C', 'BNPP vs NPP'),
        ('BNPP_root_C', 'GPP_C', 'BNPP vs GPP'),
        ('ANPP_1_C', 'GPP_C', 'ANPP vs GPP')
    ]

    # Filter to available relationships
    available_plots = [
        (x, y, label) for x, y, label in plot_configs
        if x in wide_data.columns and y in wide_data.columns
    ]

    if len(available_plots) == 0:
        print("No variable pairs available for scatter plots")
        return

    # Create subplot grid
    n_plots = len(available_plots)
    n_rows = (n_plots + 1) // 2
    n_cols = 2 if n_plots > 1 else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (var_x, var_y, label) in enumerate(available_plots):
        ax = axes[idx]

        # Get complete cases
        plot_data = wide_data[[var_x, var_y]].dropna()

        if len(plot_data) >= 10:
            # Scatter plot
            ax.scatter(plot_data[var_x], plot_data[var_y], alpha=0.6, s=50, color='forestgreen')

            # Add trend line
            z = np.polyfit(plot_data[var_x], plot_data[var_y], 1)
            p = np.poly1d(z)
            x_line = np.linspace(plot_data[var_x].min(), plot_data[var_x].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            # Correlation
            corr = plot_data[var_x].corr(plot_data[var_y])
            ax.text(0.05, 0.95, f'r = {corr:.3f}\nn = {len(plot_data)}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel(f'{var_x} (Mg C ha⁻¹ yr⁻¹)', fontsize=11)
            ax.set_ylabel(f'{var_y} (Mg C ha⁻¹ yr⁻¹)', fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'Insufficient data\n(n = {len(plot_data)})',
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(label, fontsize=12, fontweight='bold')

    # Hide unused subplots
    for idx in range(len(available_plots), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Forest Productivity Relationships (ForC Database)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'productivity_scatter_plots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Scatter plots saved to: {output_file}")
    plt.close()


def create_data_coverage_overview(long_data, output_dir):
    """
    Create overview of data coverage by productivity type.

    Args:
        long_data (pd.DataFrame): Long-format productivity data
        output_dir (str): Directory to save figure
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ForC Productivity Data Coverage', fontsize=16, fontweight='bold')

    # 1. Measurements by productivity type
    ax1 = axes[0, 0]
    type_counts = long_data['productivity_type'].value_counts()
    bars = ax1.bar(range(len(type_counts)), type_counts.values,
                   color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
    ax1.set_xlabel('Productivity Type')
    ax1.set_ylabel('Number of Measurements')
    ax1.set_title('Measurements by Type')
    ax1.set_xticks(range(len(type_counts)))
    ax1.set_xticklabels(type_counts.index, rotation=45, ha='right')

    # Add count labels
    for bar, count in zip(bars, type_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{count}', ha='center', va='bottom', fontsize=10)

    # 2. Sites by productivity type
    ax2 = axes[0, 1]
    sites_by_type = long_data.groupby('productivity_type')['sites.sitename'].nunique()
    bars = ax2.bar(range(len(sites_by_type)), sites_by_type.values,
                   color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
    ax2.set_xlabel('Productivity Type')
    ax2.set_ylabel('Number of Unique Sites')
    ax2.set_title('Unique Sites by Type')
    ax2.set_xticks(range(len(sites_by_type)))
    ax2.set_xticklabels(sites_by_type.index, rotation=45, ha='right')

    # Add count labels
    for bar, count in zip(bars, sites_by_type.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{count}', ha='center', va='bottom', fontsize=10)

    # 3. Top variables by measurement count
    ax3 = axes[1, 0]
    top_vars = long_data['variable.name'].value_counts().head(10)
    bars = ax3.barh(range(len(top_vars)), top_vars.values, color='skyblue')
    ax3.set_yticks(range(len(top_vars)))
    ax3.set_yticklabels(top_vars.index, fontsize=9)
    ax3.set_xlabel('Number of Measurements')
    ax3.set_title('Top 10 Variables by Measurement Count')
    ax3.invert_yaxis()

    # 4. Geographic distribution
    ax4 = axes[1, 1]
    if 'continent' in long_data.columns:
        continent_counts = long_data.groupby('continent')['sites.sitename'].nunique()
        if len(continent_counts) > 0:
            bars = ax4.bar(range(len(continent_counts)), continent_counts.values,
                          color='lightcoral')
            ax4.set_xlabel('Continent')
            ax4.set_ylabel('Number of Sites')
            ax4.set_title('Sites by Continent')
            ax4.set_xticks(range(len(continent_counts)))
            ax4.set_xticklabels(continent_counts.index, rotation=45, ha='right')

            for bar, count in zip(bars, continent_counts.values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{count}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    output_file = os.path.join(output_dir, 'productivity_coverage_overview.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Coverage overview saved to: {output_file}")
    plt.close()


def main():
    """Main execution function."""
    print("="*60)
    print("FORC DATABASE - ALL PRODUCTIVITY DATA EXTRACTION")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 60)

    # Check if data already exists
    data_already_exists = check_data_exists(DATA_DIR)

    if not data_already_exists:
        print("ForC data not found. Downloading...")
        download_success = download_file_from_github(
            USER, REPO, COMMIT_HASH, FOLDER_PATH_IN_REPO, DATA_DIR
        )
        if not download_success:
            print("Failed to download ForC data. Exiting.")
            return False
    else:
        print("✓ ForC data already exists. Proceeding to processing...")

    try:
        # Load ForC data
        measurements, sites, variables, methodology = load_forc_data(DATA_DIR)
        if measurements is None:
            return False

        # Get sites with BNPP measurements
        bnpp_sites = get_bnpp_sites(measurements)

        # Extract all productivity data for these sites
        productivity_data = extract_all_productivity_for_sites(
            measurements, sites, bnpp_sites, variables, methodology
        )

        # Create long-format dataset
        long_data = create_long_format_dataset(productivity_data)

        # Create wide-format dataset
        wide_data = create_wide_format_dataset(productivity_data)

        # Save datasets
        os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)

        # Save long format (all measurements)
        long_file = PROCESSED_FILE.replace('.csv', '_long.csv')
        long_data.to_csv(long_file, index=False)
        print(f"\n✓ Long-format data saved to: {long_file}")

        # Save wide format (one row per site)
        wide_file = PROCESSED_FILE.replace('.csv', '_wide.csv')
        wide_data.to_csv(wide_file, index=False)
        print(f"✓ Wide-format data saved to: {wide_file}")

        # Analyze relationships
        analyze_productivity_relationships(wide_data)

        # Create visualizations
        print(f"\n{'='*60}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*60}")

        os.makedirs(FIGURES_DIR, exist_ok=True)
        create_productivity_correlation_matrix(wide_data, FIGURES_DIR)
        create_productivity_scatter_plots(wide_data, FIGURES_DIR)
        create_data_coverage_overview(long_data, FIGURES_DIR)

        # Success summary
        print(f"\n{'='*60}")
        print("ALL PRODUCTIVITY DATA EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Long-format: {len(long_data)} measurements")
        print(f"✓ Wide-format: {len(wide_data)} sites")
        print(f"✓ Productivity types: {long_data['productivity_type'].nunique()}")
        print(f"✓ Unique variables: {long_data['variable.name'].nunique()}")
        print(f"✓ Data saved to: {OUTPUT_DIR}")
        print(f"✓ Figures saved to: {FIGURES_DIR}")

        return True

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 All productivity data extraction completed successfully!")
    else:
        print(f"\n❌ Extraction failed.")
