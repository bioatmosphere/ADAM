#!/usr/bin/env python
"""
Download CMIP6 historical nppRoot data to complement forecast data.

This script searches ESGF for historical nppRoot data matching the parameters
of your dcppB-forecast dataset and generates a wget script for download.

Usage:
    python download_historical_cmip.py --model CanESM5 --variant r9i1p2f1
    python download_historical_cmip.py --auto-detect  # Read from existing forecast file
"""

import argparse
import sys
from pathlib import Path
import subprocess
import requests
from datetime import datetime
import xarray as xr


def detect_parameters_from_forecast(file_path):
    """Extract search parameters from existing forecast NetCDF file."""
    print(f"Reading parameters from: {file_path}")

    try:
        ds = xr.open_dataset(file_path)
        params = {
            'model': ds.attrs.get('source_id', 'CanESM5'),
            'variant': ds.attrs.get('variant_label', 'r9i1p2f1'),
            'grid': ds.attrs.get('grid_label', 'gn'),
            'institution_id': ds.attrs.get('institution_id', 'CCCma'),
        }
        ds.close()

        print(f"Detected parameters:")
        print(f"  Model: {params['model']}")
        print(f"  Variant: {params['variant']}")
        print(f"  Grid: {params['grid']}")
        print(f"  Institution: {params['institution_id']}")

        return params
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def search_esgf_api(model, variant, grid='gn', variable='nppRoot', experiment='historical'):
    """Search ESGF using their REST API."""
    print(f"\nSearching ESGF for {variable} data...")
    print(f"  Experiment: {experiment}")
    print(f"  Model: {model}")
    print(f"  Variant: {variant}")
    print(f"  Grid: {grid}")

    # ESGF search endpoint
    base_url = "https://esgf-node.llnl.gov/esg-search/search"

    params = {
        'project': 'CMIP6',
        'experiment_id': experiment,
        'variable': variable,
        'source_id': model,
        'variant_label': variant,
        'grid_label': grid,
        'frequency': 'mon',
        'realm': 'land',
        'latest': 'true',
        'format': 'application/solr+json',
        'type': 'File',
        'limit': 1000,
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        num_results = data['response']['numFound']
        print(f"\nFound {num_results} files")

        if num_results == 0:
            print("\nNo files found. Try adjusting search parameters.")
            return []

        docs = data['response']['docs']
        return docs

    except requests.exceptions.RequestException as e:
        print(f"Error searching ESGF: {e}")
        return []


def generate_wget_script(files, output_path, model, variant):
    """Generate a wget script for downloading files."""
    if not files:
        print("No files to download.")
        return False

    script_lines = [
        "#!/bin/bash",
        "#",
        f"# CMIP6 Historical nppRoot Download Script",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Model: {model}",
        f"# Variant: {variant}",
        "#",
        "",
        "# Download directory",
        "DOWNLOAD_DIR=\".\"",
        "",
        "echo \"Starting download of historical nppRoot data...\"",
        "echo \"Total files: " + str(len(files)) + "\"",
        "echo \"\"",
        "",
    ]

    # Extract download URLs
    download_count = 0
    seen_files = set()  # Track unique files to avoid duplicates

    for doc in files:
        filename = doc.get('title', 'unknown.nc')

        # Skip duplicates (same file on different nodes)
        if filename in seen_files:
            continue
        seen_files.add(filename)

        # Get HTTP URL for download
        url_list = doc.get('url', [])
        http_url = None

        # Try to extract HTTP server URL
        for url_entry in url_list:
            # Split by pipe character
            parts = url_entry.split('|')
            if len(parts) >= 3:
                url = parts[0]
                service_type = parts[1]
                if 'HTTPServer' in service_type:
                    http_url = url
                    break

        # If no HTTPServer URL found, try constructing one
        if not http_url and url_list:
            # Use the data node info to construct URL
            data_node = doc.get('data_node', '')
            dataset_id = doc.get('dataset_id', '').split('|')[0]

            # Try common ESGF URL patterns
            if data_node and dataset_id:
                # Construct typical ESGF HTTP URL
                base_urls = {
                    'esgf-node.ornl.gov': 'https://esgf-node.ornl.gov/thredds/fileServer',
                    'crd-esgf-drc.ec.gc.ca': 'https://crd-esgf-drc.ec.gc.ca/thredds/fileServer',
                    'eagle.alcf.anl.gov': 'https://eagle.alcf.anl.gov/thredds/fileServer',
                }

                if data_node in base_urls:
                    # Parse dataset_id to construct path
                    dataset_parts = dataset_id.replace('CMIP6.', '').split('.')
                    if len(dataset_parts) >= 8:
                        activity, institution, model, experiment, variant, table, variable, grid = dataset_parts[:8]
                        path = f"css03_data/CMIP6/{activity}/{institution}/{model}/{experiment}/{variant}/{table}/{variable}/{grid}"
                        http_url = f"{base_urls[data_node]}/{path}/{filename}"

        if http_url:
            file_size = doc.get('size', 0)
            file_size_mb = file_size / (1024 * 1024) if file_size else 0

            download_count += 1
            script_lines.append(f"# File {download_count}: {filename} ({file_size_mb:.1f} MB)")
            script_lines.append(f"echo \"Downloading {filename}...\"")
            script_lines.append(f"wget -c \"{http_url}\" -O \"{filename}\" 2>&1 | grep -v 'saving to'")
            script_lines.append(f"if [ $? -ne 0 ]; then")
            script_lines.append(f"    echo \"  Warning: Download failed, trying alternative...\"")
            script_lines.append(f"fi")
            script_lines.append("")

    # If no direct URLs found, provide manual instructions
    if download_count == 0:
        script_lines.append("echo \"No direct download URLs found.\"")
        script_lines.append("echo \"\"")
        script_lines.append("echo \"Please download manually from ESGF:\"")
        script_lines.append("echo \"https://esgf-node.llnl.gov/search/cmip6/\"")
        script_lines.append("echo \"\"")
        script_lines.append(f"echo \"Search for: {model} {variant} historical nppRoot\"")
    else:
        script_lines.append("echo \"\"")
        script_lines.append("echo \"Download complete!\"")
        script_lines.append(f"echo \"Files saved to: $DOWNLOAD_DIR\"")

    # Write script
    with open(output_path, 'w') as f:
        f.write('\n'.join(script_lines))

    # Make executable
    output_path.chmod(0o755)

    print(f"\nWget script generated: {output_path}")
    print(f"Total download URLs: {download_count}")

    if download_count > 0:
        # Calculate total size (only unique files)
        total_size = sum(doc.get('size', 0) for doc in files if doc.get('title') in seen_files)
        total_size_mb = total_size / (1024 * 1024)
        print(f"Total download size: {total_size_mb:.2f} MB")

    return True


def generate_download_summary(files, output_path):
    """Generate a text summary of available files."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CMIP6 HISTORICAL nppRoot DATA SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total files found: {len(files)}\n\n")

        f.write("FILE LIST\n")
        f.write("-"*80 + "\n")

        for idx, doc in enumerate(files, 1):
            filename = doc.get('title', 'Unknown')
            size = doc.get('size', 0)
            size_mb = size / (1024 * 1024)
            dataset_id = doc.get('dataset_id', 'Unknown')

            f.write(f"{idx}. {filename}\n")
            f.write(f"   Size: {size_mb:.2f} MB\n")
            f.write(f"   Dataset: {dataset_id}\n")
            f.write("\n")

        total_size = sum(doc.get('size', 0) for doc in files)
        total_size_gb = total_size / (1024 ** 3)
        f.write("-"*80 + "\n")
        f.write(f"Total size: {total_size_gb:.2f} GB\n")
        f.write("="*80 + "\n")

    print(f"Summary written to: {output_path}")


def show_manual_search_info(model, variant, grid):
    """Display manual search instructions if automated search fails."""
    print("\n" + "="*80)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*80)
    print("\nIf automated search fails, you can manually search and download:")
    print("\n1. Visit ESGF Search Portal:")
    print("   https://esgf-node.llnl.gov/search/cmip6/")
    print("\n2. Enter these search criteria:")
    print(f"   - Variable: nppRoot")
    print(f"   - Experiment: historical")
    print(f"   - Source ID: {model}")
    print(f"   - Variant Label: {variant}")
    print(f"   - Grid Label: {grid}")
    print(f"   - Frequency: mon")
    print(f"   - Realm: land")
    print("\n3. Select files and click 'WGET Script' button")
    print("\n4. Save and run the wget script in the cmip/ directory")
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Download CMIP6 historical nppRoot data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect parameters from forecast file
  python download_historical_cmip.py --auto-detect

  # Specify parameters manually
  python download_historical_cmip.py --model CanESM5 --variant r9i1p2f1

  # Download alternative experiment
  python download_historical_cmip.py --model CanESM5 --variant r9i1p2f1 --experiment ssp245
        """
    )

    parser.add_argument('--auto-detect', action='store_true',
                        help='Auto-detect parameters from existing forecast file')
    parser.add_argument('--forecast-file', type=str,
                        default='cmip/nppRoot_Lmon_CanESM5_dcppB-forecast_s2024-r9i1p2f1_gn_202501-203412.nc',
                        help='Path to forecast NetCDF file for auto-detection')
    parser.add_argument('--model', type=str, default='CanESM5',
                        help='Model name (source_id)')
    parser.add_argument('--variant', type=str, default='r9i1p2f1',
                        help='Variant label (e.g., r9i1p2f1)')
    parser.add_argument('--grid', type=str, default='gn',
                        help='Grid label')
    parser.add_argument('--variable', type=str, default='nppRoot',
                        help='Variable name')
    parser.add_argument('--experiment', type=str, default='historical',
                        help='Experiment ID (default: historical)')
    parser.add_argument('--output-dir', type=str, default='cmip',
                        help='Output directory for wget script')

    args = parser.parse_args()

    # Determine parameters
    if args.auto_detect:
        print("Auto-detecting parameters from forecast file...")
        params = detect_parameters_from_forecast(args.forecast_file)
        if params:
            model = params['model']
            variant = params['variant']
            grid = params['grid']
        else:
            print("Failed to auto-detect. Using default parameters.")
            model = args.model
            variant = args.variant
            grid = args.grid
    else:
        model = args.model
        variant = args.variant
        grid = args.grid

    # Search ESGF
    files = search_esgf_api(
        model=model,
        variant=variant,
        grid=grid,
        variable=args.variable,
        experiment=args.experiment
    )

    if not files:
        print("\nNo files found from automated search.")
        show_manual_search_info(model, variant, grid)
        return 1

    # Generate output files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    wget_script = output_dir / f'wget_historical_{args.experiment}_{timestamp}.sh'
    summary_file = output_dir / f'download_summary_{args.experiment}_{timestamp}.txt'

    # Generate wget script
    success = generate_wget_script(files, wget_script, model, variant)

    if success:
        # Generate summary
        generate_download_summary(files, summary_file)

        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print(f"\n1. Review the download summary:")
        print(f"   cat {summary_file}")
        print(f"\n2. Execute the wget script:")
        print(f"   bash {wget_script}")
        print(f"\n3. Visualize the historical data:")
        print(f"   python src/visualize_cmip.py --file cmip/<historical_file.nc> --all")
        print("\n" + "="*80 + "\n")

        return 0
    else:
        show_manual_search_info(model, variant, grid)
        return 1


if __name__ == '__main__':
    sys.exit(main())
