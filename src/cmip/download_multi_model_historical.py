#!/usr/bin/env python
"""
Download historical nppRoot data from multiple CMIP6 models.

This script generates wget scripts for downloading historical nppRoot data
from multiple CMIP6 models, maintaining consistent parameters:
- Experiment: historical
- Variant: r9i1p2f1 (or r1i1p1f1 as fallback)
- Grid: gn (native grid)
- Frequency: mon (monthly)
- Variable: nppRoot

Usage:
    python download_multi_model_historical.py --all
    python download_multi_model_historical.py --models CESM2 UKESM1-0-LL
    python download_multi_model_historical.py --search-only
"""

import argparse
import sys
from pathlib import Path
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


# List of major CMIP6 models
CMIP6_MODELS = [
    'CanESM5',          # Canadian Earth System Model
    'CESM2',            # Community Earth System Model
    'UKESM1-0-LL',      # UK Earth System Model
    'MPI-ESM1-2-LR',    # Max Planck Institute Earth System Model
    'GFDL-ESM4',        # NOAA Geophysical Fluid Dynamics Laboratory
    'IPSL-CM6A-LR',     # Institut Pierre-Simon Laplace Climate Model
    'NorESM2-LM',       # Norwegian Earth System Model
    'MIROC6',           # Model for Interdisciplinary Research on Climate
    'ACCESS-ESM1-5',    # Australian Community Climate and Earth System Simulator
    'EC-Earth3',        # European Earth System Model
    'CNRM-ESM2-1',      # Centre National de Recherches Météorologiques
    'MRI-ESM2-0',       # Meteorological Research Institute
    'ACCESS-CM2',       # ACCESS Climate Model
    'CMCC-ESM2',        # Euro-Mediterranean Centre on Climate Change
    'INM-CM5-0',        # Institute for Numerical Mathematics
]


def search_esgf_for_model(model, variant='r9i1p2f1', variable='nppRoot',
                          experiment='historical', grid='gn', timeout=30):
    """Search ESGF for historical nppRoot data for a specific model."""

    print(f"Searching for {model}...", end=' ')

    base_url = "https://esgf-node.llnl.gov/esg-search/search"

    # Try primary variant first
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
        'limit': 100,
    }

    try:
        response = requests.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        num_results = data['response']['numFound']

        if num_results > 0:
            print(f"✓ Found {num_results} files")
            return {
                'model': model,
                'variant': variant,
                'found': True,
                'num_files': num_results,
                'docs': data['response']['docs']
            }
        else:
            # Try fallback variant r1i1p1f1
            print(f"(trying r1i1p1f1)...", end=' ')
            params['variant_label'] = 'r1i1p1f1'
            response = requests.get(base_url, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            num_results = data['response']['numFound']

            if num_results > 0:
                print(f"✓ Found {num_results} files with r1i1p1f1")
                return {
                    'model': model,
                    'variant': 'r1i1p1f1',
                    'found': True,
                    'num_files': num_results,
                    'docs': data['response']['docs']
                }
            else:
                print("✗ No data found")
                return {
                    'model': model,
                    'variant': variant,
                    'found': False,
                    'num_files': 0,
                    'docs': []
                }

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return {
            'model': model,
            'variant': variant,
            'found': False,
            'num_files': 0,
            'docs': [],
            'error': str(e)
        }


def generate_esgf_wget_url(model, variant='r9i1p2f1', experiment='historical'):
    """Generate ESGF wget script download URL."""
    base_url = "https://esgf-node.llnl.gov/esg-search/wget"

    # This would be the search parameters for the wget script generator
    # Users need to visit this URL to generate the actual script
    search_params = (
        f"?project=CMIP6"
        f"&experiment_id={experiment}"
        f"&variable=nppRoot"
        f"&source_id={model}"
        f"&variant_label={variant}"
        f"&grid_label=gn"
        f"&frequency=mon"
        f"&realm=land"
    )

    return base_url + search_params


def generate_download_instructions(results, output_dir):
    """Generate download instructions file for all models."""
    output_path = output_dir / 'DOWNLOAD_INSTRUCTIONS_MULTI_MODEL.md'

    with open(output_path, 'w') as f:
        f.write("# Download Instructions for Multiple CMIP6 Models\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Available Models\n\n")

        available = [r for r in results if r['found']]
        unavailable = [r for r in results if not r['found']]

        f.write(f"**Available**: {len(available)} models\n")
        f.write(f"**Unavailable**: {len(unavailable)} models\n\n")

        # Available models
        if available:
            f.write("## Models with Data Available\n\n")
            for result in available:
                f.write(f"### {result['model']}\n")
                f.write(f"- **Variant**: {result['variant']}\n")
                f.write(f"- **Files**: {result['num_files']}\n")
                f.write(f"- **ESGF Search**: [Generate wget script]({generate_esgf_wget_url(result['model'], result['variant'])})\n")
                f.write("\n**Quick download steps**:\n")
                f.write(f"1. Visit: https://esgf-node.llnl.gov/search/cmip6/\n")
                f.write(f"2. Search criteria:\n")
                f.write(f"   - Variable: `nppRoot`\n")
                f.write(f"   - Experiment: `historical`\n")
                f.write(f"   - Source ID: `{result['model']}`\n")
                f.write(f"   - Variant Label: `{result['variant']}`\n")
                f.write(f"   - Grid Label: `gn`\n")
                f.write(f"   - Frequency: `mon`\n")
                f.write(f"3. Click 'WGET Script' button\n")
                f.write(f"4. Save as `wget_{result['model']}_historical.sh`\n")
                f.write(f"5. Run: `bash wget_{result['model']}_historical.sh`\n\n")

        # Unavailable models
        if unavailable:
            f.write("## Models with No Data Available\n\n")
            for result in unavailable:
                f.write(f"- **{result['model']}**: No historical nppRoot data with variant r9i1p2f1 or r1i1p1f1\n")

        f.write("\n## Batch Download Command\n\n")
        f.write("After generating wget scripts for all models:\n\n")
        f.write("```bash\n")
        f.write("cd cmip\n")
        f.write("for script in wget_*_historical.sh; do\n")
        f.write("    echo \"Running $script...\"\n")
        f.write("    bash \"$script\"\n")
        f.write("done\n")
        f.write("```\n\n")

        f.write("## Expected File Naming Pattern\n\n")
        f.write("Files will be named like:\n")
        f.write("```\n")
        f.write("nppRoot_Lmon_<MODEL>_historical_<VARIANT>_gn_<TIMERANGE>.nc\n")
        f.write("```\n\n")

        f.write("## After Download\n\n")
        f.write("Visualize each model's data:\n\n")
        f.write("```bash\n")
        f.write("cd src/cmip\n")
        f.write("for file in ../../cmip/nppRoot_*_historical_*.nc; do\n")
        f.write("    echo \"Visualizing $file...\"\n")
        f.write("    uv run python visualize_cmip.py --file \"$file\" --all\n")
        f.write("done\n")
        f.write("```\n\n")

    print(f"\n✓ Instructions saved: {output_path}")
    return output_path


def generate_summary_table(results, output_dir):
    """Generate a summary table of all models."""
    output_path = output_dir / 'model_availability_summary.txt'

    with open(output_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("CMIP6 HISTORICAL NPPROOT DATA AVAILABILITY SUMMARY\n")
        f.write("="*100 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total models searched: {len(results)}\n\n")

        # Header
        f.write(f"{'Model':<25} {'Variant':<15} {'Files':<10} {'Status':<15}\n")
        f.write("-"*100 + "\n")

        # Sort by availability then name
        sorted_results = sorted(results, key=lambda x: (not x['found'], x['model']))

        for result in sorted_results:
            status = "✓ Available" if result['found'] else "✗ Not found"
            variant = result.get('variant', 'N/A')
            num_files = result.get('num_files', 0)

            f.write(f"{result['model']:<25} {variant:<15} {num_files:<10} {status:<15}\n")

        f.write("-"*100 + "\n")

        available = sum(1 for r in results if r['found'])
        f.write(f"\nSummary: {available}/{len(results)} models have data available\n")
        f.write("="*100 + "\n")

    print(f"✓ Summary saved: {output_path}")
    return output_path


def create_batch_search_script(available_models, output_dir):
    """Create a bash script to open ESGF search for each model."""
    output_path = output_dir / 'open_esgf_searches.sh'

    with open(output_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Open ESGF search pages for each available model\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for result in available_models:
            url = generate_esgf_wget_url(result['model'], result['variant'])
            f.write(f"# {result['model']} - {result['num_files']} files\n")
            f.write(f"echo \"Opening search for {result['model']}...\"\n")
            f.write(f"open \"{url}\" 2>/dev/null || xdg-open \"{url}\" 2>/dev/null || echo \"  URL: {url}\"\n")
            f.write(f"sleep 2\n\n")

    output_path.chmod(0o755)
    print(f"✓ Search script saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Search and download historical nppRoot data from multiple CMIP6 models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search all default models
  python download_multi_model_historical.py --all

  # Search specific models
  python download_multi_model_historical.py --models CESM2 UKESM1-0-LL MPI-ESM1-2-LR

  # Just search, don't generate instructions
  python download_multi_model_historical.py --all --search-only

  # Use different variant
  python download_multi_model_historical.py --all --variant r1i1p1f1
        """
    )

    parser.add_argument('--all', action='store_true',
                       help='Search all default CMIP6 models')
    parser.add_argument('--models', nargs='+',
                       help='Specific models to search (space-separated)')
    parser.add_argument('--variant', type=str, default='r9i1p2f1',
                       help='Variant label (default: r9i1p2f1)')
    parser.add_argument('--experiment', type=str, default='historical',
                       help='Experiment ID (default: historical)')
    parser.add_argument('--variable', type=str, default='nppRoot',
                       help='Variable name (default: nppRoot)')
    parser.add_argument('--output-dir', type=str, default='cmip',
                       help='Output directory for instructions')
    parser.add_argument('--search-only', action='store_true',
                       help='Only search, do not generate download instructions')
    parser.add_argument('--parallel', type=int, default=5,
                       help='Number of parallel searches (default: 5)')

    args = parser.parse_args()

    # Determine which models to search
    if args.all:
        models = CMIP6_MODELS
    elif args.models:
        models = args.models
    else:
        print("Error: Must specify --all or --models")
        parser.print_help()
        return 1

    print("="*80)
    print("CMIP6 MULTI-MODEL HISTORICAL NPPROOT DATA SEARCH")
    print("="*80)
    print(f"Models to search: {len(models)}")
    print(f"Variant: {args.variant}")
    print(f"Experiment: {args.experiment}")
    print(f"Variable: {args.variable}")
    print("="*80 + "\n")

    # Search for data
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(
                search_esgf_for_model,
                model,
                args.variant,
                args.variable,
                args.experiment
            ): model
            for model in models
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    print("\n" + "="*80)

    # Summary
    available = [r for r in results if r['found']]
    print(f"\nSearch complete!")
    print(f"Available: {len(available)}/{len(results)} models")

    if not args.search_only:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate instructions
        instructions_file = generate_download_instructions(results, output_dir)
        summary_file = generate_summary_table(results, output_dir)

        if available:
            search_script = create_batch_search_script(available, output_dir)

        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print(f"\n1. Review summary:")
        print(f"   cat {summary_file}")
        print(f"\n2. Read download instructions:")
        print(f"   cat {instructions_file}")

        if available:
            print(f"\n3. Generate wget scripts:")
            print(f"   - Visit ESGF search portal for each model")
            print(f"   - Or run: bash {search_script}")

        print("\n" + "="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
