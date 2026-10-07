#!/usr/bin/env python
"""
Comprehensive search for nppRoot data across all CMIP6 models and variants.

This script queries the ESGF search API to find all available nppRoot data
for the historical experiment, regardless of variant label.
"""

import requests
import json
from datetime import datetime
from collections import defaultdict
import time

# ESGF search endpoints (try multiple in case some are down)
ESGF_NODES = [
    "https://esgf-data.dkrz.de/esg-search/search",
    "https://esgf-node.ipsl.upmc.fr/esg-search/search",
    "https://esgf.ceda.ac.uk/esg-search/search",
    "https://esgf-node.llnl.gov/esg-search/search",
]

# List of CMIP6 models to search
CMIP6_MODELS = [
    'ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-CM-1-1-MR', 'BCC-CSM2-MR', 'BCC-ESM1',
    'CAMS-CSM1-0', 'CanESM5', 'CanESM5-CanOE', 'CESM2', 'CESM2-WACCM',
    'CMCC-CM2-SR5', 'CMCC-ESM2', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'CNRM-ESM2-1',
    'E3SM-1-0', 'E3SM-1-1', 'E3SM-1-1-ECA', 'EC-Earth3', 'EC-Earth3-Veg',
    'FGOALS-f3-L', 'FGOALS-g3', 'FIO-ESM-2-0', 'GFDL-CM4', 'GFDL-ESM4',
    'GISS-E2-1-G', 'GISS-E2-1-H', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM',
    'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'KIOST-ESM',
    'MCM-UA-1-0', 'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR',
    'MRI-ESM2-0', 'NESM3', 'NorCPM1', 'NorESM2-LM', 'NorESM2-MM',
    'SAM0-UNICON', 'TaiESM1', 'UKESM1-0-LL'
]


def search_esgf_for_npproot(model, node_url, timeout=30):
    """
    Search ESGF for all nppRoot datasets for a given model.

    Returns list of datasets with their variant labels.
    """
    params = {
        'format': 'application/solr+json',
        'variable_id': 'nppRoot',
        'experiment_id': 'historical',
        'source_id': model,
        'frequency': 'mon',
        'table_id': 'Lmon',
        'latest': 'true',
        'limit': 1000,
        'distrib': 'true',
        'fields': 'variant_label,grid_label,nominal_resolution,data_node,size,number_of_files,title,id,instance_id'
    }

    try:
        response = requests.get(node_url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        if 'response' in data and 'docs' in data['response']:
            docs = data['response']['docs']
            if docs:
                # Group by variant_label and grid_label
                variants = {}
                for doc in docs:
                    # Handle both list and string formats
                    def get_field(doc, field):
                        val = doc.get(field, 'unknown')
                        if isinstance(val, list) and len(val) > 0:
                            return val[0]
                        elif isinstance(val, str):
                            return val
                        return 'unknown'

                    variant = get_field(doc, 'variant_label')
                    grid = get_field(doc, 'grid_label')

                    key = f"{variant}_{grid}"
                    if key not in variants:
                        variants[key] = {
                            'variant_label': variant,
                            'grid_label': grid,
                            'nominal_resolution': get_field(doc, 'nominal_resolution'),
                            'data_node': get_field(doc, 'data_node'),
                            'number_of_files': doc.get('number_of_files', 1),
                            'size': doc.get('size', 0),
                            'example_id': get_field(doc, 'id')
                        }
                    else:
                        # Accumulate files and size if multiple entries
                        variants[key]['number_of_files'] += doc.get('number_of_files', 1)
                        variants[key]['size'] += doc.get('size', 0)

                return list(variants.values())

        return []

    except requests.exceptions.Timeout:
        print(f"  ⚠ Timeout for {model} on {node_url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  ⚠ Error for {model}: {e}")
        return None


def format_size(size_bytes):
    """Convert bytes to human-readable format."""
    if size_bytes == 0:
        return "0 B"

    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def main():
    print("="*100)
    print("COMPREHENSIVE CMIP6 NPPROOT DATA AVAILABILITY SEARCH")
    print("="*100)
    print(f"\nSearching {len(CMIP6_MODELS)} CMIP6 models for nppRoot (historical experiment)")
    print("This may take several minutes...\n")

    results = {}
    total_searched = 0
    total_with_data = 0

    # Try each ESGF node until we find one that works
    working_node = None
    for node in ESGF_NODES:
        print(f"Testing ESGF node: {node}")
        try:
            test_response = requests.get(node, params={'format': 'application/solr+json'}, timeout=10)
            if test_response.status_code == 200:
                working_node = node
                print(f"✓ Using node: {node}\n")
                break
        except:
            print(f"✗ Node unavailable: {node}")

    if not working_node:
        print("ERROR: No ESGF nodes are responding. Please try again later.")
        return

    # Search each model
    for i, model in enumerate(CMIP6_MODELS, 1):
        print(f"[{i}/{len(CMIP6_MODELS)}] Searching {model}...", end=' ', flush=True)
        total_searched += 1

        variants = search_esgf_for_npproot(model, working_node)

        if variants is None:
            print("✗ Error")
            results[model] = None
        elif len(variants) > 0:
            print(f"✓ Found {len(variants)} variant(s)")
            results[model] = variants
            total_with_data += 1
        else:
            print("✗ Not found")
            results[model] = []

        # Small delay to avoid overwhelming the server
        time.sleep(0.5)

    # Generate report
    print("\n" + "="*100)
    print("SEARCH RESULTS")
    print("="*100)

    output_file = f"../../cmip/comprehensive_npproot_availability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(output_file, 'w') as f:
        # Write header
        f.write("="*100 + "\n")
        f.write("COMPREHENSIVE CMIP6 NPPROOT DATA AVAILABILITY REPORT\n")
        f.write("="*100 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"ESGF Node: {working_node}\n")
        f.write(f"Total models searched: {total_searched}\n")
        f.write(f"Models with nppRoot data: {total_with_data}\n\n")

        # Models with data (detailed)
        f.write("="*100 + "\n")
        f.write("MODELS WITH NPPROOT DATA (DETAILED)\n")
        f.write("="*100 + "\n\n")

        models_with_data = {k: v for k, v in results.items() if v and len(v) > 0}

        if models_with_data:
            for model, variants in sorted(models_with_data.items()):
                f.write(f"\n{'─'*100}\n")
                f.write(f"MODEL: {model}\n")
                f.write(f"{'─'*100}\n")
                f.write(f"Available variants: {len(variants)}\n\n")

                for i, var in enumerate(variants, 1):
                    f.write(f"  Variant #{i}:\n")
                    f.write(f"    • Variant Label:       {var['variant_label']}\n")
                    f.write(f"    • Grid Label:          {var['grid_label']}\n")
                    f.write(f"    • Nominal Resolution:  {var['nominal_resolution']}\n")
                    f.write(f"    • Number of Files:     {var['number_of_files']}\n")
                    f.write(f"    • Total Size:          {format_size(var['size'])}\n")
                    f.write(f"    • Data Node:           {var['data_node']}\n")
                    f.write(f"\n")
        else:
            f.write("No models found with nppRoot data.\n")

        # Summary table
        f.write("\n" + "="*100 + "\n")
        f.write("SUMMARY TABLE - ALL MODELS\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Model':<25} {'Variants':<15} {'Status':<20}\n")
        f.write("─"*100 + "\n")

        for model in sorted(CMIP6_MODELS):
            if model not in results:
                status = "Not searched"
                variants = "N/A"
            elif results[model] is None:
                status = "✗ Error"
                variants = "Error"
            elif len(results[model]) == 0:
                status = "✗ Not found"
                variants = "0"
            else:
                variant_list = ', '.join([v['variant_label'] for v in results[model]])
                if len(variant_list) > 30:
                    variant_list = variant_list[:27] + "..."
                variants = f"{len(results[model])} ({variant_list})"
                status = "✓ Available"

            f.write(f"{model:<25} {variants:<15} {status:<20}\n")

        f.write("\n" + "="*100 + "\n")
        f.write(f"Summary: {total_with_data}/{total_searched} models have nppRoot data available\n")
        f.write("="*100 + "\n")

    print(f"\n✓ Report saved to: {output_file}")
    print(f"\nSummary: {total_with_data}/{total_searched} models have nppRoot data available")

    # Print quick summary to console
    if models_with_data:
        print("\nModels with nppRoot:")
        for model, variants in sorted(models_with_data.items()):
            variant_labels = ', '.join([v['variant_label'] for v in variants])
            print(f"  • {model}: {variant_labels}")


if __name__ == '__main__':
    main()
