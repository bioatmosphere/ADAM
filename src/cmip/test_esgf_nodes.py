#!/usr/bin/env python
"""
Test ESGF node connectivity and availability.

Usage:
    python test_esgf_nodes.py
    python test_esgf_nodes.py --model CanESM5 --experiment historical
"""

import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time


# Major ESGF nodes
ESGF_NODES = {
    'LLNL (USA)': 'https://esgf-node.llnl.gov',
    'ORNL (USA)': 'https://esgf-node.ornl.gov',
    'ANL (USA)': 'https://esgf.alcf.anl.gov',
    'CEDA (UK)': 'https://esgf-index1.ceda.ac.uk',
    'DKRZ (Germany)': 'https://esgf-data.dkrz.de',
    'IPSL (France)': 'https://esgf-node.ipsl.upmc.fr',
    'NCI (Australia)': 'https://esgf.nci.org.au',
}


def test_node_connectivity(name, url, timeout=10):
    """Test basic connectivity to an ESGF node."""
    try:
        start = time.time()
        response = requests.get(f"{url}/esg-search/search?limit=0",
                               timeout=timeout,
                               verify=True)
        elapsed = time.time() - start

        if response.status_code == 200:
            return {
                'name': name,
                'url': url,
                'status': 'ONLINE',
                'response_time': elapsed,
                'error': None
            }
        else:
            return {
                'name': name,
                'url': url,
                'status': 'ERROR',
                'response_time': elapsed,
                'error': f'HTTP {response.status_code}'
            }
    except requests.exceptions.Timeout:
        return {
            'name': name,
            'url': url,
            'status': 'TIMEOUT',
            'response_time': timeout,
            'error': 'Connection timeout'
        }
    except requests.exceptions.ConnectionError:
        return {
            'name': name,
            'url': url,
            'status': 'OFFLINE',
            'response_time': None,
            'error': 'Connection failed'
        }
    except Exception as e:
        return {
            'name': name,
            'url': url,
            'status': 'ERROR',
            'response_time': None,
            'error': str(e)
        }


def search_node_for_data(name, url, model, experiment, variable='nppRoot', timeout=10):
    """Search a specific node for data availability."""
    search_url = f"{url}/esg-search/search"
    params = {
        'project': 'CMIP6',
        'source_id': model,
        'experiment_id': experiment,
        'variable': variable,
        'format': 'application/solr+json',
        'limit': 1,
        'type': 'File'
    }

    try:
        response = requests.get(search_url, params=params, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            num_found = data['response']['numFound']
            return {
                'name': name,
                'available': num_found > 0,
                'count': num_found
            }
        else:
            return {
                'name': name,
                'available': False,
                'count': 0
            }
    except:
        return {
            'name': name,
            'available': None,
            'count': 0
        }


def main():
    parser = argparse.ArgumentParser(
        description='Test ESGF node connectivity and data availability'
    )
    parser.add_argument('--model', type=str, default='CanESM5',
                       help='Model to search for')
    parser.add_argument('--experiment', type=str, default='historical',
                       help='Experiment to search for')
    parser.add_argument('--variable', type=str, default='nppRoot',
                       help='Variable to search for')
    parser.add_argument('--timeout', type=int, default=10,
                       help='Timeout in seconds')
    parser.add_argument('--connectivity-only', action='store_true',
                       help='Only test connectivity, not data availability')

    args = parser.parse_args()

    print("="*80)
    print("ESGF NODE STATUS CHECK")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Timeout: {args.timeout}s")
    print()

    # Test connectivity
    print("Testing node connectivity...")
    print("-"*80)

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(test_node_connectivity, name, url, args.timeout): name
            for name, url in ESGF_NODES.items()
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # Sort by status and response time
    results.sort(key=lambda x: (
        0 if x['status'] == 'ONLINE' else 1,
        x['response_time'] if x['response_time'] else 999
    ))

    # Print results
    online_nodes = []
    for result in results:
        status_symbol = {
            'ONLINE': '✓',
            'OFFLINE': '✗',
            'TIMEOUT': '⏱',
            'ERROR': '✗'
        }.get(result['status'], '?')

        if result['status'] == 'ONLINE':
            online_nodes.append((result['name'], result['url']))
            print(f"{status_symbol} {result['name']:20s} - {result['status']:8s} ({result['response_time']:.2f}s)")
        else:
            error_msg = result['error'] if result['error'] else ''
            print(f"{status_symbol} {result['name']:20s} - {result['status']:8s} ({error_msg})")

    online_count = len(online_nodes)
    total_count = len(results)

    print("-"*80)
    print(f"Summary: {online_count}/{total_count} nodes online")
    print()

    # Test data availability
    if not args.connectivity_only and online_nodes:
        print(f"Searching for data: {args.model} / {args.experiment} / {args.variable}")
        print("-"*80)

        data_results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(
                    search_node_for_data, name, url,
                    args.model, args.experiment, args.variable, args.timeout
                ): name
                for name, url in online_nodes
            }

            for future in as_completed(futures):
                result = future.result()
                data_results.append(result)

        # Print data availability
        for result in data_results:
            if result['available'] is None:
                print(f"? {result['name']:20s} - Search failed")
            elif result['available']:
                print(f"✓ {result['name']:20s} - Data available ({result['count']} files)")
            else:
                print(f"✗ {result['name']:20s} - No data found")

        print("-"*80)
        available_count = sum(1 for r in data_results if r['available'])
        print(f"Summary: Data found on {available_count}/{len(data_results)} nodes")
        print()

    # Recommendations
    if online_nodes:
        print("RECOMMENDATIONS")
        print("-"*80)
        print("Fastest responding nodes:")
        for i, result in enumerate(results[:3], 1):
            if result['status'] == 'ONLINE':
                print(f"  {i}. {result['name']} ({result['response_time']:.2f}s)")
        print()
        print("Try these nodes first for downloads.")
    else:
        print("WARNING: No ESGF nodes are currently reachable.")
        print("This could indicate network issues or ESGF federation downtime.")

    print("="*80)


if __name__ == '__main__':
    main()
