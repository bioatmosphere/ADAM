"""Simple elevation extraction using Open Elevation API"""

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

def extract_elevations(csv_file, batch_size=50):
    """Extract elevations for coordinates in CSV file."""
    print(f"Loading coordinates from {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"Processing {len(df)} coordinate points")
    
    coordinates = list(zip(df['lat'], df['lon']))
    all_elevations = []
    
    # Process in batches
    for i in tqdm(range(0, len(coordinates), batch_size)):
        batch_coords = coordinates[i:i + batch_size]
        
        # Prepare request
        locations = [{"latitude": lat, "longitude": lon} for lat, lon in batch_coords]
        request_data = {"locations": locations}
        
        try:
            response = requests.post(
                "https://api.open-elevation.com/api/v1/lookup",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                batch_elevations = []
                for result in data.get("results", []):
                    elevation = result.get("elevation")
                    batch_elevations.append(elevation if elevation is not None else np.nan)
                all_elevations.extend(batch_elevations)
                print(f"Extracted batch {i//batch_size + 1}, got {len(batch_elevations)} elevations")
            else:
                print(f"API error for batch {i//batch_size + 1}: {response.status_code}")
                all_elevations.extend([np.nan] * len(batch_coords))
        
        except Exception as e:
            print(f"Error for batch {i//batch_size + 1}: {e}")
            all_elevations.extend([np.nan] * len(batch_coords))
        
        # Delay between batches
        time.sleep(1)
    
    # Add elevations to dataframe
    df['elevation'] = all_elevations
    
    # Save results
    output_file = "elevation_points.csv"
    df.to_csv(output_file, index=False)
    
    # Summary
    valid_elevations = pd.Series(all_elevations).dropna()
    print(f"Results saved to: {output_file}")
    print(f"Valid elevations: {len(valid_elevations)}")
    print(f"Missing elevations: {len(all_elevations) - len(valid_elevations)}")
    if len(valid_elevations) > 0:
        print(f"Elevation range: {valid_elevations.min():.1f} to {valid_elevations.max():.1f} meters")
        print(f"Mean elevation: {valid_elevations.mean():.1f} meters")

if __name__ == "__main__":
    extract_elevations("../../productivity/earth/lat_lon.csv")