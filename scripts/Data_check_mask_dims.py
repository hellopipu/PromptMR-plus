import os
import h5py
import pandas as pd
import re
from pathlib import Path

def parse_filename(filename):
    """Extract acceleration factor, ny, and nx from filename"""
    pattern = r'acc(\d+)_(\d+)x(\d+)\.h5'
    match = re.match(pattern, filename)
    if match:
        acc = int(match.group(1))
        ny = int(match.group(2))
        nx = int(match.group(3))
        return acc, ny, nx
    return None, None, None

def analyze_masks(mask_dir):
    """Analyze all h5 mask files in the directory"""
    results = []
    mask_dir = Path(mask_dir)
    
    for h5_file in mask_dir.glob('acc*.h5'):
        acc, ny, nx = parse_filename(h5_file.name)
        if acc is None:
            print(f"Warning: Could not parse filename {h5_file.name}")
            continue
            
        try:
            with h5py.File(h5_file, 'r') as f:
                # Get all keys in the h5 file
                keys = list(f.keys())
                for key in keys:
                    data = f[key][()]
                    # Get dimensions
                    dims = data.shape
                    results.append({
                        'filename': h5_file.name,
                        'acceleration': acc,
                        'ny': ny,
                        'nx': nx,
                        'key': key,
                        'dimensions': dims,
                        'shape_str': 'x'.join(map(str, dims))
                    })
        except Exception as e:
            print(f"Error reading {h5_file.name}: {str(e)}")
    
    return pd.DataFrame(results)

def main():
    # Look for masks in the data directory
    mask_dir = '/common/lidxxlab/cmrchallenge/data/CMR2024/Processed/Mask'
    
    
        
    print(f"Analyzing masks in {mask_dir}")
    df = analyze_masks(mask_dir)
    
    # Save to CSV
    output_file = '/common/lidxxlab/cmrchallenge/code/chaowei/mask_dimensions_2024.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\nSummary of mask dimensions:")
    print(df.groupby(['acceleration', 'ny', 'nx'])['shape_str'].agg(lambda x: ' | '.join(set(x))))

if __name__ == '__main__':
    main() 