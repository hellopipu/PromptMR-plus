import os
import h5py
import numpy as np
import pandas as pd
import glob
import concurrent.futures


# Set the folder to search and the output CSV file name.
folder_to_search = "/common/lidxxlab/cmrchallenge/data/CMR2025/Processed/MultiCoil"  # <-- Change this to your folder path
csv_file = "data_summary.csv"



def extract_modality(file_name):
    """Extract modality from file name as the string between the last '_' and the last '.'"""
    # Example: "Center005_UIH_15T_umr670_P016_T2map.h5" -> modality: "T2map"
    return file_name.rsplit('_', 1)[-1].rsplit('.', 1)[0]

def process_file(filepath):
    """Process a single .h5 file and return a dictionary with the extracted info."""
    parent_folder = os.path.basename(os.path.dirname(filepath))
    file_name = os.path.basename(filepath)
    modality = extract_modality(file_name)
    
    # Initialize default values
    kspace_shape = None
    reconstruction_shape = None

    # kspace statistics
    kspace_nan_inf_count = None
    kspace_min = None
    kspace_max = None
    kspace_mean = None
    kspace_std = None

    # reconstruction_rss statistics
    rec_nan_inf_count = None
    rec_min = None
    rec_max = None
    rec_mean = None
    rec_std = None

    try:
        with h5py.File(filepath, 'r') as f:
            # Process kspace dataset if available
            if "kspace" in f:
                kspace_data = f["kspace"]
                kspace_shape = kspace_data.shape
                kspace_nan_count = np.isnan(kspace_data).sum()
                kspace_inf_count = np.isinf(kspace_data).sum()
                kspace_nan_inf_count = kspace_nan_count + kspace_inf_count
                
                # If the data is complex, calculate statistics on its magnitude
                if np.iscomplexobj(kspace_data):
                    kspace_to_check = np.abs(kspace_data)
                else:
                    kspace_to_check = kspace_data

                kspace_min = np.nanmin(kspace_to_check)
                kspace_max = np.nanmax(kspace_to_check)
                kspace_mean = np.nanmean(kspace_to_check)
                kspace_std = np.nanstd(kspace_to_check)

            # Process reconstruction_rss dataset if available
            if "reconstruction_rss" in f:
                rec_data = f["reconstruction_rss"][:]
                reconstruction_shape = rec_data.shape
                rec_nan_count = np.isnan(rec_data).sum()
                rec_inf_count = np.isinf(rec_data).sum()
                rec_nan_inf_count = rec_nan_count + rec_inf_count

                # Compute statistics on magnitude if the data is complex
                if np.iscomplexobj(rec_data):
                    rec_to_check = np.abs(rec_data)
                else:
                    rec_to_check = rec_data

                rec_min = np.nanmin(rec_to_check)
                rec_max = np.nanmax(rec_to_check)
                rec_mean = np.nanmean(rec_to_check)
                rec_std = np.nanstd(rec_to_check)
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None

    return {
        "parent folder": parent_folder,
        "file name": file_name,
        "modality": modality,
        "kspace shape": str(kspace_shape),
        "reconstruction shape": str(reconstruction_shape),
        "kspace nan or inf count": kspace_nan_inf_count,
        "reconstruction nan or inf count": rec_nan_inf_count,
        "kspace min": kspace_min,
        "kspace max": kspace_max,
        "kspace mean": kspace_mean,
        "kspace std": kspace_std,
        "reconstruction min": rec_min,
        "reconstruction max": rec_max,
        "reconstruction mean": rec_mean,
        "reconstruction std": rec_std,
    }

def main():
    # Use glob to quickly gather all .h5 files recursively.
    pattern = os.path.join(folder_to_search, '**', '*.h5')
    file_list = glob.glob(pattern, recursive=True)
    print(f"Found {len(file_list)} .h5 files.")
    
    rows = []
    # Use ProcessPoolExecutor to process files in parallel.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_file, file_list))
        for res in results:
            if res is not None:
                rows.append(res)
    
    if rows:
        columns = [
            "parent folder", "file name", "modality", 
            "kspace shape", "reconstruction shape",
            "kspace nan or inf count", "reconstruction nan or inf count",
            "kspace min", "kspace max", "kspace mean", "kspace std",
            "reconstruction min", "reconstruction max", "reconstruction mean", "reconstruction std"
        ]
        df = pd.DataFrame(rows, columns=columns)
        
        # If the CSV file exists, append without header; otherwise, create a new file.
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
            print(f"Appended {len(df)} rows to existing CSV file: {csv_file}")
        else:
            df.to_csv(csv_file, index=False)
            print(f"Created CSV file {csv_file} with {len(df)} rows.")
    else:
        print("No .h5 files found or no valid data to process.")

if __name__ == "__main__":
    main()
