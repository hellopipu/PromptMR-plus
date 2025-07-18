import os
import h5py
import numpy as np
import re
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description='Process .mat files and save them with new naming convention.')
parser.add_argument('--input_folder', required=True, help='Path to input folder containing .mat files')
parser.add_argument('--output_folder', required=True, help='Path to output folder for processed .h5 files')

args = parser.parse_args()

# Configuration
input_folder = args.input_folder
output_folder = args.output_folder
os.makedirs(output_folder, exist_ok=True)

# Dictionary to store summary
summary_dict = {}

# Regex pattern to match files
pattern = re.compile(r'.*_mask_ktRadial(\d+)\.mat$')

# # Walk through the directory
# for root, dirs, files in os.walk(input_folder):
#     for file in files:
#         match = pattern.match(file)
#         if match:
#             acc = match.group(1)
#             file_path = os.path.join(root, file)

#             # Read .mat file with h5py
#             with h5py.File(file_path, 'r') as mat:
#                 data_key = list(mat.keys())[0]  # assuming first key contains data
#                 data = np.array(mat[data_key])

#                 if data.ndim == 2:
#                     print(f"Skipping 2D data: {file}")
#                     continue

#                 if data.ndim == 3:
#                     h, w = data.shape[2], data.shape[1]
#                     new_filename = f"acc{acc}_{w}x{h}.h5"
#                     new_file_path = os.path.join(output_folder, new_filename)

#                     # Check if file exists, skip overwriting
#                     if os.path.exists(new_file_path):
#                         print(f"Skipping existing file: {new_filename}")
#                         continue

#                     # Save data
#                     with h5py.File(new_file_path, 'w') as h5file:
#                         h5file.create_dataset('data', data=data)

#                     summary_dict[f'acc{acc}_{w}x{h}'] = new_filename
#                     print(f"Saved {new_filename}")

# Create summary dictionary by scanning all .h5 files in output folder
summary_path = os.path.join(output_folder, 'summary.h5')
with h5py.File(summary_path, 'w') as summary_h5:
    for file in os.listdir(output_folder):
        if file.endswith('.h5') and file != 'summary.h5':
            file_path = os.path.join(output_folder, file)
            with h5py.File(file_path, 'r') as h5file:
                data = np.array(h5file['data'])
                key = os.path.splitext(file)[0]
                summary_h5.create_dataset(key, data=data)

print("Summary file created at:", summary_path)