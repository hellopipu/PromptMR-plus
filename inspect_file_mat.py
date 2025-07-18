import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

# --- Configuration ---
# 1. Set the full path to the .mat file you want to inspect
FILE_TO_INSPECT = "/common/lidxxlab/cmrchallenge/data/debug_Chushu/debug_T1w_T2w_inference/P001_T1w.mat"

# 2. Set the name of the variable inside the .mat file you want to analyze
#    (You might need to run the script once to see the available variable names)
DATA_VARIABLE_NAME = 'kspace'
# --- End of Configuration ---

try:
    # Load the .mat file
    mat_contents = sio.loadmat(FILE_TO_INSPECT)
    print(f"✅ Successfully loaded file: {FILE_TO_INSPECT}")

    # 1. Inspect the file structure 🔬
    # A .mat file loads as a Python dictionary. Let's see what's inside.
    print("\n--- File Contents ---")
    variable_names = [key for key in mat_contents if not key.startswith('__')]
    print(f"Variables found: {variable_names}")
    
    # 2. Analyze the specific data variable 📊
    if DATA_VARIABLE_NAME in mat_contents:
        data = mat_contents[DATA_VARIABLE_NAME]
        print(f"\n--- Analysis of variable: '{DATA_VARIABLE_NAME}' ---")
        print(f"  - Shape: {data.shape}")
        print(f"  - Data Type: {data.dtype}")
        # Use np.abs for complex data statistics
        if np.iscomplexobj(data):
            abs_data = np.abs(data)
            print(f"  - Min value (magnitude): {abs_data.min():.4f}")
            print(f"  - Max value (magnitude): {abs_data.max():.4f}")
            print(f"  - Mean value (magnitude): {abs_data.mean():.4f}")
        else:
            print(f"  - Min value: {data.min():.4f}")
            print(f"  - Max value: {data.max():.4f}")
            print(f"  - Mean value: {data.mean():.4f}")
        
        # 3. Visualize the data 🖼️
        print("\nDisplaying a sample slice...")
        
        # Select a 2D slice from the data for plotting
        if data.ndim == 2:
            slice_to_show = data
        elif data.ndim == 3: # e.g., [slices, H, W]
            middle_slice = data.shape[0] // 2
            slice_to_show = data[middle_slice, :, :]
        elif data.ndim >= 4: # e.g., [time, slices, H, W]
            middle_time = data.shape[0] // 2
            middle_slice = data.shape[1] // 2
            slice_to_show = data[middle_time, middle_slice, :, :]
        else:
            slice_to_show = None

        if slice_to_show is not None:
            plt.figure(figsize=(8, 8))
            # Take the absolute value for complex data and use a grayscale map
            plt.imshow(np.abs(slice_to_show), cmap='gray')
            plt.title(f"Sample Slice from '{DATA_VARIABLE_NAME}'")
            plt.axis('off')
            plt.show()
        else:
            print("Could not display data: Unsupported number of dimensions.")

    else:
        print(f"\nError: Variable '{DATA_VARIABLE_NAME}' not found in the file.")
        print(f"Please choose one of the available variables: {variable_names}")

except FileNotFoundError:
    print(f"Error: File not found at '{FILE_TO_INSPECT}'")