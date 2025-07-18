import os
import scipy.io
import h5py
import numpy as np # Import numpy for h5py data access
import matplotlib.pyplot as plt # Added for image saving


def load_mat_files_from_directory(parent_directory):
    """
    Recursively finds and loads all .mat files in a given directory,
    handling both older and newer (v7.3+) .mat file formats.

    Args:
        parent_directory (str): The path to the parent directory to search.

    Returns:
        dict: A dictionary where keys are the full paths to the .mat files
              and values are the loaded data.
    """
    mat_data = {}
    print(f"Searching for .mat files in: {parent_directory}")

    for dirpath, _, filenames in os.walk(parent_directory):
        for filename in filenames:
            if filename.endswith('.mat'):
                file_path = os.path.join(dirpath, filename)
                print(f"Attempting to load file: {file_path}")
                try:
                    # First, try with scipy.io.loadmat
                    data = scipy.io.loadmat(file_path)
                    mat_data[file_path] = data
                    print(f"  Successfully loaded with SciPy.")

                except Exception as scipy_e:
                    # If SciPy fails, check if it's a version error, then try h5py
                    if 'Unknown mat file type' in str(scipy_e):
                        print(f"  SciPy failed (version error). Attempting with h5py...")
                        try:
                            with h5py.File(file_path, 'r') as f:
                                # For h5py, we'll store the loaded data in a dictionary
                                # to have consistent behavior with scipy.loadmat.
                                # This reads all datasets into numpy arrays.
                                data = {key: f[key][()] for key in f.keys()}
                                mat_data[file_path] = data
                                print(f"  Successfully loaded with h5py.")
                        except Exception as h5_e:
                            print(f"  h5py also failed: {h5_e}")
                    else:
                        # Handle other potential scipy errors
                        print(f"  An unknown SciPy error occurred: {scipy_e}")

    return mat_data

def save_4d_data_as_pngs(four_d_array, output_directory, file_prefix='image'):
    """
    Generates and saves 2D PNG images from a 4D NumPy array.
    Assumes input array shape is (Nt, Nz, Ny, Nx) and permutes for saving.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Permute from (Nt, Nz, Ny, Nx) to (Ny, Nx, Nz, Nt) for visualization
    try:
        # Based on earlier discussions, the data shape is (50, 4, 224, 87) -> (Nt, Nz, Ny, Nx)
        permuted_array = np.transpose(four_d_array, (2, 3, 1, 0))
    except ValueError:
        print(f"    ! Could not permute array with shape {four_d_array.shape}. Skipping image generation.")
        return

    Ny, Nx, Nz, Nt = permuted_array.shape
    
    # Normalize the entire 4D array for consistent contrast across all images
    max_val = np.max(permuted_array)
    if max_val > 0:
        normalized_array = permuted_array / max_val
    else:
        normalized_array = permuted_array
    
    print(f"    > Generating {Nz * Nt} images in '{output_directory}'...")
    # Loop through each slice (Nz) and time frame (Nt) to save the image
    for z_index in range(Nz):
        for t_index in range(Nt):
            image_slice = normalized_array[:, :, z_index, t_index]
            filename = f"{file_prefix}_slice_{str(z_index+1).zfill(2)}_time_{str(t_index+1).zfill(3)}.png"
            output_path = os.path.join(output_directory, filename)
            # Use plt.imsave for efficient saving without creating figure objects
            plt.imsave(output_path, image_slice, cmap='gray', format='png')
    
    # Clean up memory
    plt.close('all')

def save_image_slice(slice_2d, full_path):
    """
    Normalizes a single 2D slice and saves it as a PNG.
    """
    # Normalize for consistent contrast
    max_val = np.max(slice_2d)
    if max_val > 0:
        normalized_slice = slice_2d / max_val
    else:
        normalized_slice = slice_2d
    
    plt.imsave(full_path, normalized_slice, cmap='gray', format='png')


# --- USAGE EXAMPLE ---
#loading only

# UPDATE this path to the correct location on your system.
#root_directory = '/common/lidxxlab/cmrchallenge/code/Inference/predict/cmr25validation/TaskR2_legacy/reconstructions/Perfusion/ValidationSet/UnderSample_TaskR2/Center005'
#root_directory = '/common/lidxxlab/cmrchallenge/code/Inference/predict/cmr25validation/TaskR2_legacy/TaskR2/MultiCoil/Perfusion/ValidationSet/UnderSample_TaskR2/Center005'
#input kspace data
# Load all the .mat files
#all_mat_data = load_mat_files_from_directory(root_directory)

# --- Process Loaded Data ---
#if all_mat_data:
 #   print("\n--- Successfully loaded the following .mat files ---")
  #  for file_path, data in all_mat_data.items():
   #     print(f"\nFile: {os.path.basename(file_path)}")
    #    variable_names = [key for key in data.keys() if not key.startswith('__')]
     #   print(f"  Variables: {variable_names}")

        # Example of accessing data from the first variable
       # if variable_names:
        #    first_variable_name = variable_names[0]
            # The data is now a numpy array, regardless of the loading method
         #   first_variable_data = data[first_variable_name]
            # Make sure it's a numpy array before getting shape
           # if isinstance(first_variable_data, np.ndarray):
            #    print(f"  Shape of '{first_variable_name}': {first_variable_data.shape}")
            #else:
             #    print(f"  Content of '{first_variable_name}': {first_variable_data}")
#else:
   # print(f"\nNo .mat files were successfully loaded from the directory: {root_directory}")

###
# --- MAIN SCRIPT EXECUTION ---

# --- MAIN SCRIPT EXECUTION ---

# 1. SET THE TOP-LEVEL ROOT DIRECTORY HERE
#root_directory = '/common/lidxxlab/cmrchallenge/code/Inference/predict/cmr25validation/TaskR2_legacy_cropped/reconstructions'
#root_directory = '/common/lidxxlab/cmrchallenge/code/Inference/predict/cmr25validation/task2/reconstructions/T2w/ValidationSet/UnderSample_TaskR2'
#root_directory = '/common/lidxxlab/cmrchallenge/code/Inference/predict/cmr25validation/Submission_0601/TaskR2/MultiCoil/Perfusion/ValidationSet/UnderSample_TaskR2'
root_directory= '/common/lidxxlab/chushu/PromptMR-plus/output_chushu/Inference_test/reconstructions/T1w/ValidationSet/UnderSample_TaskR2'
#root_directory= '/common/lidxxlab/cmrchallenge/code/Inference/predict/test_weighted/reconstructions/T1w/ValidationSet/UnderSample_TaskR2'
# 2. SET THE MAIN OUTPUT FOLDER
main_output_folder = 'output_chushu/Inference_test/reconstruction_png_new/T1w'

# 3. LOAD ALL DATA
all_mat_data = load_mat_files_from_directory(root_directory)
print(f"\nFound and loaded data for {len(all_mat_data)} file(s).")

# 4. PROCESS EACH FILE and SAVE IMAGES
if not all_mat_data:
    print("No .mat files were found to process.")
else:
    print("\n--- Starting Image Generation Process ---")

for file_path, data in all_mat_data.items():
    print(f"\nProcessing file: {file_path}")

    # --- NEW LOGIC TO HANDLE DIFFERENT VARIABLE TYPES ---
    
    # Create the base output directory for this file
    relative_path = os.path.relpath(file_path, root_directory)
    path_without_ext, _ = os.path.splitext(relative_path)
    final_output_dir = os.path.join(main_output_folder, path_without_ext)
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)

    # CASE 1: Handle 'img4ranking' data with shape (224, 87, 2, 3)
    if 'img4ranking' in data:
        img_data = data['img4ranking']
        print(f"  > Found 'img4ranking' with shape: {img_data.shape}")
        
        if img_data.ndim == 4:
            H, W, dim_Z, dim_T = img_data.shape
            print(f"    > Generating {dim_Z * dim_T} images in '{final_output_dir}'...")
            
            # Here, Z=2 (comparison pair) and T=3 (slices)
            for z_index in range(dim_Z):
                for t_index in range(dim_T):
                    image_slice = img_data[:, :, z_index, t_index]
                    # Descriptive filename for comparison data
                    filename = f"slice_{t_index+1}_comparison_{z_index+1}.png"
                    output_path = os.path.join(final_output_dir, filename)
                    save_image_slice(image_slice, output_path)
        else:
            print(f"    ! 'img4ranking' is not 4D. Shape is {img_data.shape}. Skipping.")

    # CASE 2: Handle 'reconstruction' data with shape (50, 4, 224, 87)
    elif 'reconstruction' in data:
        recon_data = data['reconstruction']
        print(f"  > Found 'reconstruction' with shape: {recon_data.shape}")
        
        if recon_data.ndim == 4:
            # Permute from (Time, Slices, H, W) to (H, W, Slices, Time)
            permuted_data = np.transpose(recon_data, (2, 3, 1, 0))
            H, W, Nz, Nt = permuted_data.shape
            print(f"    > Generating {Nz * Nt} images in '{final_output_dir}'...")
            
            for z_index in range(Nz):
                for t_index in range(Nt):
                    image_slice = permuted_data[:, :, z_index, t_index]
                    filename = f"slice_{z_index+1}_time_{t_index+1}.png"
                    output_path = os.path.join(final_output_dir, filename)
                    save_image_slice(image_slice, output_path)
        else:
            print(f"    ! 'reconstruction' is not 4D. Shape is {recon_data.shape}. Skipping.")
            
    # CASE 3: No known variable was found
    else:
        print(f"  > No known data variable ('img4ranking' or 'reconstruction') found. Skipping.")

    plt.close('all') # Free up memory after processing each file

print("\n--- All tasks completed! ---")



"""
# 1. SET THE TOP-LEVEL ROOT DIRECTORY HERE
# The script will automatically process ALL subjects and .mat files found within this folder.
root_directory = '/common/lidxxlab/cmrchallenge/code/Inference/predict/cmr25validation/TaskR2_legacy_cropped/reconstructions/Perfusion/ValidationSet/UnderSample_TaskR2/Center005'

# 2. SET THE MAIN OUTPUT FOLDER for the generated PNGs
main_output_folder = 'reconstructions_as_pngs'

# 3. LOAD ALL DATA AUTOMATICALLY
all_mat_data = load_mat_files_from_directory(root_directory)
print(f"\nFound and loaded data for {len(all_mat_data)} file(s).")

# 4. PROCESS EACH FILE and SAVE IMAGES
# --- DIAGNOSTIC "Process Loaded Data" section ---
if not all_mat_data:
    print("No .mat files were found to process.")
else:
    print("\n--- DIAGNOSTIC MODE: Finding variable names in each file ---")

# This loop will now print the contents of each file
for file_path, data in all_mat_data.items():
    # Get all variable names in the current .mat file
    variable_names = [key for key in data.keys() if not key.startswith('__')]
    
    print(f"\nFile: {os.path.basename(file_path)}")
    print(f"  > Variables found: {variable_names}")

    # Also print the shape of the first variable found, which is likely the one we want
    if variable_names:
        first_var = variable_names[0]
        shape = data[first_var].shape
        print(f"    > Shape of '{first_var}': {shape}")

print("\n--- Diagnostic run complete. ---")
print("Note: Look at the 'Variables found' list above to identify the correct name for your 4D data array.")

"""
####
"""

if not all_mat_data:
    print("No .mat files were found to process.")
else:
    print("\n--- Starting Image Generation Process ---")

# This loop automatically processes every subject and file that was loaded
for file_path, data in all_mat_data.items():
    # ASSUMPTION: The 4D data array inside your .mat file is named 'reconstruction'.
    # If it has a different name (like 'img_grappa_sos', 'img_sense', etc.), change it here.
    variable_to_save = 'img4ranking' 

    if variable_to_save in data:
        print(f"\nProcessing file: {file_path}")

        # Get the 4D array from the loaded data
        image_data_4d = data[variable_to_save]
        print(f"  > Found variable '{variable_to_save}' with shape: {image_data_4d.shape}")

        # Create the mirrored output path for the PNGs
        relative_path = os.path.relpath(file_path, root_directory)
        path_without_ext, _ = os.path.splitext(relative_path)
        final_output_dir = os.path.join(main_output_folder, path_without_ext)

        # Call the function to save the images
        save_4d_data_as_pngs(image_data_4d, final_output_dir)
    else:
        print(f"\nSkipping file (variable '{variable_to_save}' not found): {file_path}")


print("\n--- All tasks completed! ---")

"""

