#!/bin/bash
#SBATCH --job-name=CMR2025Preprocess            # Job name
#SBATCH --output=../log/CMR2025Preprocess%j.log        # Standard output log
#SBATCH --error=../log/CMR2025Preprocess_err%j.log     # Standard error log
#SBATCH --partition=gpu                         # Specify the GPU partition
#SBATCH --cpus-per-task=12                      # Number of CPU cores per task
#SBATCH --mem=50G                               # Total memory limit
#SBATCH --time=168:00:00                        # Time limit
#SBATCH --gres=gpu:l40s:1                       # Request 1 GPU

# Load necessary modules, e.g., if you need conda or other environment modules
# module load anaconda or something similar

# Activate your conda environment (adjust as needed)
conda activate cmr

# Change to the working directory
cd /common/lidxxlab/cmrchallenge/code/PromptMR-plus

echo "Detailed GPU Information:"
nvidia-smi

gpu_info=$(nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free --format=csv)
echo "CUDA Device Information:"
echo "$gpu_info"

# Run your script
python prepare_h5_dataset_cmrxrecon_after.py \
       --input_matlab_folder $CMRROOT/data/CMR2025/ChallengeData/MultiCoil/ \
       --output_h5_folder $CMRROOT/data/CMR2025/Processed/MultiCoil \
       --split_json configs/data_split/cmr25-cardiac.json \
       --year 2025 \
       --challenge_or_after 1
