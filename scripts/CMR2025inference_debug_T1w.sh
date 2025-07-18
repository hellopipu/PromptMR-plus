#!/bin/bash
#SBATCH --job-name=CMR_check_T1w
#SBATCH --output=logs/%j.log
#SBATCH --error=errors/%j.log
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --mail-user=chushu.shen@cshs.org
#SBATCH --mail-type=END,FAIL             


# --- Initialize Conda in non-interactive shell ------------------------------

source activate cmr
echo "Detailed GPU Information:"
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# --- User-defined variables (Change these as needed!) -----------------------
# Path to the model checkpoint you want to use:
CKPT_PATH="/common/lidxxlab/cmrchallenge/code/chaowei/experiments/cmr25/promptmr-plus/CMR2025/deep_recon/uec2kxvx/checkpoints/last.ckpt"
#"/common/lidxxlab/cmrchallenge/task3/Experiments/cmr2025_task3/lpftjn48/checkpoints/best-epochepoch=11-valvalidation_loss=0.0307.ckpt"
# Path to save inference outputs:
OUTPUT_DIR="/common/lidxxlab/chushu/PromptMR-plus/output_chushu/Inference_debug_T1w"
# Path to validation data (change if needed):
DATA_PATH="/common/lidxxlab/cmrchallenge/data/CMR2025/ChallengeData/MultiCoil/T1w/TrainingSet/UnderSample_taskR2"

# --- Run inference ----------------------------------------------------------
srun python main.py predict \
       --config /common/lidxxlab/chushu/PromptMR-plus/configs/inference/pmr-plus/cmr25_T1w_debug.yaml \
       --ckpt_path "$CKPT_PATH" \
       --model.init_args.pretrain_weights_path null \
       --data.init_args.data_path "$DATA_PATH" \
       --trainer.accelerator gpu \
       --trainer.devices 1 \
       --trainer.strategy ddp \
       --trainer.callbacks '[{"class_path": "__main__.CustomWriter", "init_args": {"output_dir": "'"$OUTPUT_DIR"'", "write_interval": "batch_and_epoch"}}]'