#!/bin/bash
#SBATCH --job-name=CMR2025Train
#SBATCH --output=output_cmr25_%j.log
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:l40s:4

source activate torch

CMRROOT=/home/xup2/code/PromptMR-plus
SAVE_DIR=/common/xup2/Experiments/cmr25

echo "Detailed GPU Information:"
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

wandb login 37adfaca5f1b2086f330e07dcf05b30b2536bfbb

# Run training script with the correct LightningCLI format
python main.py fit \
       -c $CMRROOT/configs/base.yaml \
       -c $CMRROOT/configs/train/pmr-plus/cmr25-cardiac-upd.yaml \
       -c $CMRROOT/configs/model/pmr-plus.yaml \
       --trainer.devices=auto \
       --model.init_args.lr=0.0004 \
       --model.init_args.pretrain=True \
       --model.init_args.pretrain_weights_path=/common/lidxxlab/cmrchallenge/code/PromptMR-plus/weights/cmr24-cardiac/promptmr-plus-epoch=11-step=337764.ckpt \
       --trainer.logger.init_args.save_dir=$SAVE_DIR/promptmr-plus/CMR2025