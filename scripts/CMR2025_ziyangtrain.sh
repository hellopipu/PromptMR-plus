#!/bin/bash
#SBATCH --job-name=CMR2025Train
#SBATCH --output=../log/CMR2025_ziyang_Train%j.log
#SBATCH --error=../log/CMR2025_ziyang_Train_err%j.log
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:l40s:1

conda activate cmr

cd /common/lidxxlab/cmrchallenge/code/PromptMR-plus

echo "Detailed GPU Information:"
nvidia-smi

# Run training script with the correct LightningCLI format
python main.py fit \
       --config configs/base.yaml configs/train/pmr-plus/cmr25-cardiac.yaml \
       --trainer.devices=1 \
       --trainer.max_epochs=100 \
       --model.init_args.lr=1e-4 \
       --trainer.logger.init_args.save_dir=$CMRROOT/output/CMR2025 \
       --ckpt_path=$CMRROOT/output/CMR2025/checkpoints/last.ckpt