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
#SBATCH --exclude=esplhpc-cp080

source activate cmr

CMRROOT=/common/lidxxlab/chushu/PromptMR-plus
SAVE_DIR=/common/lidxxlab/chushu/PromptMR-plus/debug_T3w


echo "Detailed GPU Information:"
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

wandb login 888ea4d187a6f809d8cf6dda7de79991e057d892

#python main.py predict --config configs/inference/pmr-plus/cmr25_cardiac_TTT_with_logger_new.yaml
# Run training script with the correct LightningCLI format
python main.py predict \
       -c $CMRROOT/configs/base.yaml \
       -c $CMRROOT/configs/inference/pmr-plus/cmr25_cardiac_T1w_holdout.yaml \
       -c $CMRROOT/configs/model/pmr-plus.yaml \
       --trainer.devices=auto \
       --trainer.logger.init_args.save_dir=$SAVE_DIR \
       #--model.init_args.pretrain_weights_path=/common/lidxxlab/cmrchallenge/code/chaowei/experiments/cmr25/promptmr-plus/CMR2025/deep_recon/uec2kxvx/checkpoints/last.ckpt 
       #/common/lidxxlab/chushu/PromptMR-plus/CMR2025_output_with_early_stopping/pretrained_model_weight/02_with_self_loss_epoch=11-step=41112.ckpt
       #/common/lidxxlab/cmrchallenge/code/chaowei/experiments/cmr25/promptmr-plus/CMR2025/deep_recon/uec2kxvx/checkpoints/last.ckpt 
       #/common/lidxxlab/cmrchallenge/code/PromptMR-plus/weights/cmr24-cardiac/promptmr-plus-epoch=11-step=337764.ckpt \
              #--model.init_args.pretrain=False \
               # -c $CMRROOT/configs/train/pmr-plus/cmr25-cardiac-upd.yaml \
