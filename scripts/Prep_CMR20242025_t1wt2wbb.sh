#md5sum ChallengeDataTrain.tar.gz > trainmd5.txt || bash ~/mailsender.sh
#mkdir TestGroundtruth
#tar -xvzf ChallengeDataTest.tar.gz -C TestGroundtruth


#2024 train fix symbolic issue
# python prepare_h5_dataset_cmrxrecon_modified.py \
#         --input_matlab_folder /common/lidxxlab/cmrchallenge/data/CMR2024/ChallengeData/MultiCoil \
#         --output_h5_folder /common/lidxxlab/cmrchallenge/data/CMR2024/Processed/MultiCoil \
#         --split_json configs/data_split/cmr24-cardiac.json \
#         --year 2024 \
#         --train_valid_test train \
#         --symbol_only 1

# # 2025 train fix t1w t2w black blood dim
python prepare_h5_dataset_cmrxrecon_modified.py \
        --input_matlab_folder /common/lidxxlab/cmrchallenge/data/CMR2025/ChallengeData/MultiCoil \
        --output_h5_folder /common/lidxxlab/cmrchallenge/data/CMR2025/Processed/MultiCoil \
        --split_json configs/data_split/cmr25-cardiac.json \
        --year 2025 \
        --train_valid_test train \
        --only_specific \
        --symbol_only 1

# bash ~/mailsender.sh
