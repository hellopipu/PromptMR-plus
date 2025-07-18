#md5sum ChallengeDataTrain.tar.gz > trainmd5.txt || bash ~/mailsender.sh
#mkdir TestGroundtruth
#tar -xvzf ChallengeDataTest.tar.gz -C TestGroundtruth
cd /common/lidxxlab/cmrchallenge/code/PromptMR-plus
python prepare_h5_dataset_cmrxrecon_after.py \
        --input_matlab_folder /common/lidxxlab/cmrchallenge/data/CMR2023/ValidGroundtruth/ChallengeData_validation/MultiCoil/ \
        --output_h5_folder /common/lidxxlab/cmrchallenge/data/CMR2023/Processed/ValidGroundtruth/MultiCoil \
        --split_json configs/data_split/cmr23-cardiac.json \
        --year 2023 \
        --challenge_or_after 0 \
	--symbol_only 1 
#python prepare_h5_dataset_cmrxrecon_after.py \
#        --input_matlab_folder /common/lidxxlab/cmrchallenge/data/CMR2023/ValidGroundtruth/ChallengeData_validation/SingleCoil/ \
#        --output_h5_folder /common/lidxxlab/cmrchallenge/data/CMR2023/Processed/ValidGroundtruth/SingleCoil \
#        --split_json configs/data_split/cmr23-cardiac.json \
#        --year 2023 \
#        --challenge_or_after 0 

#bash ~/mailsender.sh
