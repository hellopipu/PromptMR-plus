#md5sum ChallengeDataTrain.tar.gz > trainmd5.txt || bash ~/mailsender.sh
#mkdir TrainingData
#tar -xvzf ChallengeDataTrain.tar.gz -C TrainingData --skip-old-files
cd /common/lidxxlab/cmrchallenge/code/PromptMR-plus
python prepare_h5_dataset_cmrxrecon_after.py  --input_matlab_folder /common/lidxxlab/cmrchallenge/data/CMR2023/TrainingData/ChallengeData/MultiCoil --output_h5_folder /common/lidxxlab/cmrchallenge/data/CMR2023/Processed/TrainingData --split_json configs/data_split/cmr23-cardiac.json --year 2023 --challenge_or_after 1 
bash ~/mailsender.sh
