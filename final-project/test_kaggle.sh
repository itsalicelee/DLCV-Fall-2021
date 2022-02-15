# Generate kaggle_csv_log and submit to kaggle
# bash test_kaggle.sh $1 
# $1 : model_path (e.g., baseline/)
# $2 : submit message in kaggle (e.g., ViT_B_16_imagenet1k)

# Freq track
python3 test_template.py -test_data_csv  food_data/testcase/sample_submission_freq_track.csv  -kaggle_csv_log $1/freq.log -load_model_path $1
kaggle competitions submit -c dlcv-fall-2021-final-challenge-3-freq-track -f $1/freq.log -m $2
# Main track
python3 test_template.py -test_data_csv  food_data/testcase/sample_submission_main_track.csv  -kaggle_csv_log $1/main.log -load_model_path $1
kaggle competitions submit -c dlcv-fall-2021-final-challenge-3 -f $1/main.log -m $2
# Common track
python3 test_template.py -test_data_csv  food_data/testcase/sample_submission_comm_track.csv  -kaggle_csv_log $1/comm.log -load_model_path $1
kaggle competitions submit -c dlcv-fall-2021-final-challenge-3-comm-track -f $1/comm.log -m $2
# Rare track
python3 test_template.py -test_data_csv  food_data/testcase/sample_submission_rare_track.csv  -kaggle_csv_log $1/rare.log -load_model_path $1
kaggle competitions submit -c dlcv-fall-2021-final-challenge-3-rare-track -f $1/rare.log -m $2
