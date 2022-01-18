# $1: testing images csv file (e.g., hw4_data/val.csv)
# $2: testing images directory (e.g., hw4_data/val)
# $3: path of test case on test set (e.g., hw4_data/val_testcase.csv)
# $4: path of output csv file (predicted labels) (e.g., output/val_pred.csv)

python3 p1/test_testcase.py --test_csv=$1 --test_data_dir=$2 --testcase_csv=$3 --output_csv=$4 --load=./p1/039.pth --seed=17 
