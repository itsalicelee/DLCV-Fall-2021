# $1: testing images csv file (e.g., hw4_data/val.csv)
# $2: testing images directory (e.g., hw4_data/val)
# $3: path of output csv file (predicted labels) (e.g., output/val_pred.csv)

python3 p2/inference.py --test_csv=$1 --test_data_dir=$2 --save_path=$3 --load=p2/model.pth --setting=inference
