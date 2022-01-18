# bash hw1_1.sh $1(testing images directory) $2path of output csv file(predicted labels)

python3 ./src/main.py --mode=test --test=./src/model.pth --test_data=$1 --prediction=$2