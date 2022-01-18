# hw1_2.sh $1(testing images directory) $2(output images directory)
wget "https://www.dropbox.com/s/ult694hblco4lra/model_unet_699.pth?dl=1" -O ./src2/model_unet_699.pth
python3 src2/main.py --mode=test --test=src2/model_unet_699.pth --model=unet --test_data=$1 --result_dir=$2