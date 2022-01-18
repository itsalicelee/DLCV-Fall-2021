# TODO: create shell script for running your ViT testing code

# Example
wget 'https://www.dropbox.com/s/xw5pd2pffjvesse/94.467.pth?dl=1' -O p1/model.pth
python3 p1/inference.py --test_data=$1 --test=./p1/model.pth --save_path=$2 --size=256 
