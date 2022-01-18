# TODO: create shell script for running your GAN model

# Example
wget 'https://www.dropbox.com/s/wlyok1j3hwomoxk/hw2_p2_G.pth?dl=1' -O p2/model.pth
python3 p2/test.py --save_dir=$1 --test=p2/model.pth
