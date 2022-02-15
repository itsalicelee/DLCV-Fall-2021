#ResNest269
#python3 train_template.py --img_size 320 --lr 1e-5 --batch_size 4 --max_epoch 3 --train_data_dir $1/train --val_data_dir $1/val --model_type RESNEST269 --model_path 33_RESNEST269_stage1
#python3 train_template_LT.py --img_size 320 --lr 1e-5 --batch_size 4 --max_epoch 4 --LT_EXP REVERSE --train_data_dir $1 --model_type RESNEST269 --model_path checkpoints --load 33_RESNEST269_stage1/model_best.pth
#mv -f checkpoints/model_best.pth checkpoints/33.pth

#Swin
#python3 train_template.py --img_size 384 --lr 1e-5 --batch_size 4 --max_epoch 10 --train_data_dir $1/train --val_data_dir $1/val --model_type SWIN --model_path 22_40_SWIN_stage1
#python3 train_template_LT.py --img_size 384 --lr 1e-5 -batch_size 4 --max_epoch 1 -LT_EXP REVERSE --train_data_dir $1 --model_type SWIN --model_path checkpoints --load 22_40_SWIN_stage1/model_best.pth
#mv -f checkpoints/model_best.pth checkpoints/22.pth
#python3 train_template_LT.py --img_size 384 --lr 1e-5 --batch_size 4 --max_epoch 14 --gradaccum_size 16 --train_data_dir $1 --model_type SWIN --param_fix MLP --LT_EXP REVERSE --model_path checkpoints --load 22_40_SWIN_stage1/model_best.pth
#mv -f checkpoints/model_best.pth checkpoints/40.pth

#Swin BBN
#python3 train_template.py --img_size 384 --lr 1e-5 --batch_size 4 --max_epoch 1--train_data_dir $1/train --val_data_dir $1/val  --model_type SWIN --model_path 38_SWIN_stage1
#python3 train_template_BBN.py --img_size 384 --lr 1e-5 --batch_size 2 --max_epoch 10 --gradaccum_size 60 --train_data_dir $1 --model_path checkpoints --load 38_SWIN_stage1/model_best.pth
#mv -f checkpoints/model_best.pth checkpoints/38.pth