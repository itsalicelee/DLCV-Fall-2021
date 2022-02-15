from logging import Logger
import os
import argparse
import torch
import torch.optim as optim
import warnings
# our module 
#from model_zoo import vgg16
#from  model_zoo.pytorch_pretrained_vit import ViT
#from model_zoo.swin.swin_transformer import get_swin
from model_zoo.swin.swin_transformer_bbn import get_swin_bbn
#from model_zoo.BBN.resnet import bbn_res50
from model_zoo.BBN.network import BNNetwork
from model_zoo.BBN.combiner import Combiner
from base.trainer import BaseTrainer,BBNTrainer
from base.dataset import FoodLTDataLoader,FoodDataset,ChunkSampler,P1_Dataset
from base.loss import LDAMLoss
from util import *
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	# training related argument
	parser.add_argument("-cont", "--cont",action="store_true", help='')
	parser.add_argument("-lr", "--lr", default=1e-5,type=float , help='')
	parser.add_argument("-period", "--period", default=1,type=int , help='')
	parser.add_argument("-batch_size", "--batch_size", default=4,type=int , help='')
	parser.add_argument("-gradaccum_size", "--gradaccum_size", default=20,type=int , help='')
	parser.add_argument("-load", "--load",default="",type=str , help='')
	parser.add_argument("-model_path", "--model_path",default="BBN_RESNET_UNIFORM",type=str , help='')
	parser.add_argument("-max_epoch", "--max_epoch",default=10,type=int, help='')
	# data related argument
	parser.add_argument("-img_size", "--img_size", default=384,type=int , help='')
	parser.add_argument("-train_data_dir","--train_data_dir", default = "food_data",type=str, help ="Training images directory")
	parser.add_argument("-val_data_dir","--val_data_dir", default = "",type=str, help ="Validation images directory")
	# experiment argument
	parser.add_argument("-CL_FLAG","--CL_FLAG", default = "UNIFORM",type=str, help ="Conventional Learning Branch") #BALANCED/UNIFORM
	args = parser.parse_args()
	#######################
	# Environment setting
	#######################
	device = model_setting()
	fix_seeds(87)
	os.makedirs(args.model_path, exist_ok=True)
	##############
	# Dataset
	##############	
	'''
	train_dataset = P1_Dataset("hw1_data/train_50",img_size=args.img_size,val_mode=False)
	val_dataset = P1_Dataset("hw1_data/val_50",img_size=args.img_size,val_mode=True)
	#train_dataset = FoodDataset(args.train_data_dir,img_size=args.img_size,mode = "train")
	train_loader = torch.utils.data.DataLoader(train_dataset,
												batch_size=args.batch_size,
												shuffle=False,
												num_workers=8,
												sampler=ChunkSampler(1024, 512))
	
	#val_dataset = FoodDataset(args.val_data_dir,img_size=args.img_size,mode = "val")
	val_loader = torch.utils.data.DataLoader(val_dataset,
												batch_size=args.batch_size,
												shuffle=False,
												num_workers=8,
												sampler=ChunkSampler(512, 0))
	train_loader_reverse = val_loader	

	'''
	train_loader = FoodLTDataLoader(data_dir=args.train_data_dir,
					 img_size=args.img_size,
					 batch_size=args.batch_size,
					 shuffle=True,
					 num_workers=8,
					 training=True, 
					 balanced= (args.CL_FLAG != "UNIFORM"),
					 reversed= False,
					 retain_epoch_size=True)
	train_loader_reverse = FoodLTDataLoader(data_dir=args.train_data_dir,
					 img_size=args.img_size,
					 batch_size=args.batch_size,
					 shuffle=True,
					 num_workers=8,
					 training=True, 
					 balanced= False,
					 reversed= True,
					 retain_epoch_size=True)
	val_loader = train_loader.split_validation()

	##############
	# Model
	##############

	# TODO define ours' model,schedular
	#model = ViT(model_name, pretrained=True,num_classes=1000,image_size=384)
	# ResNeSt50
	#model = resnest50(pretrained=False)
	#model.load_state_dict(torch.load('./model_zoo/pytorch_resnest/model_best.pth'))
	# BBN + ResNet50
	#model = bbn_res50(
	#		cfg = None,
	#		pretrain=True,
	#		pretrained_model="/home/r09021/DLCV_110FALL/final-project-challenge-3-no_qq_no_life/model_zoo/BBN/resnet50-19c8e357.pth",
	#		last_layer_stride=2
	#)
	#
	# Swin Tranformer +BBN
	model = get_swin_bbn(ckpt=args.load)
	model = BNNetwork(backbone_model=model,num_classes=1000,mode="swin") # Support swin/ResNet/ViT
	combiner = Combiner(MaxEpoch=args.max_epoch,
                     	model_type = model._get_name,
                        device = device)	
	#if args.load:
	#	model.load_state_dict(torch.load(args.load))
	#	print("model loaded from {}".format(args.load))
	
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) #,weight_decay=0.01
	criterion = torch.nn.CrossEntropyLoss()
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2)

	##############
	# Trainer
	##############
	trainer = BBNTrainer(
				 device = device, 
				 model = model,
				 combiner= combiner,
				 optimizer = optimizer,
				 scheduler = None,
				 MAX_EPOCH = args.max_epoch,
				 criterion = criterion,
				 train_loader = train_loader,
				 train_loader_reverse = train_loader_reverse,
				 val_loader = val_loader,
				 model_path = args.model_path,
				 lr = args.lr,
				 batch_size = args.batch_size, 
				 gradaccum_size = args.gradaccum_size, 
				 save_period = args.period)
			
	trainer.train()
	#trainer._valid(epoch=0)
	#trainer._valid_separate(epoch=0,branch=0)
	#trainer._valid_separate(epoch=0,branch=1)

