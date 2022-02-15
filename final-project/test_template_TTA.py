import os
import argparse
import torch
import torch.optim as optim

from torchvision import transforms
# our module 
#from model_zoo import vgg16 
from model_zoo.swin.swin_transformer import get_swin
from model_zoo.swin.swin_transformer_bbn import get_swin_bbn
from model_zoo.pytorch_resnest.resnest.torch import resnest269
from base.dataset import FoodTestDataset, FoodDataset
from base.tester import BaseTester
from util import *
#from model_zoo.BBN.resnet import bbn_res50
from model_zoo.BBN.network import BNNetwork
#from model_zoo.BBN.combiner import Combiner
# From tester
import model_zoo.ttach as tta
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-img_size", "--img_size", default=224,type=int , help='')
	parser.add_argument("-batch_size", "--batch_size", default=16,type=int , help='')
	parser.add_argument("-test_data_dir","--test_data_dir", default = "food_data/test",type=str, help ="Testing image directory")
	parser.add_argument("-test_data_csv","--test_data_csv", default = "food_data/testcase/sample_submission_rare_track.csv",type=str, help ="Testcase csv")
	parser.add_argument("-load", "--load",default="",type=str , help='')
	parser.add_argument("-model_path", "--model_path",default="BBN_RESNET",type=str , help='')
	parser.add_argument("-model_type", "--model_type",default="RESNEST269",type=str , help='')
	parser.add_argument("-valid_mat", "--valid_mat",default="",type=str , help='')
	parser.add_argument("-kaggle_csv_log", "--kaggle_csv_log",default="BBN_RESNET/log.csv",type=str , help='')
	parser.add_argument("-mode", "--mode",default="TEST",type=str , help='') # TEST or VALID
	args = parser.parse_args()
	#######################
	# Environment setting
	#######################
	device = model_setting()
	fix_seeds(87)
	##############
	# Dataset
	##############
	if args.mode == "TEST" :
		test_dataset = FoodTestDataset(args.test_data_csv,args.test_data_dir,img_size=args.img_size)
		test_loader = torch.utils.data.DataLoader(test_dataset,
												  batch_size=args.batch_size,
												  shuffle=False,
												  num_workers=8)
		val_loader = None
	elif args.mode == "VALID":
		test_loader = None
		val_dataset = FoodDataset("food_data/val",img_size=args.img_size,mode = "val")
		val_loader = torch.utils.data.DataLoader(val_dataset,
												 batch_size=args.batch_size,
												 shuffle=False,
												 num_workers=8)
	else:
		print("Wrong Flag QQ")
		assert(False)

	##############
	# Model
	##############

	# TODO define ours' model
	# model = vgg16.VGG16(num_class=1000)
	#model = get_swin(ckpt='./model_zoo/swin/swin_large_patch4_window12_384_22kto1k.pth')
	#model = resnest269(pretrained=False)	
	#model = get_swin_bbn(ckpt='./model_zoo/swin/swin_large_patch4_window12_384_22kto1k.pth')
	#model = BNNetwork(backbone_model=model,num_classes=1000,mode="swin") # Support swin/ResNet/ViT
	if args.model_type == "RESNEST269":
		model = resnest269(pretrained=False)
	elif args.model_type == "SWIN":
		model = get_swin(ckpt='./model_zoo/swin/swin_large_patch4_window12_384_22kto1k.pth')
	elif args.model_type == "SWIN_BBN":
		model = get_swin_bbn(ckpt='./model_zoo/swin/swin_large_patch4_window12_384_22kto1k.pth')
		model = BNNetwork(backbone_model=model,num_classes=1000,mode="swin") # Support swin/ResNet/ViT
	else:
		print("Wrong Model type QQ")
		assert(False)
		
	if args.load:
		model.load_state_dict(torch.load(args.load))
		print("model loaded from {}".format(args.load))

	#BBN + ResNet50
	#model = bbn_res50(
	#		cfg = None,
	#		pretrain=False,
	#		pretrained_model="/home/r09021/DLCV_110FALL/final-project-challenge-3-no_qq_no_life/model_zoo/BNN/resnet50-19c8e357.pth",
	#		last_layer_stride=2
	#		
	#)
	#model = BNNetwork(backbone_model=model,num_classes=1000,mode = "ResNet50")
	#model = torch.nn.DataParallel(model) #CUDA_VISIBLE_DEVICES
	#model.load_state_dict(torch.load(os.path.join(args.load_model_path, "model_best.pth")))

	#################################
	# TTA method
 	# TODO define our own TTA metric
	#################################
	Food_Aug = tta.Compose(
		[
		tta.HorizontalFlip(),
		#tta.VerticalFlip(),
		#tta.Rotate90(angles=[0,90,180,270]),
		#tta.FiveCrops(int(args.img_size*0.8),int(args.img_size*0.8)),
		#tta.Resize(sizes=(args.img_size,args.img_size) ),
		]
	)
	tta_model = tta.ClassificationTTAWrapper(model = model,
											 transforms = Food_Aug,
											 merge_mode="mean")
	##############
	# Trainer
	##############
	tester = BaseTester(
				 device = device, 
				 model = tta_model,
				 test_loader = test_loader,
				 val_loader = val_loader,
				 load_model_path = args.model_path,
				 mat_file= os.path.join(args.valid_mat),
				 kaggle_file= os.path.join(args.kaggle_csv_log),
     			 criterion = torch.nn.CrossEntropyLoss())
	
	# Genertate final testing files to kaggle or just generate validation files 
	if args.mode == "TEST" :
		tester.test()
	elif args.mode == "VALID":
		tester.valid_and_savemat()
	else:
		print("Wrong Flag QQ")
		assert(False)	

	




