import os
import argparse
import torch
import torch.optim as optim
import warnings
import argparse
# our module 
from model_zoo import vgg16 
#from model_zoo.swin.swin_transformer import get_swin
#from model_zoo.pytorch_resnest.resnest.torch import resnest50, resnest101, resnest200, resnest269
from base.dataset import FoodTestDataset,ChunkSampler
from base.tester import BaseTester
from util import *
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-img_size", "--img_size", default=224,type=int , help='')
    parser.add_argument("-batch_size", "--batch_size", default=16,type=int , help='')
    parser.add_argument("-test_data_dir","--test_data_dir", default = "food_data/test",type=str, help ="Testing image directory")
    parser.add_argument("-test_data_csv","--test_data_csv", default = "food_data/testcase/sample_submission_rare_track.csv",type=str, help ="Testcase csv")
    parser.add_argument("-load_model_path", "--load_model_path",default="baseline",type=str , help='')
    parser.add_argument("-kaggle_csv_log", "--kaggle_csv_log",default="baseline/log.csv",type=str , help='')
    args = parser.parse_args()
    #######################
    # Environment setting
    #######################
    device = model_setting()
    fix_seeds(87)
    ##############
    # Dataset
    ##############
    test_dataset = FoodTestDataset(args.test_data_csv,args.test_data_dir,img_size=args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=8)
    ##############
    # Model
    ##############

    # TODO define ours' model
    model = vgg16.VGG16(num_class=1000)
    # model = get_swin(ckpt='./model_zoo/swin/swin_base_patch4_window7_224.pth')
    # model = resnest50(pretrained=False) 
    
    # load model from [load_model_path]/model_best.pth
    model.load_state_dict(torch.load(os.path.join(args.load_model_path, "model_best.pth")))

    ##############
    # Trainer
    ##############
    tester = BaseTester(
                 device = device, 
                 model = model,
                 test_loader = test_loader,
                 load_model_path = args.load_model_path,
                 kaggle_file= os.path.join(args.kaggle_csv_log))
    tester.test()
    

    




