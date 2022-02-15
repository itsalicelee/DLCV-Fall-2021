import os
import argparse
import torch
from scipy.io import loadmat
from base.dataset import FoodTestDataset, FoodDataset
import csv
import pandas as pd
from util import *
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-test_data_csv","--test_data_csv", default = "food_data/testcase/sample_submission_rare_track.csv",type=str, help ="Testcase csv")
    parser.add_argument("-test_data_dir","--test_data_dir", default = "food_data/test",type=str, help ="Testing image directory")
    parser.add_argument("-img_size", "--img_size", default=384,type=int , help='')
    parser.add_argument("-load_model_mat_1", "--load_model_mat_1",default="",type=str , help='')
    parser.add_argument("-model_weight_1", "--model_weight_1",default=1.0,type=float , help='')
    parser.add_argument("-load_model_mat_2", "--load_model_mat_2",default="",type=str , help='')
    parser.add_argument("-model_weight_2", "--model_weight_2",default=1.0,type=float , help='')
    parser.add_argument("-load_model_mat_3", "--load_model_mat_3",default="",type=str , help='')
    parser.add_argument("-model_weight_3", "--model_weight_3",default=1.0,type=float , help='')
    parser.add_argument("-load_model_mat_4", "--load_model_mat_4",default="",type=str , help='')
    parser.add_argument("-model_weight_4", "--model_weight_4",default=1.0,type=float , help='')
    parser.add_argument("-load_model_mat_5", "--load_model_mat_5",default="",type=str , help='')
    parser.add_argument("-model_weight_5", "--model_weight_5",default=1.0,type=float , help='')
    parser.add_argument("-kaggle_csv_log", "--kaggle_csv_log",default="baseline/log.csv",type=str , help='')
    parser.add_argument("-mode", "--mode", default="TEST", type=str , help='') # TEST or VALID
    args = parser.parse_args()

    ##############
    # Ensemble Model
    ##############
    model1_valid = False
    model2_valid = False
    model3_valid = False
    model4_valid = False
    model5_valid = False
    if args.load_model_mat_1:
        mat1 = loadmat(args.load_model_mat_1)
        print(args.load_model_mat_1)
        print(args.model_weight_1)
        out1_softmax = mat1['out_softmax']
        model1_valid = True

    if args.load_model_mat_2:
        mat2 = loadmat(args.load_model_mat_2)
        print(args.load_model_mat_2)
        print(args.model_weight_2)
        out2_softmax = mat2['out_softmax']
        model2_valid = True

    if args.load_model_mat_3:
        mat3 = loadmat(args.load_model_mat_3)
        print(args.load_model_mat_3)
        print(args.model_weight_3)
        out3_softmax = mat3['out_softmax']
        model3_valid = True

    if args.load_model_mat_4:
        mat4 = loadmat(args.load_model_mat_4)
        print(args.load_model_mat_4)
        print(args.model_weight_4)
        out4_softmax = mat4['out_softmax']
        model4_valid = True

    if args.load_model_mat_5:
        mat5 = loadmat(args.load_model_mat_5)
        print(args.load_model_mat_5)
        print(args.model_weight_5)
        out5_softmax = mat5['out_softmax']
        model5_valid = True
    #print(out1_softmax.shape[0])
    
    out_softmax = 0
    if model1_valid:
        out_softmax += out1_softmax * args.model_weight_1
    if model2_valid:
        out_softmax += out2_softmax * args.model_weight_2
    if model3_valid:
        out_softmax += out3_softmax * args.model_weight_3
    if model4_valid:
        out_softmax += out4_softmax * args.model_weight_4
    if model5_valid:
        out_softmax += out5_softmax * args.model_weight_5

    if args.mode == "VALID" :        
        val_dataset = FoodDataset("food_data/val",img_size=args.img_size,mode = "val")        
        
        label = np.squeeze(mat1['label'])
        pred_label = np.argmax(out_softmax, axis=1)
        val_acc = (label==pred_label).sum()
        val_nums = len(label)
        val_nums_freq = 0.0
        val_nums_common = 0.0
        val_nums_rare = 0.0
        val_acc_freq = 0.0
        val_acc_common = 0.0
        val_acc_rare = 0.0
        for i in range(len(label)):
            if (val_dataset.freq_list[label[i]] == 0):
                val_acc_freq += (label[i]==pred_label[i]).item()
                val_nums_freq += 1
            elif (val_dataset.freq_list[label[i]] == 1):
                val_acc_common += (label[i]==pred_label[i]).item()
                val_nums_common += 1
            else:
                val_acc_rare += (label[i]==pred_label[i]).item()
                val_nums_rare += 1
        print("Validation accuracy (main) : {:5f}".format(val_acc/val_nums))
        print("Val_acc {:d} Val_nums {:d} (main)".format(int(val_acc),int(val_nums)) )
        val_acc_rate_freq = val_acc_freq/val_nums_freq
        val_acc_rate_common = val_acc_common/val_nums_common
        val_acc_rate_rare = val_acc_rare/val_nums_rare
        print("Validation accuracy (freq) : {:5f}".format(val_acc_rate_freq))
        print("Val_acc {:d} Val_nums {:d} (freq)".format(int(val_acc_freq),int(val_nums_freq)) )
        print("Validation accuracy (common) : {:5f}".format(val_acc_rate_common))
        print("Val_acc {:d} Val_nums {:d} (common)".format(int(val_acc_common),int(val_nums_common)) )
        print("Validation accuracy (rare) : {:5f}".format(val_acc_rate_rare))
        print("Val_acc {:d} Val_nums {:d} (rare)".format(int(val_acc_rare),int(val_nums_rare)))
    
    elif args.mode == "TEST":
        test_dataset = FoodTestDataset(args.test_data_csv,args.test_data_dir,img_size=args.img_size)
        
        pred_label = np.argmax(out_softmax, axis=1)
        image_ids = ["{:06d}".format(i) for i in test_dataset.data_df.image_id]
        df = pd.DataFrame({"image_id": image_ids, 'label': pred_label})
        df.to_csv(args.kaggle_csv_log, index=False)
        print("===> File saved as {}".format(args.kaggle_csv_log))
        
    else:
        print("Wrong Flag QQ")
        assert(False)




