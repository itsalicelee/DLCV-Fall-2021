import torch
import time
import os
from scipy.io import savemat
from tqdm import tqdm 
from util import gen_logger
import csv
import itertools
import pandas as pd
import numpy as np
class BaseTester():
    def __init__(self,
                 device,
                 model,
                 test_loader,
                 load_model_path,
                 kaggle_file):
        self.device = device 
        self.model = model.to(self.device) 
        self.test_loader = test_loader
        self.model_path = load_model_path
        self.logger = gen_logger(os.path.join(load_model_path,"test.log"))
        self.kaggle_file = kaggle_file
    def test(self):
        '''
        log kaggle submission file
        '''
        label_list = []
        self.model.eval()
        with torch.no_grad():
            for i, t_imgs in enumerate(tqdm(self.test_loader)):
                t_imgs = t_imgs.to(self.device)
                P_pred = self.model(t_imgs)
                _,pred_label=torch.max(P_pred,1)
                item = pred_label.flatten().cpu().squeeze().tolist()
                label_list.append(item)
        label_list = np.concatenate(label_list)
        image_ids = ["{:06d}".format(i) for i in self.test_loader.dataset.data_df.image_id]
        # write predictions
        df = pd.DataFrame({"image_id": image_ids, 'label': label_list})
        df.to_csv(self.kaggle_file, index=False)
        print("===> File saved as {}".format(self.kaggle_file))

