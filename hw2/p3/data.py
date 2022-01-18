import glob
import os
import torch, torchvision
from PIL import Image
import numpy as np
import imageio
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd


class DigitDataset(Dataset):
    def __init__(self, root, type, mode):
        list =  ['usps', 'mnistm', 'svhn']
        if  type not in list:
            print("Check dataset name!")
            assert False
        self.type = type
        self.root= root
        self.mode = mode
        self.files = []
        self.filenames = []
        self.transform = None
        if self.mode != 'inf':
            tmp = [file for file in os.listdir(os.path.join(self.root, self.type, self.mode))]
            self.df = pd.read_csv(os.path.join(self.root, self.type, self.mode+".csv"))
            for file in tmp:
                self.files.append(os.path.join(self.root, self.type, self.mode, file))
                self.filenames.append(file)
        else: 
            tmp = [file for file in os.listdir(root)]
            for file in tmp:
                self.files.append(os.path.join(self.root, file))
                self.filenames.append(file)
    
        self.files.sort()
        self.filenames.sort()
        self.len = len(self.files)
        

        if self.mode == "train":
            self.transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize((28, 28)),
                # transforms.RandomRotation(5),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomAdjustSharpness(1, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]) 
        else:
            self.transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize((28, 28)),

                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomAdjustSharpness(1, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]) 

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        if self.type == 'usps':
            img = Image.open(self.files[index]).convert('RGB')
        img = self.transform(img)
        
        if self.mode != 'inf':
            label = self.df['label'][index]
        else:
            label = 0

        
        return img, label

    def __len__(self):
        return self.len
