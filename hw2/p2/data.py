import glob
import os
import torch, torchvision
from PIL import Image
import numpy as np
import imageio
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms.transforms import ToTensor

class DigitDataset(Dataset):
    def __init__(self, root, type, mode):
        list =  ['upsps', 'mnistm', 'svhn']
        if  type not in list:
            assert False
        self.type = type
        self.root= root
        self.mode = mode
        self.files = []
        self.transform = None

        if self.mode == "train":
            tmp = [file for file in os.listdir(os.path.join(self.root, self.type, self.mode))]
            for file in tmp:
                self.files.append(os.path.join(self.root, self.type, self.mode, file))
            self.df = pd.read_csv(os.path.join(self.root, self.type, self.mode+".csv"))

        elif self.mode == "test":
            tmp = [file for file in os.listdir(root)]
            for i in range(len(tmp)):
                file = tmp[i]
                digit = int(file[0])
                self.files.append((os.path.join(self.root, file), digit)) #(img, label) pair

        self.files.sort()    
        self.len = len(self.files)
        if self.mode == "train":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((28, 28)),
                # transforms.RandomRotation(5),
                # transforms.ColorJitter(),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomAdjustSharpness(1, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]) 
        elif self.mode == "test":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((28, 28)),

                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomAdjustSharpness(1, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]) 

    def __getitem__(self, index):
        if self.mode == "train":
            img = torchvision.io.read_image(self.files[index])
            img = self.transform(img)
            label = self.df['label'][index]
        elif self.mode == "test":
            img = torchvision.io.read_image(self.files[index][0])
            img = self.transform(img)
            label = self.files[index][1]
        return img, label

    def __len__(self):
        return self.len
