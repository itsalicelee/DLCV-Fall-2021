import glob
import os
import torch, torchvision
from PIL import Image
import numpy as np
import imageio
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class P1Dataset(Dataset):
    def __init__(self, root, mode):
        self.root= root
        self.mode = mode
        self.files = []
        self.transform = None
    
        tmp = [file for file in os.listdir(root)]
        for file in tmp:
            self.files.append(os.path.join(root, file))
        
        self.files.sort()
        self.len = len(self.files)
        
        if self.mode == "train":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomAdjustSharpness(1, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]) 
        elif self.mode == "test":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomAdjustSharpness(1, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]) 

    def __getitem__(self, index):
        img = torchvision.io.read_image(self.files[index])
        img = self.transform(img)
        return img, img

    def __len__(self):
        return self.len
