import os
import torch
import torch.nn as nn
import argparse
import torchvision.transforms as transforms
from PIL import Image
from pytorch_pretrained_vit import ViT

class P1Dataset(nn.Module):
    def __init__(self, root, mode, size):
        self.root = root
        self.files = []
        self.filenames = []
        self.mode = mode
        self.filenames = [file for file in os.listdir(root)]
        if self.mode == "train" or self.mode == "val":
            for f in self.filenames:
                label = int(f[:f.find("_")])
                self.files.append((os.path.join(root, f), label))
                
        else:
            for f in self.filenames:
                self.files.append((os.path.join(root, f), -1))
        self.filenames.sort()
        self.files.sort()
        self.len = len(self.files)
        if self.mode == "train":
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size),
                transforms.Resize((size, size)),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])


    def __getitem__(self, idx):
        img = Image.open(self.files[idx][0]).convert('RGB')
        img = self.transform(img)
        label = self.files[idx][1]
        return img, label

    def __len__(self):
        return self.len
