import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import pickle
import numpy as np
from PIL import Image

filenameToPILImage = lambda x: Image.open(x)
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(64),
            transforms.RandomRotation(20),
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class OfficeHomeDataset(Dataset):
    def __init__(self, csv_path, data_dir, mode):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        self.mode = mode
        self.filenames = self.data_df['filename']
        self.ids = [i for i in range(len(self.data_df))]

        if self.mode == 'train':
            # labels to numerical class
            self.labels  = self.data_df['label'].unique()
            self.labels.sort()
            self.label2class = {}
            self.class2label = {}
            for idx, l in enumerate(self.labels):
                self.label2class[l] = idx
                self.class2label[idx] = l
            # save dictionary as pickle
            with open('./p2/label2class.pkl', 'wb') as f:
                pickle.dump(self.label2class, f)
            with open('./p2/class2label.pkl', 'wb') as f:
                pickle.dump(self.class2label, f)

            self.transform = transforms.Compose([
                filenameToPILImage,
                transforms.Resize((128,128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        elif self.mode == 'test':
            # load pickle dictionary
            with open('./p2/label2class.pkl', 'rb') as f:
                self.label2class = pickle.load(f)
            with open('./p2/class2label.pkl', 'rb') as f:
                self.class2label = pickle.load(f)

            self.transform = transforms.Compose([
                filenameToPILImage,
                transforms.Resize((128,128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            raise NotImplementedError("Wrong mode!")
        
    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        if self.mode == 'train':
            label = self.label2class[label]
        elif self.mode == 'test':
            label = -1
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)
