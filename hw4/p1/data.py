import os
import torch
from torch.utils.data import Sampler, Dataset
import torchvision.transforms as transforms
import random
import pandas as pd
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
            transforms.RandomRotation(20),
            transforms.CenterCrop(64),
            transforms.RandomHorizontalFlip(p=0.5),
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

class MiniImageNetDataset(Dataset):
    def __init__(self, train_csv, train_data_dir):
        self.data_df = pd.read_csv(train_csv).set_index("id")
        print(self.data_df)
        self.data_dir = train_data_dir
        self.orig_labels = self.data_df.label.to_list()
        # labels to counts
        self.labels2counts = self.data_df['label'].value_counts().to_dict()
        self.labels = list(self.labels2counts.keys())
        self.labels.sort()
        # class 2 labels & labels 2 class
        self.class2labels = {}
        self.labels2class = {}
        for c, l in enumerate(self.labels):
            self.class2labels[c] = l
            self.labels2class[l] = c
        self.classes = list(self.class2labels.keys())
        # class 2 counts
        self.class2counts = {}
        for c in self.classes:
            self.class2counts[c] = self.labels2counts[self.class2labels[c]]
        # indices of each class
        self.class2indices = {}
        for c in self.classes:
            indices = np.where(self.data_df['label'] == self.class2labels[c])[0]
            self.class2indices[c] = list(indices)
        # print(self.class2indices)
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        orig_label = self.data_df.loc[index, "label"]
        label = self.labels2class[orig_label]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class NwayKshotSampler(Sampler):
    def __init__(self, csv_path, episodes_per_epoch, n_way=5, k_shot=1, k_query=1):
        self.data_df = pd.read_csv(csv_path)
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.episodes_per_epoch = episodes_per_epoch
    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []
            episode_classes = np.random.choice(self.data_df['label'].unique(), size=self.n_way, replace=False)

            support = []
            query = []

            for k in episode_classes:
                ind = self.data_df[self.data_df['label'] == k]['id'].sample(self.k_shot + self.k_query).values
                support = support + list(ind[:self.k_shot])
                query = query + list(ind[self.k_shot:])

            batch = support + query

            yield np.stack(batch)

    def __len__(self):
        return self.episodes_per_epoch
