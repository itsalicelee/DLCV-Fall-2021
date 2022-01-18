import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
import os
import torch
from PIL import Image
import numpy as np
import imageio
from torchvision.transforms.transforms import Grayscale
from viz_mask import read_masks
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt


class P2Dataset(Dataset):
    def __init__(self, root, mode):
        self.filenames = []
        self.root = root
        self.transform = None
        self.mode = mode
        if self.mode == 'train':
            for i in range(2000):
                sat = os.path.join(root, '{:04d}_sat.jpg'.format(i))
                mask = os.path.join(root, '{:04d}_mask.png'.format(i))
                self.filenames.append((sat, mask)) # (sat, mask) pair
        elif self.mode == 'val':
            for i in range(257):
                sat = os.path.join(root, '{:04d}_sat.jpg'.format(i))
                mask = os.path.join(root, '{:04d}_mask.png'.format(i))
                self.filenames.append((sat, mask)) # (sat, mask) pair
        elif self.mode == 'test':
            self.filenames = [file for file in os.listdir(root) if file.endswith('.jpg')]
            self.filenames.sort()


        self.len = len(self.filenames)

        # transform vertical flip
        self.img_transform_vflip = transforms.Compose([
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                transforms.Resize((512, 512)),
                transforms.RandomVerticalFlip(p=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.mask_transform_vflip = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512,512)),
                transforms.RandomVerticalFlip(p=1),
        ])
        # transform horizontal flip
        self.img_transform_hflip = transforms.Compose([
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                transforms.Resize((512, 512)),
                transforms.RandomHorizontalFlip(p=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.mask_transform_hflip = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512,512)),
                transforms.RandomHorizontalFlip(p=1),
        ])
        # transform vertical & horizontal flip
        self.img_transform_vhflip = transforms.Compose([
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                transforms.Resize((512, 512)),
                transforms.RandomVerticalFlip(p=1),
                transforms.RandomHorizontalFlip(p=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.mask_transform_vhflip = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512,512)),
                transforms.RandomVerticalFlip(p=1),
                transforms.RandomHorizontalFlip(p=1),
        ])
        # no flip
        self.img_transform = transforms.Compose([
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                transforms.Resize((512,512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.mask_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512,512)),
        ])

    

    def __getitem__(self,idx):
        # read images
        if self.mode == 'train' or self.mode=='val':
            sat, mask = self.filenames[idx]
            img_sat = Image.open(sat)
            img_mask = Image.open(mask)
            img_mask = np.array(img_mask) # shape(512, 512, 3)
            
            img_mask = read_masks(img_mask, (512,512))
            img_mask = np.array(img_mask, dtype='uint8')
        elif self.mode == 'test':
            sat = self.filenames[idx]
            img_sat = Image.open(os.path.join(self.root, sat))
            img_mask = np.zeros(shape=(512,512), dtype='uint8') # shape(512, 512, 3)


        # transform
        if self.mode == 'train':
            p = np.random.randint(0,4)
            if p%4 == 0: # both flip
                img_sat = self.img_transform_vhflip(img_sat)
                img_mask = self.mask_transform_vhflip(img_mask)
                img_mask = torch.from_numpy(np.array(img_mask))
            elif p%4 == 1: # v flip
                img_sat = self.img_transform_vflip(img_sat)
                img_mask = self.mask_transform_vflip(img_mask)
                img_mask = torch.from_numpy(np.array(img_mask))
            elif p%4 == 2: # h flip
                img_sat = self.img_transform_hflip(img_sat)
                img_mask = self.mask_transform_hflip(img_mask)
                img_mask = torch.from_numpy(np.array(img_mask))        
            else : # don't flip
                img_sat = self.img_transform(img_sat)
                img_mask = torch.from_numpy(np.array(img_mask))
        elif self.mode=='val': # do not flip when validation
            img_sat = self.img_transform(img_sat)
            img_mask = torch.from_numpy(np.array(img_mask))
        elif self.mode=='test': # do not flip when test
            img_sat = self.img_transform(img_sat)
            img_mask = torch.from_numpy(np.array(img_mask))
        return img_sat, img_mask

    def __len__(self):
        return self.len
