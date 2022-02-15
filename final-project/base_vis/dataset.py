from PIL import Image
import os
import re
import glob
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import sampler,Dataset,DataLoader
import torch.nn.functional as F
from torchvision import transforms
import torchvision 

filenameToPILImage = lambda x: Image.open(x)


##############
# Dataset
##############
class FoodDataset(Dataset):
    def __init__(self,data_path,mode,img_size, class_list=None, num_per_class=None):
        """
        for training and validation data
        return data with label
        """
        # visualize #
        self.class_list = class_list
        self.num_per_class = num_per_class

        self.data_path = data_path
        self.img_size = img_size
        self.mode = mode
        self.file_list,self.label_list = self.parse_folder_visualize()
        self.num = len(self.file_list)
        print("load %d images from %s"%(self.num,self.data_path))
        if mode == "train":
            self.transform = transforms.Compose([
                filenameToPILImage,
                transforms.Resize((self.img_size,self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                torchvision.transforms.ColorJitter(),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transforms.Compose([
                filenameToPILImage,
                transforms.Resize((self.img_size,self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        self.freq_list = []
        f = open('../final-project-challenge-3-no_qq_no_life/food_data/label2name.txt', encoding='utf8')
        for line in f.readlines():
            if (line.find("f") != -1):
               self.freq_list.append(0)
            elif (line.find("c") != -1):
               self.freq_list.append(1)
            else:
               self.freq_list.append(2)         
        f.close
    def parse_folder(self):
        '''
        output : file _dict 
        '''
        file_list = []
        label_list = []
        for class_id in range(0,1000):
            str_id = str(class_id)
            sub_folder = os.path.join(self.data_path,str_id)
            sub_list = sorted([name for name in os.listdir(sub_folder) if os.path.isfile(os.path.join(sub_folder, name))])
            file_list.extend(sub_list)
            label_list.extend([str_id]*len(sub_list))
        return file_list,label_list

    
    def parse_folder_visualize(self):
        '''
        output : file _dict 
        '''
        file_list = []
        label_list = []
        for class_id in self.class_list:
            str_id = str(class_id)
            sub_folder = os.path.join(self.data_path,str_id)
            sub_list = sorted([name for name in os.listdir(sub_folder) if os.path.isfile(os.path.join(sub_folder, name))])
            if len(sub_list) > self.num_per_class:
                sub_list = sub_list[:self.num_per_class]
            file_list.extend(sub_list)
            label_list.extend([str_id]*len(sub_list))
        return file_list,label_list

    def getOriginalImage(self, index):
        img_path = os.path.join(self.data_path,self.label_list[index],self.file_list[index])
        transform_original = transforms.Resize((384, 384))
        img = Image.open(img_path)
        orginal_img = np.array(transform_original(img))
        return orginal_img
        
    def __len__(self) -> int:
        return self.num
    def __getitem__(self, index):       
        img_path = os.path.join(self.data_path,self.label_list[index],self.file_list[index])
        label = int(self.label_list[index])
        # Preprocessing -> normalize image
        img = self.transform(img_path)
        return img,label

class FoodTestDataset(Dataset):
    def __init__(self,csv_path,data_path,img_size):
        """
        for training and validation data
        return data without label
        """
        self.data_df = pd.read_csv(csv_path)
        self.data_path = data_path
        self.img_size = img_size
        self.num = len(self.data_df)
        print("load %d images from %s"%(self.num,self.data_path))
        self.transform = transforms.Compose([
                filenameToPILImage,
                transforms.Resize((self.img_size,self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    def __getitem__(self, index):       
        img_path = os.path.join(self.data_path,"{:06d}.jpg".format(self.data_df.loc[index, "image_id"]))
        img = self.transform(img_path)
        return img
    def __len__(self) -> int:
        return self.num

#############################
# Long tail food dataloader #
#############################

class FoodLTDataLoader(DataLoader):
    """
    2021 ICLR RIDE
    modified from ImageNetLT Data Loader
    counting statistics of data,and construct a list of class_num
    base on this list of class_num,we can do reweight/resample
    """
    def __init__(self, data_dir, img_size, batch_size, shuffle=True, num_workers=1, training=True, balanced=False, reversed=False, retain_epoch_size=True):
        if training:
            dataset = FoodDataset(os.path.join(data_dir,"train"),"train",img_size = img_size)
            val_dataset = FoodDataset(os.path.join(data_dir,"val"),"val",img_size = img_size)
        else: # test
            dataset = FoodTestDataset(os.path.join(data_dir,"test"),img_size = img_size)
            val_dataset = None

        self.dataset = dataset
        self.val_dataset = val_dataset

        self.n_samples = len(self.dataset)

        num_classes = len(np.unique(dataset.label_list))
        assert num_classes == 1000

        cls_num_list = [0] * num_classes
        for label in dataset.label_list:
            cls_num_list[int(label)] += 1

        self.cls_num_list = cls_num_list

        if balanced:
            if training:
                buckets = [[] for _ in range(num_classes)]
                for idx, label in enumerate(dataset.label_list):
                    buckets[int(label)].append(idx)
                sampler = BalancedSampler(buckets, retain_epoch_size)
                shuffle = False
            else:
                print("Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")
        elif reversed:
            if training:
                max_num = max(self.cls_num_list)
                class_weight = [max_num / i for i in self.cls_num_list]
                buckets = [[] for _ in range(num_classes)]
                for idx, label in enumerate(dataset.label_list):
                    buckets[int(label)].append(idx)
                sampler = ReversedSampler(buckets, retain_epoch_size, class_weight)
                shuffle = False 
            else:
                print("Test set will not be evaluated with reversed sampler")
        else:
            sampler = None
        
        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }
        self.val_init_kwargs = {
            'batch_size': batch_size,
            'shuffle': False, # For validation data,always false
            'num_workers': num_workers
        }
        super().__init__(dataset=self.dataset, **self.init_kwargs, sampler=sampler) # Note that sampler does not apply to validation set

    def split_validation(self):
        # If you do not want to validate:
        #return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.val_init_kwargs)


#######################
# hw1 dataloader
# Sanity check workload
# use hw1 data to verify correctness of training code
#######################
class P1_Dataset(Dataset):
    def __init__(self,data_path,val_mode):
        self.data_path = data_path
        self.val_mode = val_mode
        self.file_list=sorted([name for name in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, name)) ])
        self.num = len(self.file_list)
        print("load %d images from %s"%(self.num,self.data_path))
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((224,224)),
            transforms.RandomRotation(degrees=10, resample=Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    def __len__(self) -> int:
        return self.num
    def __getitem__(self, index):
        label_idx = int(re.findall(r'(\d+)_\d+.png',self.file_list[index])[0])
        # Preprocessing -> normalize image
        image_data = self.transform(os.path.join(self.data_path,self.file_list[index]))#.unsqueeze(0)
        return image_data,label_idx
##############
# Sampler
##############
###################################
# 2021 CVPR RIDE Balanced sampler #
###################################
class BalancedSampler(sampler.Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size
    
    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets]) # Acrually we need to upscale to next full batch
        else:
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num # Ensures every instance has the chance to be visited in an epoch

class ReversedSampler(sampler.Sampler):
    def __init__(self, buckets, retain_epoch_size=False, class_weight=None):
        for bucket in buckets:
            random.shuffle(bucket)
        self.class_weight = class_weight
        self.sum_weight = sum(self.class_weight)
        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size
    
    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(len(self.class_weight)):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = self.sample_class_index_by_weight()
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets]) # Acrually we need to upscale to next full batch
        else:
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num # Ensures every instance has the chance to be visited in an epoch

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset for sanity check 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start
        

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples
if __name__ == "__main__":
    import time 
    start = time.time()
    #D = FoodDataset("food_data/train","train",img_size = 384 )
    D = FoodTestDataset("food_data/testcase/sample_submission_comm_track.csv","food_data/test",img_size = 384 )
    d = D.__getitem__(0)
    print(d[0])

    print(time.time()-start)

