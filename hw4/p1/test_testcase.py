import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from model import ConvNet, MLP

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)



def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
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

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def distance(dis_type, x, y):
    '''
    x: (way*query,z)  outputs of query image
    y: (way,z)  prototype
    '''
    if x.shape[1] != y.shape[1]:
        raise AssertionError('shape mismatch! x:{}, y:{}'.format(x.shape, y.shape))
    
    n = x.shape[0] # way
    m = y.shape[0] # way*query

    if dis_type == 'euclidean':
        a = x.unsqueeze(1).expand(n, m, -1) # [way, way*query, z)
        b = y.unsqueeze(0).expand(n, m, -1)
        result = ((a - b)**2).sum(dim=2)
        return result

    elif dis_type == 'cosine':
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        result = cos(x.unsqueeze(1).expand(n, m, -1), y.unsqueeze(0).expand(n, m, -1))
        return result

    elif dis_type == 'parametric':
        x = x.unsqueeze(1).expand(n,m,-1).reshape(n*m,-1)
        y = y.unsqueeze(0).expand(n,m,-1).reshape(n*m,-1)
        z = torch.cat((x,y), dim=-1)
        mlp= MLP().to(device)
        result = mlp(z)
        return result.reshape(n, m)
    else:
        raise NotImplementedError("No such Loss function!")

def predict(args, model, data_loader):
    prediction_results = []

    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):

            # split data into support and query data
            support_input = data[:args.N_way * args.N_shot,:,:,:] 
            query_input   = data[args.N_way * args.N_shot:,:,:,:]

            # create the relative label (0 ~ N_way-1) for query data
            label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])

            # extract the feature of support and query data
            prototype = model(support_input.to(device))
            outputs = model(query_input.to(device))

            # calculate the prototype for each class according to its support data
            prototype = prototype.reshape(args.N_way, args.N_shot, -1).mean(dim=1) #  [n_way, z]

            # classify the query data depending on the its distense with each prototype
            d = distance(args.loss, outputs, prototype)
            pred = (-d).softmax(dim=1).max(1, keepdim=True)[1]

            pred = pred.reshape(-1)
            prediction_results.append(pred)
            

    return prediction_results

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N_way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N_shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N_query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, default='../hw4_data/mini/val.csv', help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, default='../hw4_data/mini/val', help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, default='../hw4_data/mini/val_testcase.csv', help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")
    parser.add_argument('--loss', type=str, default="euclidean")
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    # fix random seeds for reproducibility
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)
    np.random.seed(SEED)

    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("===> Using device: " + device)
    model = ConvNet().to(device)
    state = torch.load(args.load)
    model.load_state_dict(state['state_dict'])

    print("===> Predicting...")
    print("Using {} as loss function...".format(args.loss))
    prediction_results = predict(args, model, test_loader)

    # output prediction to csv
    print("===> Writing predictions...")
    lines = []
    with open(args.output_csv, 'w') as f:
        title = 'episode_id'
        for i in range(args.N_way * args.N_query):
            title += ',query{}'.format(i)
        lines.append(title)

        for i, pred in enumerate(prediction_results):
            # for every episode
            line = str(i)
            for p in pred:
                line += ',{}'.format(p)
            lines.append(line)

        f.writelines("%s\n" % l for l in lines)
