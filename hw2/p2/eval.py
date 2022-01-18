import os
import torchvision
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, dataloader
import torch.nn.functional as F
import parser
from data import DigitDataset
import random
from model import G
from train import fix_random_seed
import parser

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path, map_location = "cuda")
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

def eval(args, device):


    
    net = Classifier().to(device)
    net.eval()
    path = "./Classifier.pth"
    load_checkpoint(path, net)
    
    test_dataset = DigitDataset(root="./output", type="mnistm", mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)
    correct = 0 
    labels = []
    for idx, (img, label) in enumerate(test_dataloader):
        img, label = img.to(device), label.to(device)
        output = net(img)
        pred = torch.argmax(output, dim=1).detach()
        pred = pred.to(device)        
        correct += (label == pred).sum()
        pred = pred.cpu().numpy()
        labels.append(pred)
    
    labels = np.concatenate(labels)
    
    acc = (correct / len(test_dataset)) * 100
    print('Accuracy: {}/{} ({:3f}%)'.format(correct, len(test_dataset), acc))




if __name__ == '__main__':
    args = parser.arg_parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("===> Using device {}".format(device))
    fix_random_seed(args.random_seed)
    eval(args, device)