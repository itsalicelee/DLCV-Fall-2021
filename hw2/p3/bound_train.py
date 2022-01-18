import os
import torchvision
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import parser
from data import DigitDataset
import random
import model
from numpy.random import choice
import pandas as pd
import csv, sys

def load_checkpoint_F(pathF, feature_extractor, optimizer_F):
    state_F = torch.load(pathF)
    feature_extractor.load_state_dict(state_F['state_dict'])
    optimizer_F.load_state_dict(state_F['optimizer'])
def load_checkpoint_L(pathL, label_predictor, optimizer_L):
    state_L = torch.load(pathL)
    label_predictor.load_state_dict(state_L['state_dict'])
    optimizer_L.load_state_dict(state_L['optimizer'])



def init_weights(layer):
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)

# randomly flip some labels
def noisy_labels(y, p_flip):
    n_select = int(p_flip * y.shape[0])
    flip_ix = choice([i for i in range(y.shape[0])], size=n_select)
    y[flip_ix] = 1 - y[flip_ix]
    zeros = torch.zeros(y.shape, dtype=torch.double)
    y = torch.where(y>0, y, zeros)
    return y


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def onehot_encode(label, device, n_class=10):  
    eye = torch.eye(n_class, device=device) 
    return eye[label].view(-1, n_class, 1, 1)   
 
def concat_noise_label(noise, label, device):
    oh_label = onehot_encode(label, device=device)
    return torch.cat((noise, oh_label), dim=1)

def smooth_positive_labels(y):
    return np.random.uniform(low=0.8, high=1.2, size=y.shape)

def smooth_negative_labels(y):
    return np.random.uniform(low=0.0, high=0.3, size=y.shape)

def test(test_loader, test_data, feature_extractor, label_classifier, device):
    feature_extractor.eval()
    label_classifier.eval()
    labels = []
    # Use image and label for target domain
    
    correct_count = 0
    for idx, data in enumerate(test_loader):
        img, label = data[0].to(device), data[1].to(device)
        features = feature_extractor(img)
        features = features.view(features.shape[0], -1)
        output = label_classifier(features)
        pred = torch.argmax(output, dim=1).detach()
        correct_count += (label == pred).sum()
        pred = pred.cpu().numpy()
        labels.append(pred)
       
    labels = np.concatenate(labels)

    # switch model for training
    feature_extractor.train()
    label_classifier.train()
    # evaluation
    accuracy = (correct_count / len(test_data)) * 100
    print('Accuracy: {}/{} ({:3f}%)'.format(correct_count, len(test_data), accuracy))
    return accuracy, labels



def main():
    args = parser.arg_parse()
    #TODO: remove this
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("===> Using device " + device)

    fix_random_seed(args.random_seed)
    
    print("===> Preparing dataloader...")
    train_data = DigitDataset(root=args.train_data, type=args.source, mode="train")
    train_loader = DataLoader(train_data, batch_size=args.train_batch, num_workers=args.num_workers, shuffle=True)
    test_data = DigitDataset(root=args.test_data, type=args.target, mode="test")
    test_loader = DataLoader(test_data, batch_size=args.test_batch, num_workers=args.num_workers, shuffle=False)    

    

    print("===> Loading model...")
    feature_extractor = model.Extractor().to(device)
    label_predictor = model.LabelPredictor().to(device)

    print("======= Feature Extractor =======")
    print(feature_extractor)
    print("\n======= Label Predictor =======")
    print(label_predictor)


    class_criterion = nn.CrossEntropyLoss()

    optimizer_F = torch.optim.Adam(feature_extractor.parameters(), lr=args.lr_f)
    optimizer_L = torch.optim.Adam(label_predictor.parameters(), lr=args.lr_l)

    if args.ckpt_f:
        load_checkpoint_F(args.ckpt_f, feature_extractor, optimizer_F)
    if args.ckpt_l:
        load_checkpoint_L(args.ckpt_l, label_predictor, optimizer_L)

    if args.lr_scheduler:
        scheduler_F = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_F, T_max=args.epochs)
        scheduler_L = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_L, T_max=args.epochs)



    print("===> Start training...")
    best_acc = 0.0
    best_epoch = -1
    for epoch in range(1, args.epochs+1):
        total_hit = 0
        total_num = 0
        iter = 0
        for idx, (data, label) in enumerate(train_loader):
            iter += 1
            data = data.to(device)
            label = label.to(device)
            batch_size = data.size(0)

            features = feature_extractor(data)
            features = features.view(features.shape[0], -1)
            class_output = label_predictor(features)
            loss = class_criterion(class_output, label)
            loss.backward()
            optimizer_F.step()
            optimizer_L.step()

            optimizer_F.zero_grad()
            optimizer_L.zero_grad()

            total_hit += torch.sum(torch.argmax(class_output, dim=1) == label).item()
            total_num += data.shape[0]
            if (idx+1) % 10 == 0:
                print("Epoch: {} [{}/{}] | Label Loss: {:.4f} | Label Acc: {:.4f} | Best Acc: {:.4f} ".format(epoch, idx+1, len(train_loader), loss.item(), total_hit/total_num, best_acc))

        
        if epoch % 2 == 0:

            print("Saving models...")
            state = {'state_dict': label_predictor.state_dict(),
                'optimizer' : optimizer_L.state_dict()}
            torch.save(state, os.path.join(args.log_dir, "{:03d}_L.pth".format(epoch)))
            state = {'state_dict': feature_extractor.state_dict(),
                'optimizer' : optimizer_F.state_dict()}
            torch.save(state, os.path.join(args.log_dir, "{:03d}_F.pth".format(epoch)))
            





if __name__ == '__main__':
    main()
