import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import parser
from data import P2Dataset
from model import FCN32, FCN8s, UNet
from train import train, save_checkpoint
from test import inference

if __name__=='__main__':
    args = parser.arg_parse()
    print(torch.__version__)
    print(torchvision.__version__)
    # create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # fix random seed 
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    
    # dataloader 
    print('===> Preparing dataloader ...')
    if args.mode == 'train': # training mode
        train_dataset = P2Dataset(root=args.train_data, mode="train")
        val_dataset = P2Dataset(root=args.val_data, mode="val")
        trainset_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, num_workers=args.num_workers)
        valset_loader = DataLoader(val_dataset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)
        print('# images in trainset:', len(train_dataset))
        print('# images in testset:', len(val_dataset))

    elif args.mode == 'test': # testing mode
        print("===> Start testing...")
        test_dataset = P2Dataset(root=args.test_data, mode="test")
        testset_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)
    

    # set up device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("===> Using device: {}".format(device))
    
    # set up model 
    print('===> Preparing model ...')
    if args.model.lower() == 'fcn8':
        model = FCN8s().to(device)
    elif args.model.lower() == 'fcn32':
        model = FCN32(n_class=7).to(device)
    elif args.model.lower() == 'unet':
        model = UNet(n_class=7).to(device)
    
    print("===> Creating optimizer...")

    if args.optimizer.lower() == 'adam' :
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif  args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
    
    if torch.cuda.device_count()>1 and args.num_gpu:
        print("Using {} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    if args.mode == 'train': # training mode
        print('===> Start training ...')
        try:
            train(model, optimizer, trainset_loader, valset_loader, device)
        except KeyboardInterrupt:
            save_checkpoint("INTERRUPTED.pth", model, optimizer)
    
    elif args.mode == 'test': # testing mode
        print('===> Start inferencing...')
        inference(args.test, args.result_dir, model, testset_loader, test_dataset, device)
    