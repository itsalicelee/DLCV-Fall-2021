import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import parser
from data import P1Dataset
import model
from train import train
from test import inference


if __name__=='__main__':
    args = parser.arg_parse()
    print(torch.__version__)
    print(torchvision.__version__)
    # create save directory
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)

    # fix random seed 
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    

    # dataloader 
    # print('===> Preparing dataloader ...')
    if args.mode == 'train': # training mode
        train_dataset = P1Dataset(root=args.train_data, mode="train")
        test_dataset = P1Dataset(root=args.test_data, mode="test")
        trainset_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, num_workers=args.num_workers)
        testset_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)
        print('# images in trainset:', len(train_dataset))
        print('# images in testset:', len(test_dataset))

        dataiter = iter(trainset_loader)
        images, labels = dataiter.next()

        print('Image tensor in each batch:', images.shape, images.dtype)
        print('Label tensor in each batch:', labels.shape, labels.dtype)
    elif args.mode == 'test': # testing mode
        print("===> Start testing...")
        test_dataset = P1Dataset(root=args.test_data, mode="test")
        testset_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)
    

    # set up device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
   
    
    # set up model 
    print('===> Preparing model ...')
    my_model = model.get_model().to(device)
    # my_model = model.Model().to(device)
    if torch.cuda.device_count()>1 and args.num_gpu:
        print("Using {} GPUs!".format(torch.cuda.device_count()))
        my_model = nn.DataParallel(my_model)
    # train(my_model, args.epoch, trainset_loader, testset_loader, device, args.log_interval)
    if args.mode == 'train': # training mode
        print('===> Start training ...')
        train(my_model, args.epoch, trainset_loader, testset_loader, device, args.log_interval)
    elif args.mode == 'test': # testing mode
        print('===> Start inferencing...')
        inference(args.test, my_model, testset_loader, test_dataset, device)
    
