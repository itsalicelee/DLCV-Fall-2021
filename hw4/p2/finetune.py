import os
import parser
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import OfficeHomeDataset
from model import get_resnet
from tqdm import tqdm, trange

def load_checkpoint(path, model, optimizer):
    state = torch.load(path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

def test(test_loader, test_data, model, device):
    print("=====================================================")
    model.eval()
    labels = []
    correct_count = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            img, label = data[0].to(device), data[1].to(device)
            output = model(img)
            pred = torch.argmax(output, dim=1).detach()
            correct_count += (label == pred).sum()
            pred = pred.cpu().numpy()
            labels.append(pred)
    labels = np.concatenate(labels)
    # switch model for training
    model.train()
    # evaluation
    accuracy = (correct_count / len(test_data)) * 100
    print('Accuracy: {}/{} ({:3f}%)'.format(correct_count, len(test_data), accuracy))
    return accuracy, labels


if __name__ == '__main__':
    args = parser.arg_parse()
    setting = args.setting
    if setting not in ['a','b','c','d','e']:
        raise NotImplementedError("Choose setting from a-e!")
    
    '''
    # fix random seeds for reproducibility
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    '''
    # create log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("===> Using device: " + device)
    # data
    train_dataset = OfficeHomeDataset(args.train_csv, args.train_data_dir, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    test_dataset = OfficeHomeDataset(args.test_csv, args.test_data_dir, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    # model
    model = get_resnet(args, setting).to(device)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer, filter out the layers that are fixed 
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler
    if args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == "step":
        scheduler =  StepLR(optimizer, step_size=args.step)
    
    if args.resume:
        load_checkpoint(args.resume, model, optimizer)

    best_acc = 0
    best_epoch = -1
    print("===> Start training...")
    for epoch in trange(args.epochs):
        model.train()
        total_hit = 0
        total_num = 0
        for idx, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()

            optimizer.step()
            if args.lr_scheduler:
                scheduler.step()

            optimizer.zero_grad()
            total_hit += torch.sum(torch.argmax(outputs, dim=1) == label).item()
            total_num += image.shape[0]
            if idx % args.log_interval == 0 and idx != 0:
                print("Epoch: {} [{}/{}] | Loss: {:.4f} | Acc: {:.4f} | Best Acc: {:.4f} ({})".format(
                    epoch, idx, len(train_loader), loss.item(), total_hit/total_num, best_acc, best_epoch))
            
        # Evaluate after each epoch
        acc, classes = test(test_loader, test_dataset, model, device)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            # save prediction
            print("===> Saving prediction...")
            filenames = test_dataset.filenames
            labels = []
            for c in classes:
                labels.append(test_dataset.class2label[c])
            ids = test_dataset.ids
            df = pd.DataFrame({'id': ids, 'filename': filenames, 'label': labels})
            df.to_csv(args.save_path, index=False)

            print("Saving models...")
            state = {'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()}
            torch.save(state, os.path.join(args.log_dir, "{:04d}.pth".format(epoch)))

