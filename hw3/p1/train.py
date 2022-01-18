import torch
import torch.nn as nn
import numpy as np
import parser 
import os
from data import P1Dataset
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_vit import ViT
from tqdm import tqdm, trange
import pandas as pd
from model import Model
from timm import create_model

    
def load_checkpoint(path, model, optimizer):
    state = torch.load(path)
    model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer'])

def fix_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def test(test_loader, test_data, model, device):
    print("=====================================================")
    model.eval()
    labels = []
    correct_count = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            # print(data.shape)
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


def main():
    args = parser.arg_parse()
    fix_random_seed(args.random_seed)
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("===> Using device " + device)

    print("===> Preparing dataloader...")
    train_data = P1Dataset(root=args.train_data, mode="train", size=args.size)
    train_loader = DataLoader(train_data, batch_size=args.train_batch, num_workers=args.num_workers, shuffle=True)
    test_data = P1Dataset(root=args.val_data, mode="val", size=args.size)
    test_loader = DataLoader(test_data, batch_size=args.test_batch, num_workers=args.num_workers, shuffle=False)    
    
    # model = timm.create_model('resnet34', num_classes=37, pretrained=True, drop_rate=0.8)
    model_name = args.model_name
    model = ViT('B_16_imagenet1k', num_classes=37, image_size=args.size, pretrained=True)
    # model = create_model(model_name, num_classes=37, pretrained=True).to(device)
    # model = Model(model_name=model_name).to(device)
    # model = Model().to(device)
    
    print(model)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_scheduler != 'none':    
        if args.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.lr_scheduler == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        elif args.lr_scheduler == 'lambda':
            warmup_steps = int(args.warmup_ratio * len(train_loader) * args.epochs / args.accum_iter)
            noam_lambda = lambda step:(768**(-0.5) * min((step + 1) ** (-0.5), (step + 1) * warmup_steps ** (-1.5)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)
    
    if args.resume: 
        load_checkpoint(args.resume, model, optimizer)
    
    best_acc = 0
    best_epoch = -1
    for epoch in trange(1, args.epochs+1):
        model.train()
        total_hit = 0
        total_num = 0
        for idx, data in enumerate(train_loader):
            img, label = data[0].to(device), data[1].to(device)
            with torch.set_grad_enabled(True):
                outputs = model(img)
                loss = criterion(outputs, label)
                loss.backward()
            
            if((idx + 1) % args.accum_iter == 0) or ((idx + 1) == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
                if args.lr_scheduler != 'none':
                    scheduler.step()
            total_hit += torch.sum(torch.argmax(outputs, dim=1) == label).item()
            total_num += img.shape[0]
            if idx % args.log_interval == 0:
                print("Epoch: {} [{}/{}] | Loss: {:.4f} | Acc: {:.4f} | Best Acc: {:.4f} ({})".format(epoch, idx, len(train_loader), loss.item(), total_hit/total_num, best_acc, best_epoch))
        
        acc, labels = test(test_loader, test_data, model, device)
        
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            # save prediction
            print("===> Saving prediction...")
            image_names = test_data.filenames
            df = pd.DataFrame({'filename': image_names, 'label': labels})
            df.to_csv(args.save_path, index=False)

            print("Saving models...")
            state = {'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()}
            torch.save(state, os.path.join(args.log_dir, "{:03d}.pth".format(epoch)))

            
            



if __name__ == '__main__':
    main()
