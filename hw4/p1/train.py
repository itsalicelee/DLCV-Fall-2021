import os
import parser
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
from model import ConvNet, weights_init, MLP
from data import MiniImageNetDataset, NwayKshotSampler
from test_testcase import MiniDataset, worker_init_fn


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
        return result.reshape(n,m)
    else:
        raise NotImplementedError("No such Loss function!")

def load_checkpoint(path, model, optimizer):
    state = torch.load(path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

def save_checkpoint(path, model, optimizer, epoch):
    state = {'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}
    torch.save(state, os.path.join(args.log_dir, "{:03d}.pth".format(epoch)))

def train(args, device, train_loader, model, optimizer):
    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.lr_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_acc = []
    best_mean, best_epoch = -1, -1
    for epoch in range(1, args.epochs+1):
        print("====== Epoch:{} ======".format(epoch))
        model.train()
        for idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            image, label = data[0], data[1]
            image = image.to(device)
            support_image = image[:args.n_way * args.k_shot,:,:,:] 
            query_image   = image[args.n_way * args.k_shot:,:,:,:]

            label_encoder = {label[i * args.k_shot] : i for i in range(args.n_way)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in label[args.n_way * args.k_shot:]])

            # extract features
            prototype = model(support_image)
            outputs = model(query_image)
            # calculate the prototype for each class
            prototype = prototype.reshape(args.n_way, args.k_shot, -1).mean(dim=1) #  [n_way, z]
            
            # calculate loss
            d = distance(args.loss, outputs, prototype)
            loss = criterion(-d, query_label)

            # backward loss and step optimizer
            loss.backward()
            optimizer.step()
            if args.lr_scheduler:
                scheduler.step()

            pred = (-d).softmax(dim=1).max(1, keepdim=True)[1]
            pred = pred.reshape(query_label.shape)
            acc = (query_label == pred).sum() / len(query_label)
            train_acc.append(acc.item())
            if (idx + 1) % args.log_interval == 0:
                episodic_acc = np.array(train_acc)
                mean = episodic_acc.mean()
                std = episodic_acc.std()
                print("Epoch {:3d}/{:03d}({:03d}/{:03d})| Loss:{:.4f} | Acc: {:.3f}% | Best Acc: {:.3f}%({})".format(epoch, args.epochs, 
                    (idx + 1), len(train_loader), loss.item(), mean * 100, best_mean*100, best_epoch))
                if mean > best_mean:
                    best_mean = mean
                    best_epoch = epoch
                    save_checkpoint(args.log_dir, model, optimizer, epoch)


if __name__ == '__main__':
    args = parser.arg_parse()
    # fix random seeds for reproducibility
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)
    np.random.seed(SEED)
    # create log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("===> Using device: " + device)
    # data
    train_dataset = MiniDataset(args.train_csv, args.train_data_dir)
    train_sampler = NwayKshotSampler(
            csv_path=args.train_csv,
            episodes_per_epoch=args.n_batch,
            n_way=args.n_way,
            k_shot=args.k_shot,
            k_query=args.k_query
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_sampler=train_sampler,
        worker_init_fn=worker_init_fn,
)
    # load model
    print("===> Loading model...")
    model = ConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # load checkpoint or init weights
    if args.ckpt == '':
        model.apply(weights_init)
    else:
        load_checkpoint(path=args.ckpt, model=model, optimizer=optimizer)
    
    print("===> Start training...")
    train(args, device, train_loader, model, optimizer)
