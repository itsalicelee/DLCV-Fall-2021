import os
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

def load_checkpoint_F(pathF, feature_extractor, optimizer_F):
    state_F = torch.load(pathF)
    feature_extractor.load_state_dict(state_F['state_dict'])
    optimizer_F.load_state_dict(state_F['optimizer'])
def load_checkpoint_L(pathL, label_predictor, optimizer_L):
    state_L = torch.load(pathL)
    label_predictor.load_state_dict(state_L['state_dict'])
    optimizer_L.load_state_dict(state_L['optimizer'])
def load_checkpoint_D(pathD, domain_classifier, optimizer_D):
    state_D = torch.load(pathD)
    domain_classifier.load_state_dict(state_D['state_dict'])
    optimizer_D.load_state_dict(state_D['optimizer'])


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


def main():
    args = parser.arg_parse()
    #TODO: remove this
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("===> Using device " + device)

    fix_random_seed(args.random_seed)
    
    print("===> Preparing dataloader...")
    source_data = DigitDataset(root=args.train_data, type=args.source, mode="train")
    source_loader = DataLoader(source_data, batch_size=args.train_batch, num_workers=args.num_workers, shuffle=True)
    target_data = DigitDataset(root=args.train_data, type=args.target, mode="train")
    target_loader = DataLoader(target_data, batch_size=args.train_batch, num_workers=args.num_workers, shuffle=True)

    

    print("===> Loading model...")
    feature_extractor = model.Extractor().to(device)
    label_predictor = model.LabelPredictor().to(device)
    domain_classifier = model.DomainClassifier().to(device)

    print("======= Feature Extractor =======")
    print(feature_extractor)
    print("\n======= Label Predictor =======")
    print(label_predictor)
    print("\n======= Domain Classifier =======")
    print(domain_classifier)


    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss() 

    optimizer_F = torch.optim.Adam(feature_extractor.parameters(), lr=args.lr_f)
    optimizer_L = torch.optim.Adam(label_predictor.parameters(), lr=args.lr_l)
    optimizer_D = torch.optim.Adam(domain_classifier.parameters(), lr=args.lr_d)

    if args.ckpt_f:
        load_checkpoint_F(args.ckpt_f, feature_extractor, optimizer_F)
    if args.ckpt_l:
        load_checkpoint_L(args.ckpt_l, label_predictor, optimizer_L)
    if args.ckpt_d:
        load_checkpoint_D(args.ckpt_d, domain_classifier, optimizer_D)

    if args.lr_scheduler:
        scheduler_F = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_F, T_max=args.epochs)
        scheduler_L = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_L, T_max=args.epochs)
        scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=args.epochs)



    steps = min(len(source_loader), len(target_loader))
    print("===> Start training...")
    best_acc = 0.0
    best_epoch = -1
    for epoch in range(1, args.epochs+1):
        class_losses = 0.0
        domain_losses = 0.0
        total_losses = 0.0
        total_hit = 0
        total_num = 0
        iter = 0
        for idx, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_loader, target_loader)):
            '''
            source loader: use image, label
            target loader: use image only!
            '''
            iter += 1
            source_data = source_data.to(device)
            source_label = source_label.to(device)
            target_data = target_data.to(device)
            batch_size = source_data.size(0)

            # TODO: 
            lamb = 0.1
            # p = float(idx + (epoch) * steps) / (steps * args.epochs)
            # lamb = 2.0/(1.0+np.exp(-10*p)) - 1

            D_src = torch.ones(batch_size, 1).to(device) # 1 for source domain
            D_tar = torch.zeros(batch_size, 1).to(device) # 0 for target domain
            D_labels = torch.cat([D_src, D_tar], dim=0) # concat domain labels

            
            #########################
            #   Domain Classifier
            #########################
            mixed_data = torch.cat([source_data, target_data], dim=0)
            features = feature_extractor(mixed_data)
            features = features.view(features.shape[0], -1)
            if idx == 0:
                FEATURE_SHAPE = features.shape
            if features.shape != FEATURE_SHAPE:
                break
            domain_output = domain_classifier(features.detach()) # do not train feature extractor

            domain_loss = domain_criterion(domain_output, D_labels)
            domain_loss.backward()
            optimizer_D.step()

            #########################
            #   Feature Extractor
            #   &  Label Predictor
            #########################
            class_output = label_predictor(features[:batch_size]) # we only have source domain label
            domain_output = domain_classifier(features)
            loss = class_criterion(class_output, source_label) - lamb * domain_criterion(domain_output, D_labels)
            loss.backward()
            optimizer_F.step()
            optimizer_L.step()

            optimizer_D.zero_grad()
            optimizer_F.zero_grad()
            optimizer_L.zero_grad()

            total_hit += torch.sum(torch.argmax(class_output, dim=1) == source_label).item()
            total_num += source_data.shape[0]
            if (idx+1) % 10 == 0:
                print("Epoch: {} [{}/{}] | Domain Loss:{:.4f} | Label Loss: {:.4f} | Label Acc: {:.4f} | Best Acc: {:.4f} ".format(epoch, idx+1, len(source_loader), domain_loss, loss, total_hit/total_num, best_acc))

        
        
        if epoch % 2  == 0:

            print("Saving models...")
            state = {'state_dict': label_predictor.state_dict(),
                'optimizer' : optimizer_L.state_dict()}
            torch.save(state, os.path.join(args.log_dir, "{:03d}_L.pth".format(epoch)))
            state = {'state_dict': domain_classifier.state_dict(),
                'optimizer' : optimizer_D.state_dict()}
            torch.save(state, os.path.join(args.log_dir, "{:03d}_D.pth".format(epoch)))
            state = {'state_dict': feature_extractor.state_dict(),
                'optimizer' : optimizer_F.state_dict()}
            torch.save(state, os.path.join(args.log_dir, "{:03d}_F.pth".format(epoch)))
            





if __name__ == '__main__':
    main()
