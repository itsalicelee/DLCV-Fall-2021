import os
import torchvision
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import parser
from data import P1Dataset
import random
from numpy.random import choice
from model import G, D, weights_init

def load_checkpoint(pathG, pathD,  model_G, optimizer_G, model_D, optimizer_D):
    state_G = torch.load(pathG)
    state_D = torch.load(pathD)
    model_G.load_state_dict(state_G['state_dict'])
    model_D.load_state_dict(state_D['state_dict'])
    optimizer_G.load_state_dict(state_G['optimizer'])
    optimizer_D.load_state_dict(state_D['optimizer'])

    # load from previous 
    checkpoint = torch.load(pathG)
    states_to_load = {}
    for name, param in checkpoint['state_dict'].items():
        if name.startswith('main'):
            states_to_load[name] = param
    model_state = model_G.state_dict()
    model_state.update(states_to_load)
    model_G.load_state_dict(model_state)

    print('model loaded from {} and {}'.format(pathG, pathD))

# randomly flip some labels
def noisy_labels(y, p_flip):
    # determine the number of labels to flip
    n_select = int(p_flip * y.shape[0])
    # choose labels to flip
    flip_ix = choice([i for i in range(y.shape[0])], size=n_select)
    # invert the labels in place
    y[flip_ix] = 1 - y[flip_ix]
    zeros = torch.zeros(y.shape, dtype=torch.double)
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

def smooth_positive_labels(y):
    return np.random.uniform(low=0.85, high=1.0, size=y.shape)

def smooth_negative_labels(y):
    return np.random.uniform(low=0.0, high=0.2, size=y.shape)

def main():
    args = parser.arg_parse()
    
    #TODO: remove this
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("===> Using device " + device)

    # fix_random_seed(args.random_seed) # do not fix random seed in training
    
    print("===> Preparing dataloader...")
    train_data = P1Dataset(root=args.train_data, mode="train")
    train_loader = DataLoader(train_data, batch_size=args.train_batch, num_workers=args.num_workers, shuffle=True)
    print("===> Loading model...")
    model_G = G().to(device)
    model_G.apply(weights_init)
    model_D = D().to(device)
    model_D.apply(weights_init)

    print("======= Generator =======")
    print(model_G)
    print("\n======= Discriminator =======")
    print(model_D)

    criterion = nn.BCELoss()

    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.lr_d, weight_decay=args.weight_decay, betas=(0.5, 0.999))
    # optimizer_D = torch.optim.SGD(model_D.parameters(), lr=args.lr_d, weight_decay=args.weight_decay)

    if args.ckpt_g and args.ckpt_d:
        load_checkpoint(args.ckpt_g, args.ckpt_d, model_G, optimizer_G, model_D, optimizer_D)
    if args.lr_scheduler:
        scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=args.epochs)
        scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=args.epochs)
    fixed_noise = torch.randn((100, 100, 1, 1)).to(device)
    g_iter = args.g_iter
    d_iter = args.d_iter

    print("===> Start training...")
    for epoch in trange(1, args.epochs+1):
        if epoch > 100: 
            g_iter = 1
            d_iter = 1
        
        iter = 0
        loss_G = 0.0
        loss_D = 0.0
        for idx, data in enumerate(train_loader):
            if epoch < 100:
                noisy_p = 0.05 - epoch/100 * 0.03
            else:
                noisy_p = 0.02 - (args.epochs-epoch)/(args.epochs-100) * 0.02
            real_img = data[0]
            iter+=1
            model_D.train()
            model_G.train()
            batch_size = real_img.size(0) # cannot use args.train_batch since it will cause error in last batch

            ######## Train D ########

            for _ in range(d_iter):
                model_D.zero_grad()

                # forward real image 
                real_img = real_img.to(device)
                label = torch.full((batch_size,), 1., dtype=torch.float, device=device)
                # label = noisy_labels(label, p_flip=noisy_p)
                # label = smooth_positive_labels(label.cpu())
                label = label.to(device)
                output = model_D(real_img).view(-1)
                label = label.float()
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item() # average prediction acc of discriminator to real images

                # forward random noise to generate fake image
                noise = torch.randn(batch_size, 100, 1, 1, device=device)
                fake = model_G(noise)
                label.fill_(0.)
                label = smooth_negative_labels(label.cpu())
                label = torch.from_numpy(label).float().to(device)
                output = model_D(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item() # average prediction acc of discriminator to generator outputs
                errD = errD_real + errD_fake 
                optimizer_D.step()
            

            # ######## Train G ########
            
            for _ in range(g_iter):
                model_G.zero_grad()
                label.fill_(1.)
                # label = noisy_labels(label, p_flip=noisy_p)
                # label = smooth_positive_labels(label.cpu())
                label = label.float().to(device)
                output = model_D(fake).view(-1).float()
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item() # average prediction acc of discriminator to generator outputs
                optimizer_G.step()


            if (iter+1) % args.log_interval == 0:
                print("Epoch: {} [{}/{}] | G Loss:{:.4f} | D Loss: {:.4f} | D_x: {:.4f} | D_G_z1: {:.4f} | D_G_z2: {:.4f}".format(epoch, idx+1, len(train_loader), errG.item(), errD.item(),  D_x, D_G_z1, D_G_z2))
                
            
        
       
            #fake_imgs_sample = (model_G(noise).data)/2
           
        with torch.no_grad():
            model_G.eval()
            fake_imgs_sample = (model_G(fixed_noise)).detach().cpu()
            model_G.train()
            #print(fake_imgs_sample)
            #print(fake_imgs_sample.shape) #(64, 3, 64, 64)
            #fake_imgs_sample = toImage(batch_size, fake_imgs_sample
            filename = os.path.join(args.log_dir, "epoch_{:03d}.jpg".format(epoch))
            # torchvision.utils.save_image(fake_imgs_sample, filename, nrow=8)
            torchvision.utils.save_image(fake_imgs_sample, filename, nrow=10, normalize=True)
        if epoch%5 == 0:
            print("Saving model G...")
            state = {'state_dict': model_G.state_dict(),
                'optimizer' : optimizer_G.state_dict()}
            torch.save(state, os.path.join(args.log_dir, "{:03d}_G.pth".format(epoch)))
            print("Saving model D...")
            state = {'state_dict': model_D.state_dict(),
                'optimizer' : optimizer_D.state_dict()}
            torch.save(state, os.path.join(args.log_dir, "{:03d}_D.pth".format(epoch)))
            






if __name__ == '__main__':
    main()
