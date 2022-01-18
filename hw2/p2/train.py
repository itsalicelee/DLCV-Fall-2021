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
from model import G, D, weights_init
from torchvision.utils import save_image

def load_checkpoint(pathG, pathD,  model_G, optimizer_G, model_D, optimizer_D):
    state_G = torch.load(pathG)
    state_D = torch.load(pathD)
    model_G.load_state_dict(state_G['state_dict'])
    model_D.load_state_dict(state_D['state_dict'])
    optimizer_G.load_state_dict(state_G['optimizer'])
    optimizer_D.load_state_dict(state_D['optimizer'])


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


def sample_image(n_row, epoch , model_G, log_dir):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = torch.from_numpy(np.random.normal(0, 1, (n_row ** 2, 100)))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    print(labels)
    filename = os.path.join(log_dir, "{:03d}.png".format(epoch))
    gen_imgs = model_G(z, labels)
    save_image(gen_imgs.data, filename, nrow=n_row, normalize=True)

def smooth_positive_labels(y):
    return np.random.uniform(low=0.8, high=1.2, size=y.shape)

def smooth_negative_labels(y):
    return np.random.uniform(low=0.0, high=0.3, size=y.shape)
    
def main():
    args = parser.arg_parse()
    n_class = 10 # from digit 0 to digit 9 
    #TODO: remove this
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("===> Using device " + device)

    # fix_random_seed(args.random_seed)
    
    print("===> Preparing dataloader...")
    train_data = DigitDataset(root=args.train_data, type=args.type, mode="train")
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

    dis_criterion = nn.BCELoss()
    aux_criterion = nn.CrossEntropyLoss()

    optimizer_D = torch.optim.AdamW(model_D.parameters(), lr=args.lr_d, betas=(args.beta1, 0.999))
    optimizer_G = torch.optim.AdamW(model_G.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))

    if args.ckpt_g and args.ckpt_d:
        load_checkpoint(args.ckpt_g, args.ckpt_d, model_G, optimizer_G, model_D, optimizer_D)
    if args.lr_scheduler:
        scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=args.epochs)
        scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=args.epochs)
    g_iter = args.g_iter
    d_iter = args.d_iter


    
    fixed_noise = torch.randn(10, 100, 1, 1, device=device) # (10, 100, 1, 1)
    fixed_label = torch.tensor([0,1,2,3,4,5,6,7,8,9]).to(device)
    fixed_noise_label = concat_noise_label(fixed_noise, fixed_label, device) # (10, 110, 1, 1)

    print(fixed_label)

    print("===> Start training...")
    for epoch in range(1, args.epochs+1):
        if epoch > 50: 
            g_iter = 1 # train g for g_iter times when epoch < 50
            d_iter = 1 # train d for d_iter times when epoch < 50
        iter = 0
        loss_G = 0.0
        loss_D = 0.0
        D_real_acc = 0.0
        D_fake_acc = 0.0
        for idx, data in enumerate(train_loader):
            # torchvision.utils.save_image(data[0], "train.png", nrow=10, normalize=True)
            iter+=1
            real_img, real_class = data[0].to(device), data[1].to(device)

            batch_size = real_img.size(0) # cannot use args.train_batch since it will cause error in last batch
            model_D.train()
            model_G.train()

            real_target = torch.full((batch_size,), 1.0, device=device).float().to(device)
            real_target_smooth = smooth_positive_labels(real_target)
            fake_target = torch.full((batch_size,), 0.0, device=device).float().to(device)
            fake_target_smooth = smooth_negative_labels(fake_target)

            real_target_smooth = torch.from_numpy(real_target_smooth).float().to(device)
            fake_target_smooth = torch.from_numpy(fake_target_smooth).float().to(device)
            #####################
            # train Discriminator
            #####################

           
            for _ in range(d_iter):
                model_D.zero_grad()
                #### REAL image
                output_dis, output_aux = model_D(real_img)
                output_dis = torch.squeeze(output_dis)
                output_aux = torch.squeeze(output_aux)
                loss_D_dis_real = dis_criterion(output_dis, real_target_smooth) # real or fake
                loss_D_aux_real = aux_criterion(output_aux, real_class) # digit 
                D_real_acc = np.mean(((output_dis > 0.5).cpu().data.numpy() == real_target.cpu().data.numpy()))
                loss_D_real = (loss_D_dis_real + loss_D_aux_real)/2
                # loss_D_real.backward()

                ### FAKE image
                fake_class = torch.randint(n_class, (batch_size,), dtype=torch.long, device=device)
                noise = torch.randn(batch_size, 100, 1, 1, device=device).squeeze(0)
                input_z = concat_noise_label(noise, fake_class, device)  
                # print(fake_class.shape) #(B)
                # print(noise.shape) # (B,110, 1, 1)
                fake_image = model_G(input_z)

                output_dis, output_aux = model_D(fake_image.detach()) # detach for no grad
                output_dis = torch.squeeze(output_dis)
                output_aux = torch.squeeze(output_aux)
                loss_D_dis_fake = dis_criterion(output_dis, fake_target_smooth) # real or fake
                loss_D_aux_fake = aux_criterion(output_aux, fake_class) # digit TODO: real label?
                D_fake_acc = np.mean(((output_dis > 0.5).cpu().data.numpy() == fake_target.cpu().data.numpy()))
                loss_D_fake = (loss_D_dis_fake + loss_D_aux_fake)/2
                # loss_D_fake.backward()
                

                loss_D = loss_D_fake + loss_D_real
                loss_D.backward()

                optimizer_D.step()

            if args.lr_scheduler:
                scheduler_D.step()


            #####################
            # train Generator
            #####################
            for _ in range(g_iter):
                model_G.zero_grad()
                noise = torch.randn(batch_size, 100, 1, 1, device=device)
                fake_class = torch.randint(n_class, (batch_size,), dtype=torch.long, device=device)
                input_z = concat_noise_label(noise, fake_class, device)  
                fake_image = model_G(input_z)

                output_dis, output_aux = model_D(fake_image)
                output_dis = torch.squeeze(output_dis)
                output_aux = torch.squeeze(output_aux)
                loss_G_dis = dis_criterion(output_dis, real_target)  # real or fake
                loss_G_aux = aux_criterion(output_aux, fake_class)  # digit 
                loss_G = loss_G_dis + loss_G_aux
                loss_G.backward()
                optimizer_G.step()
            if args.lr_scheduler:
                scheduler_G.step()


            if (iter+1) % args.log_interval == 0:
                # print("Epoch: {} [{}/{}] | G Loss:{:.4f} | D Loss: {:.4f}".format(epoch, idx+1, len(train_loader), errG.item(), errD.item()))
                print("Epoch: {} [{}/{}] | G Loss:{:.4f} | D Loss: {:.4f} | D real Acc: {:.4f} | D fake Acc: {:4f}".format(epoch, idx+1, len(train_loader), loss_G, loss_D, D_real_acc, D_fake_acc))
                
           
        with torch.no_grad():
                model_G.eval()
                fake_imgs_sample = (model_G(fixed_noise_label)).detach().cpu()
                model_G.train()
                filename = os.path.join(args.log_dir, "{:03d}.jpg".format(epoch))
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