import argparse
import torch
import os
import numpy as np
import random
from torchvision import models
from byol_pytorch import BYOL
from data import MiniDataset, OfficeHomeDataset

def parse_args():
    parser = argparse.ArgumentParser(description="HW4_P2_Pretrain_MiniImageNet")
    parser.add_argument('--log_dir', type=str, default='ckpts/pretrained')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--load', type=str, default='')
    return parser.parse_args()

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def sample_unlabelled_images():
    all_images = [i for i in range(len(dataset))]
    indices = random.sample(all_images, args.batch_size)
    images = [dataset[idx][0] for idx in indices]
    result = torch.stack(images)
    return result

if __name__ == '__main__':
    args = parse_args()
    resnet = models.resnet50(pretrained=False)
    if args.load:
        print("Loading model from checkpoint...")
        resnet.load_state_dict(torch.load(args.load))
    
    learner = BYOL(
        resnet,
        image_size = 128,
        hidden_layer = 'avgpool'
    )

    opt = torch.optim.Adam(learner.parameters(), lr=args.lr)
    dataset = MiniDataset("../hw4_data/mini/train.csv", "../hw4_data/mini/train")

    for epoch in range(args.epochs):
        print("=====Epoch {}=====".format(epoch))
        images = sample_unlabelled_images()
        loss = learner(images)
        print("Loss:{:.4f}".format(loss.item()))
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder

        # save your improved network
        if epoch % 5 == 0:
            torch.save(resnet.state_dict(), os.path.join(args.log_dir,'ssl_model3.pth'.format(epoch)))