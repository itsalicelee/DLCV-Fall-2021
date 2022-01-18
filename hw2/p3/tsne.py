from sklearn.manifold import TSNE
import torch.nn as nn
import torch
import os
import pandas as pd
import parser
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import Extractor
from data import DigitDataset
from torch.utils.data import DataLoader, Dataset


def main():
    args = parser.arg_parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("===> Loading model...")
    state = torch.load(args.ckpt_f)
    model = Extractor().to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    print("===> Preparing Dataloader...")
    source_data = DigitDataset(root=args.train_data, type=args.source, mode="test")
    source_loader = DataLoader(source_data, batch_size=args.train_batch, num_workers=args.num_workers, shuffle=False)
    target_data = DigitDataset(root=args.train_data, type=args.target, mode="test")
    target_loader = DataLoader(target_data, batch_size=args.train_batch, num_workers=args.num_workers, shuffle=False)
    
    latent = np.zeros((4000, 48*4*4))
    gt = np.zeros((4000,))
    domain = np.zeros((4000,)) # 0 for target 1 for source
    iters = 0
    print('===> Loading source loader...')
    for idx, data in enumerate(source_loader):
        img, label = data[0], data[1]
        img = img.to(device)
        feature = model(img)
        feature = feature.view(-1, 48 * 4 * 4)
        for j in range(feature.shape[0]):
            latent[iters] = feature[j].detach().cpu().numpy()
            gt[iters] = label[j].item()
            domain[iters] = 0
            iters += 1
            if iters == 2000: 
                break
        if iters == 2000:
            break
    print('===> Loading target loader...')
    for idx, (img, label) in enumerate(target_loader):
        img = img.to(device)
        feature = model(img)
        feature = feature.view(-1, 48 * 4 * 4)
        for j in range(feature.shape[0]):
            latent[iters] = feature[j].detach().cpu().numpy()
            gt[iters] = label[j].item()
            domain[iters] = 1
            iters += 1
            if iters == 4000: 
                break
        if iters == 4000:
            break

    print('===> Creating TSNE...')
    tsne_data = TSNE(n_components=2, init='pca', learning_rate=300, n_iter=3000, n_iter_without_progress=1000).fit_transform(latent)
    sns.set(rc={'figure.figsize':(16,12)})


    # plot domain
    print('===> Creating domain TSNE...')
    df = pd.DataFrame()
    df["y"] = domain
    df["comp-1"] = tsne_data[:,0]
    df["comp-2"] = tsne_data[:,1]
    name = (args.source).upper() + " to " + (args.target).upper()
    print('===> Creating scatterplot...')
    g = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 2),
                    data=df, legend = True) 
    g.axes.set_title(name,fontsize=40)
    plt.savefig(name.replace(' ', '_') + "_domain.png")

    # plot digit
    print('===> Creating digit TSNE...')
    df = pd.DataFrame()
    df["y"] = gt
    df["comp-1"] = tsne_data[:,0]
    df["comp-2"] = tsne_data[:,1]
    print('===> Creating scatterplot...')
    g = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 10),
                    data=df, legend = True)
    g.axes.set_title(name,fontsize=40)
    plt.savefig(name.replace(' ', '_') + "_digit.png")


   

if __name__ == "__main__":
    args = parser.arg_parse()
    main()