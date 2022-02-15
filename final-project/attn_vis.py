import os
import torch
import argparse
from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import torchvision.transforms as tranforms

from model_zoo.swin.swin_transformer_vis import get_swin
from base_vis.dataset import FoodDataset,ChunkSampler,P1_Dataset
from util import *

if __name__ == '__main__':
    # print(model)
    # layers[3].blocks.mlp.fc1
    parser = argparse.ArgumentParser()
    parser.add_argument("-load", "--load",default='',type=str , help='')
    parser.add_argument("-model_path", "--model_path",default="baseline",type=str , help='')
    
    parser.add_argument("-img_size", "--img_size", default=384,type=int , help='')
    parser.add_argument("-batch_size", "--batch_size", default=1,type=int , help='')
    parser.add_argument("-val_data_dir","--val_data_dir", default = "../final-project-challenge-3-no_qq_no_life/food_data/val",type=str, help ="Validation images directory")
    args = parser.parse_args()

    device = model_setting()
    fix_seeds(87)

    if not os.path.exists(os.path.join(args.model_path, 'attn')):
        os.makedirs(os.path.join(args.model_path, 'attn'))

    raw_class_list = [558, 925, 945, 827, 880, 800, 929, 633, 515, 326]
    confuse_class_list = [610, 294, 485, 866, 88, 759, 809, 297, 936, 33]
    class_list = raw_class_list + confuse_class_list
    num_per_class = 1


    val_dataset = FoodDataset(args.val_data_dir,img_size=args.img_size,mode = "val", class_list=class_list, num_per_class=num_per_class)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=8)
    model = get_swin(ckpt='./model_zoo/swin/swin_large_patch4_window12_384_22kto1k.pth')
    # print(model)
    if args.load:
        model.load_state_dict(torch.load(args.load))
        print("model loaded from {}".format(args.load))
    model.to(device)
    model.eval()
    resize = tranforms.Resize((384,384))
    with torch.no_grad():
      for i, (data, label) in enumerate(tqdm(val_loader)):
        data = data.to(device)
        output, attn = model(data) # attn: 1, 48, 144, 144
        attn = attn.squeeze(0).cpu().numpy() # (48, 144, 144)
        avg_attn_map = attn[1, :, :]
        # avg_attn_map = np.mean(attn, axis=0) # (144, 144)
        avg_attn_map = np.mean(avg_attn_map, axis=0)
        avg_attn_map = np.reshape(avg_attn_map, (12,12))

        original_image = val_dataset.getOriginalImage(i)
        avg_attn_map = np.array(resize(Image.fromarray(avg_attn_map)))
        print(attn.shape, original_image.shape)
        plt.cla()
        plt.clf()
        plt.axis('off')
        plt.imshow(original_image)
        plt.imshow(avg_attn_map, alpha=0.5, cmap='rainbow')
        plt.savefig(os.path.join(args.model_path, 'attn, '{}.png'.format(label.item())))