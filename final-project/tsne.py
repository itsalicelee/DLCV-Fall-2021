######################################################################################################
# compute the inter output (activation layer) of specific labels (num of samples, feature dim)
# compute the tsne, output the projected vectors X_emb: (num of sample, 2), label: (num of sample,)
# draw
######################################################################################################
import os
import torch
import argparse
from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['font.serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['SimHei']
import seaborn as sns
sns.set_style("darkgrid",{"font.sans-serif":['simhei', 'Arial']})

from model_zoo.swin.swin_transformer_vis import get_swin
from base_vis.dataset import FoodDataset,ChunkSampler,P1_Dataset
from util import *

def compute_embedding(val_loader, model, args):
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.layers[3].blocks[1].mlp.fc1.register_forward_hook(get_activation('layers[3].blocks.mlp.fc1'))
    inter_outputs = []
    labels = []
    predicts = []
    print('Computing activation...')
    model.eval()
    with torch.no_grad():
      for data, label in tqdm(val_loader):
        data = data.to(device)
        output, _ = model(data)
        _,pred_label=torch.max(output,1)
        inter_output = activation['layers[3].blocks.mlp.fc1']
        inter_output = inter_output.view(data.shape[0], -1).detach().cpu().numpy().astype('float64')
        inter_outputs.append(inter_output)
        predicts.append(pred_label.view(-1, 1).cpu().numpy())
        labels.append(label.view(-1, 1).numpy())

    inter_outputs = np.vstack(inter_outputs)
    labels = np.vstack(labels).squeeze(-1)
    predicts = np.vstack(predicts).squeeze(-1)

    np.save(os.path.join(args.model_path, 'inter_output_5class.npy'), inter_outputs)
    np.save(os.path.join(args.model_path, 'label_5class.npy'), labels)
    np.save(os.path.join(args.model_path, 'predict_5class.npy'), predicts)

    print(inter_outputs.shape, labels.shape)

    print('Computing TSNE...')
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(inter_outputs)
    print(X_embedded.shape)
    np.save(os.path.join(args.model_path, 'X_embedded_5class.npy'), X_embedded)

def plot_tsne(args, class_list):
    fin = open('../final-project-challenge-3-no_qq_no_life/food_data/label2name.txt', encoding='utf8')
    lines = fin.readlines()
    fin.close()
    id2name = {}
    for line in lines:
        label, freq, name = line.split()
        if freq == 'f':
            _freq = 'frequent'
        elif freq == 'c':
            _freq = 'common'
        else:
            _freq = 'rare'
        id2name[int(label)] = (_freq, name)

    name_order = []
    for c in class_list:
        name_order.append(id2name[c][1])
    name_order.append('other')
    name_order.append('dumb')

    label = np.load(os.path.join(args.model_path, 'label_5class.npy'))
    names_label = []
    freqs_label = []
    for l in label:
        names_label.append(id2name[l][1])
        freqs_label.append(id2name[l][0])


    predict = np.load(os.path.join(args.model_path, 'predict_5class.npy'))
    names_pred = []
    freqs_pred = []
    for l in predict:
        if id2name[l][1] not in name_order:
            names_pred.append('other')
        else:
            names_pred.append(id2name[l][1])
        freqs_pred.append(id2name[l][0])

    X_embedded = np.load(os.path.join(args.model_path, 'X_embedded_5class.npy'))
    print('Plotting label...')
    df = pd.DataFrame(zip(list(X_embedded[:, 0]), list(X_embedded[:, 1]), list(label), names_label, freqs_label), columns=['x', 'y', 'label', 'name', 'frequency'])
    # df = truncate(df, raw_class_list, num_per_class)
    
    palette = sns.color_palette("tab20", 12)  #Choosing color
    palette = dict(zip(name_order, palette))
    
    g = sns.scatterplot(data=df, x="x", y="y", hue='name', style='frequency', palette=palette, legend='full')
    g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # figure = g.get_figure()    
    # figure.savefig(os.path.join(args.model_path, 'tsne_raw&confuse.png'), bbox_inches='tight')
    ax = plt.gca()
    plt.xlim([-22, 22])
    plt.ylim([-25, 25])
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # plt.axis('off')
    plt.savefig(os.path.join(args.model_path, 'tsne_raw&confuse_50_label_5class.png'), bbox_inches='tight')

    plt.clf()
    plt.cla()

    print('Plotting predict...')
    palette = sns.color_palette("tab20", 20)  #Choosing color
    palette = dict(zip(name_order, palette))
    df = pd.DataFrame(zip(list(X_embedded[:, 0]), list(X_embedded[:, 1]), list(predict), names_pred, freqs_pred), columns=['x', 'y', 'label', 'name', 'frequency'])
    # df = truncate(df, raw_class_list, num_per_class)
    
    g = sns.scatterplot(data=df, x="x", y="y", hue='name', style='frequency', palette=palette, legend='full')
    g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # figure = g.get_figure()    
    # figure.savefig(os.path.join(args.model_path, 'tsne_raw&confuse.png'), bbox_inches='tight')
    ax = plt.gca()
    plt.xlim([-22, 22])
    plt.ylim([-25, 25])
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.savefig(os.path.join(args.model_path, 'tsne_raw&confuse_50_pred_5class.png'), bbox_inches='tight')



def truncate(df, raw_class_list, num_per_class):
    df_truncate = pd.DataFrame(columns=['x', 'y', 'label'])

    for c in raw_class_list:
        df_temp = df.loc[df['label']==c]
        if len(df_temp) > num_per_class:
            df_temp = df_temp.iloc[:num_per_class]
        df_truncate = df_truncate.append(df_temp)
    print('Original df len: {}, truncated df len: {}'.format(len(df), len(df_truncate)))
    return df_truncate
if __name__ == '__main__':
    # print(model)
    # layers[3].blocks.mlp.fc1
    parser = argparse.ArgumentParser()
    parser.add_argument("-load", "--load",default='',type=str , help='')
    parser.add_argument("-model_path", "--model_path",default="baseline",type=str , help='')
    
    parser.add_argument("-img_size", "--img_size", default=384,type=int , help='')
    parser.add_argument("-batch_size", "--batch_size", default=4,type=int , help='')
    parser.add_argument("-val_data_dir","--val_data_dir", default = "../final-project-challenge-3-no_qq_no_life/food_data/val",type=str, help ="Validation images directory")
    args = parser.parse_args()

    device = model_setting()
    fix_seeds(87)

    raw_class_list = [558, 925, 945, 827, 880, 800, 929, 633, 515, 326][:5]
    confuse_class_list = [610, 294, 485, 866, 88, 759, 809, 297, 936, 33][:5]
    num_per_class = 50

    class_list = []
    for r, c in zip(raw_class_list, confuse_class_list):
        class_list.append(r)
        class_list.append(c)

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

    # compute_embedding(val_loader, model, args)
    plot_tsne(args, class_list)

    

