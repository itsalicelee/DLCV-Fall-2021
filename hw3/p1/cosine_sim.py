'''
Usage: python3 cosine_sim.py --test=[model path]
model_type: choose from models in ViT
'''
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pytorch_pretrained_vit import ViT
import parser
import torch.nn.functional as F


if __name__ == '__main__':
    args = parser.arg_parse()
    ckpt = args.test
    state = torch.load(ckpt, map_location=torch.device('cpu'))
    model = ViT('B_16_imagenet1k', num_classes=37, image_size=args.size, pretrained=True)
    model.load_state_dict(state['state_dict'])
    emb = model.positional_embedding.pos_embedding[0]
    result = []
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    for i in range(1, emb.shape[0]):
        sim = cos(emb[i:i+1], emb[1:])
        sim = sim.reshape((16, 16)).detach().cpu().numpy()
        result.append(sim)
    
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("Visualization of positional embeddings", fontsize=20)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(16, 16),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
    for ax, im in zip(grid, result):
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.imshow(im)

    plt.savefig("./results/cosine_sim.png")
