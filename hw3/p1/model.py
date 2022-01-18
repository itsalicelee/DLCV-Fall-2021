import parser
import torch
import torch.nn as nn
from pytorch_pretrained_vit import ViT
from timm import create_model

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        #self.model = ViT('B_16', num_classes=37, image_size=224, pretrained=True, )
        #self.model.fc = nn.Sequential(
        #        nn.Linear(in_features=768, out_features=37, bias=False),
        #)
        self.model.head = nn.Linear(in_features=192, out_features=37, bias=True)
        self.model.mlp = nn.Sequential(
                        nn.Linear(in_features=192, out_features=768, bias=True),
                        nn.Dropout(p=0.3, inplace=False),
                        nn.GELU(),
                        nn.Linear(in_features=768, out_features=37, bias=True),
                        )
        


    def forward(self, x):
        x = self.model(x)
        return x

        


if __name__ == '__main__':
    args = parser.arg_parse()
    
