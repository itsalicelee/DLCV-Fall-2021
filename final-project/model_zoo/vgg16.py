import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG
import numpy as np


class VGG16(nn.Module):
    def __init__(self,num_class=1000):
        super().__init__()
        self.vgg16 = torch.hub.load('pytorch/vision:v0.5.0', 'vgg16_bn', pretrained=True)
        # fix imageNet layer
        self.vgg16.classifier[6] = nn.Linear(4096,num_class)
        print(type(self.vgg16.classifier))
    def forward(self,image):
        output = self.vgg16.features.forward(image)
        output = self.vgg16.avgpool(output)
        output = output.view(output.shape[0],-1)
        features = self.vgg16.classifier[0:6](output)
        output = self.vgg16.classifier[6](features)
        return output