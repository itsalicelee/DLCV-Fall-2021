import torch
import torch.nn as nn
import torchvision.models as models

        
def get_model():
    print("===> loading resnet34...")
    model =  models.resnet34(pretrained=True)
    # model.classifier[6].out_features = 50
    # for param in model.features.parameters():
    #     param.requires_grad = False
    print("===> changing last layer...")

    # change last layer to 50
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
                nn.Dropout(0.8),
                 nn.Linear(num_features, 50)
    )

    # add dropout to layers
    # feats_list = list(model.features)
    # new_feats_list = []
    # for feat in feats_list:
    #     new_feats_list.append(feat)
    #     if isinstance(feat, nn.Conv2d):
    #         new_feats_list.append(nn.Dropout(p=0.15))
    # model.features = nn.Sequential(*new_feats_list)
    print(model)
    '''
    # num_features = model.classifier[6].in_features
    # features = list(model.classifier[6].children())[:-1] # Remove last layer
    # features.extend([nn.Linear(num_features, 50)]) # Add our layer with 50 outputs
    # model.classifier = nn.Sequential(*features) # Replace the model classifier
    # print(model)
    '''
    return model

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.model = models.resnet34(pretrained=True)
#         num_features = self.model.fc.in_features
#         self.fc =   nn.Sequential(
#                     nn.Dropout(0.8),
#                     nn.Linear(num_features, 50)
#                     )
#         self.hook = None
#     def fowward(self, x):
#         x = self.model.layer1(x)
#         x = self.model.layer2(x)
#         x = self.model.layer3(x)
#         x = self.model.layer4[0](x)
#         x = self.model.layer4[1](x)
#         x = self.model.layer4[2].conv1(x)
#         x = self.model.layer4[2].bn1(x)
#         x = self.model.layer4[2].relu(x)
#         x = self.model.layer4[2].conv2(x)
#         self.hook = x
#         x = self.model.layer4[2].bn2(x)
#         x = self.model.avgpool(x)
#         x = self.model.fc(x)
#         return x