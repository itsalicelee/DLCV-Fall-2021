import torch
import torch.nn as nn
from torchvision import models

def get_resnet(args, setting):
    resnet = models.resnet50(pretrained=False)
    resnet.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(256, 65),
    )
    print("================================")
    print(resnet)

    if setting == 'a':
        print("Setting A: \nNo pretrain\nfine tune full model (backbone + classifier)\n")
        
    elif setting == 'b':
        print("Setting B: \nProvided pretained backbone\nfine tune full model (backbone + classifier)\n")
        resnet.load_state_dict(torch.load(args.provided_ckpt), strict=False)
    
    elif setting == 'c':
        print("Setting C: \nSSL pre-trained backbone\nfine tune full model (backbone + classifier)\n")
        resnet.load_state_dict(torch.load(args.ssl_ckpt), strict=False)

    elif setting == 'd':
        print("Setting D: \nProvided pretained backbone\nFix the backbone. Train classifier only\n")
        resnet.load_state_dict(torch.load(args.provided_ckpt), strict=False)

        for name, param in resnet.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False # fix backbone


    elif setting == 'e':
        print("Setting E: \nSSL pre-trained backbone\nFix the backbone. Train classifier only\n")
        resnet.load_state_dict(torch.load(args.ssl_ckpt), strict=False)
    
        for name, param in resnet.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False # fix backbone
    
    elif setting == 'inference':
        pass

    return resnet