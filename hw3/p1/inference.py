'''
python3 inference.py --test_data=[folder] --test=[model.pth] --size=256 --save_path=[file.csv] 
'''
import torch
import torch.nn as nn
import parser
import os, sys
import numpy as np
import pandas as pd
from data import P1Dataset
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_vit import ViT

def test(args, test_loader, test_data, model, device):
    model.eval()
    labels = []
    correct_count = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            img, label = data[0].to(device), data[1].to(device)
            output = model(img)
            pred = torch.argmax(output, dim=1).detach()
            correct_count += (label == pred).sum()
            pred = pred.cpu().numpy()
            labels.append(pred)
   
    labels = np.concatenate(labels)
    filenames = test_data.filenames

    # evaluation
    accuracy = (correct_count / len(test_data)) * 100
    print('Accuracy: {}/{} ({:4f}%)'.format(correct_count, len(test_data), accuracy))

    # save predictions
    df = pd.DataFrame({'filename': filenames, 'label': labels})
    df.to_csv(os.path.join(args.save_path), index=False)


if __name__ == '__main__':
    args = parser.arg_parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # check pretrained model
    if args.test == '':
        sys.exit("Please specify pretrained model path!")
    # load data
    test_data = P1Dataset(root=args.test_data, mode="test", size=args.size)
    test_loader = DataLoader(test_data, batch_size=args.test_batch, num_workers=args.num_workers, shuffle=False)    
    # load model
    model = ViT('B_16_imagenet1k', num_classes=37, image_size=args.size, pretrained=True)
    model = model.to(device)
    state = torch.load(args.test)
    model.load_state_dict(state['state_dict'])
    test(args, test_loader, test_data, model, device)
