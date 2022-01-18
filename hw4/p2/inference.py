import os
import parser
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from data import OfficeHomeDataset
from model import get_resnet
from tqdm import tqdm, trange


if __name__ == '__main__':
    args = parser.arg_parse()
    setting = args.setting
    if setting != 'inference':
        raise NotImplementedError("Choose setting inference for evaluation!")
    # fix random seeds for reproducibility
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    # set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("===> Using device: " + device)
    # data
    test_dataset = OfficeHomeDataset(args.test_csv, args.test_data_dir, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    # model
    model = get_resnet(args, setting).to(device)
    state = torch.load(args.load)
    model.load_state_dict(state['state_dict'])
    print("===> Start testing...")
    model.eval()
    classes = []
    correct_count = 0
    with torch.no_grad():
        for idx, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            output = model(image)
            pred = torch.argmax(output, dim=1).detach()
            correct_count += (label == pred).sum()
            pred = pred.cpu().numpy()
            classes.append(pred)
    classes = np.concatenate(classes)
     # save prediction
    print("===> Saving prediction...")
    accuracy = (correct_count / len(test_dataset)) * 100
    print('Accuracy: {}/{} ({:3f}%)'.format(correct_count, len(test_dataset), accuracy))
    filenames = test_dataset.filenames
    labels = []
    for c in classes:
        labels.append(test_dataset.class2label[c])
    ids = test_dataset.ids
    df = pd.DataFrame({'id': ids, 'filename': filenames, 'label': labels})
    df.to_csv(args.save_path, index=False)


