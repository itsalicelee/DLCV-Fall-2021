import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import parser
from data import DigitDataset
import random
import model
import pandas as pd


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    args = parser.arg_parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fix_random_seed(args.random_seed)

    test_data = DigitDataset(root=args.inf_testdata, type=args.inf_target, mode="inf")
    test_loader = DataLoader(test_data, batch_size=args.test_batch, num_workers=args.num_workers, shuffle=False)    

    feature_extractor = model.Extractor().to(device)
    label_predictor = model.LabelPredictor().to(device)

    if args.improved == False:
        if args.inf_target == 'mnistm':
            state_F = torch.load('./p3/svhn2mnistm_F.pth')
            state_L = torch.load('./p3/svhn2mnistm_L.pth')
        elif args.inf_target == 'usps':
            state_F = torch.load('./p3/mnistm2usps_F.pth')
            state_L = torch.load('./p3/mnistm2usps_L.pth')
        elif args.inf_target == 'svhn':
            state_F = torch.load('./p3/usps2svhn_F.pth')
            state_L = torch.load('./p3/usps2svhn_L.pth')
    else: # improved model
        if args.inf_target == 'mnistm':
            state_F = torch.load('./p3/im_svhn2mnistm_F.pth')
            state_L = torch.load('./p3/im_svhn2mnistm_L.pth')
        elif args.inf_target == 'usps':
            state_F = torch.load('./p3/im_mnistm2usps_F.pth')
            state_L = torch.load('./p3/im_mnistm2usps_L.pth')
        elif args.inf_target == 'svhn':
            state_F = torch.load('./p3/im_usps2svhn_F.pth')
            state_L = torch.load('./p3/im_usps2svhn_L.pth')

    feature_extractor.load_state_dict(state_F['state_dict'])
    label_predictor.load_state_dict(state_L['state_dict'])

    feature_extractor.eval()
    label_predictor.eval()

    labels = []
    image_names = test_data.filenames
    cnt = 0
    for idx, data in enumerate(test_loader):
        img, label = data[0].to(device), data[1].to(device)
        features = feature_extractor(img)
        features = features.view(features.shape[0], -1)
        output = label_predictor(features)
        pred = torch.argmax(output, dim=1).detach()
        pred = pred.cpu().numpy()
        labels.append(pred)


    labels = np.concatenate(labels)
    df = pd.DataFrame({'image_name': image_names, 'label': labels})
    df.to_csv(os.path.join(args.save_dir), index=False)

if __name__ == '__main__':
    main()

