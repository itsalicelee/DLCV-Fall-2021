import os
import torchvision
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import parser
from data import DigitDataset
import random
from model import G
from train import fix_random_seed
import parser


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def onehot_encode(label, device, n_class=10):  
    eye = torch.eye(n_class, device=device) 
    return eye[label].view(-1, n_class, 1, 1)   
 
def concat_image_label(image, label, device, n_class=10):
    B, C, H, W = image.shape   
    oh_label = onehot_encode(label, device=device)
    oh_label = oh_label.expand(B, n_class, H, W)
    return torch.cat((image, oh_label), dim=1)
 
def concat_noise_label(noise, label, device):
    oh_label = onehot_encode(label, device=device)
    return torch.cat((noise, oh_label), dim=1)

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path, map_location = "cuda")
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

def eval(args, device):
    net = Classifier().to(device)
    net.eval()
    path = "./p2/Classifier.pth"
    load_checkpoint(path, net)
    
    test_dataset = DigitDataset(root="./p2/inference", type="mnistm", mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)
    correct = 0 
    labels = []
    for idx, (img, label) in enumerate(test_dataloader):
        img, label = img.to(device), label.to(device)
        output = net(img)
        pred = torch.argmax(output, dim=1).detach()
        pred = pred.to(device)        
        correct += (label == pred).sum()
        pred = pred.cpu().numpy()
        labels.append(pred)
    
    labels = np.concatenate(labels)
    
    acc = (correct / len(test_dataset)) * 100
    print('Accuracy: {}/{} ({:3f}%)'.format(correct, len(test_dataset), acc))


def main():
    args = parser.arg_parse()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("===> Using device {}".format(device))
    fix_random_seed(args.random_seed)
    
    
    model_G = G().to(device)
    state = torch.load(args.test)
    model_G.load_state_dict(state['state_dict'])
    model_G.eval()

    # define some parameters
    IMG_TOTAL = 1000
    INPUT_DIM = 100
    IMG_SHOW = 100


    with torch.no_grad():
        
        fixed_noise = torch.randn(IMG_TOTAL, INPUT_DIM, 1, 1, device=device) # (10, 100, 1, 1)
        lst = []
        for i in range(10):
            for j in range(100):
                lst.append(i) # lst[0,0,0...1,1,1...]
        # print(lst)
        fixed_label = torch.from_numpy(np.array(lst)).to(device)
        
        fixed_noise_label = concat_noise_label(fixed_noise, fixed_label, device) 
        # print(fixed_noise_label.shape)  # (1000, 110, 1, 1)
        fake = model_G(fixed_noise_label).detach().cpu()
        show = [None] * IMG_SHOW
        cnt = 0
        for i in range(10):
            for j in range(1,101):
                if j <= 10:
                    show[10*(j-1)+i] = fake[cnt]
                img = fake[cnt]
                filename = os.path.join(args.save_dir, '{}_{:03d}.png'.format(i,j))
                torchvision.utils.save_image(img, filename, normalize=True)
                cnt += 1
        # print(show)
        # grid_name = os.path.join('grid.png')
        # torchvision.utils.save_image(show, grid_name, nrow=10, normalize=True)
        
    
    # print("1000 images saved!")
    # eval(args, device) #TODO: remove



if __name__ == '__main__':
    main()
    