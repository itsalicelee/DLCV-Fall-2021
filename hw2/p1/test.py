import os
import torchvision
import torch
import parser
import random
from model import G
from train import fix_random_seed
import parser
########TODO: remove this
# import numpy as np 
# from torch.utils.data import DataLoader, Dataset
# import torch.nn as nn
# from torch.autograd import Variable
# from torch.nn import functional as F
# import torch.utils.data
# from torchvision.models.inception import inception_v3
# from numpy import asarray
# from numpy import expand_dims
# from numpy import log
# from numpy import mean
# from numpy import exp
# import torchvision.transforms as transforms
# from scipy.stats import entropy
#########################

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
    IMG_SHOW = 32 # show tthe first 32 images


    with torch.no_grad():
        noise = torch.randn(IMG_TOTAL, INPUT_DIM, 1, 1, device=device)
        
        fake = model_G(noise).detach().cpu()
        show = fake[:IMG_SHOW]
        for i in range(fake.shape[0]):
            img = fake[i]
            #print(img)
            filename = os.path.join(args.save_dir, '{:04d}.png'.format(i+1))
            torchvision.utils.save_image(img, filename, normalize=True)
        # grid_name = os.path.join(args.save_dir, 'grid.png')
        #TODO: uncomment for show gird!
        # torchvision.utils.save_image(show, grid_name, nrow=8, normalize=True)
    # print("1000 images saved!")
    # print ("Calculating Inception Score...")
    # fake_dataset = FakeDataset(fake)
    # score = inception_score(fake_dataset, cuda=True, batch_size=32, resize=True, splits=10)
    # print("Inception score {}".format(score))

'''
class FakeDataset(Dataset):
    def __init__(self,orig):
        self.orig = orig
        self.transform = transforms.Compose([
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
    def __getitem__(self, index):
        x = self.orig[index]
        x = self.transform(x)
        return x
    def __len__(self):
        return len(self.orig)


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)    
'''







if __name__ == '__main__':
    main()
    