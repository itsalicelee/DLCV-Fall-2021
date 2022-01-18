import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np

def write_images(outputs, save_dir, test_dataset):
    cls_color = {
        0:  [0, 255, 255],
        1:  [255, 255, 0],
        2:  [255, 0, 255],
        3:  [0, 255, 0],
        4:  [0, 0, 255],
        5:  [255, 255, 255],
        6: [0, 0, 0],
    }
    total = outputs.shape[0]
    for i in range(total):
        new = np.zeros((512, 512, 3), dtype=np.uint8)
        img = outputs[i]
        for c in cls_color:
            new[(img == c)] = cls_color[c]
        # print(new)
        
        result = Image.fromarray(new)
        result.save(os.path.join(save_dir, test_dataset.filenames[i][:-4] + ".png"))


def inference(checkpoint_path, save_dir, model, test_loader, test_dataset, device):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        outputs = np.zeros((len(test_loader), 512, 512))
        masks = np.zeros((len(test_loader), 512, 512))
        idx = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.long().to(device)
            output = torch.argmax(model(data), dim=1)
            target = target.squeeze(1)
            output = output.cpu()
            target = target.cpu()
            outputs[idx, :, :] = output[0]
            masks[idx, :, :] = target[0]
            idx += 1
        
        write_images(outputs, save_dir, test_dataset)
