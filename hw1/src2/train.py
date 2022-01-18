from PIL.Image import FASTOCTREE
import torch
import torch.nn as nn
import torch.nn.functional as F
import parser
import os
from torch.nn.modules.loss import BCELoss, CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

    

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path, _use_new_zipfile_serialization=False)
    

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(output, dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def load_from_fcn8(model):

    checkpoint = torch.load('fcn8s-heavy-pascal.pth')
    states_to_load = {}
    lst = ['score_fr.weight', 'score_fr.bias',
            'score_pool3.weight', 'score_pool3.bias',
            'score_pool4.weight', 'score_pool4.bias',
            'upscore2.weight', 'upscore2.bias', 
            'upscore8.weight', 'upscore_pool4.weight']
    for name, param in checkpoint.items():
        if name not in lst:
            states_to_load[name] = param
    model_state = model.state_dict()
    model_state.update(states_to_load)
    model.load_state_dict(model_state)

def train(model, optimizer, train_loader,  test_loader, device):
    # torch.cuda.empty_cache()
    train_losses,  train_iou, train_acc = [], [], []
    args = parser.arg_parse()
    epoch = args.epoch
    log_interval = args.log_interval

    if args.lr_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=epoch, last_epoch=-1)
    if args.resume != '':
        load_checkpoint(args.resume, model, optimizer)
    
    # load_from_fcn8(model)

    criterion = CrossEntropyLoss()
    cur_best_epoch = -1
    best_acc = 0.0
    for ep in range(epoch):
        model.train() 
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target = target.long().to(device)
            optimizer.zero_grad()
            output = model(data)
            target = target.squeeze(1) 

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if args.lr_scheduler:
                scheduler.step()
            
        
            if batch_idx % log_interval == 0 and batch_idx > 0:  
                # print("Epoch:{} | Loss:{:.4f} | mIoU:{:.4f} | Train Acc:{:.4f}| Best Val Acc:{:4f}".format(ep, running_loss/len(train_loader), iou_score/len(train_loader), accuracy/len(train_loader), best_acc))
                print("Epoch:{} | Loss:{:.4f} | Best Val Acc:{:4f} from epoch {}".format(ep, loss.item(), best_acc, cur_best_epoch))
        

        acc = test(model, test_loader, device) # Evaluate at the end of each epoch
        if acc > best_acc:
            best_acc = acc
            cur_best_epoch = ep
            print("Saving best model... Best Acc is: {:.4f}".format(best_acc))
            save_checkpoint(os.path.join(args.save_dir, 'model_best_{}.pth'.format(ep)), model, optimizer) 

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        # print(tp_fp, tp_fn, tp)
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

def test(model, test_loader, device):
    model.eval()
    iou_score = 0.0
    with torch.no_grad(): 
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
        iou_score = mean_iou_score(outputs, masks)

    print("\n[Testing] mIoU:{:.4f}".format(iou_score))
            
    return iou_score

