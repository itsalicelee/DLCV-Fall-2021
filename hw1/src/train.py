import torch
import torch.nn as nn
import parser
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path, _use_new_zipfile_serialization=False)

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def train(model, epoch, train_loader,  test_loader, device, log_interval):
    args = parser.arg_parse()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, last_epoch=-1)
    
    if args.resume != '':
        load_checkpoint(args.resume, model, optimizer)


    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for ep in range(epoch):
        model.train() 
        correct = 0.0
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            train_loss += loss.item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            loss.backward()
            optimizer.step()
            if args.lr_scheduler:
                scheduler.step()
            
            acc = 100. * correct / len(train_loader.dataset)
            if batch_idx % log_interval == 0 and batch_idx > 0:  
                print("Epoch:{} Train Acc: {:3f} ({}/{}) | Loss:{:.4f} | Best Val Acc:{:4f}".format(ep, 100.* correct/(batch_idx * len(data)),int(correct), batch_idx * len(data),loss.item(), best_acc))
            
        acc = test(model, test_loader, device) # Evaluate at the end of each epoch
        if acc > best_acc:
            best_acc = acc
            print("Saving best model... Best Acc is: {:.3f}".format(best_acc))
            save_checkpoint(os.path.join(args.save_dir, 'model_best_{}.pth'.format(ep)), model, optimizer) 

def test(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return acc
