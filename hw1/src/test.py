import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import parser
import matplotlib.pyplot as plt
import os


def inference(checkpoint_path, model, test_loader, test_dataset, device):
    args = parser.arg_parse()

    print("===> Loading model...")
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])

    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    predict = []
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # inference 
            _, test_pred = torch.max(output, 1) # get the index of the class with the highest probability
            for y in test_pred.cpu().numpy():
                predict.append(y)
                
    # write and save file
    save_csv(predict, test_dataset, args.prediction)

    # t-SNE
    # plot_tsne(model, test_loader, device)
    
def save_csv(prediction, test_dataset, filepath="prediction.csv"):
    print("===> Writing predictions...")
    img_id = create_ids(test_dataset)
    assert len(img_id) == len(prediction), "Length are not the same!"
    dict = {
        "image_id": img_id,
        "label": prediction
        }
    pd.DataFrame(dict).to_csv(filepath, index=False)
    

def create_ids(test_dataset):
    filenames = []
    for i in range(len(test_dataset)):
        filenames.append(test_dataset.filenames[i])
    return filenames


def check_result(predict):
    count = 0
    idx = 0
    for i in range(50):
        for _ in range(50):
            if i == predict[idx]:
                count += 1
            idx += 1           
    print("Congrats! Acc: {}".format(count/2500))

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

'''
def plot_tsne(model,test_loader,device):
    
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            if i == 0: 
                X = data      
            else:
                X = np.vstack((X,data))
               
        X = torch.FloatTensor(X).to(device)
        print(X.shape)
        # model.layer4[2].conv2.register_forward_hook(get_activation('layer4[2].conv2'))
        model.avgpool.register_forward_hook(get_activation('avgpool'))
        output = model(X)
    print(activation)
    activation['avgpool'] = activation['avgpool'].squeeze()

    X_tsne = TSNE(n_components=2).fit_transform(activation['avgpool'].cpu())

    #Normalize
    X_min, X_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - X_min) / (X_max - X_min) 

    # create labels
    y = []
    for i in range(50):
        for j in range(50):
            y.append(i)

    df = pd.DataFrame()
    df["y"] = y
    df["comp-2"] = X_norm[:,1]
    df["comp-1"] = X_norm[:,0]


    # seaborn
    g = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 50),
                    data=df, legend = False).set(title="T-SNE projection") 
    plt.savefig("TSNE.png")
'''
