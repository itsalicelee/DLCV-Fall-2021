import torch
import time
import os
from scipy.io import savemat
from tqdm import tqdm 
from util import gen_logger
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

class BaseTrainer():
    def __init__(self,
                 device,
                 model,
                 optimizer,
                 scheduler,
                 MAX_EPOCH,
                 criterion,
                 train_loader,
                 val_loader,
                 lr,
                 batch_size,
                 gradaccum_size, 
                 model_path,
                 save_period):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.MAX_EPOCH = MAX_EPOCH
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_path = model_path
        self.scheduler = scheduler
        self.save_period = save_period
        self.lr = lr
        self.batch_size = batch_size
        self.gradaccum_size = gradaccum_size
        self.logger = gen_logger(os.path.join(model_path,"console.log"))
        # training related
        self.val_best_acc = -1.0
        self.info_list = {"train_loss":[],
                          "val_loss":[],
                          "train_acc":[],
                          "val_acc":[],
                          "lr":lr,
                          "batch_size":batch_size
                         }
    def train(self):
        for epoch in range(self.MAX_EPOCH):
            start = time.time()
            self.logger.info("====================================================")
            # self.logger.info("Epoch {}: train".format(epoch))
            # self._train(epoch)
            # self.logger.info("Epoch {}: validation".format(epoch))
            self._valid(epoch)
            # self._valid_visualize(epoch)
            self.logger.info("Total {:5f} sec per epoch".format(time.time()-start))
            # log training info per epoch
            self.save_info_list()
    def _train(self,epoch):
        self.model.train()
        train_nums = 0.0
        train_loss = 0.0
        train_acc = 0.0
        # train
        loss_step = 1
        for batch_idx,(data,label) in enumerate(tqdm(self.train_loader)):
            data,label =data.to(self.device),label.to(self.device)
            output = self.model(data)
            loss = self.criterion(output,label)
            train_loss +=loss.item()
            # self.logger.info("Epoch:{} | batch: {}| loss: {}".format(epoch,batch_idx,train_loss))
            # update: backpropagation,lr schedule
            loss = loss / self.gradaccum_size
            loss.backward()
            if (loss_step % self.gradaccum_size == 0) or ((batch_idx + 1) == len(self.train_loader)):
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_step += 1
            _,pred_label=torch.max(output,1)
            train_acc+=(label==pred_label).sum().item()
            train_nums+=pred_label.size(0)
        if self.scheduler is not None:
            self.scheduler.step()
        train_acc_rate = train_acc/train_nums
        self.logger.info("Training loss:{:5f},Training Accuracy:{:5f}".format(train_loss,train_acc_rate) )
        # save info about loss,accurracy
        self.info_list["train_acc"].append(train_acc_rate)
        self.info_list["train_loss"].append(train_loss)
    def _valid(self,epoch):
        val_nums = 0.0
        val_loss = 0.0
        val_acc = 0.0
        val_nums_freq = 0.0
        val_nums_common = 0.0
        val_nums_rare = 0.0
        val_acc_freq = 0.0
        val_acc_common = 0.0
        val_acc_rare = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch_idx,(data,label) in enumerate(tqdm(self.val_loader)):
                data,label =data.to(self.device),label.to(self.device)
                output =self.model(data)
                val_loss += self.criterion(output,label).item()
                _,pred_label=torch.max(output,1)
                val_acc+=(label==pred_label).sum().item()
                val_nums+=pred_label.size(0)
                for i in range(len(label)):
                    if (self.val_loader.dataset.freq_list[label[i]] == 0):
                        val_acc_freq += (label[i]==pred_label[i]).item()
                        val_nums_freq += 1
                    elif (self.val_loader.dataset.freq_list[label[i]] == 1):
                        val_acc_common += (label[i]==pred_label[i]).item()
                        val_nums_common += 1
                    else:
                        val_acc_rare += (label[i]==pred_label[i]).item()
                        val_nums_rare += 1      
        val_acc_rate = val_acc/val_nums
        self.logger.info("Validation loss:{:5f} Validation accuracy:{:5f}".format(val_loss,val_acc_rate ) )
        self.logger.info("Val_acc {:d} Val_nums {:d}".format(int(val_acc),int(val_nums)) )
        val_acc_rate_freq = val_acc_freq/val_nums_freq
        val_acc_rate_common = val_acc_common/val_nums_common
        val_acc_rate_rare = val_acc_rare/val_nums_rare
        self.logger.info("Validation accuracy (freq) : {:5f}".format(val_acc_rate_freq))
        self.logger.info("Val_acc {:d} Val_nums {:d} (freq)".format(int(val_acc_freq),int(val_nums_freq)) )
        self.logger.info("Validation accuracy (common) : {:5f}".format(val_acc_rate_common))
        self.logger.info("Val_acc {:d} Val_nums {:d} (common)".format(int(val_acc_common),int(val_nums_common)) )
        self.logger.info("Validation accuracy (rare) : {:5f}".format(val_acc_rate_rare))
        self.logger.info("Val_acc {:d} Val_nums {:d} (rare)".format(int(val_acc_rare),int(val_nums_rare)) )
        # save info about loss,accurracy
        self.info_list["train_acc"].append(val_acc_rate)
        self.info_list["val_loss"].append(val_loss)
        # if self.val_best_acc < val_acc_rate:
        #     self.val_best_acc = val_acc_rate
        #     torch.save(self.model.state_dict(), os.path.join(self.model_path,"model_best.pth")) 
        #     self.logger.info("Save best model")
        # if epoch % self.save_period == 0:
        #     torch.save(self.model.state_dict(), os.path.join(self.model_path,"model_{:d}.pth".format(epoch))) 

    def save_info_list(self):
        savemat("{}/Loss.mat".format(self.model_path), self.info_list)
        self.logger.info("Save learning history to {}/Loss.mat".format(self.model_path))

    def _valid_visualize(self, epoch):
        val_nums = 0.0
        val_loss = 0.0
        val_acc = 0.0
        val_nums_freq = 0.0
        val_nums_common = 0.0
        val_nums_rare = 0.0
        val_acc_freq = 0.0
        val_acc_common = 0.0
        val_acc_rare = 0.0
        self.model.eval()

        y_true = []
        y_pred = []
        label2correct = dict(zip([i for i in range(1000)], [0 for i in range(1000)]))
        label2total = dict(zip([i for i in range(1000)], [0 for i in range(1000)]))
        with torch.no_grad():
            for batch_idx,(data,label) in enumerate(tqdm(self.val_loader)):
                data,label =data.to(self.device),label.to(self.device)
                output =self.model(data)
                val_loss += self.criterion(output,label).item()
                _,pred_label=torch.max(output,1)
                val_acc+=(label==pred_label).sum().item()
                val_nums+=pred_label.size(0)
                y_pred.extend(pred_label.cpu().tolist())
                y_true.extend(label.cpu().tolist())
                for i in range(len(label)):
                    label2correct[label[i].item()] += (label[i]==pred_label[i]).item()
                    label2total[label[i].item()] += 1
                    if (self.val_loader.dataset.freq_list[label[i]] == 0):
                        val_acc_freq += (label[i]==pred_label[i]).item()
                        val_nums_freq += 1
                    elif (self.val_loader.dataset.freq_list[label[i]] == 1):
                        val_acc_common += (label[i]==pred_label[i]).item()
                        val_nums_common += 1
                    else:
                        val_acc_rare += (label[i]==pred_label[i]).item()
                        val_nums_rare += 1
        acc_list = []
        for correct, total in zip(list(label2correct.values()), list(label2total.values())):
            acc_list.append(round(correct/total, 2))
        freq_list = []
        name_list = []
        f = open('../final-project-challenge-3-no_qq_no_life/food_data/label2name.txt', encoding='utf8')
        for line in f.readlines():
            _id, freq, name = line.split()
            freq_list.append(freq)
            name_list.append(name)
        # label, freq/comm/rare, name, total, correct, acc
        # df = pd.DataFrame(list(zip(list(range(1000)), freq_list, name_list, list(label2total.values()), list(label2correct.values()), acc_list)),
        #                     columns=['label', 'frequency', 'name', 'total', 'correct', 'accuracy'])
        # df.to_csv(os.path.join(self.model_path, 'valid_info.csv'), index=False)
        # confusion matrix
        print('computing confusion matrix...')
        cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(1000)])
        np.save(os.path.join(self.model_path, 'cm.npy'), cm)
        
        
        val_acc_rate = val_acc/val_nums
        self.logger.info("Validation loss:{:5f} Validation accuracy:{:5f}".format(val_loss,val_acc_rate ) )
        self.logger.info("Val_acc {:d} Val_nums {:d}".format(int(val_acc),int(val_nums)) )
        val_acc_rate_freq = val_acc_freq/val_nums_freq
        val_acc_rate_common = val_acc_common/val_nums_common
        val_acc_rate_rare = val_acc_rare/val_nums_rare
        self.logger.info("Validation accuracy (freq) : {:5f}".format(val_acc_rate_freq))
        self.logger.info("Val_acc {:d} Val_nums {:d} (freq)".format(int(val_acc_freq),int(val_nums_freq)) )
        self.logger.info("Validation accuracy (common) : {:5f}".format(val_acc_rate_common))
        self.logger.info("Val_acc {:d} Val_nums {:d} (common)".format(int(val_acc_common),int(val_nums_common)) )
        self.logger.info("Validation accuracy (rare) : {:5f}".format(val_acc_rate_rare))
        self.logger.info("Val_acc {:d} Val_nums {:d} (rare)".format(int(val_acc_rare),int(val_nums_rare)) )
        # save info about loss,accurracy
        self.info_list["train_acc"].append(val_acc_rate)
        self.info_list["val_loss"].append(val_loss)
        if self.val_best_acc < val_acc_rate:
            self.val_best_acc = val_acc_rate
            torch.save(self.model.state_dict(), os.path.join(self.model_path,"model_best.pth")) 
            self.logger.info("Save best model")
        if epoch % self.save_period == 0:
            torch.save(self.model.state_dict(), os.path.join(self.model_path,"model_{:d}.pth".format(epoch))) 
    def train_seperate(self):
        for epoch in range(self.MAX_EPOCH):
            start = time.time()
            self.logger.info("====================================================")
            self.logger.info("Epoch {}: train".format(epoch))
            self._train_seperate(epoch)
            self.logger.info("Epoch {}: validation".format(epoch))
            self._valid_seperate(epoch)
            self.logger.info("Total {:5f} sec per epoch".format(time.time()-start))
            # log training info per epoch
            self.save_info_list()
    
    def _train_seperate(self,epoch):
        # seperately update different layer's loss
        self.model.train()
        train_nums = 0.0
        train_loss = 0.0
        train_acc = 0.0
        # train
        for batch_idx,(data,label) in enumerate(tqdm(self.train_loader)):
            data,label =data.to(self.device),label.to(self.device)
            self.optimizer.zero_grad()
            output1, output2, output3, output4 = self.model(data)
            loss1 = self.criterion(output1, label)
            loss2 = self.criterion(output2, label)
            loss3 = self.criterion(output3, label)
            loss4 = self.criterion(output4, label)
            loss = (loss1 + loss2 + loss3 + loss4) / 4
            train_loss += (loss1.item() + loss2.item() + loss3.item() + loss4.item()) / 4
            # self.logger.info("Epoch:{} | batch: {}| loss: {}".format(epoch,batch_idx,train_loss))
            # update: backpropagation,lr schedule
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            logit = (F.softmax(output1, dim=-1) + F.softmax(output2, dim=-1) + F.softmax(output3, dim=-1) + F.softmax(output4, dim=-1)) / 4
            _,pred_label=torch.max(logit, 1)
            train_acc+=(label==pred_label).sum().item()
            train_nums+=pred_label.size(0)
            if batch_idx % 100 == 0 and batch_idx > 0:
                print("Training loss:{:5f},Training Accuracy:{:5f}".format(loss.item(),train_acc/train_nums))
        train_acc_rate = train_acc/train_nums
        self.logger.info("Training loss:{:5f},Training Accuracy:{:5f}".format(train_loss,train_acc_rate) )
        # save info about loss,accurracy
        self.info_list["train_acc"].append(train_acc_rate)
        self.info_list["train_loss"].append(train_loss)

    def _valid_seperate(self,epoch):
        val_nums = 0.0
        val_loss = 0.0
        val_acc = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch_idx,(data,label) in enumerate(self.val_loader):
                data,label =data.to(self.device),label.to(self.device)
                output1, output2, output3, output4 = self.model(data)
                loss1 = self.criterion(output1, label)
                loss2 = self.criterion(output2, label)
                loss3 = self.criterion(output3, label)
                loss4 = self.criterion(output4, label)
                loss = (loss1 + loss2 + loss3 + loss4) / 4
                val_loss += (loss1.item() + loss2.item() + loss3.item() + loss4.item()) / 4
                logit = (F.softmax(output1, dim=-1) + F.softmax(output2, dim=-1) + F.softmax(output3, dim=-1) + F.softmax(output4, dim=-1)) / 4
                _,pred_label=torch.max(logit, 1)
                val_acc+=(label==pred_label).sum().item()
                val_nums+=pred_label.size(0)
        val_acc_rate = val_acc/val_nums
        self.logger.info("Validation loss:{:5f} Validation accuracy:{:5f}".format(val_loss,val_acc_rate ) )
        self.logger.info("Val_acc {:d} Val_nums {:d}".format(int(val_acc),int(val_nums)) )
        # save info about loss,accurracy
        self.info_list["train_acc"].append(val_acc_rate)
        self.info_list["val_loss"].append(val_loss)
        if self.val_best_acc < val_acc_rate:
            self.val_best_acc = val_acc_rate
            torch.save(self.model.state_dict(), os.path.join(self.model_path,"model_best.pth")) 
            self.logger.info("Save best model")
        if epoch % self.save_period == 0:
            torch.save(self.model.state_dict(), os.path.join(self.model_path,"model_{:d}.pth".format(epoch))) 

