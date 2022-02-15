import torch
import os
from scipy.io import savemat
from tqdm import tqdm 
from util import gen_logger
import csv
import pandas as pd

class BaseTester():
	def __init__(self,
				 device,
				 model,
				 test_loader,
				 val_loader,
				 load_model_path,
				 mat_file,
				 kaggle_file,
     			 criterion):
		self.device = device 
		self.model = model.to(self.device) 
		self.test_loader = test_loader
		self.val_loader = val_loader
		self.model_path = load_model_path
		self.logger = gen_logger(os.path.join(load_model_path,"valid.log"))
		#self.logger = gen_logger(kaggle_file)
		self.mat_file = mat_file
		self.kaggle_file = kaggle_file
		self.criterion = criterion
		self.output_list= {"out":[],
						   "out_softmax":[],
						   "pred":[],
						   "label":[]
						 }
	def test(self):
		'''
		log kaggle submission file
		'''
		label_list = []
		self.model.eval()
		output_list_out = []
		output_list_out_softmax = []
		output_list_pred = []
		with torch.no_grad():
			for i, t_imgs in enumerate(tqdm(self.test_loader)):
				t_imgs = t_imgs.to(self.device)
				P_pred = self.model(t_imgs)
				_,pred_label=torch.max(P_pred,1)
				label_list.extend(P_pred.argmax(dim=-1).cpu().numpy().tolist())
				for i in range(len(pred_label)):
					output_list_out.append(P_pred[i].cpu().numpy())
					output_list_out_softmax.append(torch.nn.functional.softmax(P_pred[i],dim=0).cpu().numpy())
					output_list_pred.append(pred_label[i].cpu().numpy())
		self.output_list["out"] = output_list_out
		self.output_list["out_softmax"] = output_list_out_softmax
		self.output_list["pred"] = output_list_pred
		self.save_info_list()
		image_ids = ["{:06d}".format(i) for i in self.test_loader.dataset.data_df.image_id]
		# write predictions
		df = pd.DataFrame({"image_id": image_ids, 'label': label_list})
		df.to_csv(self.kaggle_file, index=False)
		print("===> File saved as {}".format(self.kaggle_file))

	def valid_and_savemat(self):
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
		output_list_out = []
		output_list_out_softmax = []
		output_list_pred = []
		output_list_label = []
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
				for i in range(len(label)):
					output_list_out.append(output[i].cpu().numpy())
					output_list_out_softmax.append(torch.nn.functional.softmax(output[i],dim=0).cpu().numpy())
					output_list_pred.append(pred_label[i].cpu().numpy())		
					output_list_label.append(label[i].cpu().numpy())
		self.output_list["out"] = output_list_out
		self.output_list["out_softmax"] = output_list_out_softmax
		self.output_list["pred"] = output_list_pred
		self.output_list["label"] = output_list_label
		self.save_info_list()
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

	def save_info_list(self):
		if self.mat_file:
			savemat(self.mat_file, self.output_list)
			self.logger.info("Save output .mat to {}".format(self.mat_file))