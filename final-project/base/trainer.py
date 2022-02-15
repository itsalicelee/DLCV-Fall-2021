import torch
import time
import os
from scipy.io import savemat
from tqdm import tqdm 
from util import gen_logger

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
			self.logger.info("Epoch {}: train".format(epoch))
			self._train(epoch)
			self.logger.info("Epoch {}: validation".format(epoch))
			self._valid(epoch)
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
			#self.logger.info("Epoch:{:d} | batch: {:d}| loss: {:5f}".format(epoch,batch_idx,train_loss))
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
		self.info_list["val_acc"].append(val_acc_rate)
		self.info_list["val_loss"].append(val_loss)
		if self.val_best_acc < val_acc_rate:
			self.val_best_acc = val_acc_rate
			torch.save(self.model.state_dict(), os.path.join(self.model_path,"model_best.pth")) 
			self.logger.info("Save best model")
		if epoch % self.save_period == 0:
			torch.save(self.model.state_dict(), os.path.join(self.model_path,"model_{:d}.pth".format(epoch))) 

	def save_info_list(self):
		savemat("{}/Loss.mat".format(self.model_path), self.info_list)
		self.logger.info("Save learning history to {}/Loss.mat".format(self.model_path))


class BBNTrainer(BaseTrainer):
	def __init__(self,
				 device,
				 combiner,
				 model,
				 optimizer,
				 scheduler,
				 MAX_EPOCH,
				 criterion,
				 train_loader,
				 train_loader_reverse,
				 val_loader,
				 lr,
				 batch_size,
				 gradaccum_size,
				 model_path,
				 save_period):
		self.device = device
		self.model = model.to(self.device)
		self.combiner = combiner
		self.optimizer = optimizer
		self.MAX_EPOCH = MAX_EPOCH
		self.criterion = criterion
		self.train_loader = train_loader
		self.train_loader_reverse = train_loader_reverse
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
						  "balanced_val_acc":[],
						  "balanced_val_loss":[],
						  "reversed_val_acc":[],
						  "reversed_val_loss":[],
						  "lr":lr,
						  "batch_size":batch_size
						 }

	def _train(self,epoch):
		# Enter training mode
		self.model.train()
		self.combiner.reset_epoch(epoch)
		train_nums = 0.0
		train_loss = 0.0
		train_acc = 0.0
		# train
		loss_step = 1
		len_dataloader = min(len(self.train_loader), len(self.train_loader_reverse))
		data_iter = iter(self.train_loader)
		data_reverse_iter = iter(self.train_loader_reverse)
		for batch_idx in tqdm(range(len_dataloader)):
			# sameple bidirectional data
			##################
			# Balanced sampler
			##################
			try:
				data,label = data_iter.next()
			except StopIteration:
				data_iter = iter(tqdm(self.train_loader))
				data,label = data_iter.next()
			##################
			# Reverse sampler
			##################
			try:
				data_reverse,label_reverse = data_reverse_iter.next()
			except StopIteration:
				data_reverse_iter = iter(self.train_loader_reverse)
				data_reverse,label_reverse = data_reverse_iter.next()

			data,label = data.to(self.device),label.to(self.device)
			data_reverse,label_reverse = data_reverse.to(self.device),label_reverse.to(self.device)

			feature_a, feature_b = (
				self.model(data, feature_cb=True),
				self.model(data_reverse, feature_rb=True),
			)
			##################
			# alpha weight
			##################
			l = 1 - ((self.combiner.epoch - 1) / self.combiner.div_epoch) ** 2  # parabolic decay
			#l = 0.5  # fix
			#l = math.cos((self.epoch-1) / self.div_epoch * math.pi /2)   # cosine decay
			#l = 1 - (1 - ((self.epoch - 1) / self.div_epoch) ** 2) * 1  # parabolic increment
			#l = 1 - (self.epoch-1) / self.div_epoch  # linear decay
			#l = np.random.beta(self.alpha, self.alpha) # beta distribution
			#l = 1 if self.epoch <= 120 else 0  # seperated stage

			mixed_feature = 2 * torch.cat((l * feature_a, (1-l) * feature_b), dim=1)
			output = self.model(mixed_feature, classifier_flag=True)
			loss = l * self.criterion(output, label) + (1 - l) * self.criterion(output, label_reverse)

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
			###############################
			# compute average accurracy
			###############################
			train_acc+=(l*(label==pred_label)+(1-l)*(label_reverse==pred_label)).sum().item()
			train_nums+=pred_label.size(0)
		if self.scheduler is not None:
			self.scheduler.step()
		train_acc_rate = train_acc/train_nums
		self.logger.info("Training loss:{:5f},Training Accuracy:{:5f}".format(train_loss,train_acc_rate) )
		# save info about loss,accurracy
		self.info_list["train_acc"].append(train_acc_rate)
		self.info_list["train_loss"].append(train_loss)

	def _valid_separate(self,epoch,branch = -1):

		self.logger.info("==============================")
		if branch == 0:
			self.logger.info("Balanced branch model")
			branch_flag = "balanced"
			self.inference_cache = []
		elif branch == 1:
			self.logger.info("Reversed branch model")
			branch_flag = "reversed"
		else:
			assert (False)
		self.logger.info("==============================")
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
				##########################
				# Choose branch of model #
				##########################
				if branch == 0:
					# cache balanced branch
					output,output_b = self.model(data,separate_classifier_flag=True)
					self.inference_cache.append(output_b)
				elif branch == 1:
					output = self.inference_cache[batch_idx]

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
		self.info_list["{}_val_acc".format(branch_flag)].append(val_acc_rate)
		self.info_list["{}_val_loss".format(branch_flag)].append(val_loss)


	def train(self):
		for epoch in range(1,self.MAX_EPOCH+1):
			start = time.time()
			self.logger.info("====================================================")
			self.logger.info("Epoch {}: train".format(epoch))
			self._train(epoch)
			self.logger.info("Epoch {}: validation".format(epoch))
			self._valid(epoch)
			self.logger.info("Epoch {}: validation in Balanced Sampler model".format(epoch))
			self._valid_separate(epoch,branch=0)
			self.logger.info("Epoch {}: validation in Reversed Sampler model".format(epoch))
			self._valid_separate(epoch,branch=1)
			self.logger.info("Total {:5f} sec per epoch".format(time.time()-start))
			# log training info per epoch
			self.save_info_list()

	
