import torch
import torch.nn as nn
import torch.nn.functional as F
from ..pytorch_pretrained_vit import ViT


#################
# Common Module
#################
class GAP(nn.Module):
	"""Global Average pooling
		Widely used in ResNet, Inception, DenseNet, etc.
	 """

	def __init__(self):
		super(GAP, self).__init__()
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

	def forward(self, x):
		x = self.avgpool(x)
		#         x = x.view(x.shape[0], -1)
		return x

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
		
	def forward(self, x):
		return x


##################
# BNN architecture
##################
class BNNetwork(nn.Module):
	def __init__(self, backbone_model,num_classes=1000,num_features=4096,mode="swin"):
		super(BNNetwork, self).__init__()
		self.num_classes = num_classes
		if mode == "swin":
			####################
			# SWIN backbone
			####################
			self.backbone = backbone_model
			self.module = Identity()
			print(self.backbone)
		elif mode == "ViT" :
			####################
			# ViT backbone
			####################
			model_name = "B_16_imagenet1k"
			self.backbone = ViT(model_name, pretrained=True,num_classes=num_classes,image_size=324)
			#self.backbone.norm = Identity()
			#self.backbone.fc = Identity()
			self.module = GAP()
			print(self.backbone)
		elif mode == "ResNet50":
			####################
			# ResNet50 backbone
			####################
			self.backbone = backbone_model
			self.module = GAP()
			print(self.backbone)		
		else:
			print("invalid model mode QQ")
		self.num_features = num_features
		self.classifier = nn.Linear(self.num_features, self.num_classes, bias=True)
		
		## swin flag classifier
		#self.swin_classifier=nn.Sequential(
		#					 nn.Linear(self.num_features*2,num_classes)
		#					 )


	def forward(self, x, **kwargs):
		if "feature_flag" in kwargs or "feature_cb" in kwargs or "feature_rb" in kwargs:
			return self.extract_feature(x, **kwargs)
		elif "classifier_flag" in kwargs:
			return self.classifier(x)

		x = self.backbone(x)
		x = self.module(x)
		x = x.view(x.shape[0], -1)

		if "separate_classifier_flag" in kwargs:
			return self.separate_classifier(x)
		x = self.classifier(x)
		return x


	def extract_feature(self, x, **kwargs):
		if len(kwargs) > 0:
			x = self.backbone(x, **kwargs)
		else:
			x = self.backbone(x)
		x = self.module(x)
		x = x.view(x.shape[0], -1)

		return x


	def freeze_backbone(self):
		print("Freezing backbone .......")
		for p in self.backbone.parameters():
			p.requires_grad = False


	def load_backbone_model(self, backbone_path=""):
		self.backbone.load_model(backbone_path)
		print("Backbone has been loaded...")


	def load_model(self, model_path):
		pretrain_dict = torch.load(
			model_path #, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
		)
		pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
		model_dict = self.state_dict()
		from collections import OrderedDict
		new_dict = OrderedDict()
		for k, v in pretrain_dict.items():
			if k.startswith("module"):
				new_dict[k[7:]] = v
			else:
				new_dict[k] = v
		model_dict.update(new_dict)
		self.load_state_dict(model_dict)
		print("Model has been loaded...")

	def separate_classifier(self,fcfb):
		# weight of FC
		feature_len = self.classifier.weight.shape[1]//2
		Wc = torch.permute(self.classifier.weight[:,:feature_len],dims=(1,0))
		Wb = torch.permute(self.classifier.weight[:,feature_len:],dims=(1,0))
		Bias = self.classifier.bias #WcWb[1][1]
		# y=xW^T+bias
		o_Wcfc = torch.matmul(fcfb[:,:feature_len],Wc)+0.5*Bias
		o_Wbfb = torch.matmul(fcfb[:,feature_len:],Wb)+0.5*Bias
		return  o_Wcfc,o_Wbfb
