#code for custom pytorch models
import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F



class CustomNet_small(torch.nn.Module):
	def __init__(self, out_channels = 26):
		super(CustomNet_small, self).__init__()
		self.out_channels = out_channels

		self.features = nn.Sequential(
			nn.Conv2d(3, 40, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=False),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
			nn.Conv2d(40, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(20, 60, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True))

		self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(60*4*4, 500),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(500, 200),
			nn.ReLU(inplace=True),
			nn.Linear(200, self.out_channels))


	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x




class BranchingNet(torch.nn.Module):
	def __init__(self, out_channels = 26, in_channels = 3):
		super(BranchingNet, self).__init__()
		self.out_channels = out_channels
		self.in_channels = in_channels

		self.features = nn.Sequential(
			nn.Conv2d(self.in_channels, 40, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=False),
			nn.ReLU(inplace=True),
			#nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
			nn.Conv2d(40, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(20, 60, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True))

		self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(60*4*4, 500),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(500, 200),
			nn.ReLU(inplace=True),
			nn.Linear(200, self.out_channels))


	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x