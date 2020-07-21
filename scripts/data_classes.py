from PIL import Image, ImageOps
import os
import numpy as np
from torchvision import datasets, transforms, utils
import torch

from torch.utils.data import Dataset, DataLoader
from random import randint



default_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()])




class letters_validation_data(Dataset):
	

	def __init__(self, root_dir ='../data/validation/', order_file_name = '../data/validation_image_order.txt', transform = default_transform, rgb_convert=True):
		
		
		self.root_dir = root_dir

		order_file = open(order_file_name,'r')
		self.img_names = [x.strip() for x in order_file.readlines()] 
		order_file.close()

		self.label_names = []
		for img_name in self.img_names:
			label_name = img_name.split('_')[0]
			if label_name not in self.label_names:
				self.label_names.append(label_name)

		self.num_classes = len(self.label_names)
		self.transform = transform
		self.rgb_convert = rgb_convert

	def __len__(self):
		return len(self.img_names)

	def get_label_from_name(self,img_name):
		label_name = img_name.split('_')[0]
		return torch.tensor(self.label_names.index(label_name))       

	def __getitem__(self, idx):

		img_path = os.path.join(self.root_dir,self.img_names[idx])
		img = Image.open(img_path)
		if self.rgb_convert:
			img = img.convert('RGB')
		img = self.transform(img)
		label = self.get_label_from_name(self.img_names[idx])
		
		return (img,label)


class letters_train_data(Dataset):
	

	def __init__(self, root_dir ='../data/', train=True, transform = default_transform, rgb_convert=True, size_vary=True):
				
		if train:
			self.root_dir = root_dir+'train'
		else:
			self.root_dir = root_dir+'test'

		self.img_names = os.listdir(self.root_dir)
		self.img_names.sort()

		self.label_names = []
		for img_name in self.img_names:
			label_name = img_name.split('_')[0]
			if label_name not in self.label_names:
				self.label_names.append(label_name)

		self.num_classes = len(self.label_names)
		self.transform = transform
		self.rgb_convert = rgb_convert
		self.size_vary = size_vary

	def __len__(self):
		return len(self.img_names)

	def get_label_from_name(self,img_name):
		label_name = img_name.split('_')[0]
		return torch.tensor(self.label_names.index(label_name))       

	def shrink_add_border(self,img,min_size=100):
		old_size = img.size[0]
		new_size = randint(min_size/2,old_size/2)*2
		img = img.resize((new_size,new_size), Image.ANTIALIAS)
		img = ImageOps.expand(img,border=int((old_size-new_size)/2),fill='white')
		return img


	def __getitem__(self, idx):

		img_path = os.path.join(self.root_dir,self.img_names[idx])
		img = Image.open(img_path)
		if self.rgb_convert:
			img = img.convert('RGB')
		if self.size_vary:
			img = self.shrink_add_border(img)
		img = self.transform(img)
		label = self.get_label_from_name(self.img_names[idx])
		
		return (img,label)


letter_labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
