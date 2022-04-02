from PIL import Image, ImageOps
import os
from subprocess import call
import numpy as np
from torchvision import datasets, transforms, utils
import torch

from torch.utils.data import Dataset, DataLoader
from random import randint
import random
from copy import deepcopy
import random

#torchvision compose transforms

imnet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     	 std=[0.229, 0.224, 0.225])

default_transform = transforms.Compose([
        			transforms.Resize((224,224)),
        			transforms.ToTensor()])


mixed_letter_transform = transforms.Compose([
        			transforms.Resize((224,224)),
        			transforms.ToTensor(),
					imnet_normalize
					])

imagenet_train_transform = transforms.Compose([
            	transforms.RandomResizedCrop(224),
            	transforms.RandomHorizontalFlip(),
            	transforms.ToTensor(),
            	imnet_normalize
        		])

imagenet_test_transform = transforms.Compose([
                	transforms.Resize(256),
            		transforms.CenterCrop(224),
            		transforms.ToTensor(),
					imnet_normalize
        			])

lenet_transform = transforms.Compose([
        			transforms.Resize((32,32)),
        			transforms.ToTensor(),
					imnet_normalize
					])


class AddGaussianNoise(object):
	def __init__(self, mean=0., std=1.):
		self.std = std
		self.mean = mean

	def __call__(self, tensor):
		return tensor + torch.randn(tensor.size()) * self.std + self.mean

	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


augmentation_transform = transforms.Compose([
								AddGaussianNoise(0,.01)
								])

topil = transforms.ToPILImage()
totensor = transforms.ToTensor()


def add_img_ratio_margin(pil_img, color='white',ratio=.7):
    width, height = pil_img.size
    margin = int(ratio*(width+height)/2)
    #print('width')
    #print(width)
    #print('height')
    #print(height)
    #print('margin')
    #print(margin)
    
    if width>height:
        new_width = width + 2*margin 
        new_height = height + 2*margin + (width-height)
    else:
        new_width = width + 2*margin + (height-width) 
        new_height = height + 2*margin

    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (int(new_width/2-width/2), int(new_height/2-height/2)))
    return result





class letters_expstim_data(Dataset):
	
	def __init__(self, root_dir ='../../data/expstim/', order_file_name = '../../data/expstim_image_order.txt', transform = default_transform, rgb_convert=True):
		
		
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


class mix_imagenet_letter_data(Dataset):
	
	def __init__(self, root_dir ='../data/mixed_imnet_letter/', train=True, rgb_convert=True, size_vary=True):
		self.train = train	
		if train:
			self.root_dir = root_dir+'train/'
		else:
			self.root_dir = root_dir+'test/'

		self.label_names = os.listdir(self.root_dir)
		self.label_names.sort()

		#tuples of img_name with label_name
		self.img_label_pairs = []
		for label_name in self.label_names:
			img_names = os.listdir(os.path.join(self.root_dir,label_name))
			img_names.sort()
			for img_name in img_names:
				self.img_label_pairs.append((img_name,label_name))

		self.num_classes = len(self.label_names)
		self.rgb_convert = rgb_convert
		self.size_vary = size_vary

	def __len__(self):
		return len(self.img_label_pairs)

	def get_label_from_name(self,label_name):
		return torch.tensor(self.label_names.index(label_name))       

	def shrink_add_border(self,img,min_size=100):
		old_size = img.size[0]
		new_size = randint(min_size/2,old_size/2)*2
		img = img.resize((new_size,new_size), Image.ANTIALIAS)
		img = ImageOps.expand(img,border=int((old_size-new_size)/2),fill='white')
		return img


	def __getitem__(self, idx):
		label_name = self.img_label_pairs[idx][1]
		label = self.get_label_from_name(label_name)

		img_name = self.img_label_pairs[idx][0]
		img_path = os.path.join(self.root_dir,label_name,img_name)
		img = Image.open(img_path)
		
		if self.rgb_convert:
			img = img.convert('RGB')
		if self.size_vary and label.item() > 999:
			img = self.shrink_add_border(img)
		if label.item() > 999:
			img = mixed_letter_transform(img)
		else:
			if self.train:
				img = imagenet_train_transform(img)
		#img = imagenet_transform(img)
			else:
				img = imagenet_test_transform(img)
		
		return (img,label)




class augmentation_letters_data(Dataset):
	
	def __init__(self, root_dir ='../data/combined_data/', train=True,use_augment=True, imagenet_normalize=True,return_augmentations = False, transform_prob_dict=None):
				
		if train:
			self.root_dir = root_dir+'/train'
		elif 'expstim' not in root_dir:
			self.root_dir = root_dir+'/test'
		else:
			self.root_dir = root_dir
		

		self.img_names = os.listdir(self.root_dir)
		self.img_names.sort()

		self.label_names = []
		for img_name in self.img_names:
			label_name = img_name.split('_')[0]
			if label_name not in self.label_names:
				self.label_names.append(label_name)

		self.num_classes = len(self.label_names)

		
		self.shrink = {'font':random.uniform(.6,.2),'NIST':random.uniform(.9,.35)}
		self.shear = {'font':random.uniform(.7,1/.7),'NIST':random.uniform(.7,1/.7)}
		self.rot = {'font':random.uniform(-.6,.6),'NIST':random.uniform(-.1,.1)}

		self.use_augment = use_augment
		self.imagenet_normalize = imagenet_normalize

		self.return_augmentations = return_augmentations

		if transform_prob_dict is None:
			self.transform_prob_dict = {'color':.5,
										'size':.5,
										'noise':.5
										}
		else:
			self.transform_prob_dict = transform_prob_dict

	def __len__(self):
		return len(self.img_names)

	def get_label_from_name(self,img_name):
		label_name = img_name.split('_')[0]
		return torch.tensor(self.label_names.index(label_name))       

	def augment(self,img,img_path):
		augmentations = [0,0,0]
		img = img.resize((224,224), Image.ANTIALIAS)

		if 'NIST' in img_path:
			shrink,shear,rot = self.shrink['NIST'],self.shear['NIST'],self.rot['NIST']
		else:
			shrink,shear,rot = self.shrink['font'],self.shear['font'],self.rot['font']

		if np.random.binomial(1,self.transform_prob_dict['size']) == 1:
			augmentations[0] = 1
			s = (int(img.size[0]*shrink),int(img.size[1]*shrink*shear))

			small_img = img.resize(s, Image.ANTIALIAS)

			#rotate (might want 'expand' to be '1' if we notice letters getting cut off )
			small_img = small_img.rotate(30*rot, Image.BILINEAR, expand = 1, fillcolor='white')

			#center
			#x_anch = int(img.size[0]/2 - small_img.size[0]/2)
			#y_anch = int(img.size[1]/2 - small_img.size[1]/2)

			x_anch = random.randint(0,max(img.size[0]-small_img.size[0],0))
			y_anch = random.randint(0,max(img.size[1]-small_img.size[1],0))

			back_img = Image.new("L", img.size, 'white')
			back_img.paste(small_img, (x_anch, y_anch))
			#affined_img = back_img.convert('RGB')
			affined_img = back_img
		else:
			affined_img = add_img_ratio_margin(img)
			affined_img = affined_img.resize((224,224), Image.ANTIALIAS)

		#tensor stuff


		#img_data = transform(affined_img)
		if np.random.binomial(1,self.transform_prob_dict['color']) == 1:
			augmentations[1] = 1
			img_data_r = totensor(affined_img)

			img_data_r[img_data_r>.5] = 1
			img_data_r[img_data_r<1] = 0

			img_data_g = deepcopy(img_data_r)
			img_data_b = deepcopy(img_data_r)

			affined_img




			to_close = True
			far_enough = .4

			while to_close:
				black_color = []
				white_color = []
				for c in range(3):
					black_color.append(random.uniform(0,1))
					white_color.append(random.uniform(.001,1))
				Sum = 0
				for c in range(3):
					Sum += (black_color[c]-white_color[c])**2
				color_d = np.sqrt(Sum)
				if color_d > far_enough:
					to_close = False

			img_data_r[img_data_r==1] = white_color[0]
			img_data_r[img_data_r==0] = black_color[0]

			img_data_g[img_data_g==1] = white_color[1]
			img_data_g[img_data_g==0] = black_color[1]

			img_data_b[img_data_b==1] = white_color[2]
			img_data_b[img_data_b==0] = black_color[2]

			img_data = torch.cat((img_data_r,img_data_g,img_data_b))
			
			
		else:
			img_data = totensor(affined_img)
			img_data = torch.cat((img_data,img_data,img_data))

		if np.random.binomial(1,self.transform_prob_dict['noise']) == 1:
			augmentations[2] = 1
			img_data = augmentation_transform(img_data)
		
		if self.imagenet_normalize:
			img_data = imnet_normalize(img_data)

		return img_data,augmentations
			

	def __getitem__(self, idx):

		img_path = os.path.join(self.root_dir,self.img_names[idx])
		img = Image.open(img_path)

		if self.use_augment:
			img, augmentations = self.augment(img,img_path)


		else:
			img = img.convert('RGB')
			if self.imagenet_normalize:
				img = mixed_letter_transform(img)
			else:
				img = default_transform(img)

		label = self.get_label_from_name(self.img_names[idx])
		
		if self.return_augmentations:
			return (img,label,augmentations)
		else:
			return (img,label)




class mix_imagenet_aug_letter_data(Dataset):
	

	def __init__(self, root_dir ='../data/mixed_imnet_letter/', train=True,use_augment=True, imagenet_normalize=True,return_augmentations = False, transform_prob_dict=None):
		self.train = train	
		if train:
			self.root_dir = root_dir+'train/'
		else:
			self.root_dir = root_dir+'test/'


		self.shrink = {'font':random.uniform(.6,.2),'NIST':random.uniform(.9,.35)}
		self.shear = {'font':random.uniform(.7,1/.7),'NIST':random.uniform(.7,1/.7)}
		self.rot = {'font':random.uniform(-.6,.6),'NIST':random.uniform(-.3,.3)}


		self.label_names = os.listdir(self.root_dir)
		self.label_names.sort()

		#tuples of img_name with label_name
		self.img_label_pairs = []
		for label_name in self.label_names:
			img_names = os.listdir(os.path.join(self.root_dir,label_name))
			img_names.sort()
			for img_name in img_names:
				self.img_label_pairs.append((img_name,label_name))

		self.num_classes = len(self.label_names)

		self.imnet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
						std=[0.229, 0.224, 0.225])

		self.use_augment = use_augment
		self.imagenet_normalize = imagenet_normalize

		self.return_augmentations = return_augmentations

		if transform_prob_dict is None:
			self.transform_prob_dict = {'color':.5,
										'size':.5,
										'noise':.5
										}
		else:
			self.transform_prob_dict = transform_prob_dict

	def __len__(self):
		return len(self.img_label_pairs)

	def get_label_from_name(self,label_name):
		return torch.tensor(self.label_names.index(label_name))       

	def augment(self,img,img_path):
		augmentations = [0,0,0]
		img = img.resize((224,224), Image.ANTIALIAS)

		if 'NIST' in img_path:
			shrink,shear,rot = self.shrink['NIST'],self.shear['NIST'],self.rot['NIST']
		else:
			shrink,shear,rot = self.shrink['font'],self.shear['font'],self.rot['font']

		if np.random.binomial(1,self.transform_prob_dict['size']) == 1:
			augmentations[0] = 1
			s = (int(img.size[0]*shrink),int(img.size[1]*shrink*shear))

			small_img = img.resize(s, Image.ANTIALIAS)

			#rotate (might want 'expand' to be '1' if we notice letters getting cut off )
			small_img = small_img.rotate(30*rot, Image.BILINEAR, expand = 1, fillcolor='white')

			#center
			#x_anch = int(img.size[0]/2 - small_img.size[0]/2)
			#y_anch = int(img.size[1]/2 - small_img.size[1]/2)

			x_anch = random.randint(0,max(img.size[0]-small_img.size[0],0))
			y_anch = random.randint(0,max(img.size[1]-small_img.size[1],0))

			back_img = Image.new("L", img.size, 'white')
			back_img.paste(small_img, (x_anch, y_anch))
			#affined_img = back_img.convert('RGB')
			affined_img = back_img
		else:
			affined_img = add_img_ratio_margin(img)
			affined_img = affined_img.resize((224,224), Image.ANTIALIAS)

		#tensor stuff


		#img_data = transform(affined_img)
		if np.random.binomial(1,self.transform_prob_dict['color']) == 1:
			augmentations[1] = 1
			img_data_r = totensor(affined_img)

			img_data_r[img_data_r>.5] = 1
			img_data_r[img_data_r<1] = 0

			img_data_g = deepcopy(img_data_r)
			img_data_b = deepcopy(img_data_r)

			affined_img




			to_close = True
			far_enough = .4

			while to_close:
				black_color = []
				white_color = []
				for c in range(3):
					black_color.append(random.uniform(0,1))
					white_color.append(random.uniform(.001,1))
				Sum = 0
				for c in range(3):
					Sum += (black_color[c]-white_color[c])**2
				color_d = np.sqrt(Sum)
				if color_d > far_enough:
					to_close = False

			img_data_r[img_data_r==1] = white_color[0]
			img_data_r[img_data_r==0] = black_color[0]

			img_data_g[img_data_g==1] = white_color[1]
			img_data_g[img_data_g==0] = black_color[1]

			img_data_b[img_data_b==1] = white_color[2]
			img_data_b[img_data_b==0] = black_color[2]

			img_data = torch.cat((img_data_r,img_data_g,img_data_b))
			
			
		else:
			img_data = totensor(affined_img)
			img_data = torch.cat((img_data,img_data,img_data))

		if np.random.binomial(1,self.transform_prob_dict['noise']) == 1:
			augmentations[2] = 1
			img_data = augmentation_transform(img_data)
		
		if self.imagenet_normalize:
			img_data = imnet_normalize(img_data)

		return img_data,augmentations


	def __getitem__(self, idx):
		label_name = self.img_label_pairs[idx][1]
		label = self.get_label_from_name(label_name)

		img_name = self.img_label_pairs[idx][0]
		img_path = os.path.join(self.root_dir,label_name,img_name)
		img = Image.open(img_path)
		
		if label.item() > 999:
			if self.train:
				if self.use_augment:
					img, augmentations = self.augment(img,img_path)
					if self.imagenet_normalize:
						img = self.imnet_norm(img)
				else:
					augmentations = [0,0,0]
					img = img.convert('RGB')
					#img = add_img_ratio_margin(img)
					img = default_transform(img)
					if self.imagenet_normalize:
						img = self.imnet_norm(img)


			else:
				#img = self.augment(img,img_path)
				#img = self.imnet_norm(img)
				
				img = img.convert('RGB')
				img = add_img_ratio_margin(img)
				img = default_transform(img)
				#img = augmentation_transform(img)
				if self.imagenet_normalize:
					img = self.imnet_norm(img)
				#img = mixed_letter_transform(img)
				augmentations = [0,0,0]

		else:
			if self.train:
				img = imagenet_train_transform(img)
		#img = imagenet_transform(img)
			else:
				img = imagenet_test_transform(img)

			augmentations = [0,0,0]
		
		if self.return_augmentations:
			return (img,label,img_name,augmentations)
		else:
			return (img,label,img_name)







letter_labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

#In the mixed dataset with letters and imagenet, we may want to train on 
#examples of letters at a higher rate than is inherently present in the dataset.
#We do that with the function below, by adding symlinked copies of the letter images 
#to the dataset. passing arg "copies = 1" will add 1 copy of each letter image to the dataset
# effectly doubling the rate at which we train on letters, passing 2 with add 2 copies,
#tripling the rate, etc. etc.   Make sure to use in conj with 'unsymlink_letters_in_mixed',
#at the end of training.
def symlink_letters_in_mixed(copies, data_dir = '../data/mixed_imnet_letter/train/'):
	print('creating symlinks for dataset')
	data_dir_abs = os.path.abspath(data_dir)
	for label in letter_labels:
		print(label)
		files = os.listdir(os.path.join(data_dir,label))
		for file in files:
			if file[-8:-5] == '_cp':
				continue

			for copy in range(copies):
				file_root, file_ext = '.'.join(file.split('.')[:-1]),file.split('.')[-1]
				new_name = file_root+'_cp'+str(copy+1)+'.'+file_ext
				if not os.path.exists(os.path.join(data_dir_abs,label,new_name)):
					call('ln -s %s %s'%(os.path.join(data_dir_abs,label,file),os.path.join(data_dir_abs,label,new_name)),shell=True)

def unsymlink_letters_in_mixed(data_dir = '../data/mixed_imnet_letter/train/'):
	print('deleting symlinked copy files')
	call('find %s -mindepth 2  -type l -delete'%data_dir,shell=True)