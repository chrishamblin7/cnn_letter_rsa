#Get all activations from model
import torch
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import data_classes
import scipy.io
import pickle
import os
from PIL import Image, ImageOps

#params
SUM = True  #sum output activation maps
FLIP = True #flip output so its featuresXimages not imagesXfeatures
output_type = 'mat'


subfolder = 'mixednet'        #(branchingnet,alexnet,smallnet)

output_names = ['pretrain_alexnet_letter_3copies_ep5_imnet0.468_letter0.972']


if subfolder == 'branchingnet':
	branch_key = {  #a dict that gives the alexnet module name from the name in the model file
	'relu1':'1',
	'relu2':'4',
	'relu3':'7',
	'relu4':'9',
	'relu5':'11'
	}
	alexnet = models.alexnet(pretrained=True)
	alexnet.eval()
	from train_branching_model import through_network


####END OF PARAMS ########

#data loader
labels = os.listdir('../data/imagenet_validation')
labels.sort()


imagenet_transform = transforms.Compose([
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
				])
def image_loader(image_folder, transform=imagenet_transform):
	img_names = os.listdir(image_folder)
	img_names.sort()
	image_list = []
	for img_name in img_names:
		image = Image.open(os.path.join(image_folder,img_name))
		image = transform(image).float()
		#image = transforms.ToTensor(image)    #make sure image is float tensor
		#image = torch.tensor(image, requires_grad=True)
		image = image.unsqueeze(0)
		image_list.append(image)
	return torch.cat(image_list,0)


for output_name in output_names:
	#model
	model = torch.load('../models/%s/%s.pt'%(subfolder,output_name))
	model.to('cpu')

	#num_ftrs = model.classifier[6].in_features
	#model.classifier[6] = nn.Linear(num_ftrs,26)       #changes size of output to 26
	model.eval()

	#get activations into dict
	activations = {}
	i=0
	for label in labels:
		print(i)
		i+=1
		data = image_loader(os.path.join('../data/imagenet_validation',label))

		if subfolder == 'branchingnet':    #we need to pass input through alexnet
			branch_point = branch_key[output_name.split('_')[1]]
			data = through_network(data, network=alexnet.features,branch_point = branch_point)
		x = data

		# get conv features
		#print('features')
		for layer, (name, module) in enumerate(model.features._modules.items()):
			#print(layer)
			x=module(x)
			activation = x.detach()
			if SUM:
				activation = activation.sum(dim=2).sum(dim=2)
			activation_average = torch.mean(activation,0).unsqueeze(0)
			#activation_average = activation_average.numpy()
			if 'features'+str(module).split('(')[0]+'_'+str(layer) not in activations.keys():
				activations['features'+str(module).split('(')[0]+'_'+str(layer)] = activation_average
			else:
				activations['features'+str(module).split('(')[0]+'_'+str(layer)] = torch.cat((activations['features'+str(module).split('(')[0]+'_'+str(layer)],activation_average))

		# get avgpool
		x = model.avgpool(x)
		activation = x.detach()
		if SUM:
			activation = activation.sum(dim=2).sum(dim=2)
		activation_average = torch.mean(activation,0).unsqueeze(0)
		#activation_average = activation_average.numpy()
		if 'avg_pool' not in activations.keys():
			activations['avg_pool'] = activation_average
		else:
			activations['avg_pool'] = torch.cat((activations['avg_pool'],activation_average))
		
		
		x = torch.flatten(x, 1)

		#classifier
		#print('classifier')
		for layer, (name, module) in enumerate(model.classifier._modules.items()):
			#print(layer)
			if str(module).split('(')[0]=='Dropout':
				continue
			x=module(x)
			activation = x.detach()
			activation_average = torch.mean(activation,0).unsqueeze(0)
			#activation_average = activation_average.numpy()
			if 'classifier'+str(module).split('(')[0]+'_'+str(layer) not in activations.keys():
				activations['classifier'+str(module).split('(')[0]+'_'+str(layer)] = activation_average
			else:
				activations['classifier'+str(module).split('(')[0]+'_'+str(layer)] = torch.cat((activations['classifier'+str(module).split('(')[0]+'_'+str(layer)],activation_average))	

	if FLIP:
		for k in activations:
			activations[k] = activations[k].numpy()
			activations[k] = np.swapaxes(activations[k],0,1)

	#output = model(data) #network final output


	#print(model)
	for key in activations:
		print(key)
		print(activations[key].shape)
	#print('output')
	#print(output.shape)

	if output_type == 'mat':
		scipy.io.savemat('../activations/%s/%s_imagenet_activations.mat'%(subfolder,output_name), activations)
		#scipy.io.savemat('../activations/%s/%s_output.mat'%(subfolder,output_name), dict(output=output.detach().numpy()))
	else:
		pickle.dump(activations, open('../activations/%s/%s_imagenet_activations.pkl'%(subfolder,output_name),'wb'))
		#pickle.dump(output.detach().numpy(), open('../activations/%s/%s_output.pkl'%(subfolder,output_name),'wb'))