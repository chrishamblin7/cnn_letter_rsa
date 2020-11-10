#Get all activations from model
import torch
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import data_classes
import scipy.io
import pickle
from scipy import spatial

#params
SUM = False  #sum output activation maps
SIMMATRIX = True
FLIP = False #flip output so its featuresXimages not imagesXfeatures
output_type = 'mat'

#output_names = ['alexnet_letter_aug_seed0_acc0.981','alexnet_letter_aug_seed1_acc0.988',
#				'alexnet_letter_aug_seed2_acc0.982','alexnet_letter_aug_seed3_acc0.989','alexnet_letter_aug_seed4_acc0.978',
#				'alexnet_letter_aug_seed5_acc0.979','alexnet_letter_aug_seed6_acc0.987','alexnet_letter_aug_seed7_acc0.992',
#				'alexnet_letter_aug_seed8_acc0.991','alexnet_letter_aug_seed9_acc0.989','customsmall_letter_aug_seed0_acc0.964',
#				'customsmall_letter_aug_seed1_acc0.973','customsmall_letter_aug_seed2_acc0.955','customsmall_letter_aug_seed3_acc0.966',
#				'customsmall_letter_aug_seed4_acc0.963','customsmall_letter_aug_seed5_acc0.968','customsmall_letter_aug_seed6_acc0.968',
#				'customsmall_letter_aug_seed7_acc0.973','customsmall_letter_aug_seed8_acc0.971','customsmall_letter_aug_seed9_acc0.965']
#output_name = 'alexnet_imagenet'

subfolder = 'mixednet'        #(branchingnet,alexnet,smallnet)

output_names = ['alexnet_mixed_letters_normed_ep5_imnet0.467_letter0.967']


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

test_loader = DataLoader(
			data_classes.letters_expstim_data(transform=data_classes.mixed_letter_transform),
			batch_size=len(data_classes.letters_expstim_data()), shuffle=False)

for output_name in output_names:
	#model
	model = torch.load('../models/%s/%s.pt'%(subfolder,output_name))
	model.to('cpu')

	#num_ftrs = model.classifier[6].in_features
	#model.classifier[6] = nn.Linear(num_ftrs,26)       #changes size of output to 26
	model.eval()

	#get activations into dict
	activations = {}
	#matrices = {}
	for data,target in test_loader:
		
		if subfolder == 'branchingnet':    #we need to pass input through alexnet
			branch_point = branch_key[output_name.split('_')[1]]
			data = through_network(data, network=alexnet.features,branch_point = branch_point)
		x = data

		# get conv features
		print('features')
		for layer, (name, module) in enumerate(model.features._modules.items()):
			print(layer)
			x=module(x)
			activation = x.detach().numpy()
			if SUM:
				activation = activation.sum(axis=2).sum(axis=2)
			if FLIP:
				activation = np.swapaxes(activation,0,1)
			if SIMMATRIX:
				activation = activation.reshape(*activation.shape[:1],-1)
				print(activation.shape)
				#activation = torch.flatten(activation,start_dim=1)
				activation = spatial.distance.pdist(activation, metric='euclidean')
				#print(spatial.distance.squareform(activation).shape)
				#print(activation.shape)
			activations['features'+str(module).split('(')[0]+'_'+str(layer)] = activation

		# get avgpool
		try:
			x = model.avgpool(x)
			activation = x.detach().numpy()
			if SUM:
				activation = activation.sum(axis=2).sum(axis=2)
			if FLIP:
				activation = np.swapaxes(activation,0,1)
			if SIMMATRIX:
				activation = activation.reshape(*activation.shape[:1],-1)
				print(activation.shape)
				#activation = torch.flatten(activation,start_dim=1)
				activation = spatial.distance.pdist(activation, metric='euclidean')
				#print(spatial.distance.squareform(activation).shape)
			activations['avg_pool'] = activation
		except:
			pass
		
		x = torch.flatten(x, 1)

		#classifier
		print('classifier')
		for layer, (name, module) in enumerate(model.classifier._modules.items()):
			print(layer)
			if str(module).split('(')[0]=='Dropout':
				continue
			x=module(x)
			activation = x.detach().numpy()
			if FLIP:
				activation = np.swapaxes(activation,0,1)
			if SIMMATRIX:
				activation = activation.reshape(*activation.shape[:1],-1)
				print(activation.shape)
				#activation = torch.flatten(activation,start_dim=1)
				activation = spatial.distance.pdist(activation, metric='euclidean')
				#print(spatial.distance.squareform(activation).shape)
			activations['classifier'+str(module).split('(')[0]+'_'+str(layer)] = activation
		
		output = model(data) #network final output


	#print(model)
	for key in activations:
		print(key)
		print(activations[key].shape)
	print('output')
	print(output.shape)

	if output_type == 'mat':
		scipy.io.savemat('../simmatrices/%s/%s_act_simmatrices.mat'%(subfolder,output_name), activations)
		#scipy.io.savemat('../activations/%s/%s_output.mat'%(subfolder,output_name), dict(output=output.detach().numpy()))
	else:
		pickle.dump(activations, open('../simmatricess/%s/%s_act_simmatrices.pkl'%(subfolder,output_name),'wb'))
		#pickle.dump(output.detach().numpy(), open('../activations/%s/%s_output.pkl'%(subfolder,output_name),'wb'))

