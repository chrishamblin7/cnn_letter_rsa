#Get all activations from model
import torch
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import data_classes
import scipy.io
import pickle

#params
SUM = True  #sum output activation maps
FLIP = True #flip output so its featuresXimages not imagesXfeatures
output_type = 'mat'
output_names = ['alexnet_letter_aug_seed0_acc0.981','alexnet_letter_aug_seed1_acc0.988',
				'alexnet_letter_aug_seed2_acc0.982','alexnet_letter_aug_seed3_acc0.989','alexnet_letter_aug_seed4_acc0.978',
				'alexnet_letter_aug_seed5_acc0.979','alexnet_letter_aug_seed6_acc0.987','alexnet_letter_aug_seed7_acc0.992',
				'alexnet_letter_aug_seed8_acc0.991','alexnet_letter_aug_seed9_acc0.989','customsmall_letter_aug_seed0_acc0.964',
				'customsmall_letter_aug_seed1_acc0.973','customsmall_letter_aug_seed2_acc0.955','customsmall_letter_aug_seed3_acc0.966',
				'customsmall_letter_aug_seed4_acc0.963','customsmall_letter_aug_seed5_acc0.968','customsmall_letter_aug_seed6_acc0.968',
				'customsmall_letter_aug_seed7_acc0.973','customsmall_letter_aug_seed8_acc0.971','customsmall_letter_aug_seed9_acc0.965']

####END OF PARAMS ########

#data loader
test_loader = DataLoader(
        data_classes.letters_validation_data(),
        batch_size=len(data_classes.letters_validation_data()), shuffle=False)

for output_name in output_names:
	#model
	model = torch.load('../models/%s.pt'%output_name)
	#model = models.alexnet(pretrained=False)
	model.to('cpu')

	#num_ftrs = model.classifier[6].in_features
	#model.classifier[6] = nn.Linear(num_ftrs,26)       #changes size of output to 26
	model.eval()

	#get activations into dict
	activations = {}
	for data,target in test_loader:
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
			activations['features'+str(module).split('(')[0]+'_'+str(layer)] = activation
		# get avgpool
		x = model.avgpool(x)
		activation = x.detach().numpy()
		if SUM:
			activation = activation.sum(axis=2).sum(axis=2)
		if FLIP:
			activation = np.swapaxes(activation,0,1)
		activations['avg_pool'] = activation
		
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
			activations['classifier'+str(module).split('(')[0]+'_'+str(layer)] = activation
		
		output = model(data) #network final output


	#print(model)
	for key in activations:
		print(key)
		print(activations[key].shape)
	print('output')
	print(output.shape)

	if output_type == 'mat':
		scipy.io.savemat('../activations/%s_activations.mat'%output_name, activations)
		scipy.io.savemat('../activations/%s_output.mat'%output_name, dict(output=output.detach().numpy()))
	else:
		pickle.dump(activations, open('../activations/%s_activations.pkl'%output_name,'wb'))
		pickle.dump(output.detach().numpy(), open('../activations/%s_output.pkl'%output_name,'wb'))





