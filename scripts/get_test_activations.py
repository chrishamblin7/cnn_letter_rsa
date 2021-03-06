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
#output_names = ['alexnet_letter_aug_seed0_acc0.981','alexnet_letter_aug_seed1_acc0.988',
#				'alexnet_letter_aug_seed2_acc0.982','alexnet_letter_aug_seed3_acc0.989','alexnet_letter_aug_seed4_acc0.978',
#				'alexnet_letter_aug_seed5_acc0.979','alexnet_letter_aug_seed6_acc0.987','alexnet_letter_aug_seed7_acc0.992',
#				'alexnet_letter_aug_seed8_acc0.991','alexnet_letter_aug_seed9_acc0.989','customsmall_letter_aug_seed0_acc0.964',
#				'customsmall_letter_aug_seed1_acc0.973','customsmall_letter_aug_seed2_acc0.955','customsmall_letter_aug_seed3_acc0.966',
#				'customsmall_letter_aug_seed4_acc0.963','customsmall_letter_aug_seed5_acc0.968','customsmall_letter_aug_seed6_acc0.968',
#				'customsmall_letter_aug_seed7_acc0.973','customsmall_letter_aug_seed8_acc0.971','customsmall_letter_aug_seed9_acc0.965']
#output_name = 'alexnet_imagenet'

subfolder = 'branchingnet'        #(branchingnet,alexnet,smallnet)

output_names = [
'branchingnet_relu1_letter_seed0_acc0.968','branchingnet_relu1_letter_seed8_acc0.969','branchingnet_relu2_letter_seed6_acc0.979','branchingnet_relu3_letter_seed4_acc0.983',  
'branchingnet_relu4_letter_seed2_acc0.972','branchingnet_relu5_letter_seed0_acc0.973','branchingnet_relu5_letter_seed8_acc0.969','branchingnet_relu1_letter_seed1_acc0.971',  
'branchingnet_relu1_letter_seed9_acc0.963','branchingnet_relu2_letter_seed7_acc0.978','branchingnet_relu3_letter_seed5_acc0.978','branchingnet_relu4_letter_seed3_acc0.974',  
'branchingnet_relu5_letter_seed1_acc0.972','branchingnet_relu5_letter_seed9_acc0.971','branchingnet_relu1_letter_seed2_acc0.976','branchingnet_relu2_letter_seed0_acc0.977',
'branchingnet_relu2_letter_seed8_acc0.976','branchingnet_relu3_letter_seed6_acc0.983','branchingnet_relu4_letter_seed4_acc0.975','branchingnet_relu5_letter_seed2_acc0.971',
'branchingnet_relu1_letter_seed3_acc0.972','branchingnet_relu2_letter_seed1_acc0.975','branchingnet_relu2_letter_seed9_acc0.975','branchingnet_relu3_letter_seed7_acc0.981',  
'branchingnet_relu4_letter_seed5_acc0.967','branchingnet_relu5_letter_seed3_acc0.967','branchingnet_relu1_letter_seed4_acc0.972','branchingnet_relu2_letter_seed2_acc0.978',  
'branchingnet_relu3_letter_seed0_acc0.978','branchingnet_relu3_letter_seed8_acc0.981','branchingnet_relu4_letter_seed6_acc0.975','branchingnet_relu5_letter_seed4_acc0.972',
'branchingnet_relu1_letter_seed5_acc0.967','branchingnet_relu2_letter_seed3_acc0.979','branchingnet_relu3_letter_seed1_acc0.982','branchingnet_relu3_letter_seed9_acc0.98', 
'branchingnet_relu4_letter_seed7_acc0.973','branchingnet_relu5_letter_seed5_acc0.972','branchingnet_relu1_letter_seed6_acc0.969','branchingnet_relu2_letter_seed4_acc0.977', 
'branchingnet_relu3_letter_seed2_acc0.983','branchingnet_relu4_letter_seed0_acc0.976','branchingnet_relu4_letter_seed8_acc0.976','branchingnet_relu5_letter_seed6_acc0.973',
'branchingnet_relu1_letter_seed7_acc0.963','branchingnet_relu2_letter_seed5_acc0.977','branchingnet_relu3_letter_seed3_acc0.982','branchingnet_relu4_letter_seed1_acc0.977',  
'branchingnet_relu4_letter_seed9_acc0.976','branchingnet_relu5_letter_seed7_acc0.971']


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
        data_classes.letters_validation_data(),
        batch_size=len(data_classes.letters_validation_data()), shuffle=False)

for output_name in output_names:
	#model
	model = torch.load('../models/%s/%s.pt'%(subfolder,output_name))
	model.to('cpu')

	#num_ftrs = model.classifier[6].in_features
	#model.classifier[6] = nn.Linear(num_ftrs,26)       #changes size of output to 26
	model.eval()

	#get activations into dict
	activations = {}
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
		scipy.io.savemat('../activations/%s/%s_activations.mat'%(subfolder,output_name), activations)
		#scipy.io.savemat('../activations/%s/%s_output.mat'%(subfolder,output_name), dict(output=output.detach().numpy()))
	else:
		pickle.dump(activations, open('../activations/%s/%s_activations.pkl'%(subfolder,output_name),'wb'))
		#pickle.dump(output.detach().numpy(), open('../activations/%s/%s_output.pkl'%(subfolder,output_name),'wb'))





