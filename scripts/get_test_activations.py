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
output_name = 'alexnet_letter_aug_acc0.982'

####END OF PARAMS ########

#data loader
test_loader = DataLoader(
        data_classes.letters_validation_data(),
        batch_size=len(data_classes.letters_validation_data()), shuffle=False)

#model
model = torch.load('../models/%s.pt'%output_name)
model.to('cpu')
#model = models.alexnet(pretrained=True)
#num_ftrs = model.classifier[6].in_features
#model.classifier[6] = nn.Linear(num_ftrs,26)       #changes size of output to 26
model.eval()

#get activations into dict
activations = {}
for data,target in test_loader:
	x = data
	for layer, (name, module) in enumerate(model.features._modules.items()):
		print(layer)
		x=module(x)
		activation = x.detach().numpy()
		if SUM:
			activation = activation.sum(axis=2).sum(axis=2)
		if FLIP:
			activation = np.swapaxes(activation,0,1)
		activations[str(layer)+'_'+str(module).split('(')[0]] = activation

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





