#train models on letter discrimination task

from __future__ import print_function
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_classes
from torchvision import datasets, models, transforms
import pickle
import custom_models

### PARAMS ###
#Just stuff you might want to quickly change

output_name = 'customsmall_letter_aug'   # base name for various outputs, like the print log and the saved model
use_cuda = True   #use GPU acceleration


criterion = nn.CrossEntropyLoss()
epochs = 50
torch.manual_seed(2)
batch_size = 100 # size of training minibatches
test_batch_size = 200 # size of test minibatches
save_model = True    # Should the model be saved?
save_model_interval = 100 #Save model every X epochs
num_classes = 26        # number of classes in dataset, 26 for letters in alphabet   

#model
#model = models.alexnet(pretrained=False) #load a model from the models.py script, or switch this to torch.load(path_to_model) to load a model from a .pt file   
#num_ftrs = model.classifier[6].in_features
#model.classifier[6] = nn.Linear(num_ftrs,num_classes)

model = torch.load('../models/customsmall_letter_aug_acc0.944.pt')
#model = custom_models.CustomNet_small()

optimizer = optim.Adam(model.parameters(), lr=.001)  # adam optimizer

label_names = data_classes.letter_labels

### End of Params ###

def train(model, device, train_loader, optimizer, criterion, epoch):
	
	model.train()

	for batch_idx, (data,target) in enumerate(train_loader):

		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output,target)
		loss.backward()
		optimizer.step()
		

		if batch_idx % 5 == 0:
			print_out('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx * len(data), len(train_loader.dataset),
			100. * batch_idx / len(train_loader), loss.item()),print_log)                

def test(model, device, test_loader, criterion, epoch, max_acc):

	model.eval()
	test_loss = 0
	correct = 0
	indiv_acc_dict = {}
	#There is a lot of code here because in the log we record f1 scores and accuracies for each class
	for i in range(num_classes):
		indiv_acc_dict[i] = [0,0,0]
	with torch.no_grad():
		for data,target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += criterion(output, target).item()
			pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
			
			correct += pred.eq(target.view_as(pred)).sum().item()

			for i in range(len(pred)):
				indiv_acc_dict[int(pred[i])][2] += 1          
				indiv_acc_dict[int(target.view_as(pred)[i])][0] += 1
				if int(pred[i]) == int(target.view_as(pred)[i]):
					indiv_acc_dict[int(pred[i])][1] += 1


	test_loss /= len(test_loader.dataset)
	total_acc = correct / len(test_loader.dataset)
	print_out('epoch: ' + str(epoch),print_log)
	print_out('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * total_acc),print_log)

	if ((epoch-1)%save_model_interval == 0 or (total_acc > max_acc and total_acc > .94)) and save_model == True:       #condition to save model
		print('SAVING MODEL')
		torch.save(model,'../models/%s_acc%s.pt'%(output_name,str(round(total_acc,3))))

	if total_acc > max_acc:
		max_acc = total_acc

	print_out('class    total     guessed    accuracy    f1-score',print_log)
	for prednum in indiv_acc_dict:
		if indiv_acc_dict[prednum][0] == 0:
			print_out('no samples for class %s'%str(label_names[prednum]),print_log)
		else:
			total = indiv_acc_dict[prednum][0]
			guessed = indiv_acc_dict[prednum][2]
			accuracy = round(indiv_acc_dict[prednum][1]/indiv_acc_dict[prednum][0],3)
			f1 = round(total*accuracy/(total+guessed)*2,3)
			print_out('%s        %s         %s        %s         %s'%(label_names[prednum],str(total),str(guessed),str(accuracy),str(f1)),print_log)
			#indiv_acc_dict[prednum].append(','.join([str(round(accuracy,3)),str(guessed),str(round(f1,3))]))
	print_out('\n',print_log)
	return max_acc


def print_out(text,log):
	print(text)
	log.write(str(text))
	log.flush()

if __name__ == '__main__':

	print_log = open('../training_logs/'+output_name+'_log.txt','a+') #file logs everything that prints to console

	#Using GPU
	use_cuda = use_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	model.to(device)
	
	#Data Loaders
	kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

	train_loader = torch.utils.data.DataLoader(
			data_classes.letters_train_data(train=True),
			batch_size=batch_size, shuffle=True, **kwargs)

	test_loader = torch.utils.data.DataLoader(
			data_classes.letters_train_data(train=False, size_vary=False),
			batch_size=test_batch_size, shuffle=True, **kwargs)

	print_out('OPTIMIZER',print_log)
	print_out(optimizer,print_log)
	print_out('MODEL',print_log)
	print_out(model,print_log)

	start_time = time.time()

	max_acc = 0 # maximum accuracy achieved so far, used in test to decide if we should save the model after every epoch
	for epoch in range(1, epochs + 1):
		train(model, device, train_loader, optimizer, criterion, epoch)
		max_acc = test(model, device, test_loader, criterion, epoch, max_acc)


	print_out('Total Run Time:',print_log)
	print_out("--- %s seconds ---" % (time.time() - start_time),print_log)