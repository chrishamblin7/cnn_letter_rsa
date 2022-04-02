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
from subprocess  import call
import argparse

### PARAMS ###
#Just stuff you might want to quickly change

output_name = 'alexnet_letters'   # base name for various outputs, like the print log and the saved model
use_cuda = True   #use GPU acceleration


criterion = nn.CrossEntropyLoss()
epochs = 500
seed = 0 
lr = .001
device = 'cuda:1'
batch_size = 64 # size of training minibatches
test_batch_size = 200 # size of test minibatches
save_model = True    # Should the model be saved?
acc_thresh = .90     #how good does the model need to be to save?
save_only_best = True # if true only keep the model after the epoch with the best accuracy
save_model_interval = 100 #Save model every X epochs
num_classes = 26        # number of classes in dataset, 26 for letters in alphabet   



use_command_line = False         #set this to true to change arguments as you call this script from the command-line (set to true in conjunction with the batch submission script)
#command Line argument parsing
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--output-name", type = str, default = 'mystery')
	parser.add_argument("--branch-point", type = str, default = '1')
	parser.add_argument('--seed', type=int, default=0, metavar='S',
						help='random seed (default set in parameters file)')
	parser.add_argument('--save-only-best', action='store_true', default=False,
						help='only keep single highest accuracy model after training')


	args = parser.parse_args()
	return args

args = get_args()
if use_command_line:
	output_name = args.output_name
	seed=args.seed
	branch_point = args.branch_point
	save_only_best = args.save_only_best

torch.manual_seed(seed)

#model
#model = custom_models.LeNet5(n_classes=num_classes) #load a model from the models.py script, or switch this to torch.load(path_to_model) to load a model from a .pt file 
model = models.alexnet(pretrained=False)  
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs,num_classes)


#model = torch.load('../models/alexnet_pretrained_letters_acc0.893.pt')
#model = custom_models.BranchingNet()

optimizer = optim.Adam(model.parameters(), lr=lr)  # adam optimizer
#scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = optim.lr_scheduler.CyclicLR(optimizer, .0001,.005,cycle_momentum=False)

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
	print_out('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * total_acc),print_log)

	if ((epoch-1)%save_model_interval == 0 or total_acc > acc_thresh) and save_model == True:       #condition to save model
		if not save_only_best:
			print('SAVING MODEL')
			torch.save(model,'../models/%s_acc%s.pt'%(output_name,str(round(total_acc,3))))
		elif total_acc > max_acc:
			if os.path.exists('../models/%s_acc%s.pt'%(output_name,str(round(max_acc,3)))):
				call('rm ../models/%s_acc%s.pt'%(output_name,str(round(max_acc,3))),shell=True)
			print('SAVING MODEL')
			torch.save(model,'../models/%s_acc%s.pt'%(output_name,str(round(total_acc,3))))			
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
	log.write('\n')
	log.flush()

if __name__ == '__main__':

	print_log = open('../training_logs/'+output_name+'_log.txt','a+') #file logs everything that prints to console

	#Using GPU
	use_cuda = use_cuda and torch.cuda.is_available()
	#device = torch.device("cuda:0" if use_cuda else "cpu")
	model.to(device)
	
	#Data Loaders
	kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

	train_loader = torch.utils.data.DataLoader(
			data_classes.augmentation_letters_data(train=True),
			batch_size=batch_size, shuffle=True, **kwargs)

	# test_loader = torch.utils.data.DataLoader(
	# 		data_classes.augmentation_letters_data(train=False),
	# 		batch_size=test_batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
	 		data_classes.letters_train_data(train=False,root_dir = '../../data/', size_vary=False),
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