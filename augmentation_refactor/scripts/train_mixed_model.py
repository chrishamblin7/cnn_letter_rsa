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

import pdb

### PARAMS ###
#Just stuff you might want to quickly change

output_name = 'alexnet_mixed_mixedaug'   # base name for various outputs, like the print log and the saved model
device = 'cuda:0'   #use GPU acceleration


criterion = nn.CrossEntropyLoss()
print_indiv_accuracies = False
epochs = 200
seed = 0 
lr = .0001
batch_size = 128 # size of training minibatches
test_batch_size = 128 # size of test minibatches
save_model = True    # Should the model be saved?
#letter_acc_thresh = .96     #how good does the model need to be to save?
save_only_best = True # if true only keep the model after the epoch with the best accuracy
#save_model_interval = 100 #Save model every X epochs
num_classes = 1026        # number of classes in dataset, 26 for letters in alphabet   

num_letter_copies = 2    # number of times to copy each letter image in dataset, 
#more copies == higher rate of training on letter examples

#Data Parallel
#main_gpu = 2
#secondary_gpu = 1
#data_parallel = True


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
model = models.alexnet(pretrained=True) #load a model from the models.py script, or switch this to torch.load(path_to_model) to load a model from a .pt file   
num_ftrs = model.classifier[6].in_features
letter_linear = nn.Linear(num_ftrs,26)
torch.nn.init.xavier_uniform_(letter_linear.weight)

new_classifier_weights = torch.cat((model.classifier[6].weight, letter_linear.weight), 0)
new_classifier_bias = torch.cat((model.classifier[6].bias, letter_linear.bias), 0)
new_classifier = nn.Linear(num_ftrs,1026)
new_classifier.weight = torch.nn.Parameter(new_classifier_weights)
new_classifier.bias = torch.nn.Parameter(new_classifier_bias)
model.classifier[6] = new_classifier

model_path = '../models/alexnet_mixed_letters_normed_ep5_imnet0.55_letter0.004.pt'
model = torch.load(model_path)
 #this epoch reading should be replaced with metadata saving
start_epoch = 0


#model = custom_models.BranchingNet()
#model = nn.DataParallel(model)


optimizer = optim.Adam(model.parameters(), lr=lr)  # adam optimizer
scheduler = optim.lr_scheduler.CyclicLR(optimizer, .00001,.0005,cycle_momentum=False)
### End of Params ###

def train(model, device, train_loader, optimizer, criterion, epoch):
	
	model.train()

	for batch_idx, (data,target,img_name) in enumerate(train_loader):
		#import pdb; pdb.set_trace()
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output,target)
		loss.backward()
		optimizer.step()
		

		if batch_idx % 500 == 0:

			print_out('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx * len(data), len(train_loader.dataset),
			100. * batch_idx / len(train_loader), loss.item()),print_log)                

			pred = output.max(1, keepdim=True)[1]
			imnet_correct = 0
			letter_correct = 0
			imnet_total = 0
			letter_total = 0
			imnet_guessed = 0
			letter_guessed = 0
			for i in range(len(target)):
				if target[i] < 1000:
					imnet_total+=1
					if (target[i] == pred[i]).item():
						imnet_correct += 1
				else:
					letter_total+=1
					if (target[i] == pred[i]).item():
						letter_correct += 1
				if pred[i] < 1000:
					imnet_guessed += 1
				else:
					letter_guessed += 1

			print_out('\nImagenet Accuracy: {}/{} ({:.2f}%)'.format(
				imnet_correct, imnet_total,
				100. * imnet_correct/imnet_total),print_log)

			print_out('\nLetter Accuracy: {}/{} ({:.2f}%)'.format(
				letter_correct, letter_total,
				100. * letter_correct/letter_total),print_log)

			print_out('\nLetter Guessed: %s\n'%str(letter_guessed)
					,print_log)
				

def test(model, device, test_loader, criterion, epoch, max_acc):
	print('testing model')
	model.eval()
	test_loss = 0
	total_correct = 0
	imnet_correct = 0
	letter_correct = 0
	imnet_total = 0
	letter_total = 0

	with torch.no_grad():
		for (data,target,img_name) in test_loader:
			
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += criterion(output, target).item()
			pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
			
			total_correct += pred.eq(target.view_as(pred)).sum().item()

			for i in range(len(target)):
				if target[i] < 1000:
					imnet_total+=1
					if (target[i] == pred[i]).item():
						imnet_correct += 1

				else:
					letter_total+=1
					if (target[i] == pred[i]).item():
						letter_correct += 1


	test_loss /= len(test_loader.dataset)
	total_acc = total_correct / len(test_loader.dataset)
	imnet_acc = imnet_correct / imnet_total
	letter_acc = letter_correct / letter_total

	print_out('epoch: ' + str(epoch),print_log)
	print_out('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
		test_loss, total_correct, len(test_loader.dataset),
		100. * total_acc),print_log)
	print_out('\nImagenet Accuracy: {}/{} ({:.2f}%)\n'.format(
		imnet_correct, imnet_total,
		100. * imnet_acc),print_log)

	print_out('\nLetter Accuracy: {}/{} ({:.2f}%)\n'.format(
		letter_correct, letter_total,
		100. * letter_acc),print_log)

	#if ((epoch-1)%save_model_interval == 0 or total_acc > acc_thresh) and save_model == True:       #condition to save model
	if not save_only_best:
		print('SAVING MODEL')
		torch.save(model,'../models/mixednet/%s_ep%s_imnet%s_letter%s.pt'%(output_name,str(epoch),str(round(imnet_acc,3)),str(round(letter_acc,3))))
	elif letter_acc > max_acc:
	#		if os.path.exists('../models/%s_acc%s.pt'%(output_name,str(round(max_acc,3)))):
	#			call('rm ../models/%s_acc%s.pt'%(output_name,str(round(max_acc,3))),shell=True)
	#		print('SAVING MODEL')
		torch.save(model,'../models/%s_ep%s_imnet%s_letter%s.pt'%(output_name,str(epoch),str(round(imnet_acc,3)),str(round(letter_acc,3))))			
	#		max_acc = total_acc

	return max_acc


def print_out(text,log):
	print(text)
	log.write(str(text))
	log.write('\n')
	log.flush()

if __name__ == '__main__':

	print_log = open('../training_logs/'+output_name+'_log.txt','a+') #file logs everything that prints to console


	model.to(device)
	#symlink duplicate letter examples
	#data_classes.symlink_letters_in_mixed(num_letter_copies)

	#Data Loaders
	kwargs = {'num_workers': 8, 'pin_memory': True} if ('cuda' in device) else {}

	train_loader = torch.utils.data.DataLoader(
			data_classes.mix_imagenet_aug_letter_data(train=True),
			batch_size=batch_size, shuffle=True, **kwargs)

	test_loader = torch.utils.data.DataLoader(
			data_classes.mix_imagenet_aug_letter_data(train=False),
			batch_size=test_batch_size, shuffle=False, **kwargs)



	print_out('output name: %s'%output_name,print_log)
	print_out('OPTIMIZER',print_log)
	print_out(optimizer,print_log)
	print_out('MODEL',print_log)
	print_out(model,print_log)

	start_time = time.time()

	max_acc = 0 # maximum accuracy achieved so far, used in test to decide if we should save the model after every epoch
	print('PRETRAINING TEST:')
	max_acc = test(model, device, test_loader, criterion, 1, max_acc)
	for epoch in range(start_epoch, start_epoch + epochs + 1):
		train(model, device, train_loader, optimizer, criterion, epoch)
		max_acc = test(model, device, test_loader, criterion, epoch, max_acc)

	#remove symlinks in letter dataset
	#data_classes.unsymlink_letters_in_mixed()

	print_out('Total Run Time:',print_log)
	print_out("--- %s seconds ---" % (time.time() - start_time),print_log)
