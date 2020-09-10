#misc utility functions
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def add_more_classes_to_model(model_layer,num_new_classes): #model_layer := model.classifier[6] for alexnet
    num_ftrs = model_layer.in_features 
    num_old_classes = model_layer.out_features
    num_combined_classes = num_new_classes+num_old_classes
    new_linear = nn.Linear(num_ftrs,num_new_classes)
    new_classifier_weights = torch.cat((model_layer.weight, new_linear.weight), 0)
    new_classifier_bias = torch.cat((model_layer.bias, new_linear.bias), 0)
    new_classifier = nn.Linear(num_ftrs, num_combined_classes)
    new_classifier.weight = torch.nn.Parameter(new_classifier_weights)
    new_classifier.bias = torch.nn.Parameter(new_classifier_bias)
    return new_classifier