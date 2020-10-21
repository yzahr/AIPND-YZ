#this script will take care of the argparse inputs
import argparse
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn, optim
import ufunc
import netfunc
from ufunc import load_data
from netfunc import built_network, model_validation, model_train, model_test, save_checkpoint

#main parser
aparser = argparse.ArgumentParser(description= 'Network Inputs')

#arg for data directory
aparser.add_argument('data_dir', action = 'store_true' , default = '/flowers')

#arg for checkpoint
aparser.add_argument('--save_dir', action = 'store_true', default = 'checkpoint.pth', dest= 'save_dir', help = 'Input path for checkpoint')

#arg for gpu option
aparser.add_argument('--gpu', action = 'store_true', default = 'gpu', help = 'GPU is enabled by default')

# arg for architecture input
aparser.add_argument('--arch', dest = 'arch', default = 'vgg13', type = str, help=' input for your desired architecture model')

#arg for hidden units
aparser.add_argument('--hlayers', type = int, default = 500, help = 'input for your desired hidden layers integer')

#arg for dropout input
aparser.add_argument('--dropout', dest= 'dropout', type = float, default = 0.1, help = 'input for your desired dropout float')

#arg for learning rate
aparser.add_argument('--learning_rate', dest = 'learning_rate', type = float, default = 0.001, help = 'input for your desired learning rate float')

#arg fot epochs
aparser.add_argument('--epochs', dest= 'epochs', default = 1, help = 'input for your desired epochs integer', type = int)

#easing variables
parser = aparser.parse_args()

data_dir = parser.data_dir
checkpoint_dir = parser.save_dir
device_mode = parser.gpu
architecture = parser.arch
hlayers = parser.hlayers
dropout = parser.dropout
learning_rate = parser.learning_rate
epochs = parser.epochs

#code for choosing model

model = getattr(models, architecture)(pretrained = True)
return model

#processing data
test_loader, train_loader, valid_loader, train_set, test_set, valid_set = ufunc.load_data(data_dir)

#loading model parameters
features = model.classifier[0].in_features
output = 102
netfunc.built_network(model, features, hlayers, dropout, output)

# was reccommended by udacity to use the error function
criterion = nn.NLLLoss()

# Choosing Adam as the optimizer
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

#training model
netfunc.model_train(model, valid_loader, train_loader, criterion, optimizer, epochs, device_mode)

#testing model
netfunc.model_test(model, test_loader, device_mode)

#checkpointing model
netfunc.save_checkpoint(model, train_set, epochs, optimizer, checkpoint_dir)


print("Training Complete")








