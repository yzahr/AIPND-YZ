import argparse
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn, optim
import ufunc
import netfunc
from PIL import Image
import json

#main parser
aparser = argparse.ArgumentParser(description= 'Prediction parser')

#image data argument
aparser.add_argument('image_path' , default = 'home/workspace/ImageClassifier/flowers/test/20/image_04910.jpg', action = 'store_true')

#directory argument
aparser.add_argument('data_dir', action = 'store_true' , default = '/flowers')

#checkpoint argument
aparser.add_argument('--save_dir', action = 'store_true', default = 'checkpoint.pth', dest= 'save_dir', help = 'Input path for checkpoint')

# top 5 classes argument
aparser.add_argument('--top_k', type= int, default = 5,dest = 'top_k', help = 'input for how many top classes you want to see')

# category names margument
aparser.add_argument('--cat_names', action = 'store_true', default= 'home/workspace/ImageClassifier/cat_to_name.json' , dest = 'cat_names', help = ' purposed for real name categories')

#arg for gpu option
aparser.add_argument('--gpu', action = 'store_true', default = 'gpu', help = 'GPU is enabled by default')

# arg for architecture input
aparser.add_argument('--arch', dest = 'arch', default = 'vgg13', type = str, help=' input for your desired architecture model')


#easing variables
parser = aparser.parse_args()

arch =  parser.arch
data_dir = parser.data_dir
checkpoint_dir = parser.save_dir
device_mode = parser.gpu
cat_names = parser.cat_names
image_path = parser.image_path
top_k = parser.top_k

#setting model template
model = getattr(models, arch)(pretrained=True)

#loading checkpoint
loaded_model = netfunc.load_checkpoint(checkpoint_dir)

#mapping labels
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
#loading data
test_loader, train_loader, valid_loader, train_set, test_set, valid_set = ufunc.load_data(data_dir)


probs,classes = netfunc.predict(image_path, model, top_k)
print(probs)
print(classes)

# Getting prediction
probs,classes = predict(image_path_test, model_path, topk=5)

# Converting classes to names
names = []
for index in classes:
    names += [cat_to_name[index]]
    
print(f"prediction generated this name for the inputed flower:'{names[0]}' with a probability of {round(probs[0]*100,4)}%")


