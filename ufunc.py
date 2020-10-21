#this script will contain loading and procressing functions
import argparse
import torch
import numpy as np
from torchvision import datasets, transforms, models
from PIL import Image
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

def arguments_parser_for_train():
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
    return aparser.parse_args()

def picking_arch(architecture):
    model = getattr(models, architecture)(pretrained = True)
    return model

def load_data(data_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    
    train_set = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_set = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_set = datasets.ImageFolder(valid_dir, transform=valid_transforms)


    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=100)
    return test_loader, train_loader, valid_loader, train_set, test_set, valid_set

def process_image(image_path):
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image_path)
    
    # Process a PIL image for use in a PyTorch model
    pre_process = transforms.Compose([transforms.Resize(256),
                                     transforms.RandomResizedCrop(244),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])
    pre_processed_img = pre_process(img)
    
    return pre_processed_img

