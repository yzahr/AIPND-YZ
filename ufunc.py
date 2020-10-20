#this script will contain loading and procressing functions
import torch
import numpy as np
from torchvision import datasets, transforms, models
from PIL import Image
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
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

def process_image(image):
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    # Process a PIL image for use in a PyTorch model
    pre_process = transforms.Compose([transforms.Resize(256),
                                     transforms.RandomResizedCrop(244),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])
    pre_processed_img = pre_process(img)
    
    return pre_processed_img

