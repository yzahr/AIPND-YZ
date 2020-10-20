import torch
import numpy as np
import torch.nn.functional as F
import ufunc
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

#building the network function 
def built_network(model, features, hlayers, dropout, output):
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(features, hlayers)),
                          ('relu', nn.ReLU()), 
                          ('d1', nn.Dropout(p=0.1)),
                          ('fc2', nn.Linear(hlayers, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model = model.classifier
    return model

#building the validation function 
def model_validation (model, valid_loader, criterion, device_mode):
    # Initiate the validation accuracy & validation loss parameters with 0.
    valid_accuracy = 0
    valid_loss = 0
    
    # using gpu if available
    if torch.cuda.is_available() and device_mode == 'gpu':
        model.to('cuda')
    
    # Looping through every batch
    for inputs, labels in valid_loader:
        
        # using gpu if available
        if torch.cuda.is_available() and device_mode == 'gpu':
        # Moving parameters tensors to the device.
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        # Forward pass
        output = model.forward(inputs)
        
        # setting the validation loss parameter
        valid_loss += criterion(output, labels).item()

        # taking the exponatinal since I used log-softmax function
        ps = torch.exp(output)
        
        # max probability will be the predicted class, and will be comapred with the label
        equality = (labels.data == ps.max(dim=1)[1])
        
        # taking the mean to provide accuracy
        valid_accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, valid_accuracy

#building the train function 

def model_train (model, valid_loader, train_loader, criterion, optimizer, epochs, device_mode):
    steps = 0
    print_every = 40

    # using gpu if available
    if torch.cuda.is_available() and device_mode == 'gpu':
        model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        # Iterating over to perform training
        for ii, (images, labels) in enumerate(train_loader):
            steps += 1
            # using gpu if available
            if torch.cuda.is_available() and device_mode == 'gpu':
                images, labels = images.to('cuda'), labels.to('cuda') 
            # zeroing parameter gradients
            optimizer.zero_grad()   
            # Forward and backward passes
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()   
            running_loss += loss.item()  
            # validation process
            if steps % print_every == 0:         
                # setting model to evaluation mode during validation
                model.eval()          
                # Gradients are turned off
                with torch.no_grad():
                    validation_loss, accuracy = model_validation(model, valid_loader, criterion, device_mode)
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(valid_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loader)))           
                running_loss = 0           
                # Turning training back on
                model.train()
    return model

#building the test function
def model_test (model, test_loader, device_mode):
    correct = 0
    total = 0
    # using gpu if available
    if torch.cuda.is_available() and device_mode == 'gpu':
        model.to('cuda')
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # using gpu if available
            if torch.cuda.is_available() and device_mode == 'gpu':
                images, labels = images.to('cuda'), labels.to('cuda')
            # probabilities pasrsing
            outputs = model(images)
            # Turn probabilities into predictions
            _, predicted_result = torch.max(outputs.data, 1)
            # Total n. of images
            total += labels.size(0)
            # Counting n. of cases in which predictions are correct
            correct += (predicted_result == labels).sum().item()

    print(f"Accuracy of model performed on the test images: {round(100 * correct / total,2)}%")
    
    
# saving the model after it was trained
def save_checkpoint (model, train_set, epochs, optimizer, checkpoint_dir):
    checkpoint = {'classifier': model.classifier, 'state_dict': model.state_dict(),
                         'optimizer' : optimizer.state_dict(),
                         'class_to_idx':train_set.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')
    print('checkpoint was marked')
    
#predict function
    
def load_checkpoint(checkpoint_dir):
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_dir)
    else:
        checkpoint = torch.load(checkpoint_dir, map_location = 'cpu')
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def predict(image_path, model, top_k):

    loaded_model = load_checkpoint(model).cpu()
 
    img = ufunc.process_image(image_path)
    
    #unsequeezing the added image
    image_added_dim = img.unsqueeze_(0)  
  
    loaded_model.eval()
    
    # Turning off gradients
    with torch.no_grad():
        
        # Forward pass 
        output = loaded_model.forward(image_added_dim)
        
    # Taking exponential to get the probabilities from log softmax funtion
    ps = torch.exp(output)
    
    # listing the most predicted probablities
    probs, top_k_indices = ps.topk(topk)
    
    classes = []
    for index in top_k_indices.cpu()[0]:
        classes.append(list(loaded_model.class_to_idx)[index.numpy()]) # Here we take the class from the index.
    
    return probs.cpu()[0].numpy(), classes

     

