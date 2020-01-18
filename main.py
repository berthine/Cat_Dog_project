import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms,models
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torch.optim import lr_scheduler
from pathlib import Path
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from matplotlib.image import imread
import os
import glob
import os.path as osp
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torch.utils import data as D
from Dataset import CatDogDataset
from Model import CNN_1, CNN_2


# function to count number of parameters
def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return nppip


accuracy_list = []

def train(epoch, model, perm=torch.arange(0, 50176).long()):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader_train):
        

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader_train.dataset),
                100. * batch_idx / len(dataloader_train), loss.item()))
            
def test(model, perm=torch.arange(0, 50176).long()):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in dataloader_test:
        # permute pixels
        data = data.view(-1, 224*224)
        data = data[:, perm]
        data = data.view(-1, 3, 224, 224)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss                                                               
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(dataloader_test.dataset)
    accuracy = 100. * correct / len(dataloader_test.dataset)
    accuracy_list.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataloader_train.dataset),
        accuracy))

image_size = (224, 224)
image_row_size = image_size[0] * image_size[1]
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
                                transforms.Resize(image_size), 
                                transforms.ToTensor(), 
                                transforms.Normalize(mean, std)])

path    = './Data/train'
data_train = CatDogDataset(path, transform=transform)

path    = './Data/test'
data_test = CatDogDataset(path, transform=transform)

shuffle     = True
batch_size  = 64
num_workers = 0

dataloader_train  = DataLoader(dataset=data_train, 
                         shuffle=shuffle, 
                         batch_size=batch_size, 
                         num_workers=num_workers)


dataloader_test  = DataLoader(dataset=data_test, 
                         shuffle=shuffle, 
                         batch_size=batch_size, 
                         num_workers=num_workers)

input_size  = 224*224*3   # images size 
output_size = 2           #number of classes      
# Training settings CNN_1
n_features = 6 # number of feature maps

model_CNN = CNN_1(input_size, n_features, output_size)
optimizer = optim.SGD(model_CNN.parameters(), lr=0.01, momentum=0.5)
print('Number of parameters: {}'.format(get_n_params(model_CNN)))

for epoch in range(0, 1):
    train(epoch, model_CNN)
    test(model_CNN)


# Training settings CNN_2

n_features = 6 # number of feature maps

model_CNN = CNN_2(input_size, n_features, output_size)
optimizer = optim.SGD(model_CNN.parameters(), lr=0.01, momentum=0.5)
print('Number of parameters: {}'.format(get_n_params(model_CNN)))

for epoch in range(0, 1):
    train(epoch, model_CNN)
    test(model_CNN)