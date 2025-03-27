#!/usr/bin/env python

# ----------------------------------------------
# Research Assignment for my COS IW
# ----------------------------------------------

import createSpectrograms
import random
import numpy as np 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchviz import make_dot

# ----------------------------------------------

# Set the device to use
# CUDA refers to the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(str(device))

## Hyperparameters
num_epochs = 10
labels = ['Kick', 'Snare', 'Cymbals', 'Toms']
num_classes = len(labels)  # use the number of labels
batch_size = 32

# Create a dictionary to map each label to a unique integer
label_map = {label: i for i, label in enumerate(labels)}

## Importing Testing and Training Datasets
training_X, training_y, testing_X, testing_y = createSpectrograms.create_spectrograms()

# Convert the string labels to integers using the label map
training_y = [label_map[label] for label in training_y]
testing_y = [label_map[label] for label in testing_y]

## Fixing Random Seed for Reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class ImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        label = self.labels[index]
        with open(file_path, 'rb') as f:
            image = Image.open(f)
            if self.transform is not None:
                image = self.transform(image)
        return image, torch.tensor(label)

    def __len__(self):
        return len(self.file_paths)
    
# transforms to apply to the data, converting the data to PyTorch tensors and normalizing the data
trans = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# datasets
train_dataset = ImageDataset(training_X, training_y, transform=trans)
test_dataset = ImageDataset(testing_X, testing_y, transform=trans)

# train_loader returns batches of training data
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,num_workers=0)

# CNN Model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2,2)

        self.size_linear = 64*77*193
        self.fc1 = nn.Linear(self.size_linear, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self.size_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# class for training the Neural Network
class Trainer():
    def __init__(self,net=None,optim=None,loss_function=None, train_loader=None, accumulation_steps=1):
        self.net = net
        self.optim = optim
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.accumulation_steps = accumulation_steps

    def train(self,epochs):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            accumulated_loss = 0.0

            # learns from data
            for i, data in enumerate(self.train_loader):
                X = data[0].to(device)
                y = data[1].to(device)

                self.optim.zero_grad()
                output = self.net(X)
                loss = self.loss_function(output,y)

                accumulated_loss += loss
                epoch_loss += loss.item()
                epoch_steps += 1

                if (i + 1) % self.accumulation_steps == 0:
                    accumulated_loss /= self.accumulation_steps
                    accumulated_loss.backward()
                    self.optim.step()
                    accumulated_loss = 0.0

            # average loss of epoch
            losses.append(epoch_loss / epoch_steps)
            print("epoch [%d]: loss %.3f" % (epoch+1, losses[-1]))

        return losses

# variables to set up machine learning
learning_rate = 0.01

net = ConvNet()
net = net.to(device)
opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
loss_function = nn.CrossEntropyLoss()

trainer = Trainer(net=net, optim=opt, loss_function=loss_function, train_loader=train_loader, accumulation_steps=4)

losses = trainer.train(num_epochs)

###ASSERTS
assert(losses[-1] < 0.03)
assert(len(losses)==num_epochs)