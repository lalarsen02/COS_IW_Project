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

# Set the device to use
# CUDA refers to the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(str(device))

## Hyperparameters
num_epochs = 10
num_classes = 4  # there are 4 drums: kick, snare, cymbals, and toms
batch_size = 128

## Fixing Random Seed for Reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

## Importing Testing and Training Datasets
training_X, training_y, testing_X, testing_y = createSpectrograms.create_spectrograms()

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
        return image, label

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

# train_loader returns batches of training data. See how train_loader is used in the Trainer class later
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,num_workers=0)

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
    
class Trainer():
    def __init__(self,net=None,optim=None,loss_function=None, train_loader=None):
        self.net = net
        self.optim = optim
        self.loss_function = loss_function
        self.train_loader = train_loader

    def train(self,epochs):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            for data in self.train_loader:

                # Note that X has shape (batch_size, number of channels, height, width)
                # which is equal to (128,1,__,__) since our default batch_size = 128 and 
                # the image has only 1 channel
                X = data[0]
                print(X.shape)
                y = data[1]
                
                # ACT11-Zero the gradient in the optimizer i.e. self.optim
                ################
                self.optim.zero_grad()
                ################

                # ACT12-Getting the output of the Network
                ################
                output = self.net(X)
                ################

                # ACT13-Computing loss using loss function i.e. self.loss_function
                ################
                loss = self.loss_function(output,y)
                ################

                # ACT14-Backpropagate to compute gradients of parameteres
                ################
                loss.backward()
                ################

                # ACT15-Call the optimizer i.e. self.optim
                ################
                self.optim.step()
                ################

                epoch_loss += loss.item()
                epoch_steps += 1
            # average loss of epoch
            losses.append(epoch_loss / epoch_steps)
            print("epoch [%d]: loss %.3f" % (epoch+1, losses[-1]))
        return losses
    
learning_rate = 0.05

net = ConvNet()
net = net.to(device)
opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
loss_function = nn.CrossEntropyLoss()

trainer = Trainer(net=net, optim=opt, loss_function=loss_function, train_loader=train_loader)

losses = trainer.train(num_epochs)
###ASSERTS
assert(losses[-1] < 0.03)
assert(len(losses)==num_epochs)  # because you record the loss after each epoch