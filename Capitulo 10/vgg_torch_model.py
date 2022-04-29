from tkinter.messagebox import NO
from traceback import format_exc
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.datasets import Flowers102
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import random_split, TensorDataset, Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.optim as optim
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

X , Y = oxflower17.load_data(one_hot=True) 

class DatasetFlower(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        return self.X[ix], self.Y[ix]

dataset = DatasetFlower(X, Y)

dataloader = DataLoader(dataset, batch_size=64,shuffle=True)
print("Tamaño del dataset: {}".format(len(dataset)))

class VGGTorch(nn.Module):
    def __init__(self) -> None:
        super(VGGTorch, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=224, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.fc1 = nn.Linear(64, 8192)
        self.fc2 = nn.Linear(8192, 4096)
        self.fc3 = nn.Linear(4096, 17)

    
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1) # Este actua como el Flatten
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x

class VggNetTorch():
    def __init__(self, learning_rate, epochs):
        self.model = nn.Sequential(
                    nn.Conv2d(in_channels=224, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                    nn.BatchNorm1d(64), # Primer bloque

                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                    nn.BatchNorm1d(128), # Segundo Bloque

                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                    nn.BatchNorm1d(256), # Tercer Bloque

                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                    nn.BatchNorm1d(512), # Cuarto Bloque

                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                    nn.BatchNorm1d(512), # Cuarto Bloque
                    nn.Flatten(),
                    nn.Linear(512, 4096),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(4096, 17),
                    nn.Softmax(dim=1)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs

    def train(self):
        self.model = self.model.cuda(device=device)
        total_step = len(dataloader)
        for epoch in range(self.epochs):
            for i, (images, labels) in enumerate(dataloader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, self.epochs, i+1, total_step, loss.item()))

        # Test the model
        self.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

vgg = VggNetTorch(0.001,50)
vgg.train()