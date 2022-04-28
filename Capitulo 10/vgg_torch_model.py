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
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image
import torch.optim as optim
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split


X , Y = oxflower17.load_data(one_hot=True) 

x_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(Y)

val_size = 17 * 80 * 0.1
train_size = 17 *80 - val_size

dataset = TensorDataset(x_tensor, y_tensor)
val_size = int(len(dataset)*0.1)
train_size = len(dataset)- int(len(dataset)*0.1)

train_ds, val_ds = random_split(x_tensor, [train_size, val_size])


train_iter = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_iter = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=True)

for batch_idx, samples in enumerate(test_iter):
      print(batch_idx, type(samples))



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)

Y_train = torch.tensor(Y_train, dtype=torch.float).view(-1, 1)
Y_test = torch.tensor(Y_test, dtype=torch.float).view(-1, 1)

datasets = torch.utils.data.TensorDataset(X_train,Y_train)
tests = torch.utils.data.TensorDataset(X_test, Y_test)

train_iter = torch.utils.data.DataLoader(datasets, batch_size=64, shuffle=True)
test_iter = torch.utils.data.DataLoader(tests, batch_size=64, shuffle=True)

class VGGTorch(nn.Module):
    def __init__(self) -> None:
        super(VGGTorch, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
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

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 4096)
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
        self.model = VGGTorch()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #training with either cpu or cuda
        self.model = self.model.to()
        for epoch in range(50): #I decided to train the model for 50 epochs
            loss_ep = 0
            
            for batch_idx, (data, targets) in enumerate(train_iter):
                data = data.to(device=device)
                targets = targets.to(device=device)
                ## Forward Pass
                self.optimizer.zero_grad()
                scores = self.model(data)
                loss = self.criterion(scores,targets)
                loss.backward()
                self.optimizer.step()
                loss_ep += loss.item()
            print(f"Loss in epoch {epoch} :::: {loss_ep/len(train_iter)}")

            with torch.no_grad():
                num_correct = 0
                num_samples = 0
                for batch_idx, (data,targets) in enumerate(test_iter):
                    data = data.to(device=device)
                    targets = targets.to(device=device)
                    ## Forward Pass
                    scores = self.model(data)
                    _, predictions = scores.max(1)
                    num_correct += (predictions == targets).sum()
                    num_samples += predictions.size(0)
                print(
                    f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
                )


vgg = VggNetTorch(0.001, 250)
vgg.train()