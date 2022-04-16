import torch
import torch.nn as nn
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

bos = load_boston()

df = pd.DataFrame(bos.data)
df.columns = bos.feature_names
df['Price'] = bos.target

data = df[df.columns[:-1]]
data = data.apply(
    lambda x: (x - x.mean()) / x.std()
)

data['Price'] = df.Price

X = data.drop('Price', axis=1).to_numpy()
Y = data['Price'].to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)

Y_train = torch.tensor(Y_train, dtype=torch.float).view(-1, 1)
Y_test = torch.tensor(Y_test, dtype=torch.float).view(-1, 1)

class RegressionModelTorch():

    def __init__(self, input_size, output_size, batch_size, learning_rate, epoch):

        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            nn.Linear(16, output_size)
        )

        self.batch_size = batch_size
        self.lr = learning_rate

        self.criterion = nn.MSELoss()

        self.datasets = torch.utils.data.TensorDataset(X_train, Y_train)
        self.tests = torch.utils.data.TensorDataset(X_test, Y_test)

        self.train_iter = torch.utils.data.DataLoader(self.datasets, batch_size=8, shuffle=True)
        self.test_iter = torch.utils.data.DataLoader(self.tests, batch_size=8, shuffle=True)

        self.epochs = epoch

    def train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.train_loss = []
        self.val_loss = []

        for e in range(self.epochs):
            running_loss = 0
            rc = 0
            for x, y in self.train_iter:

                output = self.model(x)

                loss = self.criterion(output, y)
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()

            for x, y in self.test_iter:

                output = self.model(x)

                lossc = self.criterion(output, y)
                rc += lossc.item()


            self.val_loss.append(rc/len(X_test))
            self.train_loss.append(running_loss/len(X_train))
            print('Epoch {} loss train: {:.4f} loss val: {:.4f}'.format(e + 1, running_loss/len(X_train), rc/len(X_test)))

    def getTrain_Loss(self):
        return self.train_loss

    def getVal_Loss(self):
        return self.val_loss



