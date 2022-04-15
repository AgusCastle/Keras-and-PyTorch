import torch
import torch.nn as nn
from torch import optim, nn
from torchvision import datasets
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from time import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


input_size = 784 # 28x28
output_size = 10
batch_size = 128
learning_rate = 0.001

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)

test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
valloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)

class Net_Torch():

    def __init__(self, input_size, output_size, batch_size, learning_rate, epochs):
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
                      nn.ReLU(),
                      nn.BatchNorm1d(64),
                      nn.Linear(64, 64),
                      nn.ReLU(),
                      nn.BatchNorm1d(64),
                      nn.Linear(64, 64),
                      nn.ReLU(),
                      nn.BatchNorm1d(64),
                      nn.Dropout(0.2),
                      nn.Linear(64, output_size),
                      nn.Softmax(dim=1)
        )

        self.batch_size = batch_size
        self.lr = learning_rate

        self.criterion = nn.CrossEntropyLoss()
        images, labels = next(iter(trainloader))
        images = images.view(images.shape[0], -1)

        self.logps = self.model(images)
        self.loss = self.criterion(self.logps, labels) 
        self.epochs = epochs
        self.loss.backward()

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        time0 = time()
        
        total = 0
        correct = 0

        self.train_acc = []
        self.train_loss = []

        for e in range(self.epochs):
            running_loss = 0
            for images, labels in trainloader:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)
            
                # Training pass
                optimizer.zero_grad()
                
                output = self.model(images)
                loss = self.criterion(output, labels)
                
                #This is where the model learns by backpropagating
                loss.backward()
                
                #And optimizes its weights here
                optimizer.step()
                
                running_loss += loss.item()


                _, predicted = output.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()


            else:
                self.train_loss.append(abs(running_loss/len(train_data)))
                self.train_acc.append(correct/total)
                print("Epoch {} - Training loss: {} - Accuracy {}".format(e, running_loss/len(train_data), correct/total))


        print("\nTraining Time (in minutes) =",(time()-time0)/60)

    def eval(self):

        count = 0
        maount = 0
        self.val_acc = []
        self.val_loss = []

        for e in range(self.epochs):
            running_loss = 0
            for images, labels in valloader:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)
                output = self.model(images)

                '''
                ps = torch.exp(output)
                probab = list(ps.detach().numpy()[0])
                pred_label = probab.index(max(probab))
                true_label = labels.numpy()[0]

                if true_label == pred_label:
                count += 1
                maount += 1
                '''

                loss = self.criterion(output, labels)
                _, predicted = output.max(1)
                count += labels.size(0)
                maount += predicted.eq(labels).sum().item()

                running_loss += loss.item()
            else:
                self.val_loss.append(abs(running_loss/len(test_data)))
                self.val_acc.append(maount/count)
                print("Epoch {} - Eval loss: {} - Accuracy - {}".format(e,running_loss/len(test_data), maount/count))
            
    def getAccuracyLoss_Train(self):
        return self.train_acc, self.train_loss

    def getAccuracyLoss_Val(self):
        return self.val_acc, self.val_loss