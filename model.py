import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BatchNorm1d


# --------------- Define NN Models -----------------------


class SimpleFCNN(nn.Module):

    def __init__(self, num_features, dropout_rate=0.5):
        super(SimpleFCNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.activ = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.activ(x)
        x = self.drop(x)

        x = self.fc3(x)
        x = self.activ(x)
        x = self.drop(x)

        x = self.fc4(x)
        x = self.activ(x)
        x = self.drop(x)

        x = self.fc5(x)
        return x
    
# Modify the model architecture to include dropout and batch normalization
class ModifiedSimple1DCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ModifiedSimple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)  # Adding dropout with a probability of 0.5
        self.fc1 = nn.Linear(32 * (num_features // 4), 64)  # Adjusting the input size to the fully connected layer
        self.fc2 = nn.Linear(64, num_classes)
        self.bn = nn.BatchNorm1d(32)  # Adding batch normalization

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)  # Applying dropout
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)  # Applying dropout
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # No need to squeeze the output

'''
class Simple1DCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Simple1DCNN, self).__init__()  # initialize
        # super() is a built-in Python function used to call methods of a superclass (or parent class) in a subclass (or child class).
        # conv1 one convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()  # using ReLU function
        # pool max-pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # fc1 fully connected layer
        self.fc1 = nn.Linear(16 * (num_features // 2), num_classes)

    def forward(self, x):   # forward pass of the model
        x = x.unsqueeze(1)  # Add a dimension for the 'in_channels' expected by Conv1d
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
        #return x.squeeze(1)  # Squeeze the output to match the target size
'''


class Simple1DCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Simple1DCNN, self).__init__()  
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * (num_features // 2), num_classes)

    def forward(self, x):   
        x = x.unsqueeze(1)  
        x = self.conv1(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x




# ------------------------- Define training and testing loops ------------------


def train_one_epoch(model, device, train_loader, optimizer, criterion, acc_metric):
    model.train()

    total_loss = 0
    accuracy_metric = acc_metric.to(device)
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.squeeze(1)  # flatten the output
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        accuracy_metric(output, target)

    avg_loss = total_loss/len(train_loader)
    avg_acc = accuracy_metric.compute()
    return avg_loss, avg_acc



def test(model, device, test_loader, criterion, acc_metric):
    model.eval()

    total_loss = 0
    accuracy_metric = acc_metric.to(device)
    with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)

          output = model(data)
          output = output.squeeze(1)  # flatten the output
          loss = criterion(output, target)

          total_loss += loss.item()
          accuracy_metric(output, target)

    avg_loss = total_loss/len(test_loader)
    avg_acc = accuracy_metric.compute()
    return avg_loss, avg_acc


# train and test loops for 1D_CNN
def train1D(model, device, train_loader, optimizer, criterion, acc_metric):
    model.train()

    total_loss = 0
    total_accuracy = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output.squeeze(), target.float())  # Squeeze the output to remove the extra dimension
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy
        accuracy = acc_metric(output.squeeze(), target.int())  # Squeeze the output here as well
        total_accuracy += accuracy.item()
                
        # Print predictions and targets
        #print("Predictions:", output.squeeze().detach().cpu().numpy())
        #print("Targets:", target.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)
    return avg_loss, avg_accuracy


def test1D(model, device, test_loader, criterion, acc_metric):
    model.eval()

    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output.squeeze(), target.float())  # Squeeze the output to remove the extra dimension

            total_loss += loss.item()

            # Compute accuracy
            accuracy = acc_metric(output.squeeze(), target.int())  # Squeeze the output here as well
            total_accuracy += accuracy.item()

    avg_loss = total_loss / len(test_loader)
    avg_accuracy = total_accuracy / len(test_loader)
    return avg_loss, avg_accuracy
