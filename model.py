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
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        # self.fc4 = nn.Linear(64, 1)
        self.activ = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.activ(x)
        x = self.drop(x)

        # x = self.fc3(x)
        # x = self.activ(x)
        # x = self.drop(x)

        x = self.fc3(x)
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
        accuracy_metric(output, target.int())

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
          accuracy_metric(output, target.int())

    avg_loss = total_loss/len(test_loader)
    avg_acc = accuracy_metric.compute()
    return avg_loss, avg_acc


# training loop for 1D_CNN
def train1D(model, criterion, optimizer, train_loader, device):
    # Set the model to training mode
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    # Iterate over the training dataset
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()   # .backward() compute gradients of the loss function
        # Update the parameters
        optimizer.step()  # .step() performs a single optimization step and updates the parameters according to the update rule defined by the optimizer.

        # Compute training loss and accuracy
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = correct / total
    return train_loss, train_accuracy


# testing loop for 1D_CNN
def validate1D(model, criterion, val_loader, device):
    # Set the model to evaluation mode
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predicted = []

    # Disable gradient computation for efficiency during validation
    with torch.no_grad():
        # Iterate over the validation dataset
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store targets and predicted values for later analysis
            all_targets.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

    val_loss = val_loss / len(val_loader.dataset)   # Calculate average validation loss
    val_accuracy = correct / total    # Calculate validation accuracy
    return val_loss, val_accuracy, all_targets, all_predicted
