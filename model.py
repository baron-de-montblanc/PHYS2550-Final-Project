import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BatchNorm1d
from torch_geometric.nn import GATConv


# --------------- Define NN Models -----------------------


class SimpleFCNN(nn.Module):

    def __init__(self, num_features):
        super(SimpleFCNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)
        self.activ = nn.LeakyReLU()
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.activ(x)
        x = self.drop(x)

        x = self.out(x)
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
    



class GATRegression(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, out_features=1):
        super(GATRegression, self).__init__()
        # GAT layer
        self.gat1 = GATConv(num_features, hidden_dim, heads=1)  # Assuming 1 head for simplicity
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1)
        
        # Output linear layer for regression
        self.out = nn.Linear(hidden_dim, out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Pass through GAT layers
        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
        
        # Global mean pooling
        x = torch.mean(x, dim=0, keepdim=True)
        
        # Regression output
        x = self.out(x)
        return x







# ------------------------- Define training and testing loops ------------------


def run_model(model, model_type, device, data_loader, criterion, optimizer=None, train=False):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0

    with torch.set_grad_enabled(train):
        for batch in data_loader:
            if model_type == 'fcnn':
                data, target = batch
            elif model_type == 'gat':
                data, target = batch, batch.y
            else:
                raise ValueError("Unknown model type provided")

            data, target = data.to(device), target.to(device)

            if train:
                optimizer.zero_grad()

            output = model(data)
            loss = criterion(output.squeeze(), target.float())

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)

    return avg_loss


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




# ---------------------- Get predictions on the test set --------------------------------------



def get_predictions(test_set, model, device):
    """
    Obtain model predictions on the test set
    """

    label_list = []
    pred_list = []
    for data, label in test_set:
        data = data.to(device)
        label_list.append(label.item())
        pred = model(data)
        pred_list.append(pred.item())

    return label_list, pred_list  # x,y
