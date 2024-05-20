import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BatchNorm1d
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool


# --------------- Define NN Models -----------------------


class SimpleFCNN(nn.Module):

    def __init__(self, num_features):
        super(SimpleFCNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)
        self.activ = nn.LeakyReLU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.activ(x)
        x = self.drop(x)

        x = self.out(x)
        return x
    



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
    def __init__(self, num_features, hidden_dim, num_heads, out_features=1):
        super(GATRegression, self).__init__()

        # # GAT layers
        self.gat1 = GATConv(num_features, hidden_dim, heads=num_heads)
        self.gat2 = GATConv(hidden_dim*num_heads, hidden_dim, heads=num_heads)
  
        # Linear layers
        self.fc1 = Linear(hidden_dim*num_heads, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.out = Linear(64, out_features)

        # Other layers
        self.dropout = nn.Dropout(0.2)
        self.activ = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # GAT layers
        x = self.gat1(x, edge_index)
        x = self.activ(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        x = self.activ(x)
        x = self.dropout(x)

        # Global pooling
        x = global_add_pool(x, data.batch)

        # Linear layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activ(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activ(x)
        
        x = self.out(x)
        return x







# ------------------------- Define training and testing loops ------------------


def run_model(model, model_type, device, data_loader, criterion, optimizer=None, train=False):
    model.to(device)
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
            loss = criterion(output.squeeze(), target)

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



def get_predictions(test_set, model, device, model_type):
    """
    Obtain model predictions on the test set

    model_type is one of 'fcnn' or 'gat' (str)
    """

    label_list = []
    pred_list = []
    for batch in test_set:
        if model_type == 'fcnn' or model_type=="cnn":
            data, label = batch
        elif model_type == 'gat':
            data, label = batch, batch.y
        elif model_type == 'Simple1DCNN':
            data, label = batch
            data = data.unsqueeze(0).to(device)  # Add a batch dimension and move to device
        else: raise ValueError("Unknown model type")
 
        data = data.to(device)
        label_list.append(label.item())
        pred = model(data)
        pred_list.append(pred.item())

    return label_list, pred_list  # x,y

