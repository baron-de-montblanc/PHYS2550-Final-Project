import numpy as np
import torch
import torch.nn as nn


# --------------- Define NN Models -----------------------


class SimpleFCNN(nn.Module):

    def __init__(self, num_features, dropout_rate=0.4):
        super(SimpleFCNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 1)
        self.activ = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.activ(x)
        x = self.drop2(x)

        x = self.fc3(x)
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