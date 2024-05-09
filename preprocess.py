import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.nn import knn_graph
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader as BasicDataLoader
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as GeometricDataLoader

# ------------------------ Pre-processing data ------------------------------


def load(csv, mode, zmax=None):
    """
    Given the CSV file path, load in the data, labels, and feature names as
    numpy arrays given the specified mode

    mode: one of "all", "mag", "err"
    zmax: maximum (true) redshift value to include; defaults to None to return all data
    """

    df = pd.read_csv(csv)

    labels = df["spec_z"].values  # grab the labels...
    df = df.drop("spec_z", axis=1)  # ... and remove the labels column

    if zmax:
        mask = labels < zmax
        labels = labels[mask]
        df = df[mask]

    # Hard-coded column name mapping:
    if mode == "mag":
        req_cols = ['res',
                    'u_cmodel_mag', 
                    'g_cmodel_mag', 
                    'r_cmodel_mag', 
                    'i_cmodel_mag', 
                    'z_cmodel_mag', 
                    'Y_cmodel_mag']
        df = df[req_cols]

    elif mode == "err":
        req_cols = ['u_cmodel_magerr', 
                    'g_cmodel_magerr', 
                    'r_cmodel_magerr', 
                    'i_cmodel_magerr', 
                    'z_cmodel_magerr', 
                    'Y_cmodel_magerr']
        df = df[req_cols]

    else: 
        if mode != "all": raise ValueError("mode must be one of 'all', 'mag', 'err'")

    # **if** there are still NaN's, replace with zero
    # df = df.fillna(0)

    data = df.values
    return data, labels, np.array(df.columns)



def preprocess_split(data, labels, train_split, val_split):
    """
    Given loaded data and labels (numpy arrays), 
    apply some preprocessing steps, cast them to TensorDataset,
    and split into training, val, and test

    Returns: train_dataset, val_dataset, test_dataset (Tensor Subset objects)
    """
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    dataset = TensorDataset(data, labels)

    total_size = len(dataset)
    train_size = int(train_split * total_size)
    valid_size = int(val_split * total_size)
    test_size = total_size - train_size - valid_size  # ensure all data is used

    # Split the dataset
    train_set, val_set, test_set = random_split(dataset, [train_size, valid_size, test_size])

    # Calculate mean and std only from training data
    train_loader = BasicDataLoader(train_set, batch_size=len(train_set), shuffle=False)
    train_data, _ = next(iter(train_loader))
    mean = train_data.mean(dim=0)
    std = train_data.std(dim=0)

    # Standardize the data
    for dataset in [train_set, val_set, test_set]:
        for i in range(len(dataset)):
            dataset[i][0][:] = (dataset[i][0] - mean) / std

    return train_set, val_set, test_set



def prepare_graphs(data, labels, k, device, batch_size=64):
    """
    Given our input data and labels, construct graphs with
    the KNN algorithm

    input data has shape [input_size, num_features]
    labels has shape [input_size,]
    k is an integer corresponding to the desired number of neighbors
    """

    # Convert data to PyTorch tensors and move to the appropriate device
    data_tensor = torch.tensor(data, dtype=torch.float).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

    # Standardize the data (mean=0, std=1)
    mean = data_tensor.mean(dim=0, keepdim=True)
    std = data_tensor.std(dim=0, keepdim=True)
    data_tensor = (data_tensor - mean) / std

    # # get rid of NaNs and infs --> cast to zeros
    # data_tensor = torch.nan_to_num(data_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    # Create a dataset and dataloader for batch processing
    dataset = TensorDataset(data_tensor, labels_tensor)
    loader = GeometricDataLoader(dataset, batch_size=batch_size, shuffle=False)

    graphs = []
    for data, label in tqdm(loader):
        # Construct graphs for each sample in the batch
        for i in range(data.size(0)):
            edge_index = knn_graph(data[i], k=k, loop=False)
            graph = Data(x=data[i], edge_index=edge_index, y=label[i].unsqueeze(0))
            graphs.append(graph.to("cpu"))  # for memory management, don't store these on GPU

    return graphs



def split_graphs(graph_list, training_split, val_split):
    """
    Given a GraphDataset object, split it into training, validation, and test sets
    """

    # Calculate sizes for each split
    total_count = len(graph_list)
    train_count = int(training_split * total_count)
    valid_count = int(val_split * total_count)
    test_count = total_count - train_count - valid_count

    # Perform the split using random_split
    train_dataset, valid_dataset, test_dataset = random_split(graph_list, [train_count, valid_count, test_count])

    return train_dataset, valid_dataset, test_dataset




# ---------------- random helper functions --------------------------


def time_elapsed(t0,t):
    """
    print time elapsed between t0 and t, where t0 and t are time.time() instances
    """
    delta_t = t-t0
    
    t_mins = round(delta_t/60, 2)
    
    return t_mins
