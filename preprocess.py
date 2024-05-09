import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


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
    train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
    train_data, _ = next(iter(train_loader))
    mean = train_data.mean(dim=0)
    std = train_data.std(dim=0)

    # Standardize the data
    for dataset in [train_set, val_set, test_set]:
        for i in range(len(dataset)):
            dataset[i][0][:] = (dataset[i][0] - mean) / std

    return train_set, val_set, test_set

