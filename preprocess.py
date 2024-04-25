import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split


# ------------------------ Pre-processing data ------------------------------


def load(csv, mode):
    """
    Given the CSV file path, load in the data, labels, and feature names as
    numpy arrays given the specified mode

    mode: one of "all", "mag", "err"
    """

    df = pd.read_csv(csv)

    labels = df["spec_z"].values  # grab the labels...
    df = df.drop("spec_z", axis=1)  # ... and remove the labels column

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
    df = df.fillna(0)

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

    # TODO: Apply some extra preprocessing steps (normalization and anything else)

    dataset = TensorDataset(data, labels)

    total_size = len(dataset)
    train_size = int(train_split * total_size)
    valid_size = int(val_split * total_size)
    test_size = total_size - train_size - valid_size  # ensure all data is used

    # Split the dataset
    return random_split(dataset, [train_size, valid_size, test_size])

