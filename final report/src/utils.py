import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os
import torch
from scipy.spatial import distance


def load_data(scenario: str, file_name: str):
    '''
    Loads data from a specified file and returns it as a pandas DataFrame.

    Parameters:
    ----------
    scenario : str
        The scenario name which is used to construct the file path. 
        "Bottleneck_Data" or "Corridor_Data"
    file_name : str
        The name of the file (without extension) to be loaded.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the loaded data with columns: ['ID', 'FRAME', 'X', 'Y', 'Z'].
    '''
    file_path = f"../data/{scenario}/{file_name}.txt"
    df = pd.read_csv(file_path, delimiter=' ')
    df.columns = ['ID', 'FRAME', 'X', 'Y', 'Z']
    return df


def add_speed(data: pd.DataFrame):
    '''
    Calculate and add speed to each pedestrian's trajectory data based on their coordinates.

    Parameters:
    - data (pd.DataFrame): DataFrame containing trajectory data with columns ['ID', 'FRAME', 'X', 'Y', 'Z'].

    Returns:
    - pd.DataFrame: Updated DataFrame with added 'speed' column, representing speed in m/s.
    '''
    # if using Normalization, then ignore this line
    data = data.drop(columns="Z")

    data_groupedbyID = {pid: ped for (pid, ped) in data.groupby("ID")}

    for pedestrian in data_groupedbyID.values():
        coordinate_diff = pedestrian[['X', 'Y']].apply(
            lambda row: row.diff(), axis=0)
        distance_calculation = np.sqrt(
            coordinate_diff['X']**2 + coordinate_diff['Y']**2)
        # Let the first and second frame have the same speed
        distance_calculation.iloc[0] = distance_calculation.iloc[1]
        # The unit of the speed is m/s. The unit of distance_calculation is cm.
        #  0.16(m/cm/s) = 0.01(m/cm) / ((1/16)s)
        pedestrian["speed"] = distance_calculation * 0.16

    data_with_speed = pd.concat(
        [ped for ped in data_groupedbyID.values()], ignore_index=True)

    return data_with_speed


def generate_data(df_with_speed: pd.DataFrame, K: int = 10):
    '''
    Generate a dataset containing features and targets based on pedestrian trajectory data.

    Parameters:
    - df_with_speed (pd.DataFrame): DataFrame containing pedestrian trajectory data 
                                    with columns ['ID', 'FRAME', 'X', 'Y', 'speed'].
    - K (int, optional): Number of nearest neighbors to consider for feature extraction. 
                         Default is 10.

    Returns:
    - np.ndarray: Array containing the generated dataset with features and target values.
                  The shape of the array will be (n_samples, n_features), where n_features 
                  includes the mean spacing ('sk'), X-Y coordinate differences with 
                  K nearest neighbors (20 columns), and 'speed' as the target variable.
    '''
    df_groupbyF = {fra: ped for (fra, ped) in df_with_speed.groupby("FRAME")}

    frame_samples = []

    # find nearest neighbors
    for frame in df_groupbyF.values():

        if frame.shape[0] <= K:
            continue

        points = frame[['X', 'Y']].values
        neighbors = NearestNeighbors(
            n_neighbors=K+1, metric='euclidean').fit(points)
        k_distances, k_indices = neighbors.kneighbors(points)
        nearest_diffs = np.array(
            [points[k_indices[i]][1:] - points[i] for i in range(len(points))])
        flattened_diffs = nearest_diffs.reshape(nearest_diffs.shape[0], -1)
        sk = k_distances.sum(axis=1) / K  # mean spacing
        features = np.concatenate((sk.reshape(-1, 1), flattened_diffs), axis=1)
        frame_samples.append(np.concatenate(
            (features, frame[['speed']].values), axis=1))

    frame_samples = np.vstack(frame_samples)

    print("The shape of the dataset (containing features and targets) is ",
          frame_samples.shape, ".")
    # It has 22 columns.(sk, features(20 columns: The x,y coordinate differences with 10 neighbors), speed)
    return frame_samples


def weidmann_model(x, v0, l, T):
    '''
    Weidmann Model Function
    :param x: mean distance to k nearest neighbors
    :param v0: desired speed
    :param l: pedestrian size
    :param T: time gap
    '''
    return v0 * (1 - np.exp((l - x) / v0 / T))


def train(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs):  # , early_stopping
    '''
        """
    Train a neural network model using the provided data loaders and parameters.

    Parameters:
    - model (torch.nn.Module): The neural network model to be trained.
    - train_loader (torch.utils.data.DataLoader): DataLoader providing training data.
    - test_loader (torch.utils.data.DataLoader): DataLoader providing validation data.
    - criterion (torch.nn.Module): Loss function used for training.
    - optimizer (torch.optim.Optimizer): Optimizer algorithm used to update model parameters.
    - scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler to adjust learning rate during training.
    - num_epochs (int): Number of training epochs.

    Returns:
    - train_mse_losses (list): List containing Mean Squared Error (MSE) losses for each epoch during training.
    - val_mse_losses (list): List containing MSE losses for each epoch during validation.
    '''
    train_mse_losses = []
    val_mse_losses = []

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()

        scheduler.step()  # Step the learning rate scheduler

        train_mse_losses.append(loss.item())
        print(
            f'Epoch [{epoch+1}/{num_epochs}], MSE (train): {loss.item():.5g}')

        # Validation step
        model.eval()
        with torch.no_grad():
            val_total_loss = 0.0
            val_total_samples = 0
            for inputs, labels in test_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, labels.view(-1, 1))
                val_total_loss += val_loss.item() * labels.size(0)
                val_total_samples += labels.size(0)
        val_mse_loss = val_total_loss / val_total_samples
        val_mse_losses.append(val_mse_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], MSE (val): {val_mse_loss:.5g}')

    return train_mse_losses, val_mse_losses
