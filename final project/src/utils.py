import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os

from scipy.spatial import distance




def load_data(scenario: str, file_name: str):
    

    file_path = f"../data/{scenario}/{file_name}.txt"
    df = pd.read_csv(file_path, delimiter = ' ')
    df.columns = ['ID', 'FRAME', 'X', 'Y', 'Z']
    
    return df




'''
Preliminary data preprocessing, not sure if it is sufficient or necessary
def Normalization_bottleneck(data):

    # Delete z position from data
    data = data.drop(columns='Z')
    # Convert x y positions to meters
    data[:, 2:] = data[:, 2:] / 100
    
    data = data[data[:, 2] >= 0.0]
    data = data[data[:, 2] <= 1.8]
    
    data = data[data[:, 3] >= 0.0]
    data = data[data[:, 3] <= 8.0]
    return data


def Normalization_corridor(data):

    # Delete z position from data
    data = data.drop(columns='Z')
    # Convert x y positions to meters
    data[:, 2:] = data[:, 2:] / 100
    #
    data = data[data[:, 2] >= 0.0]
    data = data[data[:, 2] <= 1.8]
    
    data = data[data[:, 3] >= 0.0]
    data = data[data[:, 3] <= 6.0]
    return data
'''

def add_speed(data: pd.DataFrame):

    data = data.drop(columns="Z") # if ues Normalization, then ignore this line
    
    data_groupedbyID = {pid: ped for (pid, ped) in data.groupby("ID")}

    for pedestrian in data_groupedbyID.values():
        coordinate_diff = pedestrian[['X', 'Y']].apply(lambda row: row.diff(), axis=0)
        distance_calculation = np.sqrt(coordinate_diff['X']**2 + coordinate_diff['Y']**2)
        distance_calculation.iloc[0] = distance_calculation.iloc[1]  # Let the first and second frame have the same speed
        pedestrian["speed"] = distance_calculation * 0.16  # The unit is m/s

    data_with_speed = pd.concat([ped for ped in data_groupedbyID.values()], ignore_index=True)
    
    return data_with_speed

def generate_data(df_with_speed: pd.DataFrame, K: int =10):
    df_groupbyF = {fra: ped for (fra, ped) in df_with_speed.groupby("FRAME")}

    frame_samples = []

    # find nearest neighbors
    for frame in df_groupbyF.values():

        if frame.shape[0] <= K:
            continue

        points = frame[['X', 'Y']].values
        neighbors = NearestNeighbors(n_neighbors=K+1, metric='euclidean').fit(points)
        k_distances, k_indices = neighbors.kneighbors(points)
        nearest_diffs = np.array([points[k_indices[i]][1:] - points[i] for i in range(len(points))])
        flattened_diffs = nearest_diffs.reshape(nearest_diffs.shape[0], -1)
        sk = k_distances.sum(axis=1) / K  # mean spacing
        features = np.concatenate((sk.reshape(-1, 1), flattened_diffs), axis=1)
        frame_samples.append(np.concatenate((features, frame[['speed']].values), axis=1))


    frame_samples = np.vstack(frame_samples)

   

    print("The shape of the dataset (containing features and targets) is ", frame_samples.shape, ".")
    # it has 22 columns.(sk,features(20 columns: The x,y coordinate differences with 10 neighbors),speed)
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
