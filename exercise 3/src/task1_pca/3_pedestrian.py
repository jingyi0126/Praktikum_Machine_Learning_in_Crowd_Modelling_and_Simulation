import numpy as np
import utils
from pathlib import Path

""" Task 1.3: In this script, we apply principal component analysis to pedestrian trajectory data. 
We need functions defined in utils.py for this script.
"""

# Load trajectory data in data_DMAP_PCA_Vadere.txt. (Hint: You may need to use a space as delimiter)
dataset = (
    Path(__file__)
    .resolve()
    .parents[2]
    .joinpath("data/data_DMAP_PCA_vadere.txt")
)
data = np.loadtxt(dataset)

# Center the data by subtracting the mean
center_data = utils.center_data(data)

# Extract positions of pedestrians 1 and 2
pedestrian_1 = data[:, 0:2]
pedestrian_2 = data[:, 2:4]

# Visualize trajectories of first two pedestrians (Hint: You can optionally use utils.visualize_traj_two_pedestrians() )
utils.visualize_traj_two_pedestrians(
    pedestrian_1,
    pedestrian_2,
    ("trajectories of the first 2 pedestrians", "x", "y"),
)

# Compute SVD of the data using utils.compute_svd()
U, S, V_t = utils.compute_svd(center_data)

# Reconstruct data by truncating SVD using utils.reconstruct_data_using_truncated_svd()
reconstruct_data = utils.reconstruct_data_using_truncated_svd(
    U, S, V_t, n_components=2
)

# Visualize trajectories of the first two pedestrians in the 2D space defined by the first two principal components
re_pedestrian_1 = reconstruct_data[:, 0:2]
re_pedestrian_2 = reconstruct_data[:, 2:4]
utils.visualize_traj_two_pedestrians(
    re_pedestrian_1,
    re_pedestrian_2,
    ("trajectories of the first 2 pedestrians", "x", "y"),
)

# Answer the questions in the worksheet with the help of utils.compute_cumulative_energy(),
# utils.compute_num_components_capturing_threshold_energy()
percent = utils.compute_cumulative_energy(S, 2)

utils.compute_num_components_capturing_threshold_energy(S, 0.9)
