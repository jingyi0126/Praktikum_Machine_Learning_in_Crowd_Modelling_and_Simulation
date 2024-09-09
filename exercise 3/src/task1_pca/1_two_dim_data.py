import utils
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

""" Task 1.1: In this script, we apply principal component analysis to two-dimensional data. 
We need functions defined in utils.py for this script.
"""

# Load the dataset from the file pca_dataset.txt
dataset = Path(__file__).resolve().parents[2].joinpath("data/pca_dataset.txt")
data = np.loadtxt(dataset)

# Compute mean of the data
mean_data = np.mean(data, axis=0)

# Center data
center_data = utils.center_data(data=data)

# Compute SVD
U, S, V_t = utils.compute_svd(center_data)

# Plot principal components
first_component = V_t[0]
second_component = V_t[1]

ax = plt.axes()
plt.scatter(data[:, 0], data[:, 1])
plt.quiver(
    mean_data[0],
    mean_data[1],
    first_component[0],
    first_component[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="red",
)
plt.quiver(
    mean_data[0],
    mean_data[1],
    second_component[0],
    second_component[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="green",
)
plt.xlabel("X_0")
plt.ylabel("X_1")
plt.title("principal components directions")
plt.show()

# Analyze the energy captured by the first two principal components using utils.compute_energy()
print(utils.compute_energy(data, c=1))
