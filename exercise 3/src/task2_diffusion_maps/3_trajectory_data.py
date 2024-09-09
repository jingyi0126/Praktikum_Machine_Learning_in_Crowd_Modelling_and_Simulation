import numpy as np
import matplotlib.pyplot as plt
import utils
from pathlib import Path

"""Task 2.3: In this script, we apply the Diffusion Map algorithm to pedestrian trajectory data. 
We need functions defined in utils.py for this script.
"""

# Load trajectory data in data_DMAP_PCA_Vadere.txt
dataset = (
    Path(__file__)
    .resolve()
    .parents[2]
    .joinpath("data/data_DMAP_PCA_vadere.txt")
)
data = np.loadtxt(dataset)

# Visualize data-set
# Note: This is another visualization than in the report and portrays all 15 pedestrians
fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(8, 12))

for i in range(5):
    for j in range(3):
        axs[int(i / 2), i % 2].plot(
            data[:, 3 * i + 2 * j],
            data[:, 3 * i + 2 * j + 1],
            label=("Pedestrian " + str(3 * i + j + 1)),
        )
    axs[int(i / 2), i % 2].legend()

plt.title("Trajectories of the pedestrians")
fig.supxlabel("x")
fig.supylabel("y")
fig.delaxes(axs[2, 1])
plt.show()

# Compute first ten eigenfunctions (corresponding to 10 largest eigenvalues) of the
# Laplace Beltrami operator on the pedestrian data
_, vecs = utils.diffusion_map(data, 10)

# Plot the first non-constant eigenfunction ϕ1 against the other eigenfunctions
fig, axs = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(10, 10))

for i in [(x, y) for x in range(5) for y in range(2)]:
    if i == (0, 0):
        axs[i[0], i[1]].plot(vecs[:, 1], vecs[:, 0])
        axs[i[0], i[1]].set_ylabel("$\\phi_0$")
        continue

    axs[i[0], i[1]].plot(vecs[:, 1], vecs[:, 2 * i[0] + i[1] + 1])
    axs[i[0], i[1]].set_ylabel("$\\phi_{" + str(2 * i[0] + i[1] + 1) + "}$")

fig.supxlabel("$\\phi_1$")
plt.show()
