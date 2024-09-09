import numpy as np
import matplotlib.pyplot as plt
import utils

""" Task 2.1: In this script, we demonstrate the similarity of Diffusion Maps and Fourier analysis
using a periodic dataset. We need functions defined in utils.py for this script.
"""

# Create a periodic dataset with the details described in the task-sheet
N = 1000
k = np.linspace(1, N, N)
tk = 2 * np.pi * k / (N + 1)
X = np.array((np.cos(tk), np.sin(tk)))
X = X.transpose()

# Visualize data-set
plt.figure(figsize=(4, 4))
plt.plot(X[:, 0], X[:, 1])
plt.show()

# Compute 5 eigenfunctions associated to the largest eigenvalues
_, vecs = utils.diffusion_map(X, 5)

# Plot eigenfunctions
plt.figure(figsize=(6, 6))
plt.plot(tk, vecs)
plt.xlabel("$t_k$")
plt.legend(["$\\phi_" + str(i) + "$" for i in range(6)], loc="upper left")
plt.show()
