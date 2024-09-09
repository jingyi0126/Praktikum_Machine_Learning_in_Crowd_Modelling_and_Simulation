import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
import utils

""" Task 2.2: In this script, we compute eigenfunctions of the Laplace Beltrami operator on the
“swiss roll” manifold.  We need functions defined in utils.py for this script.
Note: The latter half of this script requires the software datafold. If it is not installed, the
first half can still be executed and the second half will be ignored.
"""

# Generate swiss roll dataset
X, t = make_swiss_roll(5000)

# Visualize data-set
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection="3d")
ax.view_init(10, -70)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()

# Compute first ten eigenfunctions (corresponding to 10 largest eigenvalues) of the
# Laplace Beltrami operator on the “swiss roll” manifold
_, vecs = utils.diffusion_map(X, 10)

# Plot the first non-constant eigenfunction ϕ1 against the other eigenfunctions
fig, axs = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(10, 10))

for i in [(x, y) for x in range(5) for y in range(2)]:
    if i == (0, 0):
        axs[i[0], i[1]].scatter(vecs[:, 1], vecs[:, 0], c=t)
        axs[i[0], i[1]].set_ylabel("$\\phi_0$")
        continue

    axs[i[0], i[1]].scatter(vecs[:, 1], vecs[:, 2 * i[0] + i[1] + 1], c=t)
    axs[i[0], i[1]].set_ylabel("$\\phi_{" + str(2 * i[0] + i[1] + 1) + "}$")

fig.supxlabel("$\\phi_1$")
plt.show()


"""Bonus: In the following part, the eigenvectors of the "swiss roll" manifold are computed again,
using the datafold software. The code is adapted from one of the datafold tutorials. It can be found
at https://datafold-dev.gitlab.io/datafold/tutorial_03_dmap_scurve.html.
A datafold installation is required. Otherwise, the script will skip this part.
"""

try:
    import datafold.dynfold as dfold
    import datafold.pcfold as pfold
    from datafold.utils.plot import plot_pairwise_eigenvector

    # Optimize kernel parameters
    X_pcm = pfold.PCManifold(X)
    X_pcm.optimize_parameters()

    # Compute first ten eigenfunctions (corresponding to 10 largest eigenvalues) of the
    # Laplace Beltrami operator on the “swiss roll” manifold
    dmap = dfold.DiffusionMaps(
        kernel=pfold.GaussianKernel(
            epsilon=X_pcm.kernel.epsilon, distance=dict(cut_off=X_pcm.cut_off)
        ),
        n_eigenpairs=11,
    )
    dmap = dmap.fit(X_pcm)
    evecs = dmap.eigenvectors_

    # Plot the first non-constant eigenfunction ψ1 against the other eigenfunctions
    # Note: Datafold refers to the eigenfunctions with ψ instead of ϕ
    plot_pairwise_eigenvector(
        eigenvectors=evecs,
        n=1,
        fig_params=dict(figsize=[10, 10]),
        scatter_params=dict(cmap=plt.cm.Spectral, c=t),
    )
    plt.show()

except ModuleNotFoundError:
    print("Datafold not found. Skipping second half of script.")
