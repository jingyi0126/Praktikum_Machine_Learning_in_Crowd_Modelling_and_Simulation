import numpy as np
import matplotlib.pyplot as plt
import scipy.datasets
import scipy.linalg
from skimage.transform import resize
import numpy.typing as npt

""" This script contains all the utility functions for the exercise on principal component analysis. 
Functions defined in this script are to be used in the respective examples.
"""


#################################################
# Utility functions for 1st exercise
#################################################


def center_data(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Center data by subtracting the mean of the data.

    Args:
        data (npt.NDArray[np.float64]):
            Data matrix.

    Returns:
        npt.NDArray[np.float64]:
            centered data.
    """

    mean_data = np.mean(data, axis=0)
    data_center = data - mean_data

    return data_center


def compute_svd(
    data: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """Compute (reduced) SVD of the data matrix. Set (full_matrices=False).

    Args:
        data (npt.NDArray[np.float]):
            data matrix.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            U, S, V_t.
    """

    U, S, V_t = scipy.linalg.svd(data, full_matrices=False)
    return U, S, V_t


def compute_energy(S: npt.NDArray[np.float64], c: int = 1) -> np.float64:
    """
    Percentage of total “energy” (explained variance) of (only) the i-th principal component of
    singular value on the diagonal of the matrix S.
        Note that it is NOT a sum of first 'c' components!

    Args:
        S (npt.NDArray[np.float64]):
            Array containing the singular values of the data matrix
        c (int):
            Component of SVD (Starts from 1, NOT 0). E.g. set c = 1 for first component. Defaults to 1.

    Returns:
        np.float64:
            percentage energy in the c-th principal component
    """

    if c >= len(S):
        print("exceeds the existing components")
    sigma2 = S[c - 1] ** 2
    energy = sigma2 / np.sum(S**2) * 100

    return np.float64(energy)


def compute_cumulative_energy(
    S: npt.NDArray[np.float64], c: int = 1
) -> np.float64:
    """
    Percentage of total “energy” (explained variance) of the sum of 'c' principal component
    of singular value on the diagonal of the matrix S.

    Args:
        S (npt.NDArray[np.float64]):
            Array containing the singular values of the data matrix
        c (int):
            Component of SVD (Starts from 1, NOT 0). E.g. set c = 1 for first component. Defaults to 1.

    Returns:
        np.float64:
            percentage energy in the first c principal components
    """
    energy = S**2
    sum_energy = np.sum(energy)
    cum_energy = np.cumsum(energy)

    return cum_energy[c - 1] / sum_energy * 100


#################################################
# Utility functions for 2nd exercise
#################################################


def load_resize_image() -> npt.NDArray[np.float64]:
    """Load data and RESIZE! the image to appropriate dimensions mentioned in the task description

    Returns:
        npt.NDArray[np.float64]:
            Return the image array
    """

    image = scipy.misc.face(gray=True)
    resized_image = resize(image, (185, 249), anti_aliasing=True)
    img_array = np.asarray(resized_image)

    return img_array


def reconstruct_data_using_truncated_svd(
    U: npt.NDArray[np.float64],
    S: npt.NDArray[np.float64],
    V_t: npt.NDArray[np.float64],
    n_components: int,
):
    """This function takes in the SVD of the data matrix and reconstructs the data matrix by retaining
    only 'n_components' SVD components. In other words, it computes a low-rank approximation with
    (rank = n_components) of the data matrix.

    Args:
        U (npt.NDArray[np.float64]):
            Matrix whose columns contain left singular vectors
        S (npt.NDArray[np.float64]):
            Matrix with singular values
        V_t (npt.NDArray[np.float64]):
            Matrix whose rows contain right singular vectors
        n_components (int):
            no. of principal components retained in the low-rank approximation

    Returns:
        npt.NDArray[np.float64]:
            Reconstructed matrix using first 'n_components' principal components.
    """

    U_k = U[:, :n_components]
    S_k = np.diag(S[:n_components])
    VT_k = V_t[:n_components, :]
    reconstructed_images = np.dot(U_k, np.dot(S_k, VT_k))

    return reconstructed_images


def reconstruct_images(
    U: npt.NDArray[np.float64],
    S: npt.NDArray[np.float64],
    V_t: npt.NDArray[np.float64],
) -> None:
    """Construct plots with different number of principal components

    Args:
        U (npt.NDArray[np.float64]):
            Matrix whose columns contain left singular vectors
        S (npt.NDArray[np.float64]):
            Matrix with singular values
        V_t (npt.NDArray[np.float64]):
            Matrix whose rows contain right singular vectors
    """

    # Create images with different numbers of principal components
    n_c = [len(S), 120, 50, 10]
    images = []
    for i, num_c in enumerate(n_c):
        image_r = reconstruct_data_using_truncated_svd(U, S, V_t, num_c)
        images.append(image_r)

    # Plot the images
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(images[0], cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(images[1], cmap="gray")
    plt.title("Reconstructed image with 120 components")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(images[2], cmap="gray")
    plt.title("Reconstructed image with 50 components")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(images[3], cmap="gray")
    plt.title("Reconstructed image with 10 components")
    plt.axis("off")
    plt.show()


def compute_num_components_capturing_threshold_energy(
    S: npt.NDArray[np.float64], energy_threshold: float = 0.99
) -> int:
    """Matrix containing the singular values of the data matrix

    Args:
        S (npt.NDArray[np.float64]):
            Singular values
        energy_threshold (float):
            Energy threshold. Defaults to 0.99.

    Returns:
        int:
            No. of principal components where energy loss is smaller than the energy threshold
    """

    # Compute total “energy” (explained variance) contained in the sum of first 'c' principal components
    # Note that it is NOT the energy in (only) c-th component!
    total_energy = np.sum(S**2)

    # Find the number of components where energy loss is smaller than the energy threshold
    cumulative_energy = np.cumsum(S**2)
    percent = cumulative_energy / total_energy
    num_components = np.where(percent >= energy_threshold)[0][0] + 1

    # Plot the results
    plt.plot(range(len(S)), percent)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Energy Loss")
    plt.scatter([num_components], [percent[num_components]], color="red")
    plt.annotate(
        f"({num_components}, {percent[num_components]:.2f})",
        xy=(num_components, percent[num_components]),
        xytext=(num_components, percent[num_components] - 0.05),
        arrowprops=dict(facecolor="red", shrink=0.05, width=0.5, headwidth=4),
    )
    plt.show()

    return num_components


#################################################
# Utility functions for 3rd exercise
#################################################


def visualize_traj_two_pedestrians(
    p1: npt.NDArray[np.float64],
    p2: npt.NDArray[np.float64],
    title_axes_labels: tuple[str, str, str],
) -> None:
    """This function can be used to plot the trajectories of two pedestrians.

    Args:
        p1 (npt.NDArray[np.float64]):
            data of the first pedestrian
        p2 (npt.NDArray[np.float64]):
            data of the second pedestrian
        title_axes_labels (tuple [str, str, str]):
            Title of the plot, x-label and y-label
    """

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(p1[:, 0], p1[:, 1], label="Pedestrian 1")
    plt.plot(p2[:, 0], p2[:, 1], label="Pedestrian 2")
    plt.title(title_axes_labels[0])
    plt.xlabel(title_axes_labels[1])
    plt.ylabel(title_axes_labels[2])
    plt.legend()
    plt.show()
