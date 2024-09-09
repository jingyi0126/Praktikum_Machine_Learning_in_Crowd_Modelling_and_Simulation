import scipy as sp
from scipy.linalg import eigh
import numpy as np

""" This script contains the implementation of the diffusion map algorithm. 
"""


def create_distance_matrix(X, max_distance=200):
    """Compute a sparse distance matrix using scipy.spatial.KDTree. Set max_distance as 200.
    (Step 1 of the algorithm mentioned in the worksheet.)

    Args:
        X (npt.NDArray[np.float]):
            Data matrix.
        max_distance (int, optional):
            Computes a distance matrix leaving as zero any distance greater than max_distance.
            Defaults to 200.

    Returns:
        npt.NDArray[np.float]:
            Distance Matrix. (output shape = (np.shape(D)[0], np.shape(D)[0]))
    """

    KDTree = sp.spatial.KDTree(X)
    D = KDTree.sparse_distance_matrix(KDTree, max_distance).toarray()
    return D


def set_epsilon(p, distance_matrix):
    """Set scalar epsilon as 'p' % of the diameter of the dataset.
    (Step 2 of the algorithm mentioned in the worksheet.)

    Args:
        p (np.float64):
            percentage.
        distance_matrix (npt.NDArray[np.float]):
            Distance matrix.

    Returns:
        np.float64:
            returns epsilon.
    """

    epsilon = p / 100 * np.max(distance_matrix)
    return epsilon


def create_kernel_matrix(D, eps):
    """Create the Kernel matrix.
    (Steps 3-5 of the algorithm mentioned in the worksheet.)

    Args:
        D (npt.NDArray[np.float]):
            Distance matrix.
        eps (np.float64):
            epsilon.

    Returns:
        npt.NDArray[np.float]:
            Kernel matrix. (output shape = (np.shape(D)[0], np.shape(D)[0]))
    """

    # Form the kernel matrix W (Step 3 of the algorithm from the worksheet)
    W = np.exp(-np.power(D, 2) / eps)

    # Form the diagonal normalization matrix P (Step 4 of the algorithm from the worksheet)
    P = np.diag(np.sum(W, axis=1))

    # Normalize W to form the kernel matrix K (Step 5 of the algorithm from the worksheet)
    P_inv = np.linalg.inv(P)
    K = P_inv @ W @ P_inv

    return K


def diffusion_map(X, n_eig_vals=5):
    """Implementation of the diffusion map algorithm.
        Please refer to the algorithm in the worksheet for the following.
        The step numbers in the following refer to the steps of the algorithm in the worksheet.

    Args:
        X (npt.NDArray[np.float]):
            Data matrix (each row represents one data point).
        n_eig_vals (int, optional):
            The number of eigenvalues and eigenvectors of the Laplace-Beltrami operator defined
            on the manifold close to the data to be computed. Default is 10.

    Returns:
        tuple(npt.NDArray[np.float], npt.NDArray[np.float]):
            eigenvalues, eigenvector of the Laplace-Beltrami operator.
            (output shapes: (n_eig_vals + 1, ), (np.shape(X)[0], n_eig_vals + 1))
    """

    # Compute distance matrix (Step 1 from the algorithm in the worksheet)
    D = create_distance_matrix(X)

    # Set epsilon to 5% of the diameter of the dataset (Step 2 from the algorithm in the worksheet).
    epsilon = set_epsilon(p=5, distance_matrix=D)

    # Form kernel matrix K (Steps 3-5 from the algorithm in the worksheet)
    K = create_kernel_matrix(D, epsilon)

    # Form the diagonal normalization matrix Q (Step 6 from the algorithm in the worksheet)
    Q = np.diag(np.sum(K, axis=1))

    # Form symmetric matrix T_hat (Step 7 from the algorithm in the worksheet)
    Q_inv_sqrt = np.sqrt(np.linalg.inv(Q))
    T_hat = Q_inv_sqrt @ K @ Q_inv_sqrt

    # Find the L + 1 largest eigenvalues and the corresponding eigenvectors of T_hat
    # (Step 8 from the algorithm in the worksheet)
    m = n_eig_vals + 1
    eigenvalues, eigenvectors = eigh(T_hat)
    indices = np.argsort(eigenvalues)[::-1]
    a_l = eigenvalues[indices[:m]]
    v_l = eigenvectors[:, indices[:m]]

    # Compute the eigenvalues of T_hat^(1/ε) in descending order
    # (Step 9 from the algorithm in the worksheet)
    lambda_l = np.power(a_l, 0.5 / epsilon)

    # Compute the eigenvectors of the matrix T (Step 10 from the algorithm in the worksheet)
    phi_l = Q_inv_sqrt @ v_l

    return lambda_l, phi_l
