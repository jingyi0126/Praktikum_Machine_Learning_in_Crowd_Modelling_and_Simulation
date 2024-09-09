import numpy as np
from scipy.spatial.distance import cdist
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import lstsq
from numpy.typing import NDArray


def time_delay(data, col: int, delta_t: int, out_dim: int):
    """
    Create time delay coordinates from the data.
    :param col: The column/dimension no. of the data array to use.
    :param delta_t: No. of time-steps used for delayed measurement.
    :param out_dim: The target output dimension (2 or 3) for the output coordinates.
    :return: A numpy array of time-delay coordinates.

    Hint: 
    For out_dim = 2, 
        - shape of delayed coordinates = (np.shape(data)[0] - delta_t, 2)
        - Column 1: data with rows starcting from 0 to -delta_t, col = col
        - Column 2: data with rows starting from delta_t to the last row, col = col
        - Stack the 2 columns to form the delayed coordinates

    For out_dim = 3, 
        - Column 1: data with rows starting from 0 to -2*delta_t, col = col
        - Column 2: data with rows starting from delta_t to -delta_t, col = col
        - Column 3: data with rows starting from 2*delta_t till the end, col = col
        - Stack the 3 columns to form the delayed coordinates    
    """

    if out_dim == 2:
        output = np.vstack((data[:-delta_t, col], data[delta_t:, col])).T
    elif out_dim == 3:
        output = np.vstack(
            (data[:-2 * delta_t, col], data[delta_t: -delta_t, col], data[2 * delta_t:, col])).T
    else:
        raise ValueError("The parameter out_dim must be 2 or 3.")

    return output


def lorenz_ODE(state, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Defines the differential equations for the Lorenz system.

    Parameters:
        state : array
            Vector of the state variables: [x, y, z]
        sigma : float
            Parameter representing the Prandtl number
        rho : float
            Parameter representing the Rayleigh number
        beta : float
            Parameter that represents a physical property of the system

    Returns:
        list
            List containing the rates of change for [x, y, z]
    """
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]


def lorenz_traj(initial, t=np.linspace(0, 1000, 100000)):
    """
    Computes the trajectory of the Lorenz system from given initial conditions.

    Parameters:
        initial : array
            Initial conditions for [x, y, z]
        t : array
            Array of time points at which to solve the ODE

    Returns:
        array
            Array containing the computed trajectory of the Lorenz system
    """
       
    traj = odeint(lorenz_ODE, initial, t)

    return traj


def ivp(initial, t_span=(0, 1000), t_points=np.linspace(0, 1000, 100000)):
    """
    Solves the Lorenz system differential equations over a specified time interval.

    Parameters:
        initial (array): Initial conditions for the state variables [x, y, z].
        t_span (tuple): The time interval (start, end) for the integration.
        t_points (array): Points at which to store the computed solutions.

    Returns:
        OdeResult: An object containing the solution of the Lorenz system.
    """
    
    # Define the Lorenz system's differential equations as an inner function.
    def lorenz_ODE(t, y, sigma, rho, beta):
        """
        Represents the differential equations of the Lorenz system.

        Parameters:
            t (float): Current time point (required by solve_ivp, not used here).
            y (array): Current values of the state variables [x, y, z].
            sigma (float): Prandtl number, affects the problem's stability.
            rho (float): Rayleigh number, representing the buoyancy force.
            beta (float): A physical parameter, part of the standard Lorenz parameters.

        Returns:
            list: Rates of change [dx/dt, dy/dt, dz/dt].
        """
        x, y, z = y
        dxdt = sigma * (y - x)  # Change in x
        dydt = x * (rho - z) - y  # Change in y
        dzdt = x * y - beta * z  # Change in z
        return [dxdt, dydt, dzdt]

    # Parameters for the Lorenz system
    sigma = 10.0  # Prandtl number
    rho = 28.0    # Rayleigh number
    beta = 8.0 / 3.0  # Physical constant

    # Solve the Lorenz system differential equations using solve_ivp
    sol = solve_ivp(lorenz_ODE, t_span, initial, args=(sigma, rho, beta), t_eval=t_points)

    return sol


def create_delay_embedding(
        data: NDArray, delays: int, offset: int, areas: int, windows: int
) -> NDArray:
    """
    Create one delay embedding.

    Parameters:
        data: NDArray
            The parsed MI data.
        delays: int
            The number of delays.
        offset: int
            The time step (offset) of the first window.
        areas: int
            Specifies how many columns of the data should be included.
            The included columns will always be the first n.
        windows: int
            The number of windows.

    Returns:
        NDArray
            The matrix of this delay embedding. Shape (areas * (delays + 1), windows)
    """

    if offset + delays + windows >= data.shape[0] or areas >= data.shape[1] or windows < 1:
        raise ValueError("Illegal input arguments")

    embedding = np.zeros((areas * (delays + 1), ), dtype=int)
    for i in range(windows):
        window = data[offset + i:offset + delays + i + 1, 1:areas + 1].flatten("F")
        embedding = np.vstack((embedding, window))

    return embedding[1:]


def color_points(pca: NDArray, original: NDArray, offset: int):
    """
    Plot a coloring of the embedded point s for all measurement areas.

    Parameters:
        pca: NDArray
            The embedding space.
        original: NDArray
            The original MI data.
        offset: int
            The offset of the first time step in the embedding w.r.t. to the data.
    """

    fig, axs = plt.subplots(5, 2, figsize=(10, 20), subplot_kw=dict(projection="3d"))

    for i in range(9):
        axs[i // 2, i % 2].scatter(*pca.T, s=1, c=original[offset:offset + pca.shape[0], i + 1])
        axs[i // 2, i % 2].set_title("Measurement area " + str(i + 1))

    fig.delaxes(axs[4][1])
    fig.suptitle("Points in the embedding space colored by measurement areas")
    plt.show()


def calculate_arclenght_velocities(embedding: NDArray, period_ends: list[int]) -> list[float]:
    """
    Calculate the velocities on arclenghts of the embedding.

    Parameters:
        embedding: NDArray
            The embedding of the data after PCA
        period_ends: list[int]
            The end of each period

    Returns:
        list[float]
            The velocities
    """

    cummulative_arclenght = 0
    period_count = -1
    velocities = []

    for i in range(embedding.shape[0] - 1):
        if i in period_ends:
            cummulative_arclenght = 0
            period_count += 1

        cummulative_arclenght += np.linalg.norm(embedding[i+1] - embedding[i])
        dt = i - period_ends[period_count] + 1
        velocities.append(cummulative_arclenght / dt)

    return velocities


def plot_arclength_velocity(velocity: list[float], period_end: int):
    """
    Plot the velocity on the arclength against the time step and arclength (for the first period).

    Parameters:
        velocity: list[float]
            The velocities on the arclenghts of the points.
        period_end: int
            End of first period.
    """

    # Plot velocities against time steps
    _ = plt.figure(figsize=(12, 6))
    plt.plot(np.linspace(1, len(velocity), len(velocity)), velocity)
    plt.title("Change of arclength over time")
    plt.xlabel("Time step")
    plt.ylabel("Velocity on arclength")
    plt.show()

    # Plot velocities against arclength
    velocity_period = velocity[:period_end]

    _ = plt.figure(figsize=(12, 6))
    plt.plot(np.linspace(0, 2 * np.pi, len(velocity_period)), velocity_period)
    plt.title("Change of arclength over time in the first period")
    plt.xlabel("Arclength")
    plt.ylabel("Velocity on arclength")
    plt.xticks([0, 2 * np.pi], ["0", "$2 \\pi$"])
    plt.show()


def radial_basis_function(x, x_c, eps):
    """
    Compute radial basis function values as defined in the worksheet in section 1.2 (equation (7))
    Note: This code is copied from part 1.

    Args:
    - x: Data points
    - x_c: Centers of the basis functions
    - eps: Epsilon parameter(bandwidth)

    Returns:
    - Radial basis function values evaluated at data points x
    """

    distances = cdist(x, x_c, 'euclidean')
    rbf = np.exp(-(distances**2) / (eps**2))

    return rbf


def least_squares(A, b, cond=1e-5):
    """ Returns a Least squares solution of Ax = b
    Note: This code is copied from part 1.

    Args:
        A (npt.NDArray[np.float32]): Input array for the least squres problem
        b (npt.NDArray[np.float32]): Output array for the least squares problem
    Returns:
        npt.NDArray[np.float32]: Least squares solution of Ax = b
    """

    x, residuals, rank, s = lstsq(A, b, cond=cond)

    return x, residuals, rank, s


def pred_nonlinear(data, x_l, eps, cond=1e-5):
    """ Predict a non-linear function as a set of radial basis functions.
    Note: This code is copied from part 1.

    Args:
        data: The data set
        x_l: The centers of the rbfs
        eps: The parameter epsilon for the rbf calculation
        cond: Not used
    Returns:
        Radial basis functions and their coefficients
    """

    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    rbf_values = radial_basis_function(x, x_l, eps)
    coef, residuals, rank, s = least_squares(rbf_values, y)
    y_hat = rbf_values @ coef
    return y_hat.reshape(-1, 1), coef


def rbf_ivp(t, y, x_c, eps, coef):
    """A version of the radial basis function supposed to be called by solve_ivp"""
    y = y.reshape(1, -1)
    distances = cdist(y, x_c, 'euclidean')
    rbf = np.exp(-(distances ** 2) / (eps ** 2))
    return rbf @ coef


def pred_future(data: NDArray, embedding: NDArray, period_length: int, coef: NDArray):
    """
    Predict and plot the utilization of the MI building in the next 14 days

    Parameters:
        data: NDArray
            The measured utilization data
        embedding: NDArray
            The embedded data
        period_length: int
            The length of the first period
        coef: NDArray
            The least squares coefficients
    """

    # Calculate arclengths for 14 days
    arclengths = []
    for i in range(14):
        t_eval = np.linspace(0, period_length, period_length)
        sol = solve_ivp(
            rbf_ivp,
            [0, period_length],
            np.expand_dims(np.linalg.norm(embedding[1] - embedding[0]), 0),
            t_eval=t_eval,
            args=[np.linspace(0, 20000, 2000).reshape(-1, 1), 5, coef]
        )
        arclengths.append(sol.y[0, :])
    arclengths = np.array(arclengths).flatten()

    # Create mapping from arclength to data
    y_hat, coef = pred_nonlinear(
        np.stack((arclengths[:data.shape[0]], data[:, 1])).T,
        np.linspace(0, data.shape[0], 2000).reshape(-1, 1),
        5
    )
    y_hat = np.concatenate((y_hat, y_hat))

    # Plot the prediction
    _ = plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_hat)), y_hat, label="Prediction")
    plt.plot(data[:, 1], label="Measurement")
    plt.title("Predicted utilization of the MI building")
    plt.legend()
    plt.show()
