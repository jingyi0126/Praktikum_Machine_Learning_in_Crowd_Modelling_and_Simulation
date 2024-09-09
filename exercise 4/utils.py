import math
from typing import List, Tuple, Type
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from dynamical_system import *


# Task 1
def plot_phase_portrait_task_1(
    fig: matplotlib.figure.Figure,
    ax: plt.Axes,
    X: ArrayLike,
    DX: ArrayLike,
    classification: str,
    A: ArrayLike,
) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    """Plot the phase portrait for the dynamical system of task 1

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        A matplotlib figure instance
    ax: plt.Axes
        A matplotlib axes instance
    X: ArrayLike
        Evenly spaced grid points
    DX: ArrayLike
        Velocities of X
    classification: str
        The topological classification according to Kuznetsov's book
    A: ArrayLike
        The parametrized matrix of this system

    Returns
    -------
    fig: matplotlib.figure.Figure
        A matplotlib figure instance
    ax: plt.Axes
        A matplotlib axes instance
    """

    # Retrieve the parameters from the matrix
    alpha = A[0][0]
    beta = A[0][1]

    # Create the streamplot
    ax.streamplot(X[0], X[1], DX[0], DX[1])
    ax.set_aspect("auto")
    ax.set_xlim(np.min(X[0]), np.max(X[0]))
    ax.set_ylim(np.min(X[1]), np.max(X[1]))

    # Name the figure and axes
    ax.set_title("Phase portrait of the " + classification)
    ax.set_xlabel(
        """$x_1$
        $\\alpha = $"""
        + str(np.around(alpha, 3))
        + ", $\\beta = $"
        + str(np.around(beta, 3))
        + ", $\\lambda_{1/2} = $"
        + (str(np.around(alpha, 3)) if alpha != 0 else "")
        + "$\\pm$"
        + str(np.around(np.emath.sqrt(beta * np.abs(beta)), 3))
    )
    ax.set_ylabel("$x_2$")

    return fig, ax


# Task 2
def plot_bifurcation_diagram_task_2(
    fig: matplotlib.figure.Figure,
    ax: plt.Axes,
    X: ArrayLike,
    gamma: int,
    title: str,
    line_color_stable: str = "blue",
    line_color_unstable: str = "red",
) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    """Plot the bifurcation diagram

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        A matplotlib figure instance
    ax: plt.Axes
        A matplotlib axes instance
    X: ArrayLike
        The steady states of the system
    gamma: int
        The parameter gamma of the system.
        It is used to define the range of alpha values.
    title: str
        The title of the diagram
    line_color_stable: str, optional
        The color of the stable line. Default is 'blue'.
    line_color_unstable: str, optional
        The color of the unstable line. Default is 'red'.

    Returns
    -------
    fig: matplotlib.figure.Figure
        A matplotlib figure instance
    ax: plt.Axes
        A matplotlib axes instance
    """

    # Retrieve the values for alpha
    alpha = np.linspace(-gamma, 6 - gamma, 400)

    # Plot the bifurcation diagram
    ax.plot(alpha, X, ",", color=line_color_stable, label="Stable")
    ax.plot(
        alpha, -X, ",", color=line_color_unstable, linestyle="dashed", label="Unstable"
    )

    # Name the figure and axes
    ax.set_title("equation:" + title)
    ax.set_xlabel("alpha")
    ax.set_ylabel("$x$")
    ax.grid(True)

    return fig, ax


# Task 3
def Andronov_Hopf(x1, x2, alpha):
    """Calculate the derivatives of the Andronov-Hopf system"""
    dx1 = alpha * x1 - x2 - x1 * (x1**2 + x2**2)
    dx2 = x1 + alpha * x2 - x2 * (x1**2 + x2**2)
    return dx1, dx2


def plot_phase_portrait_task_3(
    x_min: float,
    x_max: float,
    n_x: int,
    y_min: float,
    y_max: float,
    n_y: int,
    pars: ArrayLike,
):
    """Plot the phase portrait of the Andronov-Hopf system

    Parameters
    ----------
    x_min: float
        The minimal x-value
    x_max: float
        The maximal x-value
    n_x: int
        The number of samples in the range [x_min, x_max]
    y_min: float
        The minimal y-value
    y_max: float
        The maximal y-value
    n_y: int
        The number of samples in the range [y_min, y_max]
    pars: ArrayLike
        Values of alpha. One phase portrait will be created for each.
    """

    # Create the coordinate system
    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(y_min, y_max, n_y)
    X, Y = np.meshgrid(x, y)

    # Create a streamplot for each value of alpha
    for idx in range(len(pars)):
        fig, ax = plt.subplots()
        U, V = Andronov_Hopf(X, Y, pars[idx])
        ax.streamplot(X, Y, U, V, density=1.3, arrowsize=1.4)
        ax.set_title(f"Phase Portrait for $\\alpha = ${pars[idx]}")
        ax.set_aspect(1)
        plt.tight_layout()
        plt.show()


def plot_orbits(
    system_class: Type[DynamicalSystem],
    dict_par_system_class: dict,
    line_color: str | None = None,
):
    """Plot the orbit of a dynamical system

    Parameters
    ----------
    system_class: Type[DynamicalSystem]
        The class of the dynamical system
    dict_par_system_class: dict
        A dictionary containing all parameters
    line_color: str | None = None
        The line color used in the plot. Default: None,
        which leads to the usage of matplotlib's default color
    """

    # Extract the parameters from the dictionary
    pars = dict_par_system_class["pars"]
    init_states = np.array(dict_par_system_class["init_states"])
    list_par_x = dict_par_system_class["list_par_x"]
    t_eval = dict_par_system_class["t_eval"]
    discrete = dict_par_system_class["discrete"]

    # Solve the system for each value of the parameter
    dim = init_states.shape[1]
    trajectory_matrix = np.zeros((len(pars) * dim, len(t_eval)))
    for idx, par in enumerate(pars):
        init_state = init_states[idx]
        system = system_class(par=par, list_par_x=list_par_x, discrete=discrete)
        trajectory = system.solve_system(
            fun=system.fun, init_state=init_state, t_eval=t_eval
        )
        trajectory_matrix[dim * idx : dim * (idx + 1), :] = trajectory.T

    for idx in range(len(pars)):
        # Create a new figure and axes for each iteration
        fig, ax = plt.subplots()

        # Plot the trajectory for the current index
        ax.plot(
            trajectory_matrix[dim * idx, :],
            trajectory_matrix[dim * idx + 1, :],
            alpha=1,
            color=line_color,
            marker=".",
            markersize=0.5,
        )

        # Set labels and title
        ax.set_title(
            f"Orbit for $\\alpha = {pars[idx]}, starting \: at \: ({init_states[idx][0]}, {init_states[idx][1]})$"
        )
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect(1)
        plt.show()


def plot_3d_surface(X: List[np.ndarray], view_angles: Tuple[float] | None = None):
    """Visualize a 3-d surface

    Parameters
    ----------
    X: List[np.ndarray]
        2-d array data for each of the three axes
    view_angles: Tuple[float] | None = None
        Specify the angles to view the 3-d plot
    """

    # Create a scatter plot of the data
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[0], X[1], X[2], c=X[0], cmap="viridis")

    # Set labels and title
    ax.set_xlabel("$\\alpha_1$")
    ax.set_ylabel("$\\alpha_2$")
    ax.set_zlabel("x")
    ax.zaxis.labelpad = -3
    ax.set_title("Cusp Bifurcation Visualization")
    if view_angles is not None:
        ax.view_init(view_angles[0], view_angles[1], view_angles[2])


# Task 4
def plot_bifurcation_diagram_logistic_map(
    fig: matplotlib.figure.Figure,
    ax: plt.Axes,
    title: str,
    system_class: Type[DynamicalSystem],
    dict_par_system_class: dict,
    line_color: str | None = None,
) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    """Plot the bifurcation diagram of a dynamical system

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        A matplotlib figure instance
    ax: plt.Axes
        A matplotlib axes instance
    title: str
        The title of the figure
    system_class: Type[DynamicalSystem]
        The class of the dynamical system
    dict_par_system_class: dict
        A dictionary containing all necessary arguments to instantiate
        system_class and to solve the dynamical system
    line_color: str | None = None
        The color of the lines in the plot

    Returns
    -------
    fig: matplotlib.figure.Figure
        A matplotlib figure instance
    ax: plt.Axes
        A matplotlib axes instance
    """

    # Retrieve the parameters from the dictionary
    pars = dict_par_system_class["pars"]
    init_states = dict_par_system_class["init_states"]
    list_par_x = dict_par_system_class["list_par_x"]
    t_eval = dict_par_system_class["t_eval"]
    discrete = dict_par_system_class["discrete"]

    # Solve the system for each value of the parameter
    trajectory_matrix = np.zeros((len(pars), len(t_eval)))
    for idx, par in enumerate(pars):
        init_state = [init_states[idx]]
        system = system_class(par=par, list_par_x=list_par_x, discrete=discrete)
        trajectory = system.solve_system(
            fun=system.fun, init_state=init_state, t_eval=t_eval
        )
        trajectory_matrix[idx, :] = trajectory[:, 0]

    # Ste title and labels
    ax.set_title(title)
    ax.set_xlabel("r")
    ax.set_ylabel("x")

    # Plot the diagram
    for idx in range(len(t_eval)):
        ax.plot(pars, trajectory_matrix[:, idx], ",", alpha=0.25, color=line_color)

    return fig, ax


def logistic_map(r: float, x: ArrayLike) -> ArrayLike:
    """Computes the next state of the logistic map.

    Parameters
    ----------
    r: float
        Growth rate parameter. 0 < r <= 4
    x: ArrayLike
        Current state x_n. 0 <= x_n <= 1

    Returns
    ----------
    ArrayLike
        Next state x_(n+1).
    """
    return r * x * (1 - x)


def plot_x_n(r: float, x_0: ArrayLike, num_iter: int = 100):
    """Plots the logistic map for a given value of r and initial values of x.

    Parameters
    ----------
    r: float
        Growth rate parameter (0 < r <= 4).
    x_0: ArrayLike
        Initial conditions.
    num_iter: int = 100
        Number of iterations.
    """

    if r <= 0 or r > 4:
        raise ValueError("The parameter r must satisfy 0<r<=4.")

    x = np.zeros((num_iter + 1, x_0.shape[0]))
    x[0] = x_0

    # Calculate the trajectories
    for i in range(num_iter):
        x[i + 1] = logistic_map(r, x[i])

    # Plot the trajectories
    for j in range(x_0.shape[0]):
        plt.plot(np.arange(num_iter + 1), x[:, j], label=f"$x_0$ = {x[0, j]}")
    plt.xlabel("n")
    plt.ylabel("$x_n$")
    plt.title(f"Logistic Map Iterations when r={r}")
    plt.legend()
    plt.show()

    print(x)


def plot_bifurcation_diagram_logistic(num_iter: int = 100):
    """Generates a bifurcation diagram of the logistic map.

    Parameters
    ----------
    num_iter: int = 100
        Number of iterations for the diagram.
    """

    # r = 0.01, 0.02, ..., 4 (excluding 0)
    r_values = np.linspace(0, 4, 401)[1:]
    # Initial conditions x = 0, 0.01, ..., 1
    initial_values = np.linspace(0, 1, 101)
    n_plot = 15  # Number of iterations to be plotted

    plt.figure(figsize=(10, 8))

    # Compute trajectories for initial values and plot bifurcation diagram
    for r in r_values:
        for x0 in initial_values:
            x_values = np.empty(num_iter + 1)
            x_values[0] = x0
            for i in range(1, num_iter + 1):
                x_values[i] = logistic_map(r, x_values[i - 1])
            # x-coordinate is r, y-coordinate is last #n_plot iterated values for each initial x
            plt.plot([r] * n_plot, x_values[-n_plot:], ",k", alpha=0.2)

    # Set title, labels and axes
    plt.title("Bifurcation Diagram of the Logistic Map")
    plt.xlabel("r")
    plt.ylabel("x")
    plt.xlim(0, 4)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


def plot_trajectory_lorenz_attractor(
    trajectory: ArrayLike, par: ArrayLike, noisy: bool = False
) -> None:
    """Plot the given trajectory of a Lorenz attractor in a 3D plot"""

    ax = plt.figure(figsize=(15, 15)).add_subplot(projection="3d")
    ax.plot(*trajectory, lw=0.6)

    # Set title and labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(
        "Lorenz attractor with $\\sigma = $"
        + str(np.around(par[0], 3))
        + ", $\\beta = $"
        + str(np.around(par[2], 3))
        + ", $\\rho = $"
        + str(np.around(par[1], 3))
        + (", and with noise" if noisy else ", and without noise")
    )
    plt.show()


def plot_error_lorenz_attractor(
    trajectory_1: ArrayLike, trajectory_2: ArrayLike, t_eval: ArrayLike
) -> None:
    """Plot the difference between two trajectories over time"""

    difference = np.linalg.norm(trajectory_1 - trajectory_2, axis=1) ** 2

    ax = plt.figure(figsize=(6, 6)).add_subplot()
    ax.plot(t_eval, difference)

    # Set title and labels
    ax.set_xlabel("t")
    ax.set_ylabel("$||x(t) - \\^x(t)||^2$")
    ax.set_title("Difference between the trajectories $x(t)$ and $\\^x(t)$")

    # Add a horizontal reference line at difference = 1
    ax.hlines(y=1, xmin=0, xmax=1000, colors="r", linestyles="--")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()


def plot_3d_traj(trajectories: List[np.ndarray]):
    """Visualize the 3-d trajectory

    Parameters:
    -----------
    trajectories: List[np.ndarray]
        1-d array data for each of the three axes

    Returns
    -------
    None
    """
    # TODO: add proper comments. Change code and signature and docstring if
    # necessary. Make the plot look nicer by adding axes labels and title etc.
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for trajectory in trajectories:
        ax.plot(trajectory[0], trajectory[1], trajectory[2], linewidth=0.1)
