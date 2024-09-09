import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp
import math

# TODO: Define all the functions (in addition to the ones below) required to complete tasks 1-3 in this python file.

# Evaluation of a single Radial Basis Function (RBF)


def radial_basis_function(x, x_c, eps):
    """
    Compute radial basis function values as defined in the worksheet in section 1.2 (equation (7))

    Args:
    - x: Data points
    - x_c: Centers of the basis functions
    - eps: Epsilon parameter(bandwidth)

    Returns:
    - Radial basis function values evaluated at data points x
    """
    # Hint: Use cdist from scipy.spatial.distance
    distances = cdist(x, x_c, 'euclidean')
    rbf = np.exp(-(distances**2) / (eps**2))

    return rbf


def least_squares(A, b, cond=1e-5):
    """ Returns a Least squares solution of Ax = b

    Args:
        A (npt.NDArray[np.float32]): Input array for the least squres problem
        b (npt.NDArray[np.float32]): Output array for the least squares problem
    Returns:
        npt.NDArray[np.float32]: Least squares solution of Ax = b
    """
    # TODO: Implement using scipy.linalg.lstsq

    # Hint: Don't forget that lstsq also returns other parameters such as residuals, rank, singular values etc.

    x, residuals, rank, s = lstsq(A, b, cond=cond)

    return x, residuals, rank, s


def pred_linear(data):
    """
    Perform linear regression on a given dataset and return the predicted values and coefficients.
    
    Args:
    data (numpy.ndarray): The input dataset, a 2D array where the first column represents the independent variable (x)
                          and the second column represents the dependent variable (y).
                          
    Returns:
    tuple:
        y_hat (numpy.ndarray): The predicted values of y based on the linear regression model.
        coef (numpy.ndarray): The coefficients of the linear regression model.
        
    The function uses least squares to fit a linear model of the form y = b0 + b1 * x, where b0 is 
    the intercept and b1 is the slope. The matrix X is constructed by stacking a column of 
    ones (for the intercept) and the column of independent variable x.
    
    """
    x = data[:, 0]
    y = data[:, 1]
    X = np.vstack((np.ones_like(x), x)).T
    coef, residuals, rank, s = least_squares(X, y, cond=1e-5)
    y_hat = X @ coef
    return y_hat, coef


def plot_linear(data, name="Dataset"):
    """
    Plot the linear regression fit for a given dataset and display the Mean Squared Error (MSE).
    
    Parameters:
    data (numpy.ndarray): A 2D array where the first column represents the independent variable (x)
                          and the second column represents the dependent variable (y).
    name (str, optional): The name of the dataset to be displayed in the plot title. Default is "Dataset".
                          It can be "Dataset A" or "Dataset B", which depends on the used dataset.
    
    Returns:
    None
    
    The function performs the following steps:
    1. Computes the linear regression fit using the pred_linear function.
    2. Calculates the Mean Squared Error (MSE) between the actual y values and the predicted y values.
    3. Plots the original data points as a scatter plot.
    4. Plots the linear regression fit as a line plot.
    5. Displays the MSE in the plot title.
    """ 
    x = data[:, 0]
    y = data[:, 1]
    y_hat, coef = pred_linear(data)
    MSE = mean_squared_error(y, y_hat)
    print(f"MSE={MSE}")
    x_range = np.linspace(x.min()-1, x.max()+1, 1500).reshape(-1, 1)
    X_range = np.hstack((np.ones_like(x_range), x_range))
    y_hat_range = X_range @ coef

    plt.figure(figsize=(8, 5))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x, y, color='blue', label="Original data")
    plt.plot(x_range, y_hat_range, color='red',
             label='Linear approximation')
    plt.title(f'Approximating linear function for {name}, MSE={MSE:.5g}')
    plt.legend()
    plt.show()


def pred_nonlinear(data, x_l, eps, cond=1e-5):
    """ Predict a non-linear function as a set of radial basis functions.

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
    # print(rbf_values.shape)
    coef, residuals, rank, s = least_squares(rbf_values, y)
    y_hat = rbf_values @ coef
    return y_hat.reshape(-1, 1), coef


def plot_nonlinear(data, x_l, eps, cond=1e-5, name="Dataset"):
    """
    Plot the nonlinear regression fit using radial basis functions for a given dataset and display the Mean Squared Error (MSE).
    
    Parameters:
    data (numpy.ndarray): The input dataset, a 2D array where the first column represents the independent variable (x)
                          and the second column represents the dependent variable (y).
    x_l (numpy.ndarray): A 1D array of center points for the radial basis functions.
    eps (float): The width parameter (epsilon) for the radial basis functions.
    cond (float, optional): The cutoff for small singular values in the least squares solver. Default is 1e-5.
    name (str, optional): The name of the dataset to be displayed in the plot title. Default is "Dataset".
                          It can be "Dataset A" or "Dataset B", which depends on the used dataset.
    
    
    Returns:
    None
    
    The function performs the following steps:
    1. Computes the nonlinear regression fit using the pred_nonlinear function.
    2. Calculates the Mean Squared Error (MSE) between the actual y values and the predicted y values.
    3. Plots the original data points as a scatter plot.
    4. Plots the nonlinear regression fit as a line plot.
    5. Displays the MSE, number of center points (L), and epsilon in the plot title.
    """
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    y_hat, coef = pred_nonlinear(data, x_l, eps, cond=cond)
    MSE = mean_squared_error(y, y_hat)
    print(f"MSE={MSE}")
    x_range = np.linspace(x.min()-1, x.max()+1, 1500).reshape(-1, 1)
    rbf_values_range = radial_basis_function(x_range, x_l, eps)
    y_hat_range = rbf_values_range @ coef

    plt.figure(figsize=(8, 5))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x, y, color='blue', label="Original data")
    # plt.scatter(x, y_hat, color='red', marker='x', alpha=0.5, label='Nonlinear approximation')
    plt.plot(x_range, y_hat_range, color='red',
             label='Nonlinear approximation')
    plt.title(f'Approximating nonlinear function for {name},\n L={len(x_l)}, $\epsilon$={eps}, MSE={MSE:.5g}')
    plt.legend()
    plt.show()


def system(t, x, A):
    """
    Compute the time derivative of the state vector x in a linear dynamical system.
    
    Parameters:
    t (float): Time variable (not used in this case because the system is independent of time).
    x (array-like): State vector at time t.
    A (numpy.ndarray): The system matrix defining the linear transformation.
    
    Returns:
    numpy.ndarray: The time derivative of the state vector x.
    """
    #x = x.reshape(1, x.shape[0])
    return A @ x

def rbf_approx(t, x, C, centers, eps):
    """
    Compute the time derivative of the state vector x in a non-linear dynamical system.

    Args:
        t (float): Time variable (not used in this case because the system is independent of time).
        x (array-like): State vector at time t.
        C (numpy.ndarray): The system matrix defining the non-linear transformation.
        centers (numpy.ndarray): Array with centroids used for the readial basis functions.
        eps (float): Epsilon parameter for radial basis functions.

    Returns:
        numpy.ndarray: Return approximation of time derivative of state vector x for non-linear system.
    """
    x = x.reshape(1, x.shape[-1])
    rbf = radial_basis_function(x, centers, eps)
    return rbf @ C


def vector_field(X, Y, A):
    """
    Compute the vector field for a 2D linear dynamical system over a grid of points.
    
    Parameters:
    X (numpy.ndarray): x-coordinates of the grid points.
    Y (numpy.ndarray): y-coordinates of the grid points.
    A (numpy.ndarray): The system matrix (2*2) defining the linear transformation.
    
    Returns:
    tuple:
        u (numpy.ndarray): x-components of the vectors in the vector field.
        v (numpy.ndarray): y-components of the vectors in the vector field.
    
    The function computes the vector field by evaluating the system function at each grid point.
    """
    u = np.zeros(X.shape)
    v = np.zeros(Y.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            velocity = system(None, [X[i, j], Y[i, j]], A)
            u[i, j], v[i, j] = velocity
    return u, v

def closest_factors(N):
    """
    Computes two numbers p and q such that p*q = N, minimizing the difference between p, q.

    Args:
        N (int): Number of centroids desired.

    Returns:
        tuple: pair of values p, q obtained such that p*q = N, as to sample uniformly in 2D.
    """
    # Start with p and q being 1 and N
    p, q = 1, N
    # Initialize the minimum difference to a large value
    min_diff = N - 1
    
    # Iterate over possible factors up to the square root of N
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:  # i is a factor
            # Calculate the corresponding factor
            factor1, factor2 = i, N // i
            # Calculate the difference
            diff = abs(factor1 - factor2)
            # Update p, q if this pair has a smaller difference
            if diff < min_diff:
                p, q = factor1, factor2
                min_diff = diff

    return p, q

def find_best_tf(x0, x1, ti, tf, dt, M, func = system, centers = None, eps = None, verbose = True):
    """


    Args:
        x0 (numpy.ndarray): Initial state.
        x1 (numpy.ndarray): Final state.
        ti (float): Initial time for integration
        tf (float): Final time for integration
        dt (float): Time step, used for determining all points at which the solution will be tested.
        M (numpy.ndarray): Matrix used to define the linear system to be solved.

    Returns:
        tuple: Returns best solution array, best MSE (mean square error) and final time of best solution obtained.
    """

    # Create span and evaluation time arrays for solve_ivp.
    t_span = (ti, tf)
    t_eval = np.linspace(ti, tf, round((tf - ti)/dt) + 1)
    #t_eval = np.concatenate(([t_eval[0]], t_eval[-10:]))

    # Initialize best result variables.
    final_mse = None
    best_tf = None
    best_mse = float('inf')
    y_pred = []

    # Define extra arguments for left-hand-side function for solve_ivp.
    if centers is not None:
        args = (M, centers, eps)
    else:
        args = (M,)
    
    # Solve for all initial points x0 until xf.
    for i in range(len(x0)):
        sol = solve_ivp(func, t_span, x0[i], t_eval = t_eval, args = args)
        y_pred.append(sol.y)

    y_pred = np.array(y_pred)

    # Compute MSE and save results if current approximation is the best one.
    for i, ft in enumerate(t_eval):
        x_pred = y_pred[:, :, i]
        mse = np.mean(np.sum((x_pred - x1) ** 2, axis = 1))
        if mse < best_mse:
            best_tf = i
            best_mse = mse
        if i == len(t_eval) - 1:
            final_mse = mse

    if verbose:
        print(f"Best final solution x1 found for tf = {t_eval[best_tf]}.")
        print(f"Mean squared error for best final time = {best_mse}.")
        print(f"Mean square error for final time tf = {t_eval[-1]}, mse = {final_mse}.")

    return y_pred[:, :, best_tf], best_mse, best_tf

def find_best_L_and_eps(x0, x1, v_k, ti, tf, dt, num_x_l_list, eps_list, verbose = True):
    """_summary_

    Args:
        x0 (numpy.ndarray): Initial state.
        x1 (numpy.ndarray): Final state.
        v_k (numpy.ndarray): Velocity vector field.
        ti (float): Initial time for integration
        tf (float): Final time for integration
        dt (float): Time step, used for determining all points at which the solution will be tested.
        num_x_l_list (numpy.ndarray): Array with values of number of centroids to try for the rbf.
        eps_list (numpy.ndarray): Array with values of epsilon to try for the rbf.

    Returns:
        tuple: Returns best solution array, best MSE (mean square error), final time, number of centroids and epsilon of best solution obtained.
    """
    best_mse = float('inf')
    best_L = -1
    best_eps = -1
    best_x_pred = []
    d0min, d0max = math.floor(np.min(x0[:, 0])), math.ceil(np.max(x0[:, 0]))
    d1min, d1max = math.floor(np.min(x0[:, 1])), math.ceil(np.max(x0[:, 1]))
    
    # Number of points along each axis (sqrt(N) x sqrt(N))
    for L in num_x_l_list:
        print(f"Using {L} centroids")
        
        num_points_per_axis = closest_factors(L)
        
        # Generate linearly spaced points along each axis
        d0 = np.linspace(d0min, d0max, num_points_per_axis[0])
        d1 = np.linspace(d1min, d1max, num_points_per_axis[1])
        
        # Create the grid
        X, Y = np.meshgrid(d0, d1)
        
        # Flatten the grid arrays to get a list of coordinates
        centers = np.column_stack([X.ravel(), Y.ravel()])
    
        for eps in eps_list:
            rbf = radial_basis_function(x0, centers, eps)
            C, _, _, _ = least_squares(rbf, v_k, cond = 1e-5)
            x_pred, mse, current_tf = find_best_tf(x0, x1, ti, tf, dt, C, rbf_approx, centers, eps, verbose = False)
            
            # Save if current approximation is the best.
            if mse < best_mse:
                best_x_pred = x_pred
                best_tf = current_tf
                best_L = L
                best_eps = eps
                best_mse = mse
    
    if verbose:
        print(f"Best final solution x1 found for tf = {tf}, L = {best_L}, eps = {best_eps}.")
        print(f"Mean squared error for best solution = {best_mse}.")

    return best_x_pred, best_mse, best_tf, best_L, best_eps

def plot_steady_states(x0, x1, v_k, ti, tf, dt, L, eps):
    """
    Creates the plot of the final steady states after a long time tf.

    Args:
        x0 (numpy.ndarray): Initial state.
        x1 (numpy.ndarray): Final state.
        v_k (numpy.ndarray): Velocity vector field.
        ti (float): Initial time for integration
        tf (float): Final time for integration
        dt (float): Time step, used for determining all points at which the solution will be tested.
        L (int): Number of centroids to try for the rbf.
        eps (float): Epsilon to try for the rbf.

    """
    t_span = (ti, tf)
    t_eval = np.linspace(ti, tf, round((tf - ti)/dt) + 1)
    d0min, d0max = math.floor(np.min(x0[:, 0])), math.ceil(np.max(x0[:, 0]))
    d1min, d1max = math.floor(np.min(x0[:, 1])), math.ceil(np.max(x0[:, 1]))
    
    # Generate linearly spaced points along each axis
    num_points_per_axis = closest_factors(L)
    d0 = np.linspace(d0min, d0max, num_points_per_axis[0])
    d1 = np.linspace(d1min, d1max, num_points_per_axis[1])
    
    # Create the grid
    X, Y = np.meshgrid(d0, d1)
    
    # Flatten the grid arrays to get a list of coordinates
    centers = np.column_stack([X.ravel(), Y.ravel()])
    rbf = radial_basis_function(x0, centers, eps)
    C, _, _, _ = least_squares(rbf, v_k, cond = 1e-5)
    args = (C, centers, eps)

    y_pred = []

    for i in range(len(x0)):
        sol = solve_ivp(rbf_approx, t_span, x0[i], args = args)
        y_pred.append(sol.y)

    y_pred = np.array(y_pred)

    plt.figure(figsize=(8, 8))
    plt.title("Non-linear system steady states")
    plt.scatter(y_pred[:, 0], y_pred[:, 1], color='blue', alpha=1, label='xf', s=12)
    plt.xlim(-4.5, 4.5)
    plt.ylim(-4.5, 4.5)
    plt.show()