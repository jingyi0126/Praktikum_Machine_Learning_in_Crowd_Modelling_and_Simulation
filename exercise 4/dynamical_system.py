from typing import List, Callable
from numpy.typing import ArrayLike
import numpy as np


class DynamicalSystem:
    """This class defines a dynamical system

    Methods
    --------
    solve_system(fun: Callable, init_state: ArrayLike, t_eval: ArrayLike):
        Solve the dynamical system
    """

    def __init__(self, discrete: bool = True):
        """Parameters
        -------------
        discrete: bool = False
            If true, the dynamical system is time-discrete
        """
        self.discrete = discrete

    def solve_system(
        self, fun: Callable, init_state: ArrayLike, t_eval: ArrayLike
    ) -> ArrayLike:
        """Solve the dynamical system

        Given the evolution rules, the initial point, and the time steps, we
        obtain the trajectory of the point. The solving method is different
        for time-discrete system, so two methods are implemented here.

        Parameters
        ----------
        fun: Callable
            Evolution operator
        init_state: ArrayLike
            Initial state of the system
        t_eval: ArrayLike
            Time steps of the trajectory

        Returns
        -------
        trajectory: ArrayLike
            Trajectory of the initial point in time
        """
        trajectory = np.zeros((len(t_eval), len(init_state)))
        trajectory[0, :] = init_state
        delta_time = t_eval[1] - t_eval[0]
        if self.discrete:
            for k in range(1, len(t_eval)):
                trajectory[k, :] = trajectory[k - 1, :] + delta_time * fun(
                    k, trajectory[k - 1, :]
                )
        else:
            for k in range(1, len(t_eval)):
                trajectory[k, :] = trajectory[k - 1, :] + delta_time * fun(
                    k, trajectory[k - 1, :]
                )
        return trajectory

    def _set_grid_coordinates(self, list_par_x: List[List[int]]) -> List[np.ndarray]:
        """Set up the coordinates. For multidimensional cases use meshgrid"""
        """
        match len(list_par_x):
            case 1:
                return np.linspace(list_par_x[0][0], list_par_x[0][1], list_par_x[0][2])
            case 2:
                X1, X2 = np.meshgrid(
                    np.linspace(list_par_x[0][0], list_par_x[0][1], list_par_x[0][2]),
                    np.linspace(list_par_x[1][0], list_par_x[1][1], list_par_x[1][2])
                )
                return [X1, X2]
            case 3:
                X1, X2, X3 = np.meshgrid(
                    np.linspace(list_par_x[0][0], list_par_x[0][1], list_par_x[0][2]),
                    np.linspace(list_par_x[1][0], list_par_x[1][1], list_par_x[1][2]),
                    np.linspace(list_par_x[2][0], list_par_x[2][1], list_par_x[2][2])
                )
                return [X1, X2, X3]
        """

        # We use this structure since Artemis is not compatible with match statements
        if len(list_par_x) == 1:
            return np.linspace(list_par_x[0][0], list_par_x[0][1], list_par_x[0][2])
        elif len(list_par_x) == 2:
            X1, X2 = np.meshgrid(
                np.linspace(list_par_x[0][0], list_par_x[0][1], list_par_x[0][2]),
                np.linspace(list_par_x[1][0], list_par_x[1][1], list_par_x[1][2]),
            )
            return [X1, X2]
        elif len(list_par_x) == 3:
            X1, X2, X3 = np.meshgrid(
                np.linspace(list_par_x[0][0], list_par_x[0][1], list_par_x[0][2]),
                np.linspace(list_par_x[1][0], list_par_x[1][1], list_par_x[1][2]),
                np.linspace(list_par_x[2][0], list_par_x[2][1], list_par_x[2][2]),
            )
            return [X1, X2, X3]
        else:
            raise ValueError("The list_par_x length must be 1, 2, or 3.")


class Task1(DynamicalSystem):
    """This class defines the system used in task 1

    Methods
    --------
    fun(t: float, x: ArrayLike):
        Evolution operator
    """

    def __init__(self, par: ArrayLike, list_par_x: List[List[int]], *args, **kwargs):
        """Parameters
        -------------
        par: ArrayLike
            The parametrized matrix used in the evolution operator
        list_par_x: List[List[int]]
            The coordinates of the system
        """
        super().__init__(*args, **kwargs)
        self.matrix = par
        self.X = self._set_grid_coordinates(list_par_x)

    def fun(self, t: float, x: ArrayLike) -> ArrayLike:
        """Apply the evolution operator

        Parameters
        ----------
        t: float
            The current time step. Since this system is time independent, t is never
            used. However, it is still required to be compatible with the super class.
        x: ArrayLike
            The previous state of the system

        Returns
        -------
        DX: ArrayLike
            Trajectory of the previous point in this time step
        """

        # Update the state of x
        x = np.asarray(x)
        x_shape = x.shape
        x = x.reshape((2, -1))
        DX = self.matrix @ x  # velocities of x
        DX = DX.reshape(x_shape)

        return DX


class Task2(DynamicalSystem):
    """This class defines the generalized version of the systems used in task 2

    Methods
    --------
    fun():
        Calculates the steady states analytically. This class is not compatible
        with DynamicalSystem's solver, which is not required in the exercise.
    """

    def __init__(self, beta: int, gamma: int, *args, **kwargs):
        """Parameters
        -------------
        beta: int
            The coefficient of x²
        gamma: int
            The additive constant added to the system
        """
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.gamma = gamma

    def fun(self):
        """Calculate the steady states for selected values of alpha analytically"""
        alpha = np.linspace(-self.gamma, 6 - self.gamma, 400)
        X = np.sqrt((alpha + self.gamma) / self.beta)
        return X


class AndronovHopf(DynamicalSystem):
    """This class defines the Andronov-Hopf bifurcation system

    Methods
    --------
    fun(t: float, x: ArrayLike):
        Evolution operator
    """

    def __init__(self, par: float, list_par_x: List[List[int]], *args, **kwargs):
        """Parameters
        -------------
        par: float
            The parameter alpha used in the evolution operator
        list_par_x: List[List[int]]
            The coordinates of the system
        """
        super().__init__(*args, **kwargs)
        self.alpha = par
        self.X = self._set_grid_coordinates(list_par_x)

    def fun(self, t: float, x: ArrayLike) -> ArrayLike:
        """Apply the evolution operator

        Parameters
        ----------
        t: float
            The current time step.
        x: ArrayLike
            The previous state of the system

        Returns
        -------
        DX: ArrayLike
            Trajectory of the previous point in this time step
        """
        x1, x2 = x
        dx1 = self.alpha * x1 - x2 - x1 * (x1**2 + x2**2)
        dx2 = x1 + self.alpha * x2 - x2 * (x1**2 + x2**2)
        return np.array([dx1, dx2])


class LorenzAttractor(DynamicalSystem):
    """This class defines the Lorenz attractor

    Methods
    --------
    fun(t: float, x: ArrayLike):
        Evolution operator
    """

    def __init__(self, par: ArrayLike, list_par_x: List[List[int]], *args, **kwargs):
        """Parameters
        -------------
        par: ArrayLike
            The parameters sigma, rho, and beta used in the evolution operator
        list_par_x: List[List[int]]
            The coordinates of the system
        """
        super().__init__(*args, **kwargs)
        self.sigma, self.rho, self.beta = par
        self.X = self._set_grid_coordinates(list_par_x)

    def fun(self, t: float, x: ArrayLike) -> ArrayLike:
        """Apply the evolution operator

        Parameters
        ----------
        t: float
            The current time step.
        x: ArrayLike
            The previous state of the system

        Returns
        -------
        DX: ArrayLike
            Trajectory of the previous point in this time step
        """

        x1, x2, x3 = x
        dx1 = self.sigma * (x2 - x1)
        dx2 = x1 * (self.rho - x3) - x2
        dx3 = x1 * x2 - self.beta * x3

        return np.array([dx1, dx2, dx3])
