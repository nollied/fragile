from functools import partial

import numpy as np
import numpy.linalg

from fragile.core.fractalai import relativize


def numerical_grad_single_vector(theta, f, dx=1e-3, order=1):
    """return numerical estimate of the local gradient
    The gradient is computer by using the Taylor expansion approximation over
    each dimension:
        function(t + dt) = function(t) + h df/dt(t) + h^2/2 d^2f/dt^2 + ...
    The first order gives then:
        df/dt = (function(t + dt) - function(t)) / dt + O(dt)
    Note that we could also compute the backwards version by subtracting dt instead:
        df/dt = (function(t) - function(t -dt)) / dt + O(dt)
    A better approach is to use a 3-step formula where we evaluate the
    derivative on both sides of a chosen point t using the above forward and
    backward two-step formulae and taking the average afterward. We need to use the Taylor
     expansion to higher order:
        function (t +/- dt) = function (t) +/- dt df/dt + dt ^ 2 / 2  dt^2 function/dt^2 +/- \
        dt ^ 3 d^3 function/dt^3 + O(dt ^ 4)
        df/dt = (function(t + dt) - function(t - dt)) / (2 * dt) + O(dt ^ 3)
    Note: that the error is now of the order of dt ^ 3 instead of dt
    In a same manner we can obtain the next order by using function(t +/- 2 * dt):
        df/dt = (function(t - 2 * dt) - 8 function(t - dt)) + 8 function(t + dt) - \
        function(t + 2 * dt) / (12 * dt) + O(dt ^ 4)
    In the 2nd order, two additional function evaluations are required (per dimension), implying a
    more time-consuming algorithm. However, the approximation error is of the order of dt ^ 4

    INPUTS
    ------
    theta: ndarray[float, ndim=1]
        vector value around which estimating the gradient
    function: callable
        function from which estimating the gradient
    KEYWORDS
    --------
    dx: float
        pertubation to apply in each direction during the gradient estimation
    order: int in [1, 2]
        order of the estimates:
            1 uses the central average over 2 points
            2 uses the central average over 4 points
    OUTPUTS
    -------
    df: ndarray[float, ndim=1]
        gradient vector estimated at theta
    COST: the gradient estimation need to evaluates ndim * (2 * order) points (see above)
    CAUTION: if dt is very small, the limited numerical precision can result in big errors.
    """
    ndim = len(theta)
    df = np.empty(ndim, dtype=float)
    if order == 1:
        cst = 0.5 / dx
        for k in range(ndim):
            dt = np.zeros(ndim, dtype=float)
            dt[k] = dx
            df[k] = (f(theta + dt) - f(theta - dt)) * cst
    elif order == 2:
        cst = 1.0 / (12.0 * dx)
        for k in range(ndim):
            dt = np.zeros(ndim, dtype=float)
            dt[k] = dx
            df[k] = cst * (
                f(theta - 2 * dt) - 8.0 * f(theta - dt) + 8.0 * f(theta + dt) - f(theta + 2.0 * dt)
            )
    return df


def numerical_grad(theta, f, dx=1e-3, order=1):
    """return numerical estimate of the local gradient
    The gradient is computer by using the Taylor expansion approximation over
    each dimension:
        function(t + dt) = function(t) + h df/dt(t) + h^2/2 d^2f/dt^2 + ...
    The first order gives then:
        df/dt = (function(t + dt) - function(t)) / dt + O(dt)
    Note that we could also compute the backwards version by subtracting dt instead:
        df/dt = (function(t) - function(t -dt)) / dt + O(dt)
    A better approach is to use a 3-step formula where we evaluate the
    derivative on both sides of a chosen point t using the above forward and
    backward two-step formulae and taking the average afterward. We need to use the
     Taylor expansion to higher order:
    function (t +/- dt) = function (t) +/- dt df/dt + dt ^ 2 / 2  dt^2
    function/dt^2 +/- dt ^ 3 d^3 function/dt^3 + O(dt ^ 4)
        df/dt = (function(t + dt) - function(t - dt)) / (2 * dt) + O(dt ^ 3)
    Note: that the error is now of the order of dt ^ 3 instead of dt
    In a same manner we can obtain the next order by using function(t +/- 2 * dt):
        df/dt = (function(t - 2 * dt) - 8 function(t - dt)) + 8 function(t + dt) - \
        function(t + 2 * dt) / (12 * dt) + O(dt ^ 4)
    In the 2nd order, two additional function evaluations are required (per dimension), implying a
    more time-consuming algorithm. However the approximation error is of the order of dt ^ 4
    INPUTS
    ------
    theta: ndarray[float, ndim=1]
        vector value around which estimating the gradient
    function: callable
        function from which estimating the gradient
    KEYWORDS
    --------
    dx: float
        pertubation to apply in each direction during the gradient estimation
    order: int in [1, 2]
        order of the estimates:
            1 uses the central average over 2 points
            2 uses the central average over 4 points
    OUTPUTS
    -------
    df: ndarray[float, ndim=1]
        gradient vector estimated at theta
    COST: the gradient estimation need to evaluates ndim * (2 * order) points (see above)
    CAUTION: if dt is very small, the limited numerical precision can result in big errors.
    """
    ndim = theta.shape[1]
    df = np.empty_like(theta)
    if order == 1:
        cst = 0.5 / dx
        for k in range(ndim):
            dt = np.zeros(theta.shape, dtype=float)
            dt[:, k] = dx
            df[:, k] = (f(theta + dt) - f(theta - dt)) * cst
    elif order == 2:
        cst = 1.0 / (12.0 * dx)
        for k in range(ndim):
            dt = np.zeros(theta.shape, dtype=float)
            dt[:, k] = dx
            df[:, k] = cst * (
                f(theta - 2 * dt) - 8.0 * f(theta - dt) + 8.0 * f(theta + dt) - f(theta + 2.0 * dt)
            )
    return df


def leapfrog(init_pos, init_momentum, grad, step_size, function, force=None):
    """Perfom a leapfrog jump in the Hamiltonian space
    INPUTS
    ------
    init_pos: ndarray[float, ndim=1]
        initial parameter position
    init_momentum: ndarray[float, ndim=1]
        initial momentum
    grad: float
        initial gradient value
    step_size: float
        step size
    function: callable
        it should return the log probability and gradient evaluated at theta
        logp, grad = function(theta)
    OUTPUTS
    -------
    new_position: ndarray[float, ndim=1]
        new parameter position
    new_momentum: ndarray[float, ndim=1]
        new momentum
    gradprime: float
        new gradient
    new_potential: float
        new lnp
    """
    force = 0.0 if force is None else force
    # make half step in init_momentum
    new_momentum = init_momentum + 0.5 * step_size * (force - grad)
    # make new step in theta
    new_position = init_pos + step_size * new_momentum
    # compute new gradient
    new_potential, gradprime = function(new_position)
    # make half step in init_momentum again
    new_momentum = new_momentum + 0.5 * step_size * (force - gradprime)
    return new_position, new_momentum, gradprime, new_potential


def fai_leapfrog(init_pos, init_momentum, grad, step_size, function, force=None):
    """Perfom a leapfrog jump in the Hamiltonian space
    INPUTS
    ------
    init_pos: ndarray[float, ndim=1]
        initial parameter position
    init_momentum: ndarray[float, ndim=1]
        initial momentum
    grad: float
        initial gradient value
    step_size: float
        step size
    function: callable
        it should return the log probability and gradient evaluated at theta
        logp, grad = function(theta)
    OUTPUTS
    -------
    new_position: ndarray[float, ndim=1]
        new parameter position
    new_momentum: ndarray[float, ndim=1]
        new momentum
    gradprime: float
        new gradient
    new_potential: float
        new lnp
    """
    force = 0.0 if force is None else force
    potential, grad = function(init_pos)
    pot = relativize(potential).reshape(-1, 1)
    grad = (grad / numpy.linalg.norm(grad, axis=1).reshape(-1, 1)) * pot
    # make half step in init_momentum
    new_momentum = init_momentum + 0.5 * step_size * (force - grad)
    # make new step in theta
    new_position = init_pos + step_size * new_momentum
    # compute new gradient
    new_potential, gradprime = function(new_position)
    new_pot = relativize(new_potential).reshape(-1, 1)
    gradprime = (gradprime / numpy.linalg.norm(gradprime, axis=1).reshape(-1, 1)) * new_pot
    # make half step in init_momentum again
    new_momentum = new_momentum + 0.5 * step_size * (force - gradprime)
    return new_position, new_momentum, gradprime, new_potential


class GradFuncWrapper:
    """Create a function-like object that combines provided lnp and grad(lnp)
    functions into one as required by nuts6.
    Both functions are stored as partial function allowing to fix arguments if
    the gradient function is not provided, numerical gradient will be computed
    By default, arguments are assumed identical for the gradient and the
    likelihood. However you can change this behavior using set_xxxx_args.
    (keywords are also supported)
    if verbose property is set, each call will print the log-likelihood value
    and the theta point
    """

    def __init__(self, function, grad_func=None, dx=1e-3, order=1):
        self.function = function
        if grad_func is not None:
            self.grad_func = grad_func
        else:
            self.grad_func = partial(numerical_grad, f=self.function, dx=dx, order=order)
        self.verbose = False

    def __call__(self, theta):
        r = (self.function(theta), self.grad_func(theta))
        if self.verbose:
            print(r[0], theta)
        return r
