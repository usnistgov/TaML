r"""
The functions that implement the various methods for incorporating theory
into machine learning. Benchmark methods of direct (no theory) and theory
(no machine learning) are also included. Specifically, the code below works
for GPR with homoscedastic noise as implemented in GPFlow.
"""

import numpy as np
import random
import scipy

import gpflow
import tensorflow as tf
from gpflow.ci_utils import ci_niter
from gpflow.utilities import print_summary, set_trainable
from taml.mltools import optimizeGPR

from sklearn.metrics import mean_squared_error


def direct(x_train, y_train, t_train, a_train, x_test, y_test, t_test):
    """
    Run the GPR with homoscedastic noise with no knowledge of theory
    and return the mean squared error.
    """

    # put data together
    y_m_train = y_train.reshape(-1, 1)
    y_m_test = y_test.reshape(-1, 1)

    # model construction
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[3, 0.1])
    model = gpflow.models.GPR(
        (x_train,
         y_m_train),
        kernel=kernel,
        mean_function=gpflow.mean_functions.Constant(
            y_train.mean()))

    # optimize
    model = optimizeGPR(model)

    # predict
    y_pred, y_var = model.predict_f(x_test)

    return mean_squared_error(y_m_test, y_pred)


def latentvariable(x_train, y_train, t_train, a_train, x_test, y_test, t_test):
    """
    Use numerical values of the theory as an additional input to GPR with
    homoscedastic noise. Return the mean squared error.
    """

    # put data together
    y_m_train = y_train.reshape(-1, 1)
    y_m_test = y_test.reshape(-1, 1)
    x_m_train = np.hstack([x_train, t_train.reshape(-1, 1)])
    x_m_test = np.hstack([x_test, t_test.reshape(-1, 1)])

    # model construction
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[3, 0.1, 1])
    mean = gpflow.mean_functions.Constant(y_train.mean())
    model = gpflow.models.GPR((x_m_train, y_m_train),
                              kernel=kernel,
                              mean_function=mean)

    # optimize
    model = optimizeGPR(model)

    # predict
    y_pred, y_var = model.predict_f(x_m_test)

    return mean_squared_error(y_m_test, y_pred)


def difference(x_train, y_train, t_train, a_train, x_test, y_test, t_test):
    """
    Use GPR with homoscedastic noise to predict the difference between the
    training data and numerical values of the theory. Then use that GPR to
    predict the difference at new input values. Next add the difference back
    to the predicted theory to yield a prediction for the output. Return the
    mean squared error.
    """

    # put data together
    y_m_train = y_train.reshape(-1, 1) - t_train.reshape(-1, 1)
    y_m_test = y_test.reshape(-1, 1)

    # model construction
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[3, 0.1])
    mean = gpflow.mean_functions.Constant(y_train.mean() - t_train.mean())
    model = gpflow.models.GPR((x_train, y_m_train),
                              kernel=kernel,
                              mean_function=mean)

    # optimize
    model = optimizeGPR(model)

    # predict
    y_pred, y_var = model.predict_f(x_test)
    y_pred += t_test.reshape(-1, 1)

    return mean_squared_error(y_m_test, y_pred)


def quotient(x_train, y_train, t_train, a_train, x_test, y_test, t_test):
    """
    Use GPR with heteroscedastic noise to predict the quotient between the
    training data and numerical values of the theory. Then use that GPR to
    predict the quotient at new input values. Next multiply the quotient
    back to the predicted theory to yield a prediction for the output. Return
    the mean squared error.
    """

    # put data together
    y_m_train = y_train.reshape(-1, 1) / t_train.reshape(-1, 1)
    y_m_test = y_test.reshape(-1, 1)

    # model construction
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[3, 0.1])
    mean = gpflow.mean_functions.Constant(y_train.mean())
    model = gpflow.models.GPR((x_train, y_m_train),
                              kernel=kernel,
                              mean_function=mean)

    # optimize
    model = optimizeGPR(model)

    # predict
    y_pred, y_var = model.predict_f(x_test)
    y_pred = y_pred * t_test.reshape(-1, 1)

    return mean_squared_error(y_m_test, y_pred)


def linearprior(x_train, y_train, t_train, a_train, x_test, y_test, t_test):
    """
    Use GPR with homoscedastic noise for predictions. The prior is assumed to
    be a linear function of the inputs rather than a constant as the default.
    This includes the functional form of the theory in an incomplete way.
    Return the mean squared error.
    """

    # put data together
    y_m_train = y_train.reshape(-1, 1)
    y_m_test = y_test.reshape(-1, 1)

    # model construction
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[3, 0.1])
    mean = gpflow.mean_functions.Linear(A=[[0.5], [0.01]], b=y_train.mean())
    model = gpflow.models.GPR((x_train, y_m_train),
                              kernel=kernel,
                              mean_function=mean)

    # optimize
    model = optimizeGPR(model)

    # predict
    y_pred, y_var = model.predict_f(x_test)

    return mean_squared_error(y_m_test, y_pred)


def fixedprior(x_train, y_train, t_train, a_train, x_test, y_test, t_test):
    """
    Use GPR with homoscedastic noise for predictions. The prior is assumed to
    be a linear function of the inputs rather than a constant as the default.
    The coeffients in the prior are set to be that from the theory.
    This includes the functional form of the theory in an incomplete way.
    Return the mean squared error.
    """

    # put data together
    y_m_train = y_train.reshape(-1, 1)
    y_m_test = y_test.reshape(-1, 1)

    # model construction
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[3, 0.1])
    mean = gpflow.mean_functions.Linear(A=[[1], [0]], b=np.log(1 / 6))
    model = gpflow.models.GPR((x_train, y_m_train),
                              kernel=kernel,
                              mean_function=mean)

    set_trainable(model.mean_function.A, False)
    set_trainable(model.mean_function.b, False)

    # optimize
    model = optimizeGPR(model)

    # predict
    y_pred, y_var = model.predict_f(x_test)

    return mean_squared_error(y_m_test, y_pred)


def parameterization(
        x_train,
        y_train,
        t_train,
        a_train,
        x_test,
        y_test,
        t_test):
    """
    Instead of modeling the data using a single GPR with homoscedastic noise,
    make use of the functional form of the theory such that
    ln R_g^2 = 2 nu[alpha]*(ln N) + kappa[alpha, ln N] where both nu and kappa
    are Gaussian Processes. This fully incorporates the functional form of the
    theory. Note that these two Gaussain Processes can be combined in a single
    Gaussian Process using mathmatical relationships. Return the mean squared
    error.
    """

    # put data together
    y_m_train = y_train.reshape(-1, 1) - t_train.reshape(-1, 1)
    y_m_test = y_test.reshape(-1, 1)

    # model construction
    # kernel for the constant term
    kernel1 = gpflow.kernels.SquaredExponential(lengthscales=[3, .1])

    # kernel for the function
    kernel2 = gpflow.kernels.Linear(active_dims=[0])
    kernel2.variance.assign(1)

    # kernel for the slope
    kernel3 = gpflow.kernels.SquaredExponential(
        lengthscales=[3], active_dims=[1])

    # total kernel
    kernel = kernel2 * kernel3 + kernel1
    mean = gpflow.mean_functions.Constant(y_train.mean() - t_train.mean())
    model=gpflow.models.GPR((x_train, y_m_train),
                              kernel = kernel,
                              mean_function = mean)

    # optimize
    # need more iterations as more parameters to minimize over
    model=optimizeGPR(model, 15000)

    set_trainable(
        model.kernel.kernels[0].kernels[0].variance,
        False)  # fix one of the two prefactors

    # predict
    y_pred, y_var = model.predict_f(x_test)
    y_pred += t_test.reshape(-1, 1)

    return mean_squared_error(y_m_test, y_pred)
