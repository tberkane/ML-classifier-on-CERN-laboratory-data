import numpy as np
from helpers import *
from typing import Tuple

def least_squares_GD(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float)-> Tuple[np.ndarray, float]:
    """least_squares_GD: gradient descent algorithm. Uses MSE loss function.
    @input:
    - ndarray y: array that contains the correct values to be predicted.
    - ndarray tx: matrix that contains the data points.
    - ndarray initial_w: array containing the linear parameters to start from.
    - int max_iters: the maximum number of iterations to be done.
    - float gamma: the stepsize of the GD
    @output: 
    - np.ndarray w: the linear parameters.
    - float loss: the loss given w as parameters.
    """
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        grad, _ = compute_gradient(y, tx, w)
        w = w - gamma * grad
        loss = compute_loss(y, tx, w)
    return w, loss

def least_squares_SGD(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float)-> Tuple[np.ndarray, float]:
    """least_squares_SGD: stochastic gradient descent algorithm. Uses MSE loss function.
    @input:
    - ndarray y: array that contains the correct values to be predicted.
    - ndarray tx: matrix that contains the data points.
    - ndarray initial_w: array containing the linear parameters to start from.
    - int max_iters: the maximum number of iterations to be done.
    - float gamma: the stepsize of the GD
    @output: 
    - np.ndarray w: the linear parameters.
    - float loss: the loss given w as parameters.
    """
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad, _ = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
        loss = compute_loss(y, tx, w)
    return w, loss

def least_squares(y: np.ndarray, tx: np.ndarray)-> Tuple[np.ndarray, float]:
    """least_squares: computes the least squares solution.
    @input: 
    - ndarray y: array that contains the correct values to be predicted.
    - ndarray tx: matrix that contains the data points. 
    @output:
    - ndarray w: the linear parameters.
    - float loss: the loss given w as parameters.
    """
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y: np.ndarray, tx: np.ndarray, lambda_: float)-> Tuple[np.ndarray, float]:
    """ridge_regression: computes ridge linear with the given `lambda_`.
    @input: 
    - ndarray y: array that contains the correct values to be predicted.
    - ndarray tx: matrix that contains the data points. 
    - float lambda_: lambda regularization parameter
    @output: 
    - ndarray w: the linear parameters.
    - float loss: the loss given w as parameters.
    """
    lambda_p = 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1])
    a = tx.T @ tx + lambda_p
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss

def logistic_regression(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float)-> Tuple[np.ndarray, float]:
    """ logistic_regression: computes the parameters for the logistic linear.
    @input:
    - ndarray y: array that contains the correct values to be predicted.
    - ndarray tx: matrix that contains the data points. 
    - ndarray initial_w: array containing the linear parameters to start with.
    - int max_iters: the maximum number of iterations to do.
    - float gamma: gradient descent stepsize
    @output
    -ndarray w: the linear parameters.
    -float loss: the loss given w as parameters.
    """
    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)


def reg_logistic_regression(y: np.ndarray, tx: np.ndarray, lambda_: float, initial_w: np.ndarray, max_iters: int, gamma: float)-> Tuple[np.ndarray, float]:
    """reg_logistic_regression: does the regularized logistic linear.
    @input:
    - ndarray y: array that contains the correct values to be predicted.
    - ndarray tx: matrix that contains the data points. 
    - float lambda_: the lambda used for regularization.
    - ndarray initial_w: array containing the linear parameters to start with.
    - int max_iters: the maximum number of iterations to do.
    - float gamma: gradient descent stepsize.
    @output
    -ndarray w: the linear parameters.
    -float loss: the loss given w as parameters.
    """
    threshold = 1e-8
    losses = []
    w = initial_w
    for iter in range(max_iters):
        loss, w = lr_gradient_descent_step(y, tx, w, gamma, lambda_)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]