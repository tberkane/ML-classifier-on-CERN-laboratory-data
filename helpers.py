import csv
import numpy as np
from typing import Union, List, Tuple

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def compute_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray)-> np.ndarray:
    """compute_gradient: Computes the gradient of the MSE
    @input:
    -ndarray y: array that contains the correct values to be predicted.
    -ndarray tx: matrix that contains the data points. The first column is made of 1s.
    -ndarray w: array containing the parameters of the linear model, from w0 on.
    @output:
    -ndarray grad: array containing the gradient of the MSE function.
    -ndarray err: error vector(en = yn - the predicted n-th value)
    """
    err = y - tx @ w
    grad = -tx.T @ err / len(err)
    return grad, err

def calculate_mse(e: np.ndarray) -> float:
    """calculate_mse: Computes the mean square error given the error vector
    @input:
    -ndarray e: error vector
    @output:
    -float _: mean square error
    """
    return np.mean(e**2) / 2.

def calculate_mae(e: np.ndarray) -> float:
    """calculate_mae: Computes the mean absolute error given the error vector
    @input:
    -ndarray e: error vector
    @output:
    -float _: mean absolute error
    """
    return np.mean(np.absolute(e))/2

def compute_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray, f: str ='mse')-> float:
    """compute_loss: calculate the loss using either MSE or MAE for linear linear.
    @input:
    - ndarray y: array that contains the correct values to be predicted.
    - ndarray tx: matrix that contains the data points. 
    - ndarray w: array containing the linear parameters to test.
    - str f: string indicating which cost function to use; "mse" (default) or "mae".
    @output:
    - float loss: the loss for the given linear parameters.
    """
    e = y - tx @ w
    loss = (calculate_mse(e) if str == 'mse' else calculate_mae(e))
    return loss

def batch_iter(y: np.ndarray, tx: np.ndarray, batch_size: int, num_batches:int=1, shuffle:bool=True):
    """batch_iter: generates a minibatch iterator for a dataset
    @input: 
    - ndarray y: array that contains the correct values to be predicted.
    - ndarray tx: matrix that contains the data points. 
    - int batch_size: size of the desidered batches
    - int num_batches: number of desired batches; default = 1
    - bool shuffle: whether to shuffle the data or not
    @output:
    - iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`
    """
    data_size = len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def sigmoid(t: Union[float, np.ndarray])-> Union[float, np.ndarray]:
    """sigmoid: element-wise applies sigmoid function on t.
    @input
    -float or ndarray t: element onto which the sigmoid needs to be applied.
    @output
    - float or ndarray s(t): element to which the sigmoid function has been applied.
    """
    return 1. / (1 + np.exp(-t))

def lr_calculate_loss(y, tx, w):
    """"lr_calculate_loss: calculates the loss for logistic regression.
    @input:
    - ndarray y: array that contains the correct values to be predicted.
    - ndarray tx: matrix that contains the data points. 
    - ndarray w: array containing the linear parameters to test.
    @output
    - float loss: the loss for the given logistic linear parameters.
    """
    m = tx @ w
    return np.sum(np.log(1 + np.exp(m)) - y*m)

def lr_calculate_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """"lr_calculate_gradient: calculates the of logistic gradient.
    @input:
    - ndarray y: array that contains the correct values to be predicted.
    - ndarray tx: atrix that contains the data points. 
    - ndarray w: array containing the linear parameters to test.
    @output:
    - ndarray gradient: the gradient for the given logistic parameters.
    """
    return tx.T @ (sigmoid(tx @ w) - y)

def lr_gradient_descent_step(y, tx, w, gamma, lambda_):
    """lr_gradient_descent_step: computes one step of gradient descent for the logistic regression.
    @input:
    - ndarray y: array that contains the correct values to be predicted.
    - ndarray tx: matrix that contains the data points. 
    - ndarray w: array containing the linear parameters to test.
    - float gamma: the stepsize.
    - float lambda_: the lambda used for regularization. Default behavior is without regularization.
    @output:
    - ndarray w: the linear parameters.
    -float loss: the loss given w as parameters.
    """
    loss = lr_calculate_loss(y, tx, w) + lambda_/2 * np.power(np.linalg.norm(w), 2)
    gradient = lr_calculate_gradient(y, tx, w) + lambda_ * w
    w -= gamma * gradient
    return loss, w

def standardize(x: np.ndarray, m: float, d: float)-> np.ndarray:
    """standardize: standardizes numpy array
    @input:
    - ndarray x: array to be standardized
    - float m: mean of array
    - float d: standard deviation of array
    @output:
    -ndarray _: stadardized array with mean 0 and deviation 1
    """
    return (x-m)/d

def fill_missing_values(tx: np.ndarray, tx_test: np.ndarray, invalid: int, use_median: bool)-> np.ndarray:
    """fill_missing_values: replaces invalid values with the mean/median of all the values in the cooresponding feature
    @input: 
    - ndarray tx: features
    - ndarray tx_test: features of test data
    - int invalid: invalid data, here -999
    - bool use_median: true if replace with the median, false for mean
    @output:
    - ndarray tx: features with invalid data replaced 
    - ndarray tx_test: features of test data with invalid data replaced 
    """
    for i in range(tx.shape[1]):
        tx[:,i][tx[:,i] == invalid] = np.nan
        tx_test[:,i][tx_test[:,i] == invalid] = np.nan
        new_value = (np.nanmedian(tx[:,i]) if use_median else np.nanmean(tx[:,i]))
        tx[:,i][np.isnan(tx[:,i])] = new_value
        tx_test[:,i][np.isnan(tx_test[:,i])] = new_value
    return tx, tx_test

def remove_outliers(tx: np.ndarray, low_perc: np.ndarray, high_perc: np.ndarray)-> np.ndarray:
    """remove_outliers: removes features of datapoints which are more than limit standard deviations away from the mean of the feature
    @input: 
    - ndarray tx: features
    - ndarray low_perc: values to replace lower outliers with
    - ndarray high_perc: values to replace higher outliers with
    @output: 
    -ndarray tx: features with outliers removed 
    """
    
    for i in range(np.shape(tx)[1]):
        tx[:, i][tx[:, i] < low_perc[i]] = low_perc[i]
        tx[:, i][tx[:, i] > high_perc[i]] = high_perc[i]
    return tx

def poly_expansion(tx: np.ndarray, degree: int, interactions: bool = False)-> np.ndarray:
    """poly_expansion: creates polynomial expansion of features up to a certain degree
    @input: 
    - np.array(N,m) tx: features
    - int degree: degree of polynomial expansion
    - bool interactions: wether to allow mixed products
    @output: np.array(N,m) with new expanded features 
    """
    cols = [c for c in tx.T]
    mix = []
    exp = []
    if degree < 2:
        return tx
    if interactions:
        mix = list([c_1 * c_2 for i_1, c_1 in enumerate(cols[:-1]) for c_2 in cols[i_1+1:]])
    exp = [c_1 ** deg for c_1 in cols for deg in range(2,degree+1)]
    cols.extend(mix)
    cols.extend(exp)
    return np.column_stack(cols)

def get_jet_indexes(x: np.ndarray) -> List[np.ndarray]:
    """get_jet_indexes: gets the masks with the rows belonging to each of the jet number subsets.
    @input:
    - ndarray x: the array to be indexed.
    @output:
    - List[ndarray] indexes: a list of ndarrays (of booleans), each containing the mask of the relative
        subset. `indexes[0]` contains has True values on the rows with jet number 0, `indexes[1]`
        has True values on the rows with jet number 1 and `indexes[2]` has True values on the
        rows with jet numbers 2 and 3.
    """
    return [ x[:, 22] == 0, x[:, 22] == 1, np.bitwise_or(x[:, 22] == 2, x[:, 22] == 3)]

def get_correlated_idx(tx: np.ndarray)-> np.ndarray:
    """get_correlated_idx: returns indexes of highly correlated columns in the matrix
    @input:
    - ndarray tx: matrix in which we search correlated columns
    @output:
    -ndarray ind: indexes of correlated columns
    """
    cor = np.corrcoef(tx.T)
    close = np.isclose(cor, 1)
    tri = np.triu(close, 1)
    ind = np.argwhere(tri)
    return ind

def abs_transform(tx):
    """abs_transform: calculates absolute value of columns whose values are symmetrically distributed around 0, in order to help distinguish between the 2 categories
    @input:
    - ndarray tx: feature matrix
    @output:
    -ndarray: feature matrix with absolute value of columns calculated
    """
    for c in (14, 17, 24, 27):
        tx[:, c] = abs(tx[:, c])
    return tx
