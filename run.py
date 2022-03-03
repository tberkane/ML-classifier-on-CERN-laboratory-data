#!/usr/bin/env python
# coding: utf-8

# Imports
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from implementations import *
from typing import List, Tuple

# ## Load the training data into feature matrix, class labels, and event ids:

DATA_TRAIN_PATH = 'data/train.csv'
DATA_TEST_PATH = 'data/test.csv'
OUTPUT_PATH = 'data/submission.csv'
y, tx, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample=False)
y_test, tx_test, ids_test = load_csv_data(DATA_TEST_PATH)
y = np.expand_dims(y, axis=1)
y_pred=np.zeros(len(y_test))

def pre_processing(tx: np.ndarray, tx_test: np.ndarray, poly_deg: int = 2, gamma: int = 4)-> Tuple[np.ndarray, np.ndarray]:
    """pre_processing: prepares the training and test set, by cleaning and expanding data.
    @input:
    - ndarray tx: matrix that contains the data points for training.
    - ndarray tx_test: matrix that contains the data points for testing.
    - int poly_deg: degree of polynomial expansion
    - int gamma: percentile from which to start clamping outliers on both sides
    @output: 
    - ndarray tx: matrix ready for training.
    - ndarray tx_test: matrix ready for testing.
    """
    #Removing outliers
    low_perc = np.percentile(tx, gamma, axis = 0)
    high_perc = np.percentile(tx, 100 - gamma, axis = 0) 
    tx = remove_outliers(tx, low_perc, high_perc)
    tx_test = remove_outliers(tx_test, low_perc, high_perc)
    
    #Standardization of values
    mean = np.mean(tx, axis = 0)
    deviation = np.std(tx, axis = 0)
    tx = standardize(tx, mean, deviation)
    tx_test = standardize(tx_test, mean, deviation)
    
    #removing columns that are exactly proportional
    ind_cor = get_correlated_idx(tx)
    tx = np.delete(tx, ind_cor[:,], axis=1)
    tx_test = np.delete(tx_test, ind_cor[:,], axis = 1)

    #polynomial expansion
    tx = poly_expansion(tx, degree = poly_deg, interactions = True)
    tx_test = poly_expansion(tx_test, degree = poly_deg, interactions = True)
    tx = np.c_[np.ones((tx.shape[0], 1)), tx]
    tx_test = np.c_[np.ones((tx_test.shape[0], 1)), tx_test]

    return tx, tx_test 

def get_jet_indexes(x: np.ndarray)->List[np.ndarray]:
    """get_jet_indexes: gets the masks with the rows belonging to each of the jet number subsets.
    @input:
    - x: the array to be indexed.
    @output:
    -List[ndarray] indexes: list of ndarrays, each containins the mask of the relative subset.
    """
    return [x[:, 22] == 0, x[:, 22] == 1, x[:, 22] > 1]

def jet_processing(x: np.ndarray, mask: np.ndarray, delete_null_cols: bool = False)->List[np.ndarray]:
    """jet_processing: splits data according to mask
    @input:
    - ndarray x: the data array to be processed.
    - List[ndarray] mask: mask to identify subsets
    - bool delete_null_cols: delete null columns in the result
    @output:
    -List[ndarray] indexes: list of ndarrays, each containins the relative subset.
    """
    #indices of columns to drop for each set 
    col_indexes = [[4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29],  [4, 5, 6, 12, 22, 26, 27, 28], [22]]
    # indices of heavy tail columns for which to calculate log, for each set
    log_idx = [[0, 1, 2, 3, 8, 9, 13, 16, 19, 21], [0, 1, 2, 3, 8, 9, 13, 16, 19, 21, 23, 29], [0, 1, 2, 3, 8, 9, 13, 16, 19, 21, 23, 26, 29]]
    
    xs = []
    for i in range(len(mask)):
        x_i = x[mask[i]]
        if delete_null_cols:
            x_log = np.log1p(x_i[:, log_idx[i]])
            x_i = np.delete(x_i, col_indexes[i], 1)
            x_i = np.hstack((x_i, x_log))
        xs.append(x_i)
    return xs

def prepare_data(tx: np.ndarray, tx_test: np.ndarray, y: np.ndarray, process: bool = True)-> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """prepare_data: splits data into subsets and processes each one before training
    @input:
    - ndarray tx: training data
    - ndarray tx_test: testing data
    - ndarray y: labels of training data
    - bool process: wether to pre_process the data or not
    @output:
    - List[ndarray] txs: each node contains the relative subset ready for model training.
    - List[ndarray] txs_test: each node contains the relative subset ready for calculating labels.
    - List[ndarray] txs: each node contains the relative subset of labels.
    """
    print("Preparing data ...")
    #Filling missing values
    tx, tx_test = fill_missing_values(tx, tx_test, invalid = -999, use_median=True)
    
    #Absolute value of symmetric features 
    tx = abs_transform(tx)
    tx_test = abs_transform(tx_test)
    
    #Split data into subsets
    mask = get_jet_indexes(tx)
    txs = jet_processing(tx, mask, True)
    ys = jet_processing(y, mask, False)
    mask_test = get_jet_indexes(tx_test)
    txs_test = jet_processing(tx_test, mask_test, True)
    degrees = [7, 7, 5]
    gammas = [3, 6, 8]

    #Clean the subsets
    if process:
        for i in range(len(txs)):
            txs[i], txs_test[i] = pre_processing(txs[i], txs_test[i], poly_deg = degrees[i], gamma=gammas[i])

    return txs, ys, txs_test    
        

#Training the model
def train_model(txs: np.ndarray, ys: np.ndarray)-> Tuple[List[np.ndarray], List[np.ndarray]]:
    """train_model: trains the binary classifier
    @input:
    - ndarray txs: training data split into three subsets
    - ndarray y: labels of training data split into three subsets
    @output:
    - List[ndarray] ws: weights of each subsets.
    """
    print("Training model ...")
    ws = []
    lambdas = [1e-5, 1e-5, 0.0001]
    for i in range(len(txs)):
        #weights, loss = reg_logistic_regression(ys[i], txs[i], lambda_= 1, initial_w=np.zeros((txs[i].shape[1], 1)), max_iters=1000, gamma=0.01)
        weights, _ = ridge_regression(ys[i], txs[i], lambda_= lambdas[i])
        ws.append(weights)
    return ws


# Cross-Validation
def build_k_indices(num_row: int, k_fold: int, seed: int)-> np.ndarray:
    """build_k_indices: build k indices for k-fold.
    @input:
    - int num_row: total number of rows
    - int k_fold: number of folds to divide the set into
    - int seed: seed for generation of random sets
    @output:
    - k_indices: array of indices
    """
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y: np.ndarray, tx: np.ndarray, k_indices: np.ndarray, k: int, lambda_: float, deg:int, gamma: int)-> float:
    """cross_validation: computes cross validation
    @input: 
    - ndarray y: labels of dataset
    - ndarray tx: dataset
    - ndarray k_indices: array of indices for to test the model on
    - int k: iteration for cross validation
    - float lambda_: hyperparamter for training model
    - int deg: degree of polynomial expansion
    - int gamma: percentile from which to start clamping outliers on both sides
    @output:
    - float a: accuracy of cross validation
    """
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = tx[te_indice]
    x_tr = tx[tr_indice]
    x_tr, x_te = pre_processing(x_tr, x_te, poly_deg = deg, gamma=gamma)
    #w, _ = reg_logistic_regression(y_tr, x_tr, 1e-4, initial_w = np.zeros((x_tr.shape[1], 1)), max_iters = 500, gamma = 1e-4)
    w, _ = ridge_regression(y_tr, x_tr, lambda_)
    return (y_te == predict_labels(w, x_te)).mean()

def cross_validation_grid_search(txs: List[np.ndarray], ys: List[np.ndarray]):
    """cross_validation_grid_search: runs cross validation on the data with different values of hyperparameters to compare accuracy
    @input:
    -List[ndarray] txs: subsets of train dataset
    -List[ndarray] ys: labels of the different subsets
    """
    seed = 847
    k_fold = 3
    lambdas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    degrees = range(2, 8)
    gammas = range(0, 11)
    # split data in k fold
    k_indices = []
    for i in range(len(txs)):
        k_indices.append(build_k_indices(ys[i].shape[0], k_fold, seed))
    # cross validation
    for i in range(len(txs)):
        max_acc = 0
        print(f"*Set {i}")
        for l in lambdas:
            for d in degrees:
                for g in gammas:
                    pred_pcts = []
                    for k in range(k_fold):
                        pred_pct = cross_validation(ys[i], txs[i], k_indices[i], k, l, d, g)
                        pred_pcts.append(pred_pct)
                    pct = np.mean(pred_pcts)
                    if pct > max_acc:
                        max_acc = pct
                        #prints hypterparameters values if a new best accuracy is found
                        print(f">>>>Set {i}/lamdba={l}/deg={d}/gamma={g}/ACC={np.around(pct, 4)}")

def cross_validation_demo(txs: List[np.ndarray], ys: List[np.ndarray]):
    """cross_validation_demo: runs cross validation on the data and prints accuracy
    @input:
    -List[ndarray] txs: subsets of train dataset
    -List[ndarray] ys: labels of the different subsets
    """
    seed = 847
    k_fold = 3
    lambdas = [1e-5, 1e-5, 0.0001]
    degrees = [7, 7, 5]
    gammas = [3, 6, 8]
    # split data in k fold
    k_indices = []
    for i in range(len(txs)):
        k_indices.append(build_k_indices(ys[i].shape[0], k_fold, seed))
    # cross validation
    for i in range(len(txs)):        
        pred_pcts = []
        for k in range(k_fold):
            pred_pct = cross_validation(ys[i], txs[i], k_indices[i], k, lambdas[i], degrees[i], gammas[i])
            pred_pcts.append(pred_pct)
        print(f">>>>Set {i}/lamdba={lambdas[i]}/deg={degrees[i]}/gamma={gammas[i]}/ACC={np.around(np.mean(pred_pcts), 4)}")


def generate_predictions(txs_test: List[np.ndarray], ws: List[np.ndarray]):
    """generate_predictions: generates predictions and save ouput in csv format for submission
    @input: 
    - List[ndarray] txs_test: subsets of test dataset
    - List[ndarray] ws: weights of the different subsets
    """
    mask_test = get_jet_indexes(tx_test)
    for j in range(len(txs_test)):
            y_pred[mask_test[j]] = [i[0] for i in predict_labels(ws[j], txs_test[j])]
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

def main():
    submission = False
    if submission: 
        txs, ys, txs_test  = prepare_data(tx, tx_test, y)
        weights = train_model(txs, ys)
        generate_predictions(txs_test, weights)
    else:
        txs, ys, txs_test  = prepare_data(tx, tx_test, y, False)
        cross_validation_demo(txs, ys)
    return

main()






