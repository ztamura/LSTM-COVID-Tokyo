"""
Contains helper functions to preprocess data for LSTM

Author: Zen Tamura

Dependencies:
    Python 3.7.6
    Numpy 1.19.1
    Pandas 1.1.5
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def getdata(filename):
    """
    converts data in text file to 2d numpy array
    Args: 
        filename: str
    Returns: 
        data: 2d numpy array, with shape (N, 1)  (N: # inputs)
    """
    with open(filename) as fileobj:
        dlist = fileobj.readlines()
    datalist = dlist[::-1]
    data = np.array([int(s) for s in datalist])
    N = len(data)
    data = data.reshape((N, 1))
    #print("length: ", len(data))
    #print("data shape:", data.shape)
    errormessage = "shape for data incorrect; expected" + str(data.shape) + ", got (1, " + str(len(data)) + ")"
    assert data.shape == (N, 1), errormessage
    return data

def MinMaxScaling(data):
    """
    Performs min-max scaling (scales all data to [-1, 1])
    Args:
        data: numpy array of shape (N, 1)
    Returns:
        scaled: numpy array of shape (N, 1)
    """
    N = data.shape[0]
    scaled = np.zeros((N, 1))
    x_min = np.amin(data)
    x_max = np.amax(data)
    for i in range(N):
        scaled[i, 0] = (data[i, 0] - x_min) / (x_max - x_min)
    assert scaled.shape == data.shape
    return scaled, x_min, x_max

def plot_MinMaxScaling(data):
    """
    helper function that plots min max scaling results for verification
    Args:
        data: numpy array of shape (N, 1)
    Returns:
        None
    Generates plot
    """
    fig, ax = plt.subplots()
    ax.plot(data.flatten())
    plt.show()

def gen_input(data, nx):
    """
    Generates input matrix with moving window of size nx
    Args:
        data: numpy array of shape (N, 1)
        nx: int, size of input at each timestep
    Returns:
        x: numpy array of shape (nx, Tx)
    """
    assert data.shape[1] == 1
    N = data.shape[0]

    # added -1 to make Tx == Ty
    # Tx is number of LSTM cells in model (= number of total time steps)
    Tx = N - nx + 1 - 1
    #print(Tx)
    assert type(Tx) == int
    x = np.zeros((nx, Tx))
    for i in range(Tx):
        x[:, i] = data[i:(i + nx), 0]
    #print(x.shape)
    assert x.shape == (nx, Tx)
    return x

def gen_moving_window(data, window, nx):
    """
    Generates moving window
    Args:
        data: numpy array of shape (nx, 1)
        window: int, window size
        nx: int, size of input at each time step
    Returns:
        numpy array of shape (nx, Tx)
    """
    assert data.shape[1] == 1
    #nx = data.shape[0]

    # Tx is number of LSTM cells (number of time steps)
    Tx = nx - window + 1
    assert type(Tx) == int
    x = np.zeros((nx, Tx))
    for i in range(Tx):
        x[:, i] = data[i:(i + window), :]
    assert x.shape == (nx, Tx)
    return x

def sigmoid(x):
    """
    clips x to [-700, 700] for numerically stable implementation of
    sigmoid function: 1/(1 + exp(-x))
    Args:
        x: numpy array
    Returns:
        numpy array
    """
    clipped = np.clip(x, -700, 700)
    return 1/ (1 + np.exp(-clipped))

def relu(x):
    """
    Rectified linear unit, where ReLU(x) = max(0, x)
    Args:
        x: numpy array
    """
    return np.maximum(0, x)

def gen_labels(data, x, ny):
    """
    calculates labels 
    Args:
        data: data vector, numpy array of shape (N, 1)
        x: input matrix, numpy array of shape (nx, Tx)
    Returns: 
        y: label matrix, numpy array of shape (ny, Tx)
    """
    nx, Tx = x.shape
    y = np.zeros((ny, Tx))
    for i in range(Tx):
        y[:, i] = data[(nx + i): (i +nx + ny), 0]
    assert y.shape == (ny, Tx)
    return y

def checklist(lst):
    """
    checks if elements in given list are all the same
    Args:
        lst: list
    Returns:
        res: boolean
    """
    res = all(element == lst[0] for element in lst)
    return res
