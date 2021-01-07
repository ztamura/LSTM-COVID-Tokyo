"""
Full scratch Python implementation of a Long Short Term Memory (LSTM) model
DOES NOT USE deep learning libraries such as Tensorflow or Keras

Author: Zen Tamura

Dependencies:
    Python 3.7.6
    Numpy 1.19.1
    Pandas 1.1.5
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def getdata(filename):
    """
    converts data in text file to  2d numpy array
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

class LSTM:
    """LSTM model implementation"""
    def __init__(self, na, x, lr = 0.001, epochs = 100, optimizer = "adam", beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        """
        Constructor for LSTM class
        Args:
            na: int, dimension of hidden state at each time step
            x: input matrix, numpy array
            lr: learning rate, set to 0.001
            epochs: int
            optimizer: "adam" or "gradient descent", "adam" is preferred
            beta1: float, Adam optimizer parameter
            beta2: float, Adam optimizer parameter
            epsilon: float, Adam optimizer parameter
        """
        self.na = na
        self.x = x
        self.nx = x.shape[0]
        self.Tx = x.shape[1]
        self.ny = 1
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer

        # Adam hyperparameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # initializing gradients
        self.gradients = {}
        self.gradients["dWy"] = np.zeros((self.ny, self.na))
        self.gradients["dby"] = np.zeros((self.ny, 1))
        self.gradients["dWo"] = np.zeros((self.na, (self.na + self.nx)))
        self.gradients["dbo"] = np.zeros((self.na, 1))
        self.gradients["dWc"] = np.zeros((self.na, (self.na + self.nx)))
        self.gradients["dbc"] = np.zeros((self.na, 1))
        self.gradients["dWi"] = np.zeros((self.na, (self.na + self.nx)))
        self.gradients["dbi"] = np.zeros((self.na, 1))
        self.gradients["dWf"] = np.zeros((self.na, (self.na + self.nx)))
        self.gradients["dbf"] = np.zeros((self.na, 1))
        #self.gradients["dxt"] = np.zeros((self.nx, 1))

        if optimizer == "adam":
            self.adam_params = {}
            for key in self.gradients:
                self.adam_params["V" + key] = np.zeros_like(self.gradients[key])
                self.adam_params["S" + key] = np.zeros_like(self.gradients[key])

        if self.optimizer != "gradient descent" and self.optimizer != "adam":
            print("optimizer not spelled correctly")
    
    def initialize_params(self, method = "random"):
        """
        Initializes weights and biases
        Args:
            method: "random" or "xavier"
                    "random": Randomly initializes weights to small values sampled from 
                                standard normal distribution N(0,1)
                    "xavier": uses Xavier initialization
        
        In either method, biases are initialized to zero
        """
        self.params = {}
        
        if method == "random":
            #np.random.seed(1)
            #forget gate
            self.params["Wf"] = np.random.randn(self.na, (self.na + self.nx)) * 0.01
            self.params["bf"] = np.zeros((self.na, 1))
            # input gate
            self.params["Wi"] = np.random.randn(self.na, (self.na + self.nx)) * 0.01
            self.params["bi"] = np.zeros((self.na, 1))
            # output gate
            self.params["Wo"] = np.random.randn(self.na, (self.na + self.nx)) * 0.01
            self.params["bo"] = np.zeros((self.na, 1))
            # cell state
            self.params["Wc"] = np.random.randn(self.na, (self.na + self.nx)) * 0.01
            self.params["bc"] = np.zeros((self.na, 1))
            # output layer
            self.params["Wy"] = np.random.randn(self.ny, self.na) * 0.01
            self.params["by"] = np.zeros((self.ny, 1))

        if method == "xavier":
            self.params["Wf"] = np.random.uniform(-np.sqrt(3), np.sqrt(3), (self.na, (self.na + self.nx)))
            self.params["bf"] = np.zeros((self.na, 1))

            self.params["Wi"] = np.random.uniform(-np.sqrt(3), np.sqrt(3), (self.na, (self.na + self.nx)))
            self.params["bi"] = np.zeros((self.na, 1))

            self.params["Wo"] = np.random.uniform(-np.sqrt(3), np.sqrt(3), (self.na, (self.na + self.nx)))
            self.params["bo"] = np.zeros((self.na, 1))

            self.params["Wc"] = np.random.uniform(-np.sqrt(3), np.sqrt(3), (self.na, (self.na + self.nx)))
            self.params["bc"] = np.zeros((self.na, 1))

            self.params["Wy"] = np.random.uniform(-np.sqrt(3), np.sqrt(3), (self.ny, self.na))
            self.params["by"] = np.zeros((self.ny, 1))

    def clip_grads(self, minval, maxval):
        """
        clips gradients to [minval, maxval] to prevent exploding gradients
        Args:
            minval: int or float
            maxval: int or float
        Returns:
            None
        """
        for key in self.gradients.keys():
            np.clip(self.gradients[key], minval, maxval, out=self.gradients[key])
        return
    
    def reset_grads(self):
        """
        resets gradients to zeros after backpropagating through all LSTM cells
        Args: None
        Returns: None
        """
        for key in self.gradients:
            self.gradients[key].fill(0)
        return
        
    def step_forward(self, xt, a_prev, c_prev):
        """
        Performs one step of forward propagation in the LSTM
        Args:
            xt: input for time step t, numpy array of shape (nx, 1)
            a_prev: hidden state for time step t-1, numpy array of shape (na, 1)
            c_prev: cell state for time step t-1, numpy array of shape (na, 1)
        Returns:
            a_next: next hidden state, numpy array of size (na, 1)
            c_next: next cell state, numpy array of size (na, 1)
            yt_pred: prediction at time step t, numpy array of size (ny, 1)
            cache: tuple containing vals needed for backward pass
                (a_next, c_next, a_prev, c_prev, lf, ft, li, it, lo, ot, lc, cct, vt, xt)
                where:
                    lf: linear part of forget gate, numpy array of shape (na, 1)
                    ft: forget gate, numpy array of size (na, 1)
                    li: linear part of input gate, numpy array of shape (na, 1)
                    it: input gate, numpy array of size (na, 1)
                    lo: linear part of output gate, numpy array of shape (na, 1)
                    ot: output gate, numpy array of size (na, 1)
                    lc: linear part of cell state, numpy array of shape (na, 1)
                    cct: candidate value for cell state (c-tilde), numpy array of size (na, 1)
                    vt: linear part of predicted value, numpy array of shape (ny, 1)
                    xt: input value for time step t, numpy array of size (nx, 1)
        """
        zt = np.concatenate((a_prev, xt), axis = 0)
        assert zt.shape == ((self.na + self.nx), 1)

        # c tilde
        #cct = np.tanh(np.matmul(self.params["Wc"], zt) + self.params["bc"])
        lc = np.matmul(self.params["Wc"], zt) + self.params["bc"]
        cct = np.tanh(lc)
        assert lc.shape == (self.na, 1)
        assert cct.shape == (self.na, 1)
        
        # forget gate
        #ft = sigmoid(np.matmul(self.params["Wf"], zt) + self.params["bf"])
        lf = np.matmul(self.params["Wf"], zt) + self.params["bf"]
        ft = sigmoid(lf)
        assert lf.shape == (self.na, 1)
        assert ft.shape == (self.na, 1)

        # input gate
        #it = sigmoid(np.matmul(self.params["Wi"], zt) + self.params["bi"])
        li = np.matmul(self.params["Wi"], zt) + self.params["bi"]
        it = sigmoid(li)
        assert li.shape == (self.na, 1)
        assert it.shape == (self.na, 1)

        # output gate
        #ot = sigmoid(np.matmul(self.params["Wo"], zt) + self.params["bo"])
        lo = np.matmul(self.params["Wo"], zt) + self.params["bo"]
        ot = sigmoid(lo)
        assert lo.shape == (self.na, 1)
        assert ot.shape == (self.na, 1)

        # cell state for time step t
        c_next = it * cct + ft * c_prev
        assert c_next.shape == (self.na, 1)

        # hidden state for time step t
        a_next = ot * np.tanh(c_next)
        assert a_next.shape == (self.na, 1)

        # prediction for time step t
        #yt_pred = relu(np.matmul(self.params["Wy"], a_next) + self.params["by"])
        vt = np.matmul(self.params["Wy"], a_next) + self.params["by"]
        yt_pred = relu(vt)
        assert vt.shape == (self.ny, 1)
        assert yt_pred.shape == (self.ny, 1)

        cache = (a_next, c_next, a_prev, c_prev, lf, ft, li, it, lo, ot, lc, cct, vt, xt)

        return a_next, c_next, yt_pred, cache
    
    def forward(self, a0, y):
        """
        Implements forward propagation
        Args:
            a0: initial hidden state, numpy array of shape (na, 1)
            y: true vals for y for all time steps, numpy array of shape (ny, Tx)
        Returns:
            a: hidden states for all time steps, numpy array of shape (na, Tx)
            c: cell states for all time steps, numpy array of shape (na, Tx)
            y_pred: predictions for all time steps, numpy array of shape (ny, Tx)
            caches: list of all caches (See step_forward method (function)))
            loss: float, Mean squared error (MSE)
        """
        caches = []
        # Initialize a, c, y as 0
        a = np.zeros((self.na, self.Tx))
        #print(a.shape)
        y_pred = np.zeros((self.ny, self.Tx))
        c = np.zeros((self.na, self.Tx))

        a_next = a0
        c_next = np.zeros((self.na, 1))

        loss = 0
        for t in range(self.Tx):
            #print("t:", t)
            xt = self.x[:, t]
            xt = np.reshape(xt, (self.nx, 1))
            #print(xt)
            a_next, c_next, yt_pred, cache = self.step_forward(xt, a_next, c_next)
            (_, _a, a_prev, c_prev, lf, ft, li, it, lo, ot, lc, cct, vt, xt) = cache
            #print(a[:, t].shape)
            a[:, t] = a_next.flatten()
            c[:, t] = c_next.flatten()
            y_pred[:, t] = yt_pred.flatten()
            caches.append(cache)
            yt = y[:, t]
            # calculating loss
            loss += (yt_pred - yt) **2 
        loss /= 2 * self.Tx
        loss = float(loss)
        
        assert a.shape == (self.na, self.Tx)
        assert c.shape == (self.na, self.Tx)
        assert y_pred.shape == (self.ny, self.Tx)
        
        #print(a.shape)
        return a, c, y_pred, caches, loss

    def initialize_gradients(self):
        """
        Initializes gradients to 0 for Adam optimization
        """
        # self.gradients = {}
        # self.gradients["dWy"] = np.zeros((self.ny, self.na))
        # self.gradients["dby"] = np.zeros((self.ny, 1))
        # self.gradients["dWo"] = np.zeros((self.na, (self.na + self.nx)))
        # self.gradients["dbo"] = np.zeros((self.na, 1))
        # self.gradients["dWc"] = np.zeros((self.na, (self.na + self.nx)))
        # self.gradients["dbc"] = np.zeros((self.na, 1))
        # self.gradients["dWi"] = np.zeros((self.na, (self.na + self.nx)))
        # self.gradients["dbi"] = np.zeros((self.na, 1))
        # self.gradients["dWf"] = np.zeros((self.na, (self.na + self.nx)))
        # self.gradients["dbf"] = np.zeros((self.na, 1))
        # self.gradients["dxt"] = np.zeros((self.nx, 1))
        pass
    
    def step_backward(self, cache, yt, yt_pred):
        """
        Implements a single backpropagation step 
        Args:
            cache: cache storing information from forward pass
                (a_next, c_next, a_prev, c_prev, lf, ft, li, it, lo, ot, lc, cct, vt, xt)
            yt: true vals for y time step t, numpy array of shape (ny, 1)
            yt_pred: predicted vals at time step t, numpy array of shape (ny, 1)
        Returns:
            da_prev: dJ/da_t-1, numpy array of shape (na, 1)
            dc_prev: dJ/dc_t-1, numpy array of shape (na, 1)
        """
        #self.gradients = {}
        (a_next, c_next, a_prev, c_prev, lf, ft, li, it, lo, ot, lc, cct, vt, xt) = cache
        zt = np.concatenate((a_prev, xt), axis = 0)
        assert zt.shape == ((self.na + self.nx), 1)

        dvt = np.zeros((self.ny, 1))
        # predictions
        if vt >= 0:
            dvt[0, 0] = (yt_pred - yt) * (1 - yt) / (2 * self.Tx)
        else:
            dvt[0, 0] = (yt_pred - yt) * (-yt) / (2 * self.Tx)
        
        dWy = np.matmul(dvt, a_next.T)
        dby = dvt

        #print("Expected shape: " + "(" + str(self.ny) +", "+ str(1), ")")
        #print("Actual shape: ", str(dvt.shape))
        assert dvt.shape == (self.ny, 1)
        assert dWy.shape == (self.ny, self.na)
        assert dby.shape == (self.ny, 1)
        #self.gradients["dvt"] = dvt
        self.gradients["dWy"] += dWy
        self.gradients["dby"] += dby

        # hidden state
        da_next = np.matmul(self.params["Wy"].T, dvt)
        assert da_next.shape == (self.na, 1)
        
        # output gate
        dot = da_next * np.tanh(c_next)
        assert dot.shape == (self.na, 1)
        dlo = dot * lo * (1 - lo)
        assert dlo.shape == (self.na, 1)
        dWo = np.matmul(dlo, zt.T)
        assert dWo.shape == (self.na, (self.na + self.nx))
        dbo = dlo
        assert dbo.shape == (self.na, 1)

        self.gradients["dWo"] += dWo
        self.gradients["dbo"] += dbo

        # cell state
        dc_next = da_next * ot * (1 - (np.tanh(c_next)**2))
        assert dc_next.shape == (self.na, 1)
        dcct = dc_next * it
        assert dcct.shape == (self.na, 1)
        dlc = dcct * (1 - cct **2)
        assert dlc.shape == (self.na, 1)
        dWc = np.matmul(dlc, zt.T)
        assert dWc.shape == (self.na, (self.na + self.nx))
        dbc = dlc
        assert dbc.shape == (self.na, 1)

        self.gradients["dWc"] += dWc
        self.gradients["dbc"] += dbc

        # input (update) gate
        dit = dc_next * cct
        assert dit.shape == (self.na, 1)
        dli = dit * li * (1 - li)
        assert dli.shape == (self.na, 1)
        dWi = np.matmul(dli, zt.T)
        assert dWi.shape == (self.na, (self.na + self.nx))
        dbi = dli
        assert dbi.shape == (self.na, 1)

        self.gradients["dWi"] += dWi
        self.gradients["dbi"] += dbi

        # forget gate
        dft = dc_next * c_prev
        assert dft.shape == (self.na, 1)
        dlf = dft * lf * (1 - lf)
        assert dlf.shape == (self.na, 1)
        dWf = np.matmul(dlf, zt.T)
        assert dWf.shape == (self.na, (self.na + self.nx))
        dbf = dlf
        assert dbf.shape == (self.na, 1)

        self.gradients["dWf"] += dWf
        self.gradients["dbf"] += dbf

        # inputs
        dzt = np.matmul(self.params["Wf"].T, dlf) + np.matmul(self.params["Wi"].T, dli) \
            + np.matmul(self.params["Wo"].T, dlo) + np.matmul(self.params["Wc"].T, dlc)
        
        assert dzt.shape == ((self.na + self.nx), 1)
        da_prev = dzt[:self.na, :]
        assert da_prev.shape == (self.na, 1)
        dxt = dzt[self.na:, :]
        assert dxt.shape == (self.nx, 1)
        dc_prev = dc_next * ft
        assert dc_prev.shape == (self.na, 1)

        #self.gradients["dxt"] += dxt

        return da_prev, dc_prev

    def backward(self, caches, y, y_pred):
        """
        Implements full backpropagation
        Args:
            caches: cache storing info from forward pass
            y: true vals for y for all time steps, numpy array of shape (ny, Tx)
            y_pred: predicted values for all time steps, numpy array of shape (ny, Tx)
        Returns:
            None
        """
        #(a1, c1, a0, c0, lf1, f1, li1, i1, lo1, o1, lc, cc1, v1, x1) = caches[0]
        #self.reset_grads()
        for t in reversed(range(self.Tx)):
            self.step_backward(caches[t], y[:, t], y_pred[:, t])
        return None

    def gradient_descent(self):
        """
        updates parameters according to batch gradient descent
        """
        for key in self.params.keys():
            self.params[key] -= self.lr * self.gradients["d" + key]
    
    def adam(self, t):
        """
        performs one step of adam optimization algorithm
        Args:
            t: iteration number
        """
        for key in self.params.keys():
            self.adam_params["V" + "d" +key] = self.beta1 * self.adam_params["V" +"d" +key] + \
                (1-self.beta1) * self.gradients["d"+key]
            self.adam_params["S" + "d"+key] = self.beta2 * self.adam_params["S" +"d"+ key] + \
                (1-self.beta2) * (self.gradients["d"+key] ** 2)
            
            V_corrected = self.adam_params["V" + "d"+key] / (1 - self.beta1 ** t)
            S_corrected = self.adam_params["S" + "d"+key] / (1 - self.beta2 ** t)
            self.params[key] -= self.lr * V_corrected / (np.sqrt(S_corrected) + self.epsilon)

    def train(self, y, patience, max_epochs = 10000):
        """
        trains LSTM
        Args:
            y: ground truth labels, numpy array of shape (ny, Tx)
            patience: int, how many epochs to wait before quitting training
            max_epochs: max number of iterations
        """
        self.initialize_params(method = "random")
        self.initialize_gradients()

        a0 = np.zeros((self.na, 1))
        losses = []
        best_y_pred = np.zeros((self.ny, self.Tx))
        # for epoch in range(self.epochs):
        #     a, c, y_pred, caches, loss = self.forward(a0, y)
        #     losses.append(loss)
        #     self.backward(caches, y, y_pred)
        #     self.gradient_descent()
        #     if epoch == (self.epochs - 1):
        #         best_y_pred[:, :] = y_pred
        #     self.clip_grads(-10, 10)

        if self.optimizer == "gradient descent":

            loss = 1
            # bound of 0.208 w/ seed = 6 works 
            bound = 0.0015
            epoch = 0
            while loss > bound:
                a, c, y_pred, caches, loss = self.forward(a0, y)
                losses.append(loss)
                self.backward(caches, y, y_pred)
                self.clip_grads(-1, 1)
                self.gradient_descent()
                epoch += 1
                if epoch % 100 == 0:
                    #print(self.params)
                    print("Finished ", str(epoch), "epochs")
                    #print(self.gradients)
                    print("Loss: ", str(round(loss, 6)))
                    print("-------------------")
                    #self.reset_grads()
                self.reset_grads()
                if epoch % 200 == 0:
                    pass
                    #print(self.gradients)
                    #self.reset_grads()
            print("Converged in", str(epoch), "epochs")
            print("==================")
            #self.reset_grads()

        if self.optimizer == "adam":
            loss = 1
            bound = 0.004
            epoch = 0
            toll = 1
            while toll > 7e-9:
                epoch += 1
                a, c, y_pred, caches, loss = self.forward(a0, y)
                losses.append(loss)
                self.backward(caches, y, y_pred)
                self.clip_grads(-1, 1)
                self.adam(epoch)
                self.reset_grads()
                if epoch % 100 == 0:
                    #print(self.adam_params)
                    #print(self.params)
                    print("Finished ", str(epoch), "epochs")
                    print("Loss: ", str(round(loss, 6)))
                    print("-------------------")
                    #self.reset_grads()

                # checks for convergence
                if epoch <= 200:
                    toll = 1
                elif epoch > 200:
                    latest_losses = losses[-patience:]
                    prev_loss = losses[-2]
                    toll = np.abs(loss - prev_loss)
                    #print("Delta Loss: {}".format(toll))
                    convergence = checklist(latest_losses)
                    # fix convergence
                    # same losses is not convergence
                    # we want to check for same weights
                    if convergence:
                        break

                if epoch % 200 == 0:
                    pass
                    #self.reset_grads()
            print("Converged in", str(epoch), "epochs")
            #print("==================")
        
        return losses, self.params, y_pred, epoch
        
    def test(self, x_test, y_test):
        """
        tests learned parameters with test set
        Args:
            x_test: test set, numpy array
            y_test: test set, numpy array
        """
        a0 = np.zeros((self.na, 1))
        test_Tx = x_test.shape[1]
        a = np.zeros((self.na, test_Tx))
        y_pred = np.zeros((self.ny, test_Tx))
        c = np.zeros((self.na, test_Tx))
        a_next = a0
        c_next = np.zeros((self.na, 1))
        loss = 0
        for t in range(test_Tx):
            xt = x_test[:, t]
            xt = np.reshape(xt, (self.nx, 1))
            a_next, c_next, yt_pred, cache = self.step_forward(xt, a_next, c_next)
            a[:, t] = a_next.flatten()
            c[:, t] = c_next.flatten()
            y_pred[:, t] = yt_pred.flatten()
            yt = y_test[:, t]
            loss += (yt_pred - yt) **2
        loss /= 2 * test_Tx
        loss = float(loss)
        return y_pred, loss

def plot_loss(losses):
    """
    plots loss of LSTM vs # epochs
    Args:
        losses: list
    Returns:
        none
    Generates plot
    """
    fig, ax = plt.subplots()
    ax.plot(losses, label = "MSE")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE")
    ax.legend()

    plt.savefig("LSTM_Oct24_loss.pdf")
    plt.show()
        
def plot_y(y, y_pred, filename):
    """
    plots true y vals and predicted y vals
    Args:
        y: true y vals, numpy array of shape (ny, Tx)
        y_pred: predicted y vals, numpy array of shape (ny, Tx)
    """
    fig, ax = plt.subplots()
    ax.plot(y.flatten(), label = "Reported")
    ax.plot(y_pred.flatten(), label = "Estimated")
    ax.set_xlabel("No. of Days from 2020/01/24")
    ax.set_ylabel("COVID-19 Cases / Day")
    ax.set_title("LSTM Estimations of COVID Cases in Tokyo")
    ax.legend()
    
    plt.savefig(filename)
    plt.show()

def reverse_MinMax(y, y_min, y_max):
    """
    Reverses min-max scaling
    Args:
        y: numpy array of shape (ny, Tx)
        y_min: min value prior to scaling, float
        y_max: max value prior to scaling, float
    Returns:
        reverted: numpy array of shape (ny, Tx)
    """
    ny, Tx = y.shape
    #working_y = y.flatten()
    reverted = np.zeros((ny, Tx))
    for i in range(Tx):
        reverted[:, i] = (y_max - y_min) * y[:, i] + y_min
    
    assert reverted.shape == y.shape
    return reverted

def gen_mse(y, y_pred):
    """
    calculates mean squared error
    Args:
        y: true y vals, numpy array of shape (ny, Tx)
        y_pred: predicted y vals, numpy array of shape (ny, Tx)
    Returns:
        MSE: float
    """
    assert y.shape == y_pred.shape
    size = y.shape[1]
    mse = np.sum((y - y_pred)**2) / size
    return mse

def plot_from_txt(y, filename):
    """
    Generates plot for actual and predicted values from saved text file
    Args:
        y: numpy array of shape (ny, Tx), actual values
        filename: str, filename of text file containing predicted values
    """
    fig, ax = plt.subplots()
    ax.plot(y.flatten(), label = "Reported")
    y_pred = np.loadtxt(filename)
    ax.plot(y_pred.flatten(), label = "Estimated")
    ax.set_xlabel("No. of Days from 2020/01/24")
    ax.set_ylabel("COVID-19 Cases / Day")
    ax.set_title("LSTM Estimations of COVID Cases in Tokyo")
    ax.legend()
    fig_filename = filename.replace(".txt", "")
    fig_filename += ".pdf"
    plt.savefig(fig_filename)
    plt.show()

def hyperparam_montecarlo_lr(data, num_trials, patience, filename):
    """
    Random search (Monte Carlo search) for optimal # of hidden states, # inputs, and learning rate
    tunes: na, nx, lr
    Optimizes for lowest validation (Test) MSE (NOT lowest training MSE)

    na ~ U(8, 256) (int)
    nx ~ U(3, 30) (int)
    lr ~ loguniform(1e-4, 1e-1) (float)
    Args:
        data: input matrix, numpy array of shape (N, 1)
        num_trials: int, number of Monte Carlo trials to run
        patience: int
        filename: str
    """
    seed = np.random.randint(1e4)
    np.random.seed(seed)
    na_sample = np.random.randint(16, 128, (1, num_trials))
    nx_sample = np.random.randint(5, 25, (1, num_trials))
    lr_exp = np.random.uniform(-4, -2, (1, num_trials))
    lr_sample = 10 ** lr_exp
    #print(na_sample)
    #print(nx_sample)
    costs = []
    epochs = []
    test_mses = []
    print("Executing ", str(num_trials), " trials")
    print("Seed: ", seed)
    for i in range(num_trials):
        scaled, y_min, y_max = MinMaxScaling(data)
        nx_i = int(nx_sample[:, i])
        na_i = int(na_sample[:, i])
        lr_i = 0.001
        print("Trial {}".format(i + 1))
        print("na: ", na_i)
        print("nx: ", nx_i)
        print("lr: ", lr_i)

        x = gen_input(scaled, nx_i)
        y = gen_labels(scaled, x, 1)
        model = LSTM(na_i, x, lr = lr_i, optimizer = "adam")
        losses, params, best_y_pred, epoch = model.train(y, patience)
        reverted_y = reverse_MinMax(y, y_min, y_max)
        reverted_best_y_pred = reverse_MinMax(best_y_pred, y_min, y_max)
        mse = gen_mse(reverted_y, reverted_best_y_pred)
        costs.append(mse)
        epochs.append(epoch)
        print("Train MSE: ", mse)

        # testing
        testdata = getdata("data1101_1208.txt")
        scaled_testdata, testmin, testmax = MinMaxScaling(testdata)
        x_test = gen_input(scaled_testdata, nx_i)
        y_test = gen_labels(scaled_testdata, x_test, 1)
        test_y_pred, test_loss = model.test(x_test, y_test)
        reverted_y_test = reverse_MinMax(y_test, testmin ,testmax)
        reverted_test_y_pred = reverse_MinMax(test_y_pred, testmin, testmax)
        test_mse = gen_mse(reverted_y_test, reverted_test_y_pred)
        test_mses.append(test_mse)
        print("Test MSE:", test_mse)
        print("====================")

    

    fig, ax = plt.subplots()
    sc = plt.scatter(na_sample.flatten(), nx_sample.flatten(), c = np.array(test_mses), s = 5, cmap = "Spectral")
    cbar = plt.colorbar(sc)
    cbar.ax.set_title("Validation MSE")
    plt.xlabel("na")
    plt.ylabel("nx")
    scatter_plot_filename = filename + "na_nx_loss.pdf"
    seed_text = "Seed: " + str(seed)
    #plt.text(0.0000000001, 0.9999999999, seed_text, transform = ax.transAxes)
    plt.text(0.05, 0.94, seed_text, transform = plt.gcf().transFigure)
    plt.savefig(scatter_plot_filename)
    plt.show()
        # if i % 10 == 0 and i != 0:
        #     print("Finished ", str(i), " th trial")
    
    #print("MSE: ", costs)

    df = pd.DataFrame({
        "Validation MSE": test_mses,
        "Train MSE": costs,
        "na": na_sample.flatten(),
        "nx": nx_sample.flatten(),
        "Seed": seed
    })
    sorted_df = df.sort_values(by = ["Validation MSE"])
    sorted_df_filename = filename + ".csv"
    sorted_df.to_csv(path_or_buf = sorted_df_filename)

    #plotting top 10%
    sorted_validation_mses = sorted_df["Validation MSE"].tolist()
    validation_mses_top_10percent = int(len(sorted_validation_mses) * 0.1)
    best_validation_mses = sorted_validation_mses[:validation_mses_top_10percent]

    sorted_na = sorted_df["na"].tolist()
    na_top_10percent = int(len(sorted_na) * 0.1)
    best_na = sorted_na[:na_top_10percent]

    sorted_nx = sorted_df["nx"].tolist()
    nx_top_10percent = int(len(sorted_nx) * 0.1)
    best_nx = sorted_nx[:nx_top_10percent]

    assert validation_mses_top_10percent == na_top_10percent
    assert na_top_10percent == nx_top_10percent

    # plotting top 10 %
    fig, ax = plt.subplots()
    sc = plt.scatter(best_na, best_nx, c = best_validation_mses, s = 5, cmap = "Spectral")
    cbar = plt.colorbar(sc)
    cbar.ax.set_title("Validation MSE")
    plt.xlabel("na")
    plt.ylabel("nx")
    top10_scatterplot_filename = filename + "top10_na_nx_loss.pdf"
    #plt.text(0.0000000001, 0.9999999999, seed_text, transform = ax.transAxes)
    plt.text(0.05, 0.94, seed_text, transform = plt.gcf().transFigure)
    plt.savefig(top10_scatterplot_filename)
    plt.show()

    min_index = test_mses.index(min(test_mses))
    mins = {}
    mins["Trial"] = min_index + 1
    mins["na"] = na_sample[:, min_index]
    mins["nx"] = nx_sample[:, min_index]
    #mins["lr"] = lr_sample[:, min_index]
    mins["Train MSE"] = costs[min_index]
    mins["Test MSE"] = test_mses[min_index]
    mins["seed"] = seed
    return mins

def main():
    # seed = np.random.randint(1e4)
    seed = 1893
    np.random.seed(seed)
    print("Seed: ", seed)
    cycle = "train"
    ## Training data
    if cycle == "train":
        data = getdata("data0124_1031.txt")
    # Dev data
    if cycle == "dev":
        data = getdata("data1101_1208.txt")


    # =============================================
    # UNCOMMENT FOR TRAINING
    scaled, y_min, y_max = MinMaxScaling(data)
    nx = 24
    x = gen_input(scaled, nx)
    y = gen_labels(scaled, x, 1)
    reverted_y = reverse_MinMax(y, y_min, y_max)

    na = 118
    model = LSTM(na, x, epochs = 1000, lr = 0.001, optimizer = "adam")
    losses, params, best_y_pred, epoch = model.train(y, 10)
    plot_loss(losses)
    #print(params)
    reverted_best_y_pred = reverse_MinMax(best_y_pred, y_min, y_max)
    save_filename = "LSTM_predictions_Jan4_2"
    np.savetxt(save_filename + ".txt", reverted_best_y_pred)
    train_mse = gen_mse(reverted_y, reverted_best_y_pred)
    plot_y(reverted_y, reverted_best_y_pred, save_filename + ".pdf")
    if cycle == "train":
        print("Train MSE: ", train_mse)
    if cycle == "dev":
        print("Dev MSE: ", train_mse)
    
    # ================================================


    # =====================================
    ## FOR MONTE CARLO SIMULATION
    # tuning adam hyperparameters
    # randomtrial_filename = "LSTM_randomtrial_Jan4_1"
    # print(hyperparam_montecarlo_lr(data, 100, 10, randomtrial_filename))
    # ======================================


    # ==============================
    ## FOR PLOTTING FROM TXT FILE
    # plot_from_txt(reverted_y, "LSTM_predictions_MSE666.txt")
    # ====================================


    # ============================
    ## For TESTING

    # testdata = getdata("test0829_0930.txt")
    # scaled_testdata, testmin, testmax = MinMaxScaling(testdata)
    # x_test = gen_input(scaled_testdata, nx)
    # y_test = gen_labels(scaled_testdata, x_test, 1)
    # test_y_pred, test_loss = model.test(x_test, y_test)
    # reverted_y_test = reverse_MinMax(y_test, testmin ,testmax)
    # reverted_test_y_pred = reverse_MinMax(test_y_pred, testmin, testmax)
    # test_mse = gen_mse(reverted_y_test, reverted_test_y_pred)
    # plot_y(reverted_y_test, reverted_test_y_pred, "LSTM_test_predictions_Oct30_4.pdf")
    # print("Test MSE:", test_mse)

    #print("Test MSE:", test_mse)
    
    # ==============================

if __name__ == "__main__":
    start = time.time()
    main()
    #print(sigmoid(-1000000))
    end = time.time()
    print("Ran in ", str(round(end - start, 4)), " seconds")

    