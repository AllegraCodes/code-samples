from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.optimize
import numpy as np
import math

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

def unpack(w):
    idx1 = NUM_INPUT * NUM_HIDDEN
    idx2 = idx1 + NUM_HIDDEN
    idx3 = idx2 + NUM_HIDDEN * NUM_OUTPUT
    W1 = w[:idx1].reshape((NUM_INPUT, NUM_HIDDEN))
    b1 = w[idx1:idx2]
    W2 = w[idx2:idx3].reshape((NUM_HIDDEN, NUM_OUTPUT))
    b2 = w[idx3:]
    return W1, b1, W2, b2

def pack(W1, b1, W2, b2):
    w = np.hstack((W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()))
    return w

def loadData(which):
    images = np.load("mnist_{}_images.npy".format(which))
    labels = np.load("mnist_{}_labels.npy".format(which))
    return images, labels

def plotSGDPath(X, Y, ws):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # get axes using PCA
    ws = np.array(ws)
    pca = PCA(n_components=2)
    pca.fit(ws)
    d1 = pca.components_[0]
    d2 = pca.components_[1]
    # scatter plot showing the weights during SGD.
    Xs = ws.dot(d1)
    Ys = ws.dot(d2)
    Zs = np.zeros(Xs.shape[0])
    for i, w in zip(range(ws.shape[0]), ws):
        # print("i =", i)
        Zs[i] = fCE(X, Y, w)
    ax.scatter(Xs, Ys, Zs, color='r')
    # Compute the CE loss on a grid of points (corresponding to different w).
    axis1 = np.arange(math.floor(np.min(Xs))-1, math.ceil(np.max(Xs))+1, 1)
    axis2 = np.arange(math.floor(np.min(Ys))-1, math.ceil(np.max(Ys))+1, 1)
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Xaxis = Xaxis.T
    Yaxis = Yaxis.T
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in range(len(axis1)):
        for j in range(len(axis2)):
            # print("i, j =", i, j)
            Zaxis[i,j] = fCE(X, Y, Xaxis[i,j]*d1 + Yaxis[i,j]*d2)
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.
    plt.show()

def fCE(X, Y, w):
    n = Y.shape[0]
    z1, h1, z2, Yhat = forwardprop(X, w)
    product = Y * np.log(Yhat.T)
    summation = np.sum(np.sum(product, axis=1))
    cost = (-1/n) * summation
    return cost

def gradCE(X, Y, w):
    z1, h1, z2, Yhat = forwardprop(X, w)
    W1, b1, W2, b2 = unpack(w)
    # match notation in notes
    X = X.T
    Y = Y.T
    W2 = W2.T
    # backprop
    gb2 = Yhat - Y
    gW2 = gb2.dot(h1.T)
    gb1 = np.transpose(gb2.T.dot(W2) * drelu(z1.T))
    gW1 = gb1.dot(X.T)
    grad = (1/X.shape[1]) * pack(gW1.T, np.sum(gb1, axis=1), gW2.T, np.sum(gb2, axis=1)) # back to numpy notation
    return grad

def randw():
    W1 = 2 * (np.random.random(size=(NUM_INPUT, NUM_HIDDEN)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2 * (np.random.random(size=(NUM_HIDDEN, NUM_OUTPUT)) / NUM_HIDDEN ** 0.5) - 1. / NUM_HIDDEN ** 0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    return pack(W1, b1, W2, b2)

def train(trainX, trainY, testX, testY, w, E=15, ntilde=256, eps=.5, test=False):
    n = trainX.shape[0]
    numbatches = math.ceil(n / ntilde)
    ws = [np.array(w)]
    order = np.random.permutation(n)
    for e in range(E): # for each epoch
        if test:
            print("epoch:", e+1, "/", E)
        for i in range(numbatches): # for each minibatch
            # get batch
            start = i * ntilde
            if i == numbatches - 1:  # the last batch might be smaller than the rest
                end = n
            else:
                end = start + ntilde
            batchX = trainX[order, :][start:end]
            batchY = trainY[order, :][start:end]
            grad = gradCE(batchX, batchY, w) # calculate the gradient of the batch
            w -= eps * grad # update weights
        ws.append(np.array(w))
        if test:
            cost = fCE(testX, testY, w)
            acc = fPC(testX, testY, w)
            print("cost, acc =", cost, acc)
    return ws

def fPC(X, Y, w):
    z1, h1, z2, Yhat = forwardprop(X, w)
    return np.mean(np.argmax(Y, axis=1) == np.argmax(Yhat.T, axis=1))

def forwardprop(X, w):
    W1, b1, W2, b2 = unpack(w)
    # match notation in notes
    X = X.T
    W1 = W1.T
    b1 = np.outer(b1, np.ones(X.shape[1]))
    W2 = W2.T
    b2 = np.outer(b2, np.ones(X.shape[1]))
    # propagate
    z1 = W1.dot(X) + b1
    h1 = relu(np.array(z1))
    z2 = W2.dot(h1) + b2
    Yhat = np.exp(z2) / np.sum(np.exp(z2), axis=0) # softmax
    return z1, h1, z2, Yhat

def relu(x):
    x[x<0] = 0
    return x

def drelu(x):
    return x >= 0

def findBestHyperparameters():
    epochs = [20]
    sizes = [128, 256]
    hidden = [40, 50]
    rates = [1, .5, .1]
    ws = list()
    costs = list()
    global NUM_HIDDEN
    for e in epochs:
        for s in sizes:
            for h in hidden:
                NUM_HIDDEN = h
                for r in rates:
                    print("batch size =", s)
                    print("hidden layer size =", h)
                    print("learning rate =", r)
                    w = randw()
                    w = train(trainX, trainY, valX, valY, w, e, s, r)[-1]
                    ws.append((w, e, s, h, r))
                    c = fCE(valX, valY, w)
                    costs.append(c)
                    print("validation cost =", c)
    optarg = np.argmin(np.array(costs))
    return ws[optarg]

if __name__ == "__main__":
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")
        valX, valY = loadData("validation")

    print("Checking gradient function...")
    # Initialize weights randomly
    w = randw()
    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    lambda w_: gradCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    w))

    print("Optimizing hyperparameters...")
    opt = findBestHyperparameters()
    print("Optimized parameters:")
    # print("number of epochs:", opt[1])
    print("batch size:", opt[2])
    print("hidden layer size:", opt[3])
    print("learning rate:", opt[4])

    # Train the network and obtain the sequence of w's obtained using SGD.
    NUM_HIDDEN = opt[3]
    print("Training with optimized hyperparameters...")
    ws = train(trainX, trainY, testX, testY, randw(), opt[1], opt[2], opt[4], True)
    # Plot the SGD trajectory
    print("Rendering surface...")
    plotSGDPath(testX[:2500], testY[:2500], ws)
