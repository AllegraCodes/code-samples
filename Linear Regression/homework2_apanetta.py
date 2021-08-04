import numpy as np
import matplotlib.pyplot as plt

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    n = faces.shape[0]
    m = faces.shape[1]
    # flatten images to 1D
    faces = faces.reshape(n, m*m).T
    # append ones
    ones = np.ones(n)
    return np.vstack((faces, ones))

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (w, Xtilde, y):
    n = Xtilde.shape[1]
    yhat = Xtilde.T.dot(w)
    summation = np.sum((yhat - y)**2)
    return summation / (2*n)

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
    n = y.size
    wreg = np.copy(w)
    wreg[-1] = 0 # do not include the bias term during regularization
    gradient = (1/n) * Xtilde.dot(Xtilde.T.dot(w) - y) + (alpha/n) * wreg
    return gradient

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
    return np.linalg.solve(Xtilde.dot(Xtilde.T), Xtilde.dot(y))

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    return gradientDescent(Xtilde, y)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    return gradientDescent(Xtilde, y, alpha=ALPHA)

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations
    w = 0.01 * np.random.randn(Xtilde.shape[0])
    for i in range(T):
        w -= EPSILON * gradfMSE(w, Xtilde, y, alpha)
        if i%500==0:
            print("i=",i)
    return w

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    # Report fMSE cost using each of the three learned weight vectors
    print("analytical solution")
    w1 = method1(Xtilde_tr, ytr)
    print("training: ", fMSE(w1, Xtilde_tr, ytr))
    print("testing: ", fMSE(w1, Xtilde_te, yte))
    w1image = w1[:w1.size-1].reshape((48,48))
    plt.imshow(w1image)
    plt.show()

    print("\ngradient descent (unregularized)")
    w2 = method2(Xtilde_tr, ytr)
    print("training: ", fMSE(w2, Xtilde_tr, ytr))
    print("testing: ", fMSE(w2, Xtilde_te, yte))
    w2image = w2[:w2.size-1].reshape((48,48))
    plt.imshow(w2image)
    plt.show()

    print("\ngradient descent (regularized)")
    w3 = method3(Xtilde_tr, ytr)
    print("training: ", fMSE(w3, Xtilde_tr, ytr))
    print("testing: ", fMSE(w3, Xtilde_te, yte))
    w3image = w3[:w3.size-1].reshape((48,48))
    plt.imshow(w3image)
    plt.show()

    yhat = Xtilde_te.T.dot(w3)
    errors = yhat - yte
    rmse = np.mean(errors**2)**0.5
    print("RMSE: ", rmse)

    worst5args = np.flip(np.argsort(np.abs(errors)))[:5]
    worst5imgs = Xtilde_te[:, worst5args].T
    for img, arg in zip(worst5imgs, worst5args):
        print("y = ", yte[arg])
        print("yhat = ", yhat[arg])
        img = img[:img.size-1].reshape((48,48))
        plt.imshow(img)
        plt.show()
