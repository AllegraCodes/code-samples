import numpy as np
import matplotlib.pyplot as plt

def PCA(X):
    # compute mean vector
    xbar = np.mean(X, axis=1)

    # subtract mean from X
    sub = np.outer(xbar, np.ones(X.shape[1])) # repeat xbar to subtract from each x
    Xt = X - sub

    # compute eigenvalues & eigenvectors
    values, vectors = np.linalg.eig(Xt.dot(Xt.T))

    # get eigenvectors of 2 largest eigenvalues
    valsort = np.flip(np.argsort(values))
    d1 = vectors[:,valsort[0]]
    d2 = vectors[:,valsort[1]]
    return d1, d2

if __name__ == "__main__":
    # load data
    images = np.load("small_mnist_test_images.npy")
    X = images.T
    Y = np.load("small_mnist_test_labels.npy")

    # find 2 directions of highest variance
    d1, d2 = PCA(X)

    # project the data onto those directions
    p1 = X.T.dot(d1)
    p2 = X.T.dot(d2)

    # assign colors
    classes = np.argmax(Y, axis=1)
    colors = list(classes)
    for i in range(len(classes)):
        if classes[i] == 0:
            colors[i] = '#ff0000' # red
        elif classes[i] == 1:
            colors[i] = '#ff8000' # orange
        elif classes[i] == 2:
            colors[i] = '#ffff00' # yellow
        elif classes[i] == 3:
            colors[i] = '#00ff00' # green
        elif classes[i] == 4:
            colors[i] = '#00ffff' # light blue
        elif classes[i] == 5:
            colors[i] = '#0080ff' # mid blue
        elif classes[i] == 6:
            colors[i] = '#0000ff' # deep blue
        elif classes[i] == 7:
            colors[i] = '#8000ff' # purple
        elif classes[i] == 8:
            colors[i] = '#ff00ff' # pink
        elif classes[i] == 9:
            colors[i] = '#ff0080' # magenta

    # plot
    plt.scatter(p1, p2, c=colors, marker='.', s=1)
    plt.show()
