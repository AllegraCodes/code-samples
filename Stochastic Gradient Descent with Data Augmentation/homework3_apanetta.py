import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import random
import math

def fPC(Y, Yhat):
    return np.mean(np.argmax(Y, axis=1) == np.argmax(Yhat, axis=1))

def fCE(Y, Yhat):
    n = Y.shape[0]
    product = Y * np.log(Yhat)
    summation = np.sum(np.sum(product, axis=1))
    return (-1/n) * summation

def append1s(images):
    ones = np.ones(images.shape[0])
    return np.vstack((images.T, ones))

def gradCE(X, W, Y):
    Yhat = softmax(X, W)
    n = Y.shape[0]
    return (1/n) * X.dot(Yhat - Y)

def SGD(X, Y, ntilde=100, E=100, eps=0.05):
    # initialize weights to random values
    m = X.shape[0] # number of features (pixels)
    c = Y.shape[1] # number of categories
    W = 0.1 * np.random.randn(m, c)

    # randomly sample a small batch
    n = Y.shape[0]
    order = np.random.permutation(n)
    numbatches = math.ceil(n / ntilde)
    for e in range(E):
        print("epoch:", e+1, "/", E)
        for i in range(numbatches):
            start = i * ntilde
            if i == numbatches - 1: # the last batch might be smaller than the rest
                end = X.shape[1]
            else:
                end = start + ntilde
            batch = X[:, order][:, start:end]
            # calculate the gradient of the batch
            grad = gradCE(batch, W, Y[order][start:end])
            # update weights
            W -= eps * grad
        if e >= E - 20:
            Yhat = softmax(Xte, W)
            print("Loss: ", fCE(Yte, Yhat))
            print("Accuracy: ", fPC(Yte, Yhat))
    return W

def softmax(X, W):
    Z = W.T.dot(X) # step 1: X -> Z
    YhatT = np.exp(Z) / np.sum(np.exp(Z), axis=0) # step 2: Z -> Yhat
    return YhatT.T

def translation(img):
    result = img
    # shift rows
    offset = int(5 * random.random())
    if random.random() < 0.5: # shift down
        result = np.vstack((np.zeros( (offset, result.shape[1]) ), result))
        result = result[:28,:]
    else: # shift up
        result = np.vstack((result, np.zeros( (offset, result.shape[1]) )))
        result = result[offset:,:]
    # shift cols
    offset = int(5 * random.random())
    if random.random() < 0.5: # shift right
        result = np.hstack((np.zeros( (result.shape[0], offset) ), result))
        result = result[:,:28]
    else: # shift left
        result = np.hstack((result, np.zeros( (result.shape[1], offset) )))
        result = result[:,offset:]
    return result

def rotation(img):
    angle = 20 * random.random() - 10 # rotation angle from -10 to 10 degrees
    return sk.transform.rotate(img, angle)

def scaling(img):
    factor = 0.1 * random.random() + 1.1 # scale up 10%-20%
    scaled = sk.transform.rescale(img, factor)
    center = int(scaled.shape[0] / 2)
    return scaled[center-14:center+14, center-14:center+14]

def noise(img):
    noise = 0.1 * np.random.randn(img.shape[0], img.shape[1])
    result = img + noise
    return np.clip(result, 0, 1)

if __name__ == "__main__":
    trainimg = np.load("small_mnist_train_images.npy")
    Xtr = append1s(trainimg)
    Ytr = np.load("small_mnist_train_labels.npy")
    testimg = np.load("small_mnist_test_images.npy")
    Xte = append1s(testimg)
    Yte = np.load("small_mnist_test_labels.npy")

    SGD(Xtr, Ytr)

    original = trainimg[random.randrange(5000)].reshape(28, 28)
    translated = translation(original)
    rotated = rotation(original)
    scaled = scaling(original)
    noised = noise(original)
    plt.imshow(original)
    plt.show()
    plt.imshow(translated)
    plt.show()
    plt.imshow(rotated)
    plt.show()
    plt.imshow(scaled)
    plt.show()
    plt.imshow(noised)
    plt.show()

    print("Augmenting training set...")
    aug1 = np.copy(trainimg)
    for i in range(aug1.shape[0]):
        img = trainimg[i].reshape(28, 28)
        if i % 4 == 0:
            aug1[i] = translation(img).flatten()
        if i % 4 == 1:
            aug1[i] = rotation(img).flatten()
        if i % 4 == 2:
            aug1[i] = scaling(img).flatten()
        if i % 4 == 3:
            aug1[i] = noise(img).flatten()
    aug2 = np.copy(trainimg)
    for i in range(aug2.shape[0]):
        img = trainimg[i].reshape(28, 28)
        if i % 4 == 0:
            aug2[i] = noise(img).flatten()
        if i % 4 == 1:
            aug2[i] = translation(img).flatten()
        if i % 4 == 2:
            aug2[i] = rotation(img).flatten()
        if i % 4 == 3:
            aug2[i] = scaling(img).flatten()
    Xaug = np.vstack((trainimg, aug1, aug2))
    Yaug = np.vstack((Ytr, Ytr, Ytr))

    SGD(append1s(Xaug), Yaug, eps=0.01)
