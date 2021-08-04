import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random

def fPC(y, yhat):
    return np.mean(y == yhat)

def measureAccuracyOfPredictors(predictors, X, y):
    # create matrix where [i,j] is the result of predictor_i on image_j
    first = True # flag to create the array
    for p in predictors:
        if first:
            predictions = np.array([X[:, p[0], p[1]] > X[:, p[2], p[3]]])
            first = False
        else:
            predictions = np.vstack((predictions, X[:, p[0], p[1]] > X[:, p[2], p[3]]))
    # if most predictors returned 1 on an image, the whole ensemble returns 1 (True)
    yhat = np.mean(predictions, axis=0) > 0.5
    return fPC(y, yhat)

def stepwiseRegression(trainingFaces, trainingLabels):
    print("Starting stepwise regression...")
    predictors = set()
    for m in range(5): # find the next best feature in each round to use as a predictor
        #print("searching for feature ", m+1)
        best_accuracy = 0.
        best_feature = (0, 0, 0, 0)
        for r1 in range(trainingFaces.shape[1]):
            for c1 in range(trainingFaces.shape[2]):
                for r2 in range(trainingFaces.shape[1]):
                    for c2 in range(trainingFaces.shape[2]): # for each possible pair of pixels
                        if not (r1, c1, r2, c2) in predictors: # skip any we already have
                            predictors.add( (r1, c1, r2, c2) )
                            new_accuracy = measureAccuracyOfPredictors(predictors, trainingFaces, trainingLabels)
                            if new_accuracy > best_accuracy:
                                best_accuracy = new_accuracy
                                best_feature = (r1, c1, r2, c2)
                                #print("new best feature: ", best_feature," accuracy: ", best_accuracy)
                            predictors.remove( (r1, c1, r2, c2) )
        predictors.add(best_feature)
        print("feature ", m+1, ": ", best_feature, "accuracy: ", best_accuracy)
    return predictors, best_accuracy

def displayFeatures(predictors, image):
    # Show an arbitrary test image in grayscale
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')

    colors = {("red", "darkred"), ("gold", "darkgoldenrod"), ("chartreuse", "forestgreen"), ("blue", "darkblue"), ("fuchsia", "darkviolet")}
    for p, c in zip(predictors, colors):
        # Show r1,c1
        rect = patches.Rectangle((p[1],p[0]),1,1,linewidth=2,edgecolor=c[0],facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((p[3],p[2]),1,1,linewidth=2,edgecolor=c[1],facecolor='none')
        ax.add_patch(rect)
    # Display the merged result
    plt.show()

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    predictors = set()
    for n in range(400, 2001, 400):
        predictors, trainingAcc = stepwiseRegression(trainingFaces[:n,...], trainingLabels[:n])
        testingAcc = measureAccuracyOfPredictors(predictors, testingFaces, testingLabels)
        print("n trainingAcc testingAcc")
        print(n, trainingAcc, testingAcc, "\n", sep='  ')
    displayFeatures(predictors, random.choice(trainingFaces))
