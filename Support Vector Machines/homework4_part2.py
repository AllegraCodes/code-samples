"""
Allegra Panetta
apanetta@wpi.edu
"""

# linear auc = 0.8564325158905925
# non-linear auc = 0.8398707708141309

import sklearn.svm
import sklearn.metrics
import numpy as np
import pandas

# Load data
print("Loading data...")
d = pandas.read_csv('train.csv')
y = np.array(d.target)  # Labels
X = np.array(d.iloc[:,2:])  # Features

# Split into train/test folds
n = y.shape[0]
order = np.random.permutation(n)
ytr = y[order][:n//2]
yte = y[order][n//2:]
Xtr = X[order][:n//2]
Xte = X[order][n//2:]

# Linear SVM
lsvm = sklearn.svm.LinearSVC(C=1e15, dual=False)  # 1e15 -- approximate hard-margin
print("Training linear classifier...")
lsvm.fit(Xtr, ytr)

# Non-linear SVM (polynomial kernel)
c = 20 # number of classifiers
classifiers = list()
data = list()
labels = list()
groupsize = Xtr.shape[0] // c
for i in range(c):
    start = i * groupsize
    end = start + groupsize if i < c-1 else n
    classifiers.append(sklearn.svm.SVC(kernel='poly', degree=3, gamma='auto'))
    data.append(Xtr[start:end])
    labels.append(ytr[start:end])
for s, x, y, i in zip(classifiers, data, labels, range(c)):
    print("Training nonlinear classifier", i+1, "/", c)
    s.fit(x, y)

# Apply the SVMs to the test set
print("Testing linear classifier...")
yhat1 = lsvm.decision_function(Xte)  # Linear kernel
predictions = list()
for s, i in zip(classifiers, range(c)):
    print("Testing nonlinear classifier", i+1, "/", c)
    p = s.decision_function(Xte)
    predictions.append(p)
predictions = np.array(predictions)
yhat2 = np.mean(predictions, axis=0)  # Non-linear kernel

# Compute AUC
auc1 = sklearn.metrics.roc_auc_score(yte, yhat1)
auc2 = sklearn.metrics.roc_auc_score(yte, yhat2)

print(auc1)
print(auc2)
