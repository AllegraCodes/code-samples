#!/usr/bin/env python3
"""
Synella Gonzales & Allegra Panetta
Project 3
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from scipy.stats import ttest_ind


def generate_features():
    """
    Generates features of Connect 4 boards given as a csv file
    The original data with the features added to each line is
    written into the file specified by the user
    :return: 2D list containing the features of each board
             list containing the winner of each board
    """
    all_features = []
    winners = []
    for line in infile:
        if line[0].isdigit():  # line contains data
            board, winner = read_board(line)
            winners.append(winner)
            features = [
                left_corner(board),     # feature 0
                right_corner(board),    # feature 1
                center(board, 1),       # feature 2
                center(board, 2),       # feature 3
                connections(board, 1),  # feature 4
                connections(board, 2),  # feature 5
                rows(board, 1),         # feature 6
                rows(board, 2),         # feature 7
                columns(board, 1),      # feature 8
                columns(board, 2)       # feature 9
            ]
            n = len(features)
            for i in range(n):
                for j in range(i+1, n):
                    features.append(features[i] + features[j])
                    features.append(features[i] - features[j])
                    features.append(features[i] * features[j])
            all_features.append(features)
            stringout = line.rstrip('\n')
            for feature in features:
                stringout += ',' + str(feature)
            stringout += '\n'
            outfile.write(stringout)
        else:  # header line
            stringout = line.rstrip('\n') + ",x0,x1,x2,x3,x4,x5,x6,x7,x8,x9"
            for i in range(10):
                for j in range(i+1, 10):
                    stringout += ",x" + str(i) + '+' + str(j)
                    stringout += ",x" + str(i) + '-' + str(j)
                    stringout += ",x" + str(i) + '*' + str(j)
            stringout += '\n'
            outfile.write(stringout)
    return all_features, winners


def left_corner(board):
    """
    Evaluates feature 0 - control of bottom-left space
    :param board: the board to evaluate as a 2D list
    :return: 1 if controlled by Player 1, -1 if controlled by Player 2, 0 if empty
    """
    occupier = board[0][0]
    if occupier == 1:
        result = 1
    elif occupier == 2:
        result = -1
    else:
        result = 0
    return result


def right_corner(board):
    """
    Evaluates feature 1 - control of bottom-right space
    :param board: the board to evaluate as a 2D list
    :return: 1 if controlled by Player 1, -1 if controlled by Player 2, 0 if empty
    """
    occupier = board[0][6]
    if occupier == 1:
        result = 1
    elif occupier == 2:
        result = -1
    else:
        result = 0
    return result


def center(board, player):
    """
    Evaluates features 2 & 3 - control of the center
    1 point is awarded for each piece the player has in columns 2 and 4, and
    2 points for each piece in column 3
    :param board: the board to evaluate as a 2D list
    :param player: the player whose control is being evaluated
    :return: the total score
    """
    score = 0
    for row in range(6):
        pieces = board[row][2], board[row][3], board[row][3], board[row][4]  # double the center column
        for piece in pieces:
            if piece == player:
                score += 1
    return score


def connections(board, player):
    """
    Evaluates features 4 & 5 - number of connections
    :param board: the board to evaluate as a 2D list
    :param player: the player whose connections are being counted
    :return: the number of connections
    """
    total = 0
    for row in range(6):
        for col in range(7):
            if board[row][col] == player:
                if row + 1 < 6 and board[row + 1][col] == player:  # up
                    total += 1
                if row + 1 < 6 and col + 1 < 7 and board[row + 1][col + 1] == player:  # up-right
                    total += 1
                if col + 1 < 7 and board[row][col + 1] == player:  # right
                    total += 1
                if row - 1 >= 0 and col + 1 < 7 and board[row - 1][col + 1] == player:  # bottom-right
                    total += 1
    return total


def rows(board, player):
    """
    Evaluates features 6 & 7 - Control of rows
    :param board: the board as a 2D array
    :param player: the player whose row control is being evaluated
    :return: the number of rows that contain a player piece
    """
    contains = [False, False, False, False, False, False]
    for row in range(6):
        for col in range(7):
            if board[row][col] == player:
                contains[row] = True
    return contains.count(True)


def columns(board, player):
    """
    Evaluates features 8 & 9 - Control of columns
    :param board: the board as a 2D array
    :param player: the player whose column control is being evaluated
    :return: the number of columns that contain a player piece
    """
    contains = [False, False, False, False, False, False, False]
    for row in range(6):
        for col in range(7):
            if board[row][col] == player:
                contains[col] = True
    return contains.count(True)


def read_board(line):
    """
    Converts a single line representing a game board into a 2 dimensional list
    :param line: single csv line representing a game board
    :return: 2D list representing the same board,
             int representing the winner for the board (0 if draw)
    """
    board = [[0]*7 for i in range(6)]  # 7 columns, 6 rows
    row = 0
    col = 0
    for space in line.split(','):
        if col < 7:  # the line has an extra item indicating the winner
            board[row][col] = int(space)
            row += 1
            row %= 6
            if row == 0:
                col += 1
        else:  # this indicates the winner
            winner = int(space)
    return board, winner


def cross_val_test(x, y, tree, k=10):
    """
    Performs k-fold cross-validation on the accuracy of the tree
    :param x: the test data
    :param y: labels for x
    :param tree: the tree to test
    :param k: the number of folds
    :return: list containing the results of each fold
    """
    kfold = KFold(k)
    results = []
    for fold, (train_indices, val_indices) in enumerate(kfold.split(x, y)):
        x_train = []
        x_val = []
        y_train = []
        y_val = []
        for train_index, val_index in zip(train_indices, val_indices):
            x_train.append(x[train_index])
            x_val.append(x[val_index])
            y_train.append(y[train_index])
            y_val.append(y[val_index])
        tree.fit(x_train, y_train)
        predictions = tree.predict(x_val)
        for guess, answer in zip(predictions, y_val):
            results.append(guess == answer)
    return results


def plot_importance(forest, X):
    """
    Plots the importance of the features in the given tree ensemble
    Source: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    :param forest: the forest that split the data
    :param X: the data given to the forest as a numpy array
    :return: None
    """
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature Importance")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.show()


# GENERATE FEATURES
print("Loading data")
# Open files
input_name = sys.argv[1]
output_name = sys.argv[2]
infile = open(input_name, 'r')
outfile = open(output_name, 'w')
# get data
x, y = generate_features()
print("Features generated and written to " + output_name)
# Close files
infile.close()
outfile.close()

# SELECT FEATURES
print("Selecting features")
# Find a good number of features
best_k = 0
best_accuracy = 0
best_p = 0
for current_k in range(10, 146):
    x_new = SelectKBest(k=current_k).fit_transform(x, y)
    accuracy = np.mean(cross_val_test(x_new, y, DecisionTreeClassifier(random_state=0)))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = current_k
# Get the best features
skb = SelectKBest(k=best_k)
x_new = skb.fit_transform(x, y)
selected = skb.get_support(indices=True)
feature_string = ""
for feature in selected:
    feature_string += str(feature) + ' '
print("Selected " + str(best_k) + " features: " + feature_string)
# Plot feature importance
rfc = RandomForestClassifier(random_state=0)
rfc.fit(x_new, y)
plot_importance(rfc, x_new)

# EXPERIMENT 1 - Number of trees in a random forest
forests = [RandomForestClassifier(n_estimators=n, random_state=0) for n in range(1, 65)]
results = []
for i in range(len(forests)):
    print("Testing forest " + str(i+1) + "/64")
    results.append(cross_val_test(x_new, y, forests[i]))
accuracies = np.mean(results, axis=1)
t, p1 = ttest_ind(results[0], results[-1])
t, p2 = ttest_ind(results[1], results[-1])
t, p3 = ttest_ind(results[2], results[-1])
t, p4 = ttest_ind(results[3], results[-1])
t, p5 = ttest_ind(results[4], results[-1])
print("Probability of no difference between n = 1 and n = 64: " + str(p1))
print("Probability of no difference between n = 2 and n = 64: " + str(p2))
print("Probability of no difference between n = 3 and n = 64: " + str(p3))
print("Probability of no difference between n = 4 and n = 64: " + str(p4))
print("Probability of no difference between n = 5 and n = 64: " + str(p5))
plt.title("Number of Trees in Random Forest")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.plot(range(1, len(forests)+1), accuracies)
plt.show()

# EXPERIMENT 2 - Effect of feature selection
# get our 10 base features
x_base = []
for group in x:
    x_base.append(group[:10])
# selected features
sel_res = cross_val_test(x_new, y, DecisionTreeClassifier(random_state=0))
sel_acc = np.mean(sel_res)
# base features
base_res = cross_val_test(x_base, y, DecisionTreeClassifier(random_state=0))
base_acc = np.mean(base_res)
# total features
tot_res = cross_val_test(x, y, DecisionTreeClassifier(random_state=0))
tot_acc = np.mean(tot_res)
print("1. Selected feature accuracy: " + str(sel_acc))
print("2. Base feature accuracy: " + str(base_acc))
print("3. Total feature accuracy: " + str(tot_acc))
t, p12 = ttest_ind(sel_res, base_res)
t, p13 = ttest_ind(sel_res, tot_res)
t, p23 = ttest_ind(base_res, tot_res)
print("p-value 1-2: " + str(p12))
print("p-value 1-3: " + str(p13))
print("p-value 2-3: " + str(p23))
