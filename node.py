import copy

import numpy as np
import random


class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature_idx = None
        self.feature_value = None
        self.node_prediction = None

    def gini_best_score(self, y, possible_splits):
        best_gain = -np.inf
        best_idx = 0

        for idx in possible_splits:
            left_positive = 0
            left_negative = 0
            right_positive = 0
            right_negative = 0
            left_node = y[:idx]
            right_node = y[idx:]
            if len(left_node) == 0 or len(right_node) == 0:
                continue
            for value in left_node:
                if value == 1:
                    left_positive += 1
                else:
                    left_negative += 1
            for value in right_node:
                if value == 1:
                    right_positive += 1
                else:
                    right_negative += 1
            gini_left = 1 - pow((left_positive / (left_positive + left_negative)), 2) - pow(
                (left_negative / (left_positive + left_negative)), 2)
            gini_right = 1 - pow((right_positive / (right_positive + right_negative)), 2) - pow(
                (right_negative / (right_positive + right_negative)), 2)
            left = left_positive + left_negative
            right = right_negative + right_positive
            gini_gain = 1 - left / (left + right) * gini_left - right / (left + right) * gini_right
            if gini_gain > best_gain:
                best_gain = gini_gain
                best_idx = idx
        return best_idx, best_gain

    def split_data(self, X, y, idx, val):
        left_mask = X[:, idx] < val
        return (X[left_mask], y[left_mask]), (X[~left_mask], y[~left_mask])

    def find_possible_splits(self, data):
        possible_split_points = []
        for idx in range(data.shape[0] - 1):
            if data[idx] != data[idx + 1]:
                possible_split_points.append(idx)
        return possible_split_points

    def find_best_split(self, X, y, feature_subset):
        best_gain = -np.inf
        best_split = None
        feature_idx = None

        # TODO implement feature selection
        if feature_subset is not None:
            feature_idx = random.sample(range(6), feature_subset)

        for d in range(X.shape[1]):
            if feature_idx is not None:
                if d in feature_idx:
                    order = np.argsort(X[:, d])
                    y_sorted = y[order]
                    possible_splits = self.find_possible_splits(X[order, d])
                    idx, value = self.gini_best_score(y_sorted, possible_splits)
                    if value > best_gain:
                        best_gain = value
                        best_split = (d, [idx, idx + 1])
            else:
                order = np.argsort(X[:, d])
                y_sorted = y[order]
                possible_splits = self.find_possible_splits(X[order, d])
                idx, value = self.gini_best_score(y_sorted, possible_splits)
                if value > best_gain:
                    best_gain = value
                    best_split = (d, [idx, idx + 1])

        if best_split is None:
            return None, None

        best_value = np.mean(X[best_split[1], best_split[0]])

        return best_split[0], best_value

    def predict(self, x):
        if self.feature_idx is None:
            return self.node_prediction
        if x[self.feature_idx] < self.feature_value:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

    def train(self, X, y, params):

        self.node_prediction = np.mean(y)
        if X.shape[0] == 1 or self.node_prediction == 0 or self.node_prediction == 1:
            return True

        self.feature_idx, self.feature_value = self.find_best_split(X, y, params["feature_subset"])
        if self.feature_idx is None:
            return True

        (X_left, y_left), (X_right, y_right) = self.split_data(X, y, self.feature_idx, self.feature_value)

        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            self.feature_idx = None
            return True

        # max tree depth
        if params["depth"] is not None:
            params["depth"] -= 1
        if params["depth"] == 0:
            self.feature_idx = None
            return True

        # create new nodes
        self.left_child, self.right_child = Node(), Node()
        self.left_child.train(X_left, y_left, copy.deepcopy(params))
        self.right_child.train(X_right, y_right, copy.deepcopy(params))
