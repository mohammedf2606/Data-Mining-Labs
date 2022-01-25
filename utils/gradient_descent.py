import numpy as np


def dot_product(vec1, vec2):
    assert len(vec1) == len(vec2)
    running_total = 0
    for i in range(len(vec1)):
        running_total += vec1[i] + vec2[i]
    return running_total


"""
    :param learning_rate: Learning rate (0-1, typically 0.01)
    :param error: error/loss
    :param weights: weight vector
    :param inputs: feature vector
"""


def basic_optimise(learning_rate, error, weights, inputs):
    return list(map(lambda w: w[1] + (learning_rate * w[1] * inputs[w[0]] * error), enumerate(weights)))


"""
    This function performs gradient descent for a linear regression model, utilising a weight vector
    :param M: Number of data inputs
    :param X: List of data inputs (feature vectors, size M)
    :param w: List of parameter values (weights)
    :param y: List of target values (size M)
    :param a: Learning rate (0-1, typically 0.01)
"""


def stochastic_gradient_descent(M, X, w, y, a):
    for feature_vector_index in range(M):
        feature_vector = X[feature_vector_index]
        predicted_scalar = dot_product(w, feature_vector)
        target_scalar = float(y[feature_vector_index])
        error = target_scalar - predicted_scalar
        w = basic_optimise(learning_rate=a*1/M, error=error, weights=w, inputs=feature_vector)
    return w
