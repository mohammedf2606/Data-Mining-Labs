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
    # return list(map(lambda w: w[1] + (learning_rate * w[1] * inputs[w[0]] * error), enumerate(weights)))
    return list(map(lambda w: w[1] + (learning_rate * w[1] * error), enumerate(weights)))


"""
Takes a vector of target values (across an entire dataset) and a vector of outputs from the model,

returns a loss, mean sum of squares.
"""
def compute_basic_loss(predicted, actual):
    assert len(predicted) == len(actual)
    running_total = 0

    for i in range(len(actual)):
        running_total += (actual[i] - predicted[i]) ** 2

    return (float) running_total / len(actual) 


"""
Optimise weights via BGD using mean sum of squares as loss function.

"""
def mss_gradient_descent_optimiser(learning_rate, weights, feature_vectors, targets, predictions):
    assert len(feature_vectors) > 0
    assert len(feature_vectors) == len(targets)
    assert len(targets) == len(predictions) 
    assert len(feature_vectors[0]) == len(weights)

    new_weights = [0 for _ in range(len(weights))]
    M = len(feature_vectors)
    for j in range(len(weights)):
        #calculate derivative of loss function (over all datapoints)
        running_total = 0
        for i in range(M):
            running_total += feature_vectors[i][j] * (targets[i] - predictions[i]) 

        derivative_j = (-2/(float)number_of_predictions) * running_total
        new_weights[j] = weights[j] - (learning_rate * derivative_j)
        #work out new weights by xj - (learning_rate * D)
    return new_weights

"""
Basic batch gradient descent for a linear regression model using mean sum of squares as loss function 
"""
def batch_gradient_descent(M, X, w, y, a):
    predicted = []

    for feature_vector_index in range(M):
        feature_vector = X[feature_vector_index]
        predicted_scalar = dot_product(w, feature_vector)
        predicted.append(predicted_scalar)

    
    return mss_gradient_descent_optimiser(a, w, X, y, predicted)

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

